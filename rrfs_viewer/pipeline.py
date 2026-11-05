"""Fetch (byte-range) -> decode -> regrid -> derive, one ensemble source at a time.

Per member and 3-hour block the pipeline downloads only the GRIB2 messages it needs:

* hourly (3 per block): 1-h QPF and the rain / snow / sleet / freezing-rain flags  -> exact precipitation-type accounting,
* at the block end: run-total ASNOW, 2 m T, 10 m wind, gust, visibility, surface pressure, max upward motion,
  and 11 pressure levels of temperature + 8 of relative humidity (Kuchera, max-T-aloft, DGZ, warm nose).

The REFS mean file is small (no profiles, no gust), so only its published fields are fetched.
"""
from __future__ import annotations

import logging
import re
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np

from . import grib, winter
from .config import FILE_2D, FILE_PRS, MEMBERS, RH_LEVELS, T_LEVELS, frames
from .s3 import S3Source

log = logging.getLogger(__name__)
Req = namedtuple("Req", "kind fhr var level info")

IN_PER_M = 39.3700787
KG_TO_IN = 1.0 / winter.MM_PER_IN


# ============================================================================== fetching
class Fetcher:
    def __init__(self, src: S3Source, cycle: datetime, source: str, regrid: grib.Regrid, threads: int = 16):
        self.src, self.cycle, self.source, self.rg, self.threads = src, cycle, source, regrid, threads
        self.missing: dict[str, int] = {}

    def _one(self, r: Req) -> bytes | None:
        loc = self.src.locate(r.kind, self.source, self.cycle, r.fhr)
        if loc is None:
            return None
        return self.src.fetch_message(loc, r.var, r.level, r.info)

    def bulk(self, reqs: list[Req]) -> dict[Req, bytes | None]:
        reqs = list(dict.fromkeys(reqs))
        with ThreadPoolExecutor(self.threads) as pool:
            return dict(zip(reqs, pool.map(self._one, reqs)))

    def decode(self, blobs: dict[Req, bytes | None]) -> dict[Req, np.ndarray | None]:
        """Decode sequentially (eccodes is not thread-safe) and map onto the display raster."""
        out = {}
        for r, data in blobs.items():
            if data is None:
                out[r] = None
                self.missing[f"{r.var}:{r.level}"] = self.missing.get(f"{r.var}:{r.level}", 0) + 1
            else:
                out[r] = self.rg(grib.decode(data)[0])
        return out


def _acc(h: int, total: bool = False) -> str:
    return rf"^0-{h} hour acc" if total else rf"^{h - 1}-{h} hour acc"


# ============================================================================== ensemble members
MEMBER_KEYS = ("qpf", "snow_model", "snow_10", "snow_kuchera", "snow_maxt", "slr_kuchera", "slr_maxt", "frzr", "sleet",
               "psnow", "psleet", "pfrzr", "prain", "warmnose", "dgz_depth", "dgz_sat", "dgz_rhi", "uvv", "gust", "wind10",
               "vis", "blizzard_now", "blizzard_dur", "t2", "windchill")


def _member_requests(f: int) -> tuple[dict, list[Req]]:
    R: dict = {}
    for h in range(f - 2, f + 1):
        R[("apcp", h)] = Req(FILE_2D, h, "APCP", "surface", _acc(h))
        for name, var in (("sn", "CSNOW"), ("ip", "CICEP"), ("zr", "CFRZR")):
            R[(name, h)] = Req(FILE_2D, h, var, "surface", None)
    R["asnow"] = Req(FILE_2D, f, "ASNOW", "surface", _acc(f, total=True))
    R["t2"] = Req(FILE_2D, f, "TMP", "2 m above ground", None)
    R["u"] = Req(FILE_2D, f, "UGRD", "10 m above ground", None)
    R["v"] = Req(FILE_2D, f, "VGRD", "10 m above ground", None)
    R["gust"] = Req(FILE_2D, f, "GUST", "surface", None)
    R["vis"] = Req(FILE_2D, f, "VIS", "surface", None)
    R["ps"] = Req(FILE_2D, f, "PRES", "surface", None)
    R["uvv"] = Req(FILE_2D, f, "MAXUVV", "100-1000 mb", None)
    for p in T_LEVELS:
        R[("T", p)] = Req(FILE_PRS, f, "TMP", f"{p} mb", None)
    for p in RH_LEVELS:
        R[("RH", p)] = Req(FILE_PRS, f, "RH", f"{p} mb", None)
    return R, list(R.values())


def process_member(src: S3Source, cycle: datetime, member: str, rg: grib.Regrid, max_fhr: int,
                   progress=None) -> tuple[dict[str, np.ndarray], dict]:
    """All member products on the display raster: ``{product: (n_frames, ny, nx) float32}`` (NaN = missing)."""
    F = frames(max_fhr)
    ny, nx = rg.shape
    nan = lambda: np.full((len(F), ny, nx), np.nan, np.float32)
    out = {k: nan() for k in MEMBER_KEYS}
    f = Fetcher(src, cycle, member, rg)

    z = lambda: np.zeros((ny, nx), np.float32)
    qpf, swe_sn, swe_ip, swe_zr = z(), z(), z(), z()           # mm
    snow10, snowk, snowm = z(), z(), z()                       # inches
    clock = winter.BlizzardClock((ny, nx), step=3)
    prev_flags = None
    broken = False                                             # a missing hourly QPF hole makes all later totals unreliable

    plan = [_member_requests(fr) for fr in F]
    pool = ThreadPoolExecutor(1)
    nxt = pool.submit(f.bulk, plan[0][1])
    for k, fr in enumerate(F):
        R, _ = plan[k]
        blobs = nxt.result()
        if k + 1 < len(F):
            nxt = pool.submit(f.bulk, plan[k + 1][1])
        A = f.decode(blobs)
        g = lambda key: A.get(R[key])
        del blobs

        # ---------------------------------------------------------------- 3-hourly (instantaneous) diagnostics
        t2k, ps_pa = g("t2"), g("ps")
        u, v = g("u"), g("v")
        wind = None if u is None or v is None else np.hypot(u, v) * winter.MPH_PER_MS
        gust = None if g("gust") is None else g("gust") * winter.MPH_PER_MS
        vis = None if g("vis") is None else np.clip(g("vis") * winter.MI_PER_M, 0, 12)
        t2f = None if t2k is None else (t2k - 273.15) * 1.8 + 32
        if t2f is not None:
            out["t2"][k] = t2f
            if wind is not None:
                out["windchill"][k] = winter.wind_chill_f(t2f, wind)
        if wind is not None:
            out["wind10"][k] = wind
        if gust is not None:
            out["gust"][k] = gust
        if vis is not None:
            out["vis"][k] = vis
        if g("uvv") is not None:
            out["uvv"][k] = np.clip(g("uvv"), 0, None)

        slr_k = slr_m = None
        have_T = [g(("T", p)) for p in T_LEVELS]
        if t2k is not None and ps_pa is not None and all(a is not None for a in have_T):
            ps = ps_pa / 100.0
            Tk = np.stack(have_T)
            slr_k = winter.slr_kuchera(winter.tmax_column_k(t2k, Tk, T_LEVELS, ps))
            slr_m = winter.slr_max_t_aloft(winter.tmax_aloft_c(t2k, Tk, T_LEVELS, ps))
            out["slr_kuchera"][k], out["slr_maxt"][k] = slr_k, slr_m
            out["warmnose"][k] = winter.warm_nose_c(Tk, T_LEVELS, ps)
            rh = np.stack([g(("RH", p)) if p in RH_LEVELS and g(("RH", p)) is not None else np.full((ny, nx), np.nan, np.float32)
                           for p in T_LEVELS])
            d, s, r = winter.dgz_metrics(Tk - 273.15, rh, T_LEVELS, ps)
            out["dgz_depth"][k], out["dgz_sat"][k], out["dgz_rhi"][k] = d, s, r

        # ---------------------------------------------------------------- hourly precipitation accounting
        for h in range(fr - 2, fr + 1):
            apcp = g(("apcp", h))
            flags = [g((n, h)) for n in ("sn", "ip", "zr")]
            if apcp is None or any(a is None for a in flags):
                broken = True
                continue
            fr_now = winter.precip_fractions(*flags)
            fr_use = fr_now if prev_flags is None else tuple(0.5 * (a + b) for a, b in zip(fr_now, prev_flags))
            prev_flags = fr_now
            mm = np.nan_to_num(apcp, nan=0.0)
            qpf += mm
            swe_sn += mm * fr_use[0]
            swe_ip += mm * fr_use[1]
            swe_zr += mm * fr_use[2]
            sn_in = mm * fr_use[0] * KG_TO_IN                               # liquid equivalent of snow, inches
            snow10 += sn_in * 10.0
            snowk += sn_in * (slr_k if slr_k is not None else np.nan)
            snowm += sn_in * (slr_m if slr_m is not None else np.nan)

        if not broken:
            out["qpf"][k] = qpf * KG_TO_IN
            out["frzr"][k] = swe_zr * KG_TO_IN
            out["sleet"][k] = swe_ip * KG_TO_IN
            out["snow_10"][k], out["snow_kuchera"][k], out["snow_maxt"][k] = snow10, snowk, snowm
        if g("asnow") is not None:
            out["snow_model"][k] = g("asnow") * IN_PER_M

        # ---------------------------------------------------------------- precipitation type at the frame time + blizzard
        fl = [g((n, fr)) for n in ("sn", "ip", "zr")]
        ap = g(("apcp", fr))
        if ap is not None and all(a is not None for a in fl):
            sn, ip, zr = winter.precip_fractions(*fl)
            wet = np.nan_to_num(ap, nan=0.0) >= 0.1
            out["psnow"][k] = 100.0 * (sn * wet)
            out["psleet"][k] = 100.0 * (ip * wet)
            out["pfrzr"][k] = 100.0 * (zr * wet)
            out["prain"][k] = 100.0 * (wet & ((sn + ip + zr) == 0))
            if vis is not None and gust is not None and wind is not None:
                cond = winter.blizzard_now(vis, gust, wind, sn + ip)
                out["blizzard_now"][k] = 100.0 * cond
                out["blizzard_dur"][k] = clock.update(cond)
        if progress:
            progress(member, fr)
    pool.shutdown()
    info = {"missing": f.missing, "bytes": src.bytes_downloaded, "requests": src.requests}
    return out, info


# ============================================================================== REFS mean
REFS_KEYS = ("qpf", "snow_model", "snow_10", "psnow", "psleet", "pfrzr", "prain", "wind10", "vis", "t2", "windchill")


def process_refs(src: S3Source, cycle: datetime, rg: grib.Regrid, max_fhr: int, progress=None):
    """Official REFS mean (``refs.*/ensprod/refs.tCCz.mean.fFF.conus.grib2``): fields that exist in the file only.

    Accumulations use the run-total record when the file has one; otherwise the 1-h records are summed over all hours.
    """
    F = frames(max_fhr)
    ny, nx = rg.shape
    out = {k: np.full((len(F), ny, nx), np.nan, np.float32) for k in REFS_KEYS}
    f = Fetcher(src, cycle, "refs", rg)
    info: dict = {"mode": {}}

    def f2(h, var, level, inf=None):
        return Req(FILE_2D, h, var, level, inf)            # the kind is irrelevant for "refs": one file per hour

    # does this file carry run totals?  (decided once, on the first frame)
    loc0 = src.locate(FILE_2D, "refs", cycle, F[0])
    from .s3 import find_entry
    totals = {v: bool(loc0 and find_entry(loc0.entries, v, "surface", _acc(F[0], True))) for v in ("APCP", "WEASD")}
    info["mode"] = {v: "total" if t else "hourly-sum" for v, t in totals.items()}

    run = {"APCP": np.zeros((ny, nx), np.float32), "WEASD": np.zeros((ny, nx), np.float32)}
    last_h = 0
    for k, fr in enumerate(F):
        reqs = {
            "t2": f2(fr, "TMP", "2 m above ground"), "wind": f2(fr, "WIND", "10 m above ground"),
            "vis": f2(fr, "VIS", "surface"), "asnow": f2(fr, "ASNOW", "surface", _acc(fr, True)),
            "sn": f2(fr, "CSNOW", "surface"), "ip": f2(fr, "CICEP", "surface"), "zr": f2(fr, "CFRZR", "surface"),
            "rn": f2(fr, "CRAIN", "surface"),
        }
        for v in ("APCP", "WEASD"):
            if totals[v]:
                reqs[v] = f2(fr, v, "surface", _acc(fr, True))
            else:
                for h in range(last_h + 1, fr + 1):
                    reqs[(v, h)] = f2(h, v, "surface", _acc(h))
        A = f.decode(f.bulk(list(reqs.values())))
        g = lambda key: A.get(reqs[key])
        for v in ("APCP", "WEASD"):
            if totals[v]:
                run[v] = g(v) if g(v) is not None else np.full((ny, nx), np.nan, np.float32)
            else:
                for h in range(last_h + 1, fr + 1):
                    a = g((v, h))
                    run[v] = run[v] + (a if a is not None else np.nan)
        last_h = fr
        out["qpf"][k] = run["APCP"] * KG_TO_IN
        out["snow_10"][k] = run["WEASD"] * KG_TO_IN * 10.0
        if g("asnow") is not None:
            out["snow_model"][k] = g("asnow") * IN_PER_M
        for key, name in (("sn", "psnow"), ("ip", "psleet"), ("zr", "pfrzr"), ("rn", "prain")):
            if g(key) is not None:
                out[name][k] = np.clip(100.0 * g(key), 0, 100)
        if g("vis") is not None:
            out["vis"][k] = np.clip(g("vis") * winter.MI_PER_M, 0, 12)
        if g("t2") is not None:
            t2f = (g("t2") - 273.15) * 1.8 + 32
            out["t2"][k] = t2f
            if g("wind") is not None:
                w = g("wind") * winter.MPH_PER_MS
                out["wind10"][k] = w
                out["windchill"][k] = winter.wind_chill_f(t2f, w)
        if progress:
            progress("refs", fr)
    info.update(missing=f.missing, bytes=src.bytes_downloaded, requests=src.requests)
    return out, info


# ============================================================================== cycle selection
def find_cycle(src: S3Source, now: datetime, max_fhr: int, lookback: int = 8) -> datetime | None:
    """Newest 00/06/12/18 UTC cycle whose *last* forecast hour is published for the REFS mean and every member."""
    now = now.replace(minute=0, second=0, microsecond=0)
    base = now - timedelta(hours=now.hour % 6)
    for i in range(lookback):
        cyc = base - timedelta(hours=6 * i)
        keys = [FILE_2D.key_for("refs", cyc, max_fhr) + ".idx"]
        keys += [FILE_PRS.key_for(m, cyc, max_fhr) + ".idx" for m in MEMBERS]
        keys += [FILE_2D.key_for(m, cyc, max_fhr) + ".idx" for m in MEMBERS]
        if all(src.exists(k) for k in keys):
            return cyc
    return None
