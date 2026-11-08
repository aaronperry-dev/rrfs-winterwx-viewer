"""Fetch (byte-range) -> decode -> regrid -> derive, one ensemble source at a time.

Per member and 3-hour block the pipeline downloads only the GRIB2 messages it needs:

* hourly (3 per block): 1-h QPF and the rain / snow / sleet / freezing-rain flags  -> exact precipitation-type accounting,
* at the block end: run-total and 1-h ASNOW, 2 m T, 10 m wind, gust, visibility, surface pressure, max upward motion,
  1000/850/700 hPa heights, and 11 pressure levels of temperature + 8 of relative humidity (Kuchera, max-T-aloft, thickness,
  DGZ, warm nose).

Accumulation windows are discovered from each file's .idx (``Acc``), so a file that carries 1-hour windows, run totals or
3-hour windows is handled without configuration.

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
from .catalog import STORE
from .config import COBB_UPPER, FILE_2D, FILE_PMMN, FILE_PRS, LIFT_LEVELS, MEMBERS, RH_LEVELS, T_LEVELS, frames
from .s3 import S3Source, find_entry

log = logging.getLogger(__name__)
Req = namedtuple("Req", "kind fhr var level info")

MAX_RANGE = 8_000_000                  # bytes: longest run of adjacent messages merged into one request
IN_PER_M = 39.3700787
KG_TO_IN = 1.0 / winter.MM_PER_IN


# ============================================================================== fetching
class Fetcher:
    def __init__(self, src: S3Source, cycle: datetime, source: str, regrid: grib.Regrid, threads: int = 16):
        self.src, self.cycle, self.source, self.rg, self.threads = src, cycle, source, regrid, threads
        self.missing: dict[str, int] = {}
        self.hints: dict[str, dict] = {}            # what the file *does* contain, for fields that were not found

    def _resolve(self, r: Req):
        """(request, located file, idx entry) - the entry is None when the field is not in the file (the hint says what is)."""
        loc = self.src.locate(r.kind, self.source, self.cycle, r.fhr)
        if loc is None:
            self.hints.setdefault(f"file:f{r.fhr:03d}", {"problem": "no .idx / file"})
            return r, None, None
        e = find_entry(loc.entries, r.var, r.level, r.info)
        if e is None:
            same_var = [x for x in loc.entries if x.var == r.var]
            self.hints.setdefault(f"{r.var}:{r.level}", {
                "wanted": r.info, "fhr": r.fhr,
                "infos_at_level": sorted({x.info for x in same_var if x.level == r.level})[:8],
                "other_levels": sorted({x.level for x in same_var})[:8]})
        return r, loc, e

    def bulk(self, reqs: list[Req]) -> dict[Req, bytes | None]:
        """Fetch the messages: wanted messages that sit back to back in a file (the four precipitation-type flags, runs of pressure levels) go in ONE range
        request, so there are fewer round trips and not one byte more than before.  Ranges are capped (``MAX_RANGE``) so a long run still downloads in parallel."""
        reqs = list(dict.fromkeys(reqs))
        out: dict[Req, bytes | None] = {r: None for r in reqs}
        with ThreadPoolExecutor(self.threads) as pool:
            groups: dict[str, list] = {}
            for r, loc, e in pool.map(self._resolve, reqs):                    # .idx files first (cached per file, one small request each)
                if e is not None:
                    groups.setdefault(loc.key, []).append((e, r))
            jobs = []                                                          # (file key, [(entry, request), ...] contiguous)
            for key, items in groups.items():
                items.sort(key=lambda t: t[0].start)
                for e, r in items:
                    cur = jobs[-1][1] if jobs and jobs[-1][0] == key else None
                    last = cur[-1][0] if cur else None
                    if cur and last.end is not None and e.start == last.end + 1 and (e.end is None or e.end - cur[0][0].start < MAX_RANGE):
                        cur.append((e, r))
                    else:
                        jobs.append((key, [(e, r)]))
            for items, data in pool.map(lambda j: (j[1], self._range(*j)), jobs):
                base = items[0][0].start
                for e, r in items:
                    msg = data[e.start - base: (e.end - base + 1) if e.end is not None else None]
                    if msg[:4] != b"GRIB":
                        raise ValueError(f"byte range of {r.var}:{r.level} did not return a GRIB message")
                    out[r] = msg
        return out

    def _range(self, key: str, items: list) -> bytes:
        first, last = items[0][0], items[-1][0]
        return self.src.get_range(key, first.start, last.end)

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


# ============================================================================== accumulation windows
_WIN = re.compile(r"^(\d+)-(\d+) hour acc")
Plan = tuple  # (list[Req], combine(list[ndarray]) -> ndarray)


class Acc:
    """Accumulation windows ("a-b hour acc") of one variable, read from the .idx files themselves."""

    def __init__(self, src: S3Source, source: str, cycle: datetime, kind, var: str, level: str = "surface"):
        self.src, self.source, self.cycle, self.kind, self.var, self.level = src, source, cycle, kind, var, level

    def windows(self, h: int) -> set[tuple[int, int]]:
        loc = self.src.locate(self.kind, self.source, self.cycle, h) if h >= 1 else None
        out = set()
        for e in (loc.entries if loc else []):
            if e.var == self.var and e.level == self.level and (m := _WIN.match(e.info)):
                out.add((int(m[1]), int(m[2])))
        return out

    def req(self, h: int, a: int, b: int) -> Req:
        return Req(self.kind, h, self.var, self.level, rf"^{a}-{b} hour acc")

    def inc1(self, h: int) -> Plan | None:
        """The accumulation in hour (h-1, h]: a 1-hour window, or the difference of two run totals."""
        w = self.windows(h)
        if (h - 1, h) in w:
            return [self.req(h, h - 1, h)], lambda a: a[0]
        if h > 1 and (0, h) in w and (0, h - 1) in self.windows(h - 1):
            return [self.req(h, 0, h), self.req(h - 1, 0, h - 1)], lambda a: np.maximum(a[0] - a[1], 0.0)
        return None

    def total(self, h: int) -> Plan | None:
        return ([self.req(h, 0, h)], lambda a: a[0]) if (0, h) in self.windows(h) else None


def evaluate(plan: Plan | None, A: dict) -> np.ndarray | None:
    if plan is None:
        return None
    vals = [A.get(r) for r in plan[0]]
    return None if any(v is None for v in vals) else plan[1](vals)


class RunTotal:
    """Run-total accumulation at each frame, from a run-total window if the file has one, otherwise by summing windows."""

    def __init__(self, acc: Acc, shape):
        self.acc, self.shape = acc, shape
        self.run, self.covered, self.dead = np.zeros(shape, np.float32), 0, False

    def plan(self, fr: int):
        t = self.acc.total(fr)
        if t is not None:
            return t[0], (lambda A, t=t: evaluate(t, A))
        if self.dead:
            return [], lambda A: None
        if (self.covered, fr) in self.acc.windows(fr):               # a window that starts where we stopped
            r = self.acc.req(fr, self.covered, fr)
            return [r], lambda A, r=r, fr=fr: self._add(A.get(r), fr)
        plans = [self.acc.inc1(h) for h in range(self.covered + 1, fr + 1)]      # hour by hour
        if any(p is None for p in plans):
            self.dead = True
            return [], lambda A: None
        return [r for p in plans for r in p[0]], lambda A, plans=plans, fr=fr: self._add(self._sum(plans, A), fr)

    @staticmethod
    def _sum(plans, A):
        parts = [evaluate(p, A) for p in plans]
        return None if any(x is None for x in parts) else np.sum(parts, axis=0)

    def _add(self, inc, fr):
        if inc is None:
            self.dead = True
            return None
        self.run = self.run + inc
        self.covered = fr
        return self.run.copy()


# ============================================================================== ensemble members
MEMBER_KEYS = tuple(p.key for p in STORE)
SNOW_METHODS = ("snow_model", "snow_12", "snow_kuchera", "snow_maxt", "snow_thick")      # the five that make the multi-method mean
EXTRA_METHODS = ("snow_cobb", "snow_cobblift", "snow_nbmthick", "snow_byun", "snow_clim")   # literature methods shown on their own
# "snow_cobb" runs without its lift term (the public RRFS member files carry no omega); "snow_cobblift" weights the layers with the
# deterministic RRFS run's DZDT (process_lift, fetched once per cycle) on each member's own temperature and humidity.  Its melt variant and the NBM blends are not in the
# dashboard: winter.py keeps melt_factor / wet_bulb_c / slr_blend with tests, see README.
COBB_T_LEVELS = tuple(sorted(set(T_LEVELS) | set(COBB_UPPER), reverse=True))
COBB_RH_LEVELS = tuple(sorted(set(RH_LEVELS) | set(COBB_UPPER), reverse=True))
# Spread = max - min over the physically based forecast methods.  Byun (fitted to Korean data, capped near 19:1) and the
# climatology (not a forecast) are shown but left out so they do not set the spread.
SPREAD_METHODS = SNOW_METHODS + ("snow_cobb", "snow_nbmthick")
Z_LEVELS = (1000, 850, 700)
REFC_LEVEL = "entire atmosphere (considered as a single layer)"          # as the real member 2dfld files spell it (tests/data)


def _method_mean(arrs: list[np.ndarray | None], min_n: int = 1) -> np.ndarray | None:
    """Mean over the methods that are available at each point: any one is enough, so a missing input shrinks the mean instead of blanking it."""
    ref = next(a for a in arrs if a is not None)
    st = np.stack([a if a is not None else np.full_like(ref, np.nan) for a in arrs])
    n = np.isfinite(st).sum(axis=0)
    mean = np.where(n > 0, np.nansum(st, axis=0) / np.maximum(n, 1), np.nan)
    return np.where(n >= min_n, mean, np.nan).astype(np.float32)


def _method_range(arrs: list[np.ndarray | None], min_n: int = 2) -> np.ndarray | None:
    """max - min over the methods that are available at each point (a range needs at least two of them)."""
    ref = next(a for a in arrs if a is not None)
    st = np.stack([a if a is not None else np.full_like(ref, np.nan) for a in arrs])
    n = np.isfinite(st).sum(axis=0)
    with np.errstate(all="ignore"):
        rng = np.nanmax(st, axis=0) - np.nanmin(st, axis=0)
    return np.where(n >= min_n, rng, np.nan).astype(np.float32)


def _plan_member(F, k, acc_p: Acc, acc_s: Acc):
    fr = F[k]
    R: dict = {}
    plans: dict = {}
    for h in range(fr - 2, fr + 1):
        plans[("apcp", h)] = acc_p.inc1(h)
        for name, var in (("sn", "CSNOW"), ("ip", "CICEP"), ("zr", "CFRZR")):
            R[(name, h)] = Req(FILE_2D, h, var, "surface", None)
    plans["asnow"], plans["asnow1"] = acc_s.total(fr), acc_s.inc1(fr)
    plans["asnow_blk"] = None
    if plans["asnow"] is None:                                 # no run total in the file: this frame's block from a window, or hour by hour
        prev = F[k - 1] if k else 0
        if (prev, fr) in acc_s.windows(fr):
            plans["asnow_blk"] = ([acc_s.req(fr, prev, fr)], lambda a: a[0])
        else:
            for h in range(prev + 1, fr + 1):
                plans[("asnow_h", h)] = acc_s.inc1(h)
    R["t2"] = Req(FILE_2D, fr, "TMP", "2 m above ground", None)
    R["u"] = Req(FILE_2D, fr, "UGRD", "10 m above ground", None)
    R["v"] = Req(FILE_2D, fr, "VGRD", "10 m above ground", None)
    R["gust"] = Req(FILE_2D, fr, "GUST", "surface", None)
    R["vis"] = Req(FILE_2D, fr, "VIS", "surface", None)
    R["ps"] = Req(FILE_2D, fr, "PRES", "surface", None)
    R["uvv"] = Req(FILE_2D, fr, "MAXUVV", "100-1000 mb", None)
    R["refc"] = Req(FILE_2D, fr, "REFC", REFC_LEVEL, None)
    for p in T_LEVELS:
        R[("T", p)] = Req(FILE_PRS, fr, "TMP", f"{p} mb", None)
    for p in COBB_RH_LEVELS:
        R[("RH", p)] = Req(FILE_PRS, fr, "RH", f"{p} mb", None)
    for p in COBB_UPPER:
        R[("T", p)] = Req(FILE_PRS, fr, "TMP", f"{p} mb", None)
    for p in Z_LEVELS:
        R[("Z", p)] = Req(FILE_PRS, fr, "HGT", f"{p} mb", None)
    reqs = list(R.values()) + [r for p in plans.values() if p for r in p[0]]
    return R, plans, reqs


# ============================================================================== deterministic lift
def grid_matches(rg: grib.Regrid, lat: np.ndarray, lon: np.ndarray, tol_deg: float = 0.06) -> bool:
    """Is this native grid the one ``rg`` was built from?  Compares the native cell each display cell points at with the display cell."""
    if lat.shape != rg.native_shape:
        return False
    ok = rg.idx >= 0
    dlat = np.abs(lat.ravel()[rg.idx[ok]] - rg.lat[ok])
    dlon = np.abs(((lon.ravel()[rg.idx[ok]] - rg.lon[ok]) + 180.0) % 360.0 - 180.0)
    return bool(ok.any() and np.percentile(dlat, 99) < tol_deg and np.percentile(dlon, 99) < tol_deg)


def process_lift(src: S3Source, cycle: datetime, rg: grib.Regrid, max_fhr: int, progress=None) -> tuple[np.ndarray | None, dict]:
    """Geometric vertical velocity ``DZDT`` [m/s, up] of the deterministic RRFS on the display raster, every 25 hPa from 925 to 300.

    Returns ``(array (n_frames, 26, ny, nx) float16, NaN where missing)`` and an info dict; the array is None when the deterministic
    file is unavailable or is not on the members' grid.  One run, so it is fetched once and shared by all members.
    """
    F = frames(max_fhr)
    ny, nx = rg.shape
    f = Fetcher(src, cycle, "det", rg)
    info: dict = {"frames_ok": 0}
    loc0 = next((l for l in (src.locate(FILE_PRS, "det", cycle, h) for h in F) if l is not None), None)
    if loc0 is None:
        info.update(bytes=src.bytes_downloaded, requests=src.requests, problem="no deterministic file for this cycle")
        return None, info
    msg = src.fetch_message(loc0, "DZDT", f"{LIFT_LEVELS[0]} mb")
    if msg is None:
        info.update(bytes=src.bytes_downloaded, requests=src.requests, problem="no DZDT in the deterministic file")
        return None, info
    _, (glat, glon) = grib.decode(msg, want_geometry=True)
    if not grid_matches(rg, glat, glon):
        info.update(bytes=src.bytes_downloaded, requests=src.requests, problem="deterministic grid differs from the members' grid")
        return None, info
    out = np.full((len(F), len(LIFT_LEVELS), ny, nx), np.nan, np.float16)
    for k, fr in enumerate(F):
        reqs = [Req(FILE_PRS, fr, "DZDT", f"{p} mb", None) for p in LIFT_LEVELS]
        src.locate(FILE_PRS, "det", cycle, fr)               # read the .idx once, before the 26 parallel range requests need it
        A = f.decode(f.bulk(reqs))
        have = [A.get(r) for r in reqs]
        if any(a is None for a in have):                    # a frame with any missing level is left empty rather than half filled
            continue
        out[k] = np.clip(np.stack(have), -20.0, 20.0)
        info["frames_ok"] += 1
        if progress:
            progress("det", fr)
    info.update(missing=f.missing, hints=f.hints, bytes=src.bytes_downloaded, requests=src.requests)
    return (out if info["frames_ok"] else None), info


# ============================================================================== quality of a member's frame
# One small integer per member and frame: how far the accumulated products of that frame can be trusted.  Nothing is withheld because of it.
Q_ESTIMATED, Q_PARTIAL, Q_INTERP, Q_EMPTY = 1, 2, 4, 8
# 1  some hours of this frame's own 3-h block were estimated (precipitation type carried over from the previous hour, or an hour filled from the run total)
# 2  lower bound: some hours are missing, so the accumulations are at least what is shown
# 4  the frame itself was missing and was interpolated between the neighbouring frames
# 8  nothing at all for this frame
MAX_GAP_FRAMES = 2                                             # longest run of empty frames that is bridged by interpolation
CUM_KEYS = tuple(p.key for p in STORE if p.kind == "cum" and not p.key.endswith("_1h"))


def fill_gaps(out: dict[str, np.ndarray], quality: list[int], max_gap: int = MAX_GAP_FRAMES, fallback: dict | None = None) -> None:
    """Bridge frames that are empty between two good ones, in every accumulated product, linearly in time.

    An accumulation only grows, and at 3-hour spacing it is smooth, so a short hole is estimated far better by its neighbours than by
    showing nothing.  Leading holes are bridged from zero (the run starts with nothing accumulated); trailing holes stay empty.
    ``fallback`` = ``{product: {frame: array}}``: lower bounds that fill the frames of a product that are still empty afterwards.
    """
    n = len(quality)
    for key in CUM_KEYS:
        a = out.get(key)
        if a is None:
            continue
        fb = (fallback or {}).get(key) or {}
        have = [bool(np.isfinite(a[k]).any()) for k in range(n)]
        k = 0
        while k < n:
            if have[k]:
                k += 1
                continue
            j = k
            while j < n and not have[j]:
                j += 1
            if j < n and j - k <= max_gap and (k > 0 or (have[j] and 0 not in fb)):      # a leading hole with a lower bound keeps that, not half the next frame
                lo = a[k - 1] if k > 0 else np.zeros_like(a[j])
                for i in range(k, j):
                    a[i] = lo + (a[j] - lo) * ((i - k + 1) / (j - k + 1))
                    quality[i] |= Q_INTERP
            k = j
        for k, arr in fb.items():
            if not np.isfinite(a[k]).any():
                a[k] = arr
                quality[k] |= Q_PARTIAL
    for k in range(n):
        if not any(np.isfinite(a[k]).any() for key in ("qpf", "snow_model", "snow_mean") if (a := out.get(key)) is not None):
            quality[k] |= Q_EMPTY


def process_member(src: S3Source, cycle: datetime, member: str, rg: grib.Regrid, max_fhr: int,
                   progress=None, lift: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """All member products on the display raster: ``{product: (n_frames, ny, nx) float32}`` (NaN = missing).

    ``lift`` is the deterministic ``DZDT`` from :func:`process_lift` (frames x 26 levels x ny x nx); without it the Cobb-with-lift
    products stay empty.
    """
    F = frames(max_fhr)
    ny, nx = rg.shape
    nan = lambda: np.full((len(F), ny, nx), np.nan, np.float32)
    out = {k: nan() for k in MEMBER_KEYS}
    f = Fetcher(src, cycle, member, rg)
    acc_p = Acc(src, member, cycle, FILE_2D, "APCP")
    acc_s = Acc(src, member, cycle, FILE_2D, "ASNOW")

    z = lambda: np.zeros((ny, nx), np.float32)
    qpf, swe_sn, swe_ip, swe_zr = z(), z(), z(), z()           # mm
    snow = {m: z() for m in SNOW_METHODS + EXTRA_METHODS if m != "snow_model"}  # inches
    clim = winter.slr_baxter(rg.lat, rg.lon)                   # regional climatological ratio: no fetch, constant in time
    clock = winter.BlizzardClock((ny, nx), step=3)
    prev_flags = None
    # Nothing here is all-or-nothing: an hour that is missing or incomplete shrinks or estimates the totals, and says so in ``quality``.
    lost_q = 0                                                 # hours whose precipitation is unknown (not in QPF)
    lost_t = 0                                                 # hours whose precipitation type is unknown (not in the type / snow totals)
    est_n = 0                                                  # hours estimated: type persisted from the previous hour, gap filled from the run total
    n_liq = n_typed = 0                                        # hours that went into QPF / into the typed totals
    last_slr: dict = {}                                        # last ratio each snow method produced, carried over a frame where it is missing
    short: set = set()                                         # snow methods that skipped some liquid because they never had a ratio yet
    seen: set = set()                                          # snow methods that have had a ratio at least once
    snow_run, snow_lost = z(), 0                               # the model's own ASNOW run total, and the hours missing from it
    quality = [0] * len(F)
    lower_bound: dict[int, np.ndarray] = {}                    # frames whose own block of ASNOW had a hole: used only if they cannot be interpolated
    hours_of = lambda k: range((F[k - 1] if k else 0) + 1, F[k] + 1)

    def block(k):
        R, plans, reqs = _plan_member(F, k, acc_p, acc_s)
        return R, plans, f.bulk(reqs)

    pool = ThreadPoolExecutor(1)
    nxt = pool.submit(block, 0)
    for k, fr in enumerate(F):
        R, plans, blobs = nxt.result()
        if k + 1 < len(F):
            nxt = pool.submit(block, k + 1)
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
        refl = None if g("refc") is None else winter.clean_dbz(g("refc"), rg.idx >= 0)
        if refl is not None:
            out["refc"][k] = refl

        slr = {m: None for m in snow}
        slr["snow_12"] = out["slr_12"][k] = np.full((ny, nx), 12.0, np.float32)
        slr["snow_clim"] = out["slr_clim"][k] = clim
        # 3-hour total precipitation (Byun's coefficient set); an hour with a hole makes the block unusable anyway
        apcps = {h: evaluate(plans[("apcp", h)], A) for h in range(fr - 2, fr + 1)}
        pr3 = None if any(a is None for a in apcps.values()) else sum(np.nan_to_num(a, nan=0.0) for a in apcps.values())
        if t2k is not None and pr3 is not None:
            slr["snow_byun"] = out["slr_byun"][k] = winter.slr_byun(t2k - 273.15, pr3)
        have_T = [g(("T", p)) for p in T_LEVELS]
        all_T = all(a is not None for a in have_T)
        if t2k is not None and ps_pa is not None and all_T:
            ps = ps_pa / 100.0
            Tk = np.stack(have_T)
            slr["snow_kuchera"] = winter.slr_kuchera(winter.tmax_column_k(t2k, Tk, T_LEVELS, ps))
            slr["snow_maxt"] = winter.slr_max_t_aloft(winter.tmax_aloft_c(t2k, Tk, T_LEVELS, ps))
            out["slr_kuchera"][k], out["slr_maxt"][k] = slr["snow_kuchera"], slr["snow_maxt"]
            out["warmnose"][k] = winter.warm_nose_c(Tk, T_LEVELS, ps)
            rh = np.stack([g(("RH", p)) if p in RH_LEVELS and g(("RH", p)) is not None else np.full((ny, nx), np.nan, np.float32)
                           for p in T_LEVELS])
            d, s_, r_ = winter.dgz_metrics(Tk - 273.15, rh, T_LEVELS, ps, units="kft")
            out["dgz_depth"][k], out["dgz_sat"][k], out["dgz_rhi"][k] = d, s_, r_
            if g("uvv") is not None:
                out["dgz_lift"][k] = winter.dgz_with_lift(s_, np.clip(g("uvv"), 0, None))
            cobb_t = [g(("T", p)) for p in COBB_RH_LEVELS]
            cobb_rh = [g(("RH", p)) for p in COBB_RH_LEVELS]
            if all(a is not None for a in cobb_t + cobb_rh):
                ct, cr = np.stack(cobb_t) - 273.15, np.stack(cobb_rh)
                slr["snow_cobb"] = out["slr_cobb"][k] = winter.slr_cobb(ct, cr, COBB_RH_LEVELS, ps)
                if lift is not None and np.isfinite(lift[k, 0]).any():
                    slr["snow_cobblift"] = out["slr_cobblift"][k] = winter.slr_cobb(ct, cr, COBB_RH_LEVELS, ps, w_ms=np.asarray(lift[k], np.float32))
        # thickness ratio: model heights when present, otherwise the hypsometric equation on the temperature levels
        Z = {p: g(("Z", p)) for p in Z_LEVELS}
        if all(a is not None for a in Z.values()):
            dz_1000_700, dz_850_700 = Z[700] - Z[1000], Z[700] - Z[850]
        elif all_T:
            Tk = np.stack(have_T)
            dz_1000_700 = winter.layer_thickness_m(Tk, T_LEVELS, 1000, 700)
            dz_850_700 = winter.layer_thickness_m(Tk, T_LEVELS, 850, 700)
        else:
            dz_1000_700 = dz_850_700 = None
        if dz_1000_700 is not None:
            slr["snow_thick"] = out["slr_thick"][k] = winter.slr_thickness_cmc(dz_1000_700, dz_850_700)
            slr["snow_nbmthick"] = out["slr_nbmthick"][k] = winter.slr_thickness_nbm(dz_850_700)

        # ---------------------------------------------------------------- hourly precipitation accounting
        for name, r in slr.items():                                         # a method without a ratio now reuses its last one
            if r is not None:
                last_slr[name] = r
                seen.add(name)
        ratio = lambda name: slr[name] if slr[name] is not None else last_slr.get(name)
        inc1 = {"qpf": None, "liq": None}
        est_before = est_n

        def add_typed(mm, fr_use):                                          # one hour (or a gap) of liquid with a precipitation-type split
            nonlocal n_typed
            n_typed += 1
            swe_sn[...] += mm * fr_use[0]
            swe_ip[...] += mm * fr_use[1]
            swe_zr[...] += mm * fr_use[2]
            sn_in = mm * fr_use[0] * KG_TO_IN                               # liquid equivalent of snow, inches
            for name in snow:
                r = ratio(name)
                if r is None:
                    short.add(name)
                else:
                    snow[name] += sn_in * r
            return sn_in

        for h in range(fr - 2, fr + 1):
            apcp = apcps[h]
            if apcp is None:
                lost_q += 1
                lost_t += 1
                continue
            mm = np.nan_to_num(apcp, nan=0.0)
            qpf += mm
            n_liq += 1
            flags = [g((n, h)) for n in ("sn", "ip", "zr")]
            if all(a is not None for a in flags):
                fr_now = winter.precip_fractions(*flags)
                fr_use = fr_now if prev_flags is None else tuple(0.5 * (a + b) for a, b in zip(fr_now, prev_flags))
                prev_flags = fr_now
            elif prev_flags is not None:
                fr_use = prev_flags                                         # type flags missing: the previous hour's type
                est_n += 1
            else:
                lost_t += 1                                                 # counted in QPF, but there is no type to split it by
                continue
            sn_in = add_typed(mm, fr_use)
            if h == fr:                                                     # the last hour of the block -> the 1-hour maps
                inc1 = {"qpf": mm * KG_TO_IN, "liq": sn_in}
        if lost_q:                                                          # hours with no precipitation at all: the run total, if the file has one, gives the amount
            t = acc_p.total(fr)
            tot = evaluate(t, f.decode(f.bulk(t[0]))) if t is not None else None
            if tot is not None:
                gap = np.maximum(np.nan_to_num(tot, nan=0.0) - qpf, 0.0)
                qpf += gap
                n_liq += lost_q
                if prev_flags is not None:                                  # type: that of the last hour with flags
                    add_typed(gap, prev_flags)
                    lost_t = max(lost_t - lost_q, 0)
                    est_n += lost_q
                lost_q = 0

        asnow = evaluate(plans["asnow"], A)
        if asnow is not None:                                               # a run total from the file is exact and repairs any earlier hole
            snow_run, snow_lost = asnow, 0
        else:
            parts = [evaluate(plans["asnow_blk"], A)]                       # this frame's block: one window ...
            if parts[0] is None:
                parts = [evaluate(plans.get(("asnow_h", h)), A) for h in hours_of(k)]       # ... or hour by hour
            got = [x for x in parts if x is not None]
            snow_lost += len(parts) - len(got)
            if got:
                snow_run = snow_run + np.sum(got, axis=0, dtype=np.float32)
                if len(got) == len(parts):
                    asnow = snow_run
                else:                                                       # a hole in this very block: keep the lower bound only if nothing better turns up
                    lower_bound[k] = snow_run
        asnow1 = evaluate(plans["asnow1"], A)
        if n_liq:
            out["qpf"][k] = qpf * KG_TO_IN
            if n_typed:
                out["frzr"][k] = swe_zr * KG_TO_IN
                out["sleet"][k] = swe_ip * KG_TO_IN
                for name in snow:
                    if name in seen:
                        out[name][k] = snow[name]
                if inc1["qpf"] is not None:
                    for name in snow:
                        if ratio(name) is not None:
                            out[name + "_1h"][k] = inc1["liq"] * ratio(name)
            if inc1["qpf"] is not None:
                out["qpf_1h"][k] = inc1["qpf"]
            liq = swe_sn * KG_TO_IN
            if asnow is not None and n_typed and not lost_t and not snow_lost:
                out["slr_model"][k] = np.where(liq >= 0.02, np.clip(asnow * IN_PER_M / np.maximum(liq, 1e-6), 0, 40), np.nan)
        if asnow is not None:
            out["snow_model"][k] = asnow * IN_PER_M
        if asnow1 is not None:
            out["snow_model_1h"][k] = asnow1 * IN_PER_M
        shown = [m for m in SNOW_METHODS if np.isfinite(out[m][k]).any()]
        if shown:
            out["snow_mean"][k] = _method_mean([out[m][k] if m in shown else None for m in SNOW_METHODS])
        spread = [m for m in SPREAD_METHODS if np.isfinite(out[m][k]).any()]
        if len(spread) > 1:
            out["snow_spread"][k] = _method_range([out[m][k] if m in spread else None for m in SPREAD_METHODS])
        one = [out[m + "_1h"][k] if np.isfinite(out[m + "_1h"][k]).any() else None for m in SNOW_METHODS]
        if any(a is not None for a in one):
            out["snow_mean_1h"][k] = _method_mean(one)
        # "estimated" marks the frames whose own block had an estimated hour (the effect on later totals is small); "lower bound" stays on while hours are missing
        quality[k] = (Q_ESTIMATED if est_n > est_before else 0) | (Q_PARTIAL if (lost_q or lost_t or (snow_lost and asnow is not None) or short & set(shown)) else 0)

        # ---------------------------------------------------------------- precipitation type at the frame time + blizzard
        fl = [g((n, fr)) for n in ("sn", "ip", "zr")]
        ap = evaluate(plans[("apcp", fr)], A)
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
        if refl is not None and all(a is not None for a in fl):                 # reflectivity coloured by this member's precipitation type
            out["refl_ptype"][k] = winter.refl_ptype_code(refl, winter.ptype_class_flags(*fl))
        if progress:
            progress(member, fr)
    pool.shutdown()
    fill_gaps(out, quality, fallback={"snow_model": lower_bound})
    info = {"missing": f.missing, "hints": f.hints, "bytes": src.bytes_downloaded, "requests": src.requests, "quality": quality}
    return out, info


# ============================================================================== REFS mean
REFS_KEYS = ("qpf", "qpf_1h", "snow_model", "snow_model_1h", "snow_12", "snow_12_1h", "psnow", "psleet", "pfrzr", "prain",
             "wind10", "vis", "t2", "windchill", "refc", "refd", "refl_ptype")


def process_refs(src: S3Source, cycle: datetime, rg: grib.Regrid, max_fhr: int, progress=None):
    """Official REFS mean (``refs.*/ensprod/refs.tCCz.mean.fFF.conus.grib2``): the fields that exist in the file only.

    Accumulation windows are discovered from each file's index (run totals, 1-hour or longer tiling windows).
    """
    F = frames(max_fhr)
    ny, nx = rg.shape
    out = {k: np.full((len(F), ny, nx), np.nan, np.float32) for k in REFS_KEYS}
    f = Fetcher(src, cycle, "refs", rg)
    shape = (ny, nx)
    accs = {v: Acc(src, "refs", cycle, FILE_2D, v) for v in ("APCP", "WEASD", "ASNOW")}
    runs = {v: RunTotal(a, shape) for v, a in accs.items()}
    info: dict = {"mode": {v: ("total" if a.total(F[0]) else "sum") for v, a in accs.items()}}

    def f2(h, var, level):
        return Req(FILE_2D, h, var, level, None)

    for k, fr in enumerate(F):
        reqs = {"t2": f2(fr, "TMP", "2 m above ground"), "wind": f2(fr, "WIND", "10 m above ground"), "vis": f2(fr, "VIS", "surface"),
                "sn": f2(fr, "CSNOW", "surface"), "ip": f2(fr, "CICEP", "surface"), "zr": f2(fr, "CFRZR", "surface"),
                "rn": f2(fr, "CRAIN", "surface")}
        tot = {v: runs[v].plan(fr) for v in runs}
        one = {v: accs[v].inc1(fr) for v in runs}
        refl_reqs = {"refc": Req(FILE_PMMN, fr, "REFC", "~entire atmosphere", None),          # the probability-matched mean file: its level wording is matched loosely
                     "refd": Req(FILE_PMMN, fr, "REFD", "~1000 m above ground", None)}
        allreq = list(reqs.values()) + list(refl_reqs.values()) + [r for p, _ in tot.values() for r in p] + [r for p in one.values() if p for r in p[0]]
        A = f.decode(f.bulk(allreq))
        g = lambda key: A.get(reqs[key])
        for key, r in refl_reqs.items():
            if A.get(r) is not None:
                out[key][k] = winter.clean_dbz(A[r], rg.idx >= 0)
        if A.get(refl_reqs["refc"]) is not None and all(g(n) is not None for n in ("sn", "ip", "zr", "rn")):
            cls = winter.ptype_class_fractions(g("rn"), g("sn"), g("ip"), g("zr"))
            out["refl_ptype"][k] = winter.refl_ptype_code(out["refc"][k], cls)
        T = {v: tot[v][1](A) for v in runs}
        I = {v: evaluate(one[v], A) for v in runs}
        for key, v, mul in (("qpf", "APCP", KG_TO_IN), ("snow_12", "WEASD", KG_TO_IN * 12.0), ("snow_model", "ASNOW", IN_PER_M)):
            if T[v] is not None:
                out[key][k] = T[v] * mul
            if I[v] is not None:
                out[key + "_1h"][k] = I[v] * mul
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
    info.update(missing=f.missing, hints=f.hints, bytes=src.bytes_downloaded, requests=src.requests)
    return out, info


# ============================================================================== cycle selection
def complete_sources(src: S3Source, cyc: datetime, max_fhr: int) -> list[str]:
    """The sources (``refs`` and the members) whose *last* forecast hour is published for this cycle."""
    ok = []
    if src.exists(FILE_2D.key_for("refs", cyc, max_fhr) + ".idx"):
        ok.append("refs")
    for m in MEMBERS:
        if all(src.exists(k.key_for(m, cyc, max_fhr) + ".idx") for k in (FILE_PRS, FILE_2D)):
            ok.append(m)
    return ok


def find_cycle(src: S3Source, now: datetime, max_fhr: int, lookback: int = 8, min_members: int | None = None) -> datetime | None:
    """Newest 00/06/12/18 UTC cycle whose *last* forecast hour is published for the REFS mean and every member.

    With ``min_members`` the rule relaxes: a cycle qualifies when that many members are complete (the REFS mean or not), so one member
    that stopped publishing cannot freeze the site on an old cycle.  The caller then builds from the sources that are complete.
    """
    now = now.replace(minute=0, second=0, microsecond=0)
    base = now - timedelta(hours=now.hour % 6)
    for i in range(lookback):
        cyc = base - timedelta(hours=6 * i)
        if min_members is None:
            keys = [FILE_2D.key_for("refs", cyc, max_fhr) + ".idx"]
            keys += [FILE_PRS.key_for(m, cyc, max_fhr) + ".idx" for m in MEMBERS]
            keys += [FILE_2D.key_for(m, cyc, max_fhr) + ".idx" for m in MEMBERS]
            if all(src.exists(k) for k in keys):
                return cyc
        elif sum(s != "refs" for s in complete_sources(src, cyc, max_fhr)) >= min_members:
            return cyc
    return None
