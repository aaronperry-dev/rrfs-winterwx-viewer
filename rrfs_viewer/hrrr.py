"""HRRR (High-Resolution Rapid Refresh) winter fields on the same display raster and 3-hourly frames as the RRFS products.

Source: the NOAA open-data bucket ``noaa-hrrr-bdp-pds``, one ``hrrr.YYYYMMDD/conus/hrrr.tCCz.wrf{sfc,prs}fFF.grib2`` file per forecast hour (surface and pressure-level
files) with a sidecar ``.idx``, fetched by byte range exactly like the RRFS files.  The 00, 06, 12 and 18 UTC cycles run to F048 and are the ones that match the RRFS cycles.

HRRR is ONE deterministic run: there is no ensemble here, so these products are single-source (stored in the single-source slot, shown as "HRRR").  Field names
and level descriptors follow the real inventories (hrrr.t00z.wrfsfcf06 / wrfprsf45 ``.idx``): ASNOW / APCP / WEASD / FRZR / FROZR accumulations "0-h hour acc fcst", SNOD,
CSNOW / CICEP / CFRZR / CRAIN, CPOFP, MSLMA, "HGT:0C isotherm", "LAYTH:261 K level - 256 K level", and TMP / RH / VVEL on 25 hPa levels.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from . import grib, winter
from .catalog import STORE
from .config import COBB_UPPER, FILE_HPRS, FILE_HSFC, LIFT_LEVELS, T_LEVELS, frames
from .pipeline import Acc, Fetcher, Req, evaluate, IN_PER_M, KG_TO_IN, COBB_RH_LEVELS
from .s3 import S3Source

log = logging.getLogger(__name__)
HRRR_BUCKET = "noaa-hrrr-bdp-pds"
HRRR_MAX_FHR = 48                                   # the 00/06/12/18 UTC cycles
HRRR_KEYS = tuple(p.key for p in STORE if p.single)
LAYTH_LEVEL = "261 K level - 256 K level"
HRRR_PRS_T = tuple(sorted(set(T_LEVELS) | set(COBB_UPPER), reverse=True))        # 1000 ... 300 hPa (13 levels), temperature and humidity
RD = 287.04


def omega_to_w(omega_pa_s: np.ndarray, t_k: np.ndarray, p_hpa: float) -> np.ndarray:
    """Geometric vertical velocity (m/s, up positive) from pressure velocity omega (Pa/s, positive down): w = -omega / (rho g), ideal gas, hydrostatic."""
    rho = p_hpa * 100.0 / (RD * t_k)
    return (-omega_pa_s / (rho * 9.80665)).astype(np.float32)


def interp_levels(values: dict[int, np.ndarray], levels_from, levels_to) -> np.ndarray:
    """Linear interpolation in pressure of fields given on ``levels_from`` (hPa) to every pressure in ``levels_to``; (len(levels_to), ny, nx)."""
    lf = sorted(levels_from)
    stack = np.stack([values[p] for p in lf])
    out = []
    for p in levels_to:
        if p <= lf[0]:
            out.append(stack[0])
        elif p >= lf[-1]:
            out.append(stack[-1])
        else:
            i = int(np.searchsorted(lf, p))
            lo, hi = lf[i - 1], lf[i]
            f = (p - lo) / (hi - lo)
            out.append(stack[i - 1] * (1 - f) + stack[i] * f)
    return np.stack(out)


def _plan(f: Fetcher, src: S3Source, cycle: datetime, fr: int, with_prs: bool):
    acc = {n: Acc(src, "hrrr", cycle, FILE_HSFC, v) for n, v in (("asnow", "ASNOW"), ("apcp", "APCP"), ("frzr", "FRZR"), ("frozr", "FROZR"), ("weasd", "WEASD"))}
    plans = {("tot", n): a.total(fr) for n, a in acc.items()}
    plans["asnow1"], plans["apcp1"] = acc["asnow"].inc1(fr), acc["apcp"].inc1(fr)
    R = {
        "snod": Req(FILE_HSFC, fr, "SNOD", "surface", None),
        "refc": Req(FILE_HSFC, fr, "REFC", "~entire atmosphere", None),
        "sn": Req(FILE_HSFC, fr, "CSNOW", "surface", None), "ip": Req(FILE_HSFC, fr, "CICEP", "surface", None),
        "zr": Req(FILE_HSFC, fr, "CFRZR", "surface", None), "rn": Req(FILE_HSFC, fr, "CRAIN", "surface", None),
        "cpofp": Req(FILE_HSFC, fr, "CPOFP", "surface", None),
        "t2": Req(FILE_HSFC, fr, "TMP", "2 m above ground", None), "td2": Req(FILE_HSFC, fr, "DPT", "2 m above ground", None),
        "rh2": Req(FILE_HSFC, fr, "RH", "2 m above ground", None), "ps": Req(FILE_HSFC, fr, "PRES", "surface", None),
        "gust": Req(FILE_HSFC, fr, "GUST", "surface", None), "vis": Req(FILE_HSFC, fr, "VIS", "surface", None),
        "mslp": Req(FILE_HSFC, fr, "MSLMA", "mean sea level", None),
        "frzlvl": Req(FILE_HSFC, fr, "HGT", "0C isotherm", None),
        "layth": Req(FILE_HSFC, fr, "LAYTH", LAYTH_LEVEL, None),
    }
    if with_prs:
        for p in HRRR_PRS_T:
            R[("T", p)] = Req(FILE_HPRS, fr, "TMP", f"{p} mb", None)
            R[("RH", p)] = Req(FILE_HPRS, fr, "RH", f"{p} mb", None)
        for p in LIFT_LEVELS:
            R[("W", p)] = Req(FILE_HPRS, fr, "VVEL", f"{p} mb", None)
    reqs = list(R.values()) + [r for p in plans.values() if p for r in p[0]]
    return R, plans, reqs


def process_hrrr(src: S3Source, cycle: datetime, rg: grib.Regrid, max_fhr: int, progress=None, with_prs: bool = True) -> tuple[dict[str, np.ndarray], dict]:
    """All HRRR products on the display raster, ``{product: (n_frames, ny, nx)}`` (NaN = missing, and for every frame after F048)."""
    F = frames(max_fhr)
    ny, nx = rg.shape
    out = {k: np.full((len(F), ny, nx), np.nan, np.float32) for k in HRRR_KEYS}
    f = Fetcher(src, cycle, "hrrr", rg)
    # snow depth at the start of the run: the reference for the depth change
    snod0 = None
    r0 = Req(FILE_HSFC, 0, "SNOD", "surface", None)
    got0 = f.decode(f.bulk([r0]))
    snod0 = got0.get(r0)
    frames_done = 0
    for k, fr in enumerate(F):
        if fr > HRRR_MAX_FHR or src.locate(FILE_HSFC, "hrrr", cycle, fr) is None:
            continue
        R, plans, reqs = _plan(f, src, cycle, fr, with_prs and src.locate(FILE_HPRS, "hrrr", cycle, fr) is not None)
        A = f.decode(f.bulk(reqs))
        g = lambda key: A.get(R[key]) if key in R else None
        tot = lambda n: evaluate(plans[("tot", n)], A)
        asnow = tot("asnow")
        if asnow is not None:
            out["hr_snow"][k] = asnow * IN_PER_M
        a1 = evaluate(plans["asnow1"], A)
        if a1 is not None:
            out["hr_snow_1h"][k] = a1 * IN_PER_M
        for n, key in (("apcp", "hr_qpf"), ("frzr", "hr_frzr"), ("frozr", "hr_sleet")):
            v = tot(n)
            if v is not None:
                out[key][k] = v * KG_TO_IN
        q1 = evaluate(plans["apcp1"], A)
        if q1 is not None:
            out["hr_qpf_1h"][k] = q1 * KG_TO_IN
        weasd = tot("weasd")
        if asnow is not None and weasd is not None:
            out["hr_slr"][k] = np.where(weasd >= 0.5, np.clip(asnow * 1000.0 / np.maximum(weasd, 1e-6), 0, 40), np.nan)
        snod = g("snod")
        if snod is not None:
            out["hr_snod"][k] = snod * IN_PER_M
            if snod0 is not None:
                out["hr_snod_chg"][k] = (snod - snod0) * IN_PER_M
        refl = None if g("refc") is None else winter.clean_dbz(g("refc"), rg.idx >= 0)
        if refl is not None:
            out["hr_refc"][k] = refl
            fl = [g("sn"), g("ip"), g("zr")]
            if all(a is not None for a in fl):
                out["hr_ptype"][k] = winter.refl_ptype_code(refl, winter.ptype_class_flags(*fl))
        if g("cpofp") is not None:
            out["hr_cpofp"][k] = np.where(g("cpofp") >= 0, np.clip(g("cpofp"), 0, 100), np.nan)
        t2, td2, ps = g("t2"), g("td2"), g("ps")
        if t2 is not None:
            out["hr_t2"][k] = (t2 - 273.15) * 1.8 + 32
            if td2 is not None and ps is not None:
                tw = winter.wet_bulb_c(t2 - 273.15, td2 - 273.15, ps / 100.0)
                out["hr_tw2"][k] = tw * 1.8 + 32
        if g("gust") is not None:
            out["hr_gust"][k] = g("gust") * winter.MPH_PER_MS
        if g("vis") is not None:
            out["hr_vis"][k] = np.clip(g("vis") * winter.MI_PER_M, 0, 12)
        if g("mslp") is not None:
            out["hr_mslp"][k] = g("mslp") / 100.0
        if g("frzlvl") is not None:
            out["hr_frzlvl"][k] = g("frzlvl") / 304.8
        if g("layth") is not None:
            out["hr_dgz"][k] = np.where(g("layth") > 0, g("layth") / 304.8, np.nan)
        # ---- pressure levels: the DGZ with HRRR's own lift, and the Cobb ratio
        if with_prs and all(g(("T", p)) is not None and g(("RH", p)) is not None for p in HRRR_PRS_T) and all(g(("W", p)) is not None for p in LIFT_LEVELS) and ps is not None:
            psh = ps / 100.0
            Tc = {p: g(("T", p)) for p in HRRR_PRS_T}
            Tl = interp_levels(Tc, HRRR_PRS_T, LIFT_LEVELS)
            w = np.stack([omega_to_w(g(("W", p)), Tl[i], float(p)) for i, p in enumerate(LIFT_LEVELS)])
            Tk = np.stack([Tc[p] for p in T_LEVELS])
            rh = np.stack([g(("RH", p)) for p in T_LEVELS])
            d, sat, _ = winter.dgz_metrics(Tk - 273.15, rh, T_LEVELS, psh, units="kft", t_sfc_c=None if t2 is None else t2 - 273.15, rh_sfc=g("rh2"))
            ct = np.stack([Tc[p] for p in COBB_RH_LEVELS]) - 273.15
            cr = np.stack([g(("RH", p)) for p in COBB_RH_LEVELS])
            out["hr_slr_cobb"][k] = winter.slr_cobb(ct, cr, COBB_RH_LEVELS, psh, w_ms=w)
            w_dgz, om_dgz = winter.dgz_ascent(ct, COBB_RH_LEVELS, psh, w)
            out["hr_dgz_omega"][k] = np.clip(om_dgz, 0.0, None)
            out["hr_dgz_lift"][k] = winter.dgz_with_lift(sat, w_dgz)
        frames_done += 1
        if progress:
            progress("hrrr", fr)
    info = {"missing": f.missing, "hints": f.hints, "bytes": src.bytes_downloaded, "requests": src.requests, "frames": frames_done}
    return out, info
