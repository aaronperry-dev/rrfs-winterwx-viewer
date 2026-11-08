"""Group-A winter diagnostics for one ensemble member: the per-frame products built from the pressure-level profile, and the series products
(timing, joint impacts) built from the stored frames.  ``pipeline.process_member`` calls these; every product here is new, none replaces another.

Profiles use the 11 pressure levels the pipeline already fetches (``T_LEVELS``) plus the 2 m values at the model terrain; the surface node is
followed by the levels that lie above the ground (:func:`winter.profile_nodes`).
"""
from __future__ import annotations

import numpy as np

from . import winter

M_PER_KFT = 304.8
FT_PER_M = 1.0 / winter.M_PER_FT
WIND_LEVELS = (925, 850, 700, 500)          # u / v fetched at these levels (hPa)


def _have(uv):
    """Both wind components present (arrays are not compared with ``None not in``: that would compare element-wise)."""
    return uv is not None and uv[0] is not None and uv[1] is not None


def _kelvin_to_f(t_k):
    return (np.asarray(t_k, np.float64) - 273.15) * 1.8 + 32.0


def frame_products(*, t_levels, Tk, RH, t2k, rh2, ps_pa, terrain_m, mslp_pa, winds, lon, lat, dgz_sat=None, dz_1000_850=None, dz_850_700=None):
    """Per-frame group-A products and the profile nodes.

    ``Tk``: (n, ny, nx) temperature on ``t_levels``; ``RH``: the same for humidity (NaN where a level was not fetched); ``rh2``: 2 m RH (or None);
    ``winds``: ``{p: (u, v)}`` earth-relative components at ``WIND_LEVELS`` (any may be missing).  Returns ``(products, nodes)``; ``nodes`` is None
    when the terrain height is missing (the level products and the profile p-type need it).
    """
    out: dict[str, np.ndarray] = {}
    ps = None if ps_pa is None else np.asarray(ps_pa, np.float64) / 100.0
    idx = {p: i for i, p in enumerate(t_levels)}
    below = lambda p: None if ps is None else (ps <= p + 0.5)            # this pressure level is not above the ground here
    nodes = None
    if t2k is not None and ps is not None and terrain_m is not None:
        nodes = winter.profile_nodes(Tk, t_levels, t2k, ps, terrain_m, rh=RH, rh2=rh2)
        tc = nodes["t"] - 273.15
        # wet bulb of each node from T and RH; a node without humidity has no wet bulb
        td = winter.td_from_rh_c(tc, nodes["rh"])
        tw = winter.wet_bulb_c(tc, td, nodes["p"])
        tw = np.where(np.isfinite(nodes["rh"]), tw, np.nan)
        if rh2 is not None:
            out["tw2"] = _kelvin_to_f(tw[0] + 273.15)
            lvl, agl = winter.level_heights(tw, nodes["z"])
            out["snowlvl"] = lvl / M_PER_KFT
            out["snowlvl_agl"] = agl * FT_PER_M
        flv, _ = winter.level_heights(tc, nodes["z"])
        out["frzlvl"] = flv / M_PER_KFT
        out["inversion"] = winter.inversion_strength_c(nodes["t"], nodes["p"])
    if mslp_pa is not None:
        out["mslp"] = np.asarray(mslp_pa, np.float32) / 100.0
    if dz_1000_850 is not None:
        out["thk_1000_850"] = np.where(below(850), np.nan, dz_1000_850 / 10.0) if ps is not None else dz_1000_850 / 10.0
    if dz_850_700 is not None:
        out["thk_850_700"] = dz_850_700 / 10.0
    for p, key in ((925, "t925"), (850, "t850")):
        if p in idx and Tk is not None:
            t = Tk[idx[p]] - 273.15
            out[key] = np.where(below(p), np.nan, t) if ps is not None else t
    for p, key in ((925, "w925"), (850, "w850")):
        uv = winds.get(p)
        if _have(uv):
            w = np.hypot(uv[0], uv[1]) * winter.MPH_PER_MS
            out[key] = np.where(below(p), np.nan, w) if ps is not None else w
    # ---- dynamics at 700 / 500 hPa on the display raster
    w7, w5 = winds.get(700), winds.get(500)
    if 700 in idx and _have(w7):
        ug, vg = winter.rotate_to_grid(w7[0], w7[1], lon)
        th = winter.theta_k(Tk[idx[700]], 700.0)
        f = winter.frontogenesis_k100km3h(th, ug, vg)
        out["fronto700"] = f
        rh7 = RH[idx[700]] if RH is not None else None
        if rh7 is not None and dgz_sat is not None:
            ok = np.isfinite(rh7) & np.isfinite(dgz_sat) & np.isfinite(f)
            band = np.where((f >= 2.0) & (rh7 >= 80.0) & (dgz_sat >= 1.0), f, 0.0)
            out["band_pot"] = np.where(ok, band, np.nan)
        if 500 in idx and _have(w5) and RH is not None and 600 in idx:
            u5, v5 = winter.rotate_to_grid(w5[0], w5[1], lon)
            mpv = winter.mpv_sat_pvu(Tk[idx[700]], Tk[idx[500]], ug, vg, u5, v5, 700.0, 500.0, lat)
            sat = (RH[idx[700]] >= 80.0) & (RH[idx[600]] >= 80.0)
            ok = np.isfinite(RH[idx[700]]) & np.isfinite(RH[idx[600]]) & np.isfinite(mpv)
            out["csi"] = np.where(ok & sat, np.clip(-mpv, 0.0, 2.5), np.nan)
    return {k: np.asarray(v, np.float32) for k, v in out.items()}, nodes


def ptype_overlap(a, b):
    """Overlap of two (snow, rain, ice pellets, freezing rain) distributions, 0..1: the probability mass they share."""
    return sum(np.minimum(x, y) for x, y in zip(a, b))


def profile_ptype(nodes, wet, model_sn_ip_zr):
    """Bourgouin type percentages and the agreement with the model's own flags, where precipitation falls (``wet``).

    Returns ``{bsnow, brain, bsleet, bfrzr, bagree}`` (percent) or {} without a profile.
    """
    if nodes is None:
        return {}
    surfw, areape, areane = winter.bourgouin_energies(nodes["p"], nodes["t"])
    sn, rn, ip, zr = winter.bourgouin_fractions(surfw, areape, areane, nodes["t"][0])
    ok = np.isfinite(sn)
    out = {}
    for key, a in (("bsnow", sn), ("brain", rn), ("bsleet", ip), ("bfrzr", zr)):
        out[key] = np.where(wet & ok, 100.0 * a, np.nan).astype(np.float32)
    if model_sn_ip_zr is not None:
        msn, mip, mzr = model_sn_ip_zr
        mrn = np.clip(1.0 - msn - mip - mzr, 0.0, 1.0)
        out["bagree"] = np.where(wet & ok, 100.0 * ptype_overlap((sn, rn, ip, zr), (msn, mrn, mip, mzr)), np.nan).astype(np.float32)
    return out


# ============================================================================== series products (after all frames of a member are in)
def series_products(out: dict[str, np.ndarray], hours) -> dict[str, np.ndarray]:
    """Timing and joint-impact products from the member's stored frames (``out`` is the member's product dict; nothing in it is modified).

    Snowfall for the timing is the multi-method mean (the model's ASNOW where that is missing); the type for the changeover is the model's own
    snow / sleet / freezing-rain / rain percentages at the frame time.
    """
    res: dict[str, np.ndarray] = {}
    n = len(hours)

    def finite(a):
        return a is not None and np.isfinite(a).any()

    base = out.get("snow_mean")
    if not finite(base):
        base = out.get("snow_model")
    if finite(base):
        cum = np.array(base, np.float32)
        if base is not out.get("snow_model") and finite(out.get("snow_model")):          # a frame without the mean: the model's total stands in
            for k in range(n):
                if not np.isfinite(cum[k]).any():
                    cum[k] = out["snow_model"][k]
        start, end, (h05, h10) = winter.snow_timing(cum, hours)
        res.update(t_start=start, t_end=end, t_rate05=h05, t_rate10=h10)
        # impacts: 3-hour increments of the multi-method total
        c = np.nan_to_num(cum, nan=0.0)
        inc = np.diff(np.concatenate([np.zeros((1,) + c.shape[1:], np.float32), np.maximum.accumulate(c, axis=0)]), axis=0)
        slr, gust = out.get("slr_kuchera"), out.get("gust")
        if finite(slr) and finite(gust):
            flag = ((inc >= 1.0) & (slr < 9.0) & (gust >= 35.0)).astype(np.float32)
            flag[~np.isfinite(slr) | ~np.isfinite(gust)] = np.nan
            res["imp_wetwind"] = winter.impact_hours(flag)
            res["imp_wetwind"][~(np.isfinite(slr) & np.isfinite(gust))] = np.nan
    ps, pi, pz, pr = (out.get(k) for k in ("psnow", "psleet", "pfrzr", "prain"))
    if all(finite(a) for a in (ps, pi, pz, pr)):
        wet = (np.nan_to_num(ps) + np.nan_to_num(pi) + np.nan_to_num(pz) + np.nan_to_num(pr)) > 0
        frac = np.where(np.isfinite(ps), ps / 100.0, np.nan)
        res["t_change"] = winter.changeover_hours(frac, wet, hours)
    fz, gust = out.get("frzr"), out.get("gust")
    if finite(fz) and finite(gust):
        cz = np.nan_to_num(fz, nan=0.0)
        incz = np.diff(np.concatenate([np.zeros((1,) + cz.shape[1:], np.float32), np.maximum.accumulate(cz, axis=0)]), axis=0)
        flag = ((incz >= 0.02) & (gust >= 30.0)).astype(np.float32)
        res["imp_ice_wind"] = winter.impact_hours(flag)
        res["imp_ice_wind"][~np.isfinite(gust)] = np.nan
    return res
