"""Winter-weather diagnostics on the display grid (pure numpy; arrays are (ny, nx) or (levels, ny, nx)).

Sources
-------
* Kuchera SLR (Kuchera 2004; see Veals et al. 2025, eq. 2): ``12 + 2 (271.16 - Tmax)`` for Tmax > 271.16 K,
  ``12 + (271.16 - Tmax)`` otherwise, Tmax = warmest temperature from the surface to 500 hPa.
* Max-T-aloft SLR (Veals et al. 2025, eq. 1, fitted to Alaska data): 5th-degree polynomial in Tmax (degC), Tmax = maximum
  temperature between 610 m AGL (~ps-70 hPa here) and 400 hPa (we have 500 hPa as the top level).
* Thickness ratio (CMC operational snow-depth guidance, "Murphy factor"): ``10 x (-0.1057 x Z1000-700[dam] + 30.6)`` replaces the
  standard 10:1 when the 850-700 hPa thickness is above 154 dam (https://meteocentre.com/numerical-weather-prediction/cmc-map/sigwx-winter/tsh_e_doc.html).
* NBM thickness ratio: ``-0.16559 x Z850-700[m] + 263.35`` (NBM v5.0 "Configuration and Technical Details of Winter Suite", VLab;
  the document does not state the thickness unit, metres is the only reading that gives sensible ratios).  The NBM-style blend is
  the 50/50 mean of the max-T-aloft and thickness ratios, which is what NBM uses for the CMC ensembles (no vertical velocity, so no Cobb).
* Byun et al. (2008, Weather and Forecasting 23, 644-658): logistic ratio ``a / (1 + exp((T - b) / c))``, T in degC, coefficients by
  3-h precipitation amount; ``a`` is the cold (dry-snow) limit, ``b`` the temperature where the ratio is ``a``/2.  Fitted to Korean
  data, so it never exceeds ~19:1: a wet-snow-sensitive lower bound, not a cold-powder estimate.
* Regional climatology (Baxter, Graves & Moore 2005, Weather and Forecasting 20, 729-744, Table 2): CWA mean ratios from 1971-2000
  COOP data, spread over the map by inverse-distance weighting.  A reference, not a forecast; there is no seasonal cycle.
* Cobb & Waldstreicher (2005) as implemented in the National Blend of Models (Cobb 2024 NBM v4.1 slides; Cobb SREF write-up for the
  2011 temperature-ratio table): every 25 hPa layer from 925 to 300 hPa contributes ``SR(T)`` weighted by ``sqrt(VV) x RH^2 / 6400``;
  the ratio is the weighted mean.  RRFS has no pressure-level omega, so ``VV`` is taken as NBM's floor value of 1 in every layer, which
  is also how NBM treats levels where vertical motion is missing: the lift term drops out and the humidity term remains.
* Cobb melt (Leone, Cobb & Rudack 2023, NBM v4.2): potential melt of falling snow ``[(0.5 Tw[F] - 15.5) / D]^1.3`` inch of liquid per
  hour, ratio multiplied by ``(QPF - melt) / QPF`` and set to zero when melt exceeds QPF.  ``D`` is 15 in the NBM-team deck (marked
  the final choice) and 10 in a later talk by Cobb; see ``MELT_DENOM``.
* Dendritic growth zone (DGZ): the layer between the -12 and -18 degC isotherms (Cobb & Waldstreicher 2005; NWS TA-1604).
* NWS wind chill: https://www.weather.gov/safety/cold-wind-chill-chart
* NWS blizzard criteria: sustained wind or frequent gusts >= 35 mph, falling/blowing snow reducing visibility to
  <= 1/4 mile, for >= 3 hours (e.g. https://www.weather.gov/btv/wwa_reference).
"""
from __future__ import annotations

import numpy as np

KUCHERA_PIVOT_K = 271.16
SLR_MIN, SLR_MAX = 3.0, 40.0                 # clip: ratios outside 3:1..40:1 are not meaningful for a model column statistic
MM_PER_IN = 25.4
MI_PER_M = 1.0 / 1609.344
MPH_PER_MS = 2.2369362921


# --------------------------------------------------------------------------- snow-to-liquid ratios
def slr_kuchera(tmax_k):
    t = np.asarray(tmax_k, np.float64)
    slr = np.where(t > KUCHERA_PIVOT_K, 12.0 + 2.0 * (KUCHERA_PIVOT_K - t), 12.0 + (KUCHERA_PIVOT_K - t))
    return np.clip(slr, SLR_MIN, SLR_MAX).astype(np.float32)


_MAXT_COEF = (4.5e-6, 4.432e-4, 0.0130903, 0.0585968, -1.8150809, 5.9805722)   # T^5 .. T^0


def slr_max_t_aloft(tmax_c):
    t = np.clip(np.asarray(tmax_c, np.float64), -35.0, 5.0)       # keep the polynomial inside its fitted range
    slr = np.polyval(_MAXT_COEF, t)
    return np.clip(slr, SLR_MIN, SLR_MAX).astype(np.float32)


def tmax_column_k(t2_k, t_k, levels, ps_hpa, p_top=500):
    """Warmest temperature (K) from the surface (2 m) up to ``p_top`` hPa; levels below ground are ignored."""
    p = np.asarray(levels, np.float32).reshape(-1, 1, 1)
    ok = (p <= ps_hpa[None]) & (p >= p_top)
    t = np.where(ok, t_k, np.nan)
    with np.errstate(all="ignore"):
        top = np.nanmax(np.concatenate([t, t2_k[None]], axis=0), axis=0)
    return top


def tmax_aloft_c(t2_k, t_k, levels, ps_hpa, p_top=500, agl_hpa=70.0):
    """Max temperature (degC) between ~610 m AGL (ps - ``agl_hpa``) and ``p_top``; falls back to 2 m where the layer is empty."""
    p = np.asarray(levels, np.float32).reshape(-1, 1, 1)
    ok = (p <= ps_hpa[None] - agl_hpa) & (p >= p_top)
    t = np.where(ok, t_k, np.nan)
    with np.errstate(all="ignore"):
        m = np.nanmax(t, axis=0) if len(levels) else np.full_like(t2_k, np.nan)
    return (np.where(np.isfinite(m), m, t2_k) - 273.15).astype(np.float32)


def warm_nose_c(t_k, levels, ps_hpa, p_lo=700, p_hi=925):
    """Warmest temperature (degC) in the 925-700 hPa layer (elevated warm layer / 'warm nose'); below-ground levels ignored."""
    p = np.asarray(levels, np.float32).reshape(-1, 1, 1)
    ok = (p <= ps_hpa[None]) & (p >= p_lo) & (p <= p_hi)
    t = np.where(ok, t_k, np.nan)
    with np.errstate(all="ignore"):
        return (np.nanmax(t, axis=0) - 273.15).astype(np.float32)


# --------------------------------------------------------------------------- layer thickness and the CMC thickness ratio
RD_OVER_G = 287.04 / 9.80665                    # m K-1


def layer_thickness_m(t_k, levels, p_bot, p_top):
    """Hypsometric thickness (m) between two of the pressure ``levels`` (descending hPa) from layer temperatures.

    Dry-air approximation (virtual temperature ignored, a few metres low); used only when height fields are not available.
    Levels below ground are *not* masked, as for any standard-level thickness chart.
    """
    p = [float(x) for x in levels]
    idx = [i for i, x in enumerate(p) if p_top <= x <= p_bot]
    dz = np.zeros(np.shape(t_k)[1:], np.float64)
    for i, j in zip(idx[:-1], idx[1:]):
        dz += 0.5 * (np.asarray(t_k[i], np.float64) + t_k[j]) * np.log(p[i] / p[j])
    return (RD_OVER_G * dz).astype(np.float32)


CMC_CRITICAL_850_700_M = 1540.0


def slr_thickness_cmc(dz_1000_700_m, dz_850_700_m):
    """10:1 normally; the Murphy relation (a ratio below 10) where the 850-700 hPa thickness exceeds 1540 m."""
    murphy = 10.0 * (-0.1057 * (np.asarray(dz_1000_700_m, np.float64) / 10.0) + 30.6)
    slr = np.where(np.asarray(dz_850_700_m, np.float64) > CMC_CRITICAL_850_700_M, np.minimum(murphy, 10.0), 10.0)
    return np.clip(slr, SLR_MIN, SLR_MAX).astype(np.float32)


NBM_THICK_SLOPE, NBM_THICK_INTERCEPT = -0.16559, 263.35


def slr_thickness_nbm(dz_850_700_m):
    """NBM linear 850-700 hPa thickness ratio (clipped 3:1..40:1; 1540 m -> 8.3, 1500 m -> 15.0, 1400 m -> 31.5)."""
    slr = NBM_THICK_SLOPE * np.asarray(dz_850_700_m, np.float64) + NBM_THICK_INTERCEPT
    return np.clip(slr, SLR_MIN, SLR_MAX).astype(np.float32)


def slr_blend(*ratios):
    """Equal-weight mean of ratio fields (NaN where any input is NaN)."""
    return np.mean([np.asarray(r, np.float64) for r in ratios], axis=0).astype(np.float32)


# Byun et al. 2008, Table of coefficients by 3-h precipitation amount [mm]: (upper bound, a, b, c)
_BYUN = ((2.0, 18.8, 0.0811, 0.6508), (3.0, 16.1, 0.2182, 0.5373), (4.0, 14.9, 0.2295, 0.5174),
         (5.0, 13.2, 0.2678, 0.5074), (np.inf, 11.9, 0.1524, 0.5174))


def slr_byun(t_c, precip_mm_3h):
    """Byun et al. logistic ratio from the surface air temperature (degC) and the 3-hour precipitation (mm).

    Amounts below 1 mm per 3 h have no coefficients in the paper, which used the 1-2 mm set for them; so do we.
    """
    t = np.asarray(t_c, np.float64)
    pr = np.asarray(precip_mm_3h, np.float64)
    a = np.full(t.shape, _BYUN[-1][1])
    b = np.full(t.shape, _BYUN[-1][2])
    c = np.full(t.shape, _BYUN[-1][3])
    for hi, ca, cb, cc in reversed(_BYUN[:-1]):
        m = pr < hi
        a, b, c = np.where(m, ca, a), np.where(m, cb, b), np.where(m, cc, c)
    with np.errstate(over="ignore"):
        sr = a / (1.0 + np.exp(np.clip((t - b) / c, -60.0, 60.0)))
    return sr.astype(np.float32)


# Baxter et al. 2005, Table 2: mean SLR of the NWS county warning areas in and around New England (lat, lon of the forecast office, mean).
BAXTER_CWA = (
    ("Caribou, ME", 46.87, -68.02, 12.3),
    ("Burlington, VT", 44.47, -73.15, 13.4),
    ("Gray/Portland, ME", 43.89, -70.25, 11.6),
    ("Albany, NY", 42.69, -73.83, 12.2),
    ("Binghamton, NY", 42.21, -75.98, 13.0),
    ("Boston/Taunton, MA", 41.91, -71.07, 10.7),
    ("Upton, NY", 40.87, -72.86, 10.6),
)


def slr_baxter(lat, lon, power: float = 2.0, floor_km: float = 60.0):
    """Climatological mean ratio at each point: inverse-distance weighting (``power``) of the CWA means in ``BAXTER_CWA``.

    Always inside the range of the inputs (10.6 - 13.4); the distance floor keeps a point at an office from collapsing onto it.
    """
    la = np.radians(np.asarray(lat, np.float64))
    lo = np.radians(np.asarray(lon, np.float64))
    num = np.zeros(la.shape)
    den = np.zeros(la.shape)
    for _, slat, slon, val in BAXTER_CWA:
        dx = (lo - np.radians(slon)) * np.cos(0.5 * (la + np.radians(slat)))
        dy = la - np.radians(slat)
        d = np.maximum(6371.0 * np.hypot(dx, dy), floor_km)
        w = d ** -power
        num += w * val
        den += w
    return (num / den).astype(np.float32)


# Cobb layer snow ratio as a function of layer temperature [degC]: the 2011 version of the Cobb-Waldstreicher curve (the 2005 version
# peaked at 50:1 in a narrow -16 degC spike; the 2011 curve is broader and peaks at 26:1).  Linear between the points.
COBB_T_C = np.array([-300, -30, -28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 1, 300], float)
COBB_SR = np.array([7.2, 7.2, 6.8, 7.0, 8.8, 12.0, 18.0, 23.0, 26.0, 22.5, 17.5, 12.0, 9.5, 9.0, 8.5, 7.0, 3.0, 0.0, 0.0], float)
COBB_P_BOT, COBB_P_TOP, COBB_P_STEP = 925, 300, 25          # NBM integrates 925-300 hPa on a 25 hPa grid
COBB_MAX = 30.0                                             # cap used for the SREF Cobb product (the curve itself tops out at 26)


def cobb_layer_sr(t_c):
    return np.interp(t_c, COBB_T_C, COBB_SR)


COBB_GRID = tuple(range(COBB_P_BOT, COBB_P_TOP - 1, -COBB_P_STEP))            # 925, 900, ..., 300 hPa (26 levels)
UBAR_S_PER_PA_S = 10.0                                                       # 1 Pa/s = 10 microbar/s
G0, RD = 9.80665, 287.05


def vv_ubar_per_s(w_ms, t_c, p_hpa):
    """Cobb's upward vertical velocity ``VV`` [microbar/s, at least 1] from geometric ``w`` [m/s, up] with ``omega = -rho g w``.

    The unit is an assumption: the NBM slides give only the floor "1" (downward motion and missing data count as 1), and microbar/s is
    the omega unit of the original Cobb & Waldstreicher / BUFKIT work; 1 microbar/s is about 1.3 cm/s of ascent near 700 hPa.
    """
    rho = 100.0 * p_hpa / (RD * (np.asarray(t_c, np.float64) + 273.15))
    vv = UBAR_S_PER_PA_S * rho * G0 * np.asarray(w_ms, np.float64)
    return np.maximum(np.where(np.isfinite(vv), vv, 1.0), 1.0)


def slr_cobb(t_c, rh, levels_hpa, ps_hpa, w_ms=None):
    """Cloud-layer Cobb ratio.  Without ``w_ms`` the vertical-velocity term is fixed at 1 (see module docstring).

    ``t_c`` (degC) and ``rh`` (% , as the model gives it) are ``(nlev, ny, nx)`` on ``levels_hpa``.  They are interpolated linearly in
    ln p to a 25 hPa grid (as NBM does for data on standard levels); layers below the ground, or outside the span of the data, are left
    out, and a layer whose lower bracketing level is underground takes the upper one.  NaN where no layer has a usable T and RH.

    ``w_ms`` (optional): geometric vertical velocity [m/s, positive up] already on the 25 hPa grid ``COBB_GRID``, shape
    ``(26, ny, nx)``.  Each layer is then weighted by ``sqrt(VV)`` (``vv_ubar_per_s``) as well; NaN or downward ``w`` gives the floor of 1.
    """
    lev = np.asarray(levels_hpa, float)
    order = np.argsort(-lev)                                   # bottom (high p) to top
    lev, t_c, rh = lev[order], np.asarray(t_c)[order].astype(np.float64), np.asarray(rh)[order].astype(np.float64)
    ps = np.asarray(ps_hpa, np.float64)
    ok = lev[:, None, None] <= ps[None] + 1e-6                 # level above the ground
    t_c, rh = np.where(ok, t_c, np.nan), np.where(ok, rh, np.nan)
    num = np.zeros(ps.shape)
    den = np.zeros(ps.shape)
    for pt in range(COBB_P_BOT, COBB_P_TOP - 1, -COBB_P_STEP):
        if pt > lev[0] + 1e-6 or pt < lev[-1] - 1e-6:
            continue                                           # outside the data: no extrapolation
        i = int(np.searchsorted(-lev, -pt, side="left"))       # first level with p <= pt
        lo, hi = (i, i) if abs(lev[i] - pt) < 1e-6 else (i - 1, i)
        w = 0.0 if lo == hi else np.log(pt / lev[lo]) / np.log(lev[hi] / lev[lo])
        def at(a):
            a_lo, a_hi = a[lo], a[hi]
            return np.where(np.isnan(a_lo), a_hi, np.where(np.isnan(a_hi), a_lo, a_lo + w * (a_hi - a_lo)))
        t, r = at(t_c), at(rh)
        good = np.isfinite(t) & np.isfinite(r) & (pt <= ps + 1e-6)
        wt = np.where(good, np.clip(r, 0.0, 100.0) ** 2 / 6400.0, 0.0)         # RHw
        if w_ms is not None:
            wt = wt * np.sqrt(vv_ubar_per_s(w_ms[(COBB_P_BOT - pt) // COBB_P_STEP], np.where(good, t, 0.0), pt))     # x VVw
        num += np.where(good, cobb_layer_sr(np.where(good, t, 0.0)), 0.0) * wt
        den += wt
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return np.minimum(out, COBB_MAX).astype(np.float32)


def wet_bulb_c(t_c, td_c, p_hpa):
    """Wet-bulb temperature from air and dew-point temperature (degC) and pressure, by Newton iteration on the psychrometric equation."""
    t = np.asarray(t_c, np.float64)
    td = np.minimum(np.asarray(td_c, np.float64), t)
    p = np.asarray(p_hpa, np.float64)
    es = lambda x: 6.112 * np.exp(17.67 * x / (x + 243.5))
    e = es(td)
    tw = 0.5 * (t + td)
    for _ in range(8):
        g = 6.62e-4 * (1.0 + 1.15e-3 * tw) * p                      # psychrometric "constant" times p [hPa / degC]
        f = es(tw) - g * (t - tw) - e
        df = es(tw) * 17.67 * 243.5 / (tw + 243.5) ** 2 + g
        tw = tw - f / df
    return np.minimum(tw, t).astype(np.float32)


MELT_DENOM = 15.0         # 15 in the NBM-team deck (Leone et al. 2023, "final selection for NBM V4.2"); 10 in Cobb's March 2024 slides
MELT_ONSET_F, MELT_CAP_F = 31.0, 40.0                      # zero at Tw = 31 F; the rate stops growing at 40 F


def melt_rate_in_per_hr(tw_f, denom: float | None = None):
    """Potential melt of falling snow [in of liquid per hour]: ``((0.5 Tw - 15.5) / D) ** 1.3`` (0.01 at 32 F, 0.05 at 34 F, 0.15 at 38 F, 0.21 at 40 F for D = 15)."""
    d = MELT_DENOM if denom is None else denom
    tw = np.clip(np.asarray(tw_f, np.float64), MELT_ONSET_F, MELT_CAP_F)
    return ((0.5 * tw - 15.5) / d) ** 1.3


def melt_factor(tw_f, snow_in_per_hr, denom: float | None = None):
    """Fraction of the snow that survives melting: ``(QPF - melt) / QPF``, 0 when the melt is at least the snow rate, 1 where no snow falls.

    Both the temperature and the rate are for the same hour; ``snow_in_per_hr`` is the snow-type liquid of that hour in inches.
    """
    q = np.asarray(snow_in_per_hr, np.float64)
    melt = melt_rate_in_per_hr(tw_f, denom)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.clip((q - melt) / q, 0.0, 1.0)
    return np.where(q > 0, f, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- humidity over ice, DGZ
def rh_ice(rh_w, t_c):
    """RH (%) with respect to ice from RH w.r.t. water and temperature (degC); Magnus-type saturation vapour pressures."""
    t = np.asarray(t_c, np.float64)
    ew = 6.112 * np.exp(17.62 * t / (243.12 + t))
    ei = 6.112 * np.exp(22.46 * t / (272.62 + t))
    return (np.asarray(rh_w, np.float64) * ew / ei).astype(np.float32)


DGZ_WARM_C, DGZ_COLD_C = -12.0, -18.0
M_PER_KFT = 304.8
DGZ_SAT_RHI = 90.0             # layer-mean RH w.r.t. ice (%) that counts as moist enough: the NWS MDL / NBM Cobb-method moisture criterion (about 75-78 % RH w.r.t. water at -12..-18 C)
DGZ_LIFT_MS = 0.10             # upward motion (m/s) that counts as "lift present": Korolev & Mazin (2003) need about 10 cm/s to keep a cloud ice-supersaturated for dendrites


def dgz_metrics(t_c, rh_w, levels, ps_hpa, units: str = "hpa", sat_rhi: float = DGZ_SAT_RHI, t_sfc_c=None, rh_sfc=None):
    """Dendritic growth zone from the level temperatures (``t_c``: (L, ny, nx) degC, ``levels`` descending hPa).

    Linear interpolation of temperature between adjacent levels gives, per layer, the fraction lying between -12 and -18 degC.
    Returns three (ny, nx) arrays:

    * ``depth``  - DGZ depth
    * ``sat``    - DGZ depth that is moist enough for dendrites: layer-mean RH with respect to ice of at least ``sat_rhi`` (90 %, the NWS MDL / NBM Cobb criterion)
    * ``rhi``    - pressure-weighted mean RH w.r.t. ice inside the DGZ, % (NaN where there is no DGZ)

    ``units="hpa"`` measures depths in hPa; ``units="kft"`` in thousands of feet of geometric height, the unit a forecaster reads off a sounding
    (hypsometric thickness of each layer from its mean temperature, dry air).  ``rh_w`` is (L, ny, nx) with NaN where RH was not fetched;
    unsaturated/unknown layers do not count towards ``sat``.

    ``t_sfc_c`` / ``rh_sfc`` (optional, 2 m values): the ground is then the bottom of the profile where it cuts between two levels, so a DGZ that sits in the
    lowest part of the column (shallow cold air, high terrain) is not lost with the part of the layer below the first level above ground.
    """
    if units not in ("hpa", "kft"):
        raise ValueError(units)
    p = np.asarray(levels, np.float32)
    ps = np.asarray(ps_hpa, np.float32)
    above = p.reshape(-1, 1, 1) <= ps[None]
    T = np.where(above, t_c, np.nan)
    depth = np.zeros(T.shape[1:], np.float32)
    sat = np.zeros_like(depth)
    rhsum = np.zeros_like(depth)
    wsum = np.zeros_like(depth)
    ts = None if t_sfc_c is None else np.asarray(t_sfc_c, np.float32)
    rs = None if rh_sfc is None else np.asarray(rh_sfc, np.float32)
    for i in range(len(p) - 1):
        ta, tb, ra_in, pa = T[i], T[i + 1], rh_w[i], np.where(above[i], p[i], ps)
        if ts is not None:                                                    # the ground cuts this layer: the surface is its bottom
            cut = ~above[i] & above[i + 1] & (ps > p[i + 1]) & np.isfinite(ts)
            ta = np.where(cut, ts, ta)
            ra_in = np.where(cut, np.nan if rs is None else rs, np.where(above[i], rh_w[i], np.nan))
            pa = np.where(above[i] | cut, pa, p[i + 1])
        dp = pa - p[i + 1]
        d = tb - ta
        with np.errstate(all="ignore"):
            t1, t2 = (DGZ_WARM_C - ta) / d, (DGZ_COLD_C - ta) / d
            f = np.clip(np.minimum(np.maximum(t1, t2), 1.0) - np.maximum(np.minimum(t1, t2), 0.0), 0.0, 1.0)
        flat = np.abs(d) < 1e-6
        f = np.where(flat, ((ta <= DGZ_WARM_C) & (ta >= DGZ_COLD_C)).astype(np.float32), f)
        f = np.where(np.isnan(ta) | np.isnan(tb), 0.0, f).astype(np.float32)
        if units == "kft":                                                    # geometric thickness of the layer, from its own mean temperature
            with np.errstate(all="ignore"):
                dz = RD_OVER_G * (0.5 * (ta + tb) + 273.15) * np.log(pa / p[i + 1]) / M_PER_KFT
            size = np.where(f > 0, dz, 0.0)
        else:
            size = dp
        depth += f * size
        ra, rb = rh_ice(ra_in, ta), rh_ice(rh_w[i + 1], tb)
        cnt = np.isfinite(ra).astype(np.float32) + np.isfinite(rb)
        known = cnt > 0
        rl = np.where(known, (np.nan_to_num(ra) + np.nan_to_num(rb)) / np.maximum(cnt, 1), 0.0)
        sat += np.where(known & (rl >= sat_rhi), f * size, 0.0)
        rhsum += f * dp * rl
        wsum += np.where(known, f * dp, 0.0)
    with np.errstate(all="ignore"):
        mean_rhi = np.where(wsum > 5.0, rhsum / wsum, np.nan).astype(np.float32)
    return depth, sat, mean_rhi


def dgz_ascent(t_c, levels_hpa, ps_hpa, w_ms):
    """Mean ascent inside the dendritic growth zone, from geometric vertical velocity on the 25 hPa grid ``COBB_GRID``.

    ``t_c`` (degC) is ``(nlev, ny, nx)`` on ``levels_hpa`` (interpolated linearly in ln p to the grid, nothing extrapolated); ``w_ms`` is DZDT in m/s,
    positive up, ``(26, ny, nx)`` on the grid.  Every grid level that lies above the ground and between -12 and -18 degC counts.  Returns two (ny, nx) arrays,
    NaN where there is no DGZ level or no velocity:

    * ``w``      - mean geometric vertical velocity over those levels, m/s (negative = sinking, kept so that the mean is honest)
    * ``omega``  - the same as a pressure vertical velocity, ``-omega = rho g w`` with the density of each level (ideal gas, hydrostatic), microbar/s,
                   positive = ascent (1 Pa/s = 10 microbar/s)
    """
    lev = np.asarray(levels_hpa, float)
    order = np.argsort(-lev)
    lev, t = lev[order], np.asarray(t_c, np.float64)[order]
    ps = np.asarray(ps_hpa, np.float64)
    t = np.where(lev[:, None, None] <= ps[None] + 1e-6, t, np.nan)
    wsum, osum, n = np.zeros(ps.shape), np.zeros(ps.shape), np.zeros(ps.shape)
    for pt in COBB_GRID:
        if pt > lev[0] + 1e-6 or pt < lev[-1] - 1e-6:
            continue
        i = int(np.searchsorted(-lev, -pt, side="left"))
        lo, hi = (i, i) if abs(lev[i] - pt) < 1e-6 else (i - 1, i)
        f = 0.0 if lo == hi else np.log(pt / lev[lo]) / np.log(lev[hi] / lev[lo])
        a, b = t[lo], t[hi]
        tl = np.where(np.isnan(a), b, np.where(np.isnan(b), a, a + f * (b - a)))
        w = np.asarray(w_ms[(COBB_P_BOT - pt) // COBB_P_STEP], np.float64)
        good = np.isfinite(tl) & np.isfinite(w) & (pt <= ps + 1e-6) & (tl <= DGZ_WARM_C) & (tl >= DGZ_COLD_C)
        rho = 100.0 * pt / (RD * (np.where(good, tl, 0.0) + 273.15))
        wsum += np.where(good, w, 0.0)
        osum += np.where(good, UBAR_S_PER_PA_S * rho * G0 * w, 0.0)
        n += good
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(n > 0, wsum / n, np.nan).astype(np.float32), np.where(n > 0, osum / n, np.nan).astype(np.float32))


def dgz_with_lift(sat_depth, uvv_ms, threshold: float = DGZ_LIFT_MS):
    """Ice-saturated DGZ depth where the column also has ascent: the three ingredients of dendrite growth together.

    ``uvv_ms`` is the ascent to test against ``threshold``: the mean vertical velocity inside the DGZ from :func:`dgz_ascent` where the deterministic run's
    DZDT is available, otherwise the hourly-maximum upward motion anywhere in the column (a generous proxy: ascent exists somewhere above the point, not
    necessarily inside the DGZ).  NaN where the depth or the motion is unknown.
    """
    s, w = np.asarray(sat_depth, np.float32), np.asarray(uvv_ms, np.float32)
    return np.where(np.isfinite(s) & np.isfinite(w), np.where(w >= threshold, s, 0.0), np.nan).astype(np.float32)


# --------------------------------------------------------------------------- wind chill, blizzard
def wind_chill_f(t_f, wind_mph):
    """NWS wind chill (degF) where T <= 50 F and wind > 3 mph, otherwise the air temperature itself."""
    t = np.asarray(t_f, np.float64)
    v = np.maximum(np.asarray(wind_mph, np.float64), 0.0)
    v016 = v ** 0.16
    wc = 35.74 + 0.6215 * t - 35.75 * v016 + 0.4275 * t * v016
    return np.where((t <= 50.0) & (v > 3.0), wc, t).astype(np.float32)


def blizzard_now(vis_mi, gust_mph, wind_mph, frozen):
    """NWS blizzard conditions at one instant: wind or gusts >= 35 mph, visibility <= 1/4 mile, falling frozen precipitation."""
    with np.errstate(invalid="ignore"):
        wind_ok = (gust_mph >= 35.0) | (wind_mph >= 35.0)
        return wind_ok & (vis_mi <= 0.25) & (frozen > 0)


class BlizzardClock:
    """Longest run of consecutive blizzard-criteria samples, in hours (samples are ``step`` hours apart).

    Two consecutive qualifying samples span ``step`` hours, so with a 3-hourly step the 3-hour NWS duration is met by
    two samples in a row.  ``hours`` = step * (longest_run - 1) for runs of at least one sample (0 otherwise).
    """

    def __init__(self, shape, step: int = 3):
        self.run = np.zeros(shape, np.int16)
        self.best = np.zeros(shape, np.int16)
        self.step = step

    def update(self, cond: np.ndarray) -> np.ndarray:
        self.run = np.where(cond, self.run + 1, 0).astype(np.int16)
        self.best = np.maximum(self.best, self.run)
        return (self.step * np.maximum(self.best - 1, 0)).astype(np.float32)


def precip_fractions(csnow, cicep, cfrzr):
    """Exclusive precipitation-type weights (rain-or-none, snow, sleet, freezing rain) from the categorical flags.

    Priority: freezing rain > sleet > snow, so a column flagged for several types is not double counted.
    """
    z = lambda a: np.nan_to_num(a, nan=0.0) > 0.5
    zr = z(cfrzr)
    ip = z(cicep) & ~zr
    sn = z(csnow) & ~zr & ~ip
    return sn.astype(np.float32), ip.astype(np.float32), zr.astype(np.float32)


# ---------------------------------------------------------------------------- radar reflectivity masked by precipitation type
REFL_FLOOR, REFL_TOP = -20.0, 81.0                  # stored range of reflectivity (dBZ): anything below 0 is "no echo"


def clean_dbz(dbz: np.ndarray, footprint: np.ndarray | None = None) -> np.ndarray:
    """Model reflectivity limited to the stored range; negative values (the model's no-echo fill) become the floor, which also compresses well.

    ``footprint``: display cells that lie inside the model domain.  A missing value there (GRIB bitmaps often mask "no echo") is no echo too: left as a hole it
    would stop the page from interpolating across the edge of every echo and make the field look blocky.  A hole outside the domain stays a hole.
    """
    d = np.asarray(dbz, np.float32)
    if footprint is not None:
        d = np.where(footprint & ~np.isfinite(d), REFL_FLOOR, d)
    return np.where(np.isfinite(d), np.clip(np.where(d < 0, REFL_FLOOR, d), REFL_FLOOR, REFL_TOP), np.nan).astype(np.float32)


PTYPE_STEP = 1.25                                    # dBZ per step of the stored echo in the type code (so a 5 dBZ colour edge is every 4th step)


def refl_ptype_code(dbz: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """One number per cell for the page: ``class * 64 + q``, q = 1 + round(dBZ / 1.25) for dBZ >= 0 (1 = 0 dBZ ... 62 = 76 dBZ), 0 = no echo.

    ``cls``: 0 rain, 1 snow, 2 sleet, 3 freezing rain.  NaN where either input is missing.  The echo keeps 1.25 dBZ resolution so the page can interpolate
    it like any other field (only the class is a label); the highest code, 254, stays below the "missing" code 255.
    """
    d = np.asarray(dbz, np.float32)
    q = np.where(np.nan_to_num(d, nan=-1.0) >= 0, np.clip(np.rint(np.nan_to_num(d, nan=0.0) / PTYPE_STEP) + 1, 1, 62), 0)
    code = np.where(q > 0, np.asarray(cls, np.float32) * 64.0 + q, 0.0)
    return np.where(np.isfinite(d) & np.isfinite(cls), code, np.nan).astype(np.float32)


def ptype_class_flags(csnow, cicep, cfrzr) -> np.ndarray:
    """Member precipitation type from the categorical flags (freezing rain > sleet > snow, otherwise rain): 0 rain, 1 snow, 2 sleet, 3 freezing rain."""
    sn, ip, zr = precip_fractions(csnow, cicep, cfrzr)
    return (zr * 3.0 + (1 - zr) * (ip * 2.0 + (1 - ip) * sn)).astype(np.float32)


def ptype_class_fractions(crain, csnow, cicep, cfrzr) -> np.ndarray:
    """Type with the largest weight in a mean (REFS) file; a tie goes to the more wintry type.  Where the weights are all 0 or missing: rain."""
    st = np.stack([np.nan_to_num(np.asarray(a, np.float32), nan=0.0) for a in (crain, csnow, cicep, cfrzr)])
    cls = 3 - np.argmax(st[::-1], axis=0)
    return np.where(st.max(axis=0) > 0, cls, 0).astype(np.float32)


# ============================================================================== winter diagnostics added for the "group A" build
# Everything below is new; nothing above this line was changed.  Profiles are handled as "nodes": the surface (2 m temperature at the model terrain)
# followed by the pressure levels that lie above the ground, lowest first.  Levels at or below the surface pressure are left out (NaN).
RD = 287.04                                     # J kg-1 K-1
GRAV = 9.80665
EARTH_OMEGA = 7.2921e-5
M_PER_FT = 0.3048


def td_from_rh_c(t_c, rh):
    """Dew point (degC) from air temperature (degC) and relative humidity (%) with respect to water (Magnus)."""
    t = np.asarray(t_c, np.float64)
    g = np.log(np.clip(np.asarray(rh, np.float64), 1.0, 100.0) / 100.0) + 17.62 * t / (243.12 + t)
    return (243.12 * g / (17.62 - g)).astype(np.float32)


def profile_nodes(t_k: np.ndarray, levels, t2_k, ps_hpa, terrain_m, rh=None, rh2=None):
    """Surface + above-ground pressure levels as stacked arrays, lowest node first.

    ``t_k``: (n_levels, ny, nx) temperature on ``levels`` (hPa, any order); ``rh``: the same for relative humidity (optional).
    Returns ``dict(p, t, z[, rh])``, each (n_nodes, ny, nx): pressure (hPa), temperature (K), height (m MSL; hypsometric from the terrain height, the
    layer-mean temperature stands in for virtual temperature).  A level that is not above the ground at a point is NaN there.
    """
    order = np.argsort(-np.asarray(levels, float))
    lv = [float(levels[i]) for i in order]
    ps = np.asarray(ps_hpa, np.float64)
    P = [ps]
    T = [np.asarray(t2_k, np.float64)]
    Z = [np.asarray(terrain_m, np.float64)]
    H = None if rh is None else [None if rh2 is None else np.asarray(rh2, np.float64)]
    prev_p, prev_t, prev_z = P[0].copy(), T[0].copy(), Z[0].copy()
    for j, i in enumerate(order):
        p, t = lv[j], np.asarray(t_k[i], np.float64)
        ok = (p < prev_p - 0.5) & np.isfinite(t)
        tm = 0.5 * (prev_t + np.where(ok, t, prev_t))
        z = prev_z + RD * tm / GRAV * np.log(prev_p / p)
        P.append(np.where(ok, p, np.nan))
        T.append(np.where(ok, t, np.nan))
        Z.append(np.where(ok, z, np.nan))
        if H is not None:
            H.append(np.where(ok, np.asarray(rh[i], np.float64), np.nan))
        prev_p, prev_t, prev_z = np.where(ok, p, prev_p), np.where(ok, t, prev_t), np.where(ok, z, prev_z)
    out = dict(p=np.stack(P).astype(np.float32), t=np.stack(T).astype(np.float32), z=np.stack(Z).astype(np.float32))
    if H is not None:
        out["rh"] = np.stack([np.full(ps.shape, np.nan) if h is None else h for h in H]).astype(np.float32)
    return out


def first_crossing_up(v: np.ndarray, z: np.ndarray, thresh: float = 0.0):
    """Lowest height (searching up the nodes) where ``v`` goes from above ``thresh`` to at-or-below it, by linear interpolation; NaN where there is none."""
    n = v.shape[0]
    found = np.zeros(v.shape[1:], bool)
    out = np.full(v.shape[1:], np.nan, np.float64)
    for i in range(n - 1):
        a, b = v[i].astype(np.float64), v[i + 1].astype(np.float64)
        hit = ~found & np.isfinite(a) & np.isfinite(b) & (a > thresh) & (b <= thresh)
        if hit.any():
            f = (a - thresh) / np.where(hit, a - b, 1.0)
            out = np.where(hit, z[i] + f * (z[i + 1] - z[i]), out)
            found |= hit
    return out.astype(np.float32)


def level_heights(v_c: np.ndarray, z_m: np.ndarray, cap_agl_m: float = 3000.0):
    """``(level MSL [m], level above ground [m])`` of the 0 degC crossing of the profile ``v_c`` (nodes, lowest first).

    Searching up from the ground: a profile that is at or below 0 at the ground has its level at the ground (AGL 0: the cold air reaches the surface);
    one that is above 0 at the ground has its level at the first crossing above it; one that never crosses (all warm) has no level (NaN MSL, AGL = cap).
    """
    ground = z_m[0].astype(np.float64)
    cross = first_crossing_up(v_c, z_m, 0.0).astype(np.float64)
    cold_sfc = np.asarray(v_c[0], np.float64) <= 0.0
    lvl = np.where(cold_sfc, ground, cross)
    agl = np.where(cold_sfc, 0.0, np.where(np.isfinite(cross), np.minimum(cross - ground, cap_agl_m), cap_agl_m))
    bad = ~np.isfinite(v_c[0])
    return np.where(bad, np.nan, lvl).astype(np.float32), np.where(bad, np.nan, agl).astype(np.float32)


def inversion_strength_c(t_k: np.ndarray, p_hpa: np.ndarray, p_top: float = 850.0):
    """Largest temperature excess (degC) of any node between the ground and ``p_top`` over the surface temperature, 0 if none is warmer."""
    t0 = t_k[0].astype(np.float64)
    best = np.zeros(t0.shape)
    for i in range(1, t_k.shape[0]):
        ok = np.isfinite(t_k[i]) & (p_hpa[i] >= p_top)
        best = np.where(ok, np.maximum(best, t_k[i] - t0), best)
    return np.where(np.isfinite(t0), best, np.nan).astype(np.float32)


# ------------------------------------------------------------------------- precipitation type from the profile (Bourgouin energy method)
BOURG_SUB = 5                         # sub-layers per layer: a layer that crosses 0 degC is split so its warm and cold parts are weighed separately
BOURG_P_TOP = 250.0                   # warm layers above this (stratosphere) do not count (as in UPP)


def _bourgouin_sublayers(p_hpa: np.ndarray, t_k: np.ndarray):
    """Yield ``(energy, warm)`` for each sub-layer from the ground up (float32 arrays; energy 0 and warm False where there is no data), one at a time so no profile-sized stack is built."""
    n = p_hpa.shape[0]
    for i in range(n - 1):
        p1, p2 = p_hpa[i].astype(np.float64), p_hpa[i + 1].astype(np.float64)
        t1, t2 = t_k[i].astype(np.float64), t_k[i + 1].astype(np.float64)
        ok = np.isfinite(p1) & np.isfinite(p2) & np.isfinite(t1) & np.isfinite(t2)
        lp1, lp2 = np.log(p1), np.log(np.where(ok, p2, p1))
        dlp = (lp1 - lp2) / BOURG_SUB
        for s in range(BOURG_SUB):
            ta = t1 + (t2 - t1) * (s / BOURG_SUB)
            tb = t1 + (t2 - t1) * ((s + 1) / BOURG_SUB)
            tm = 0.5 * (ta + tb)
            pm = np.exp(lp1 - (s + 0.5) * dlp)
            ok_s = ok & (pm > BOURG_P_TOP)
            yield np.where(ok_s, RD * tm * dlp * np.log(tm / 273.15), 0.0), (tm >= 273.15) & ok_s


def bourgouin_energies(p_hpa: np.ndarray, t_k: np.ndarray):
    """``(surfw, areape, areane)`` in J/kg from the nodes (lowest first), as the NCEP UPP routine CALWXT_BOURG defines them.

    Layer energy = Rd * Tm * ln(p1/p2) * ln(Tm / 273.15): positive for a layer warmer than 0 degC, negative for a colder one.
    ``surfw``: warm layers touching the ground (the unbroken warm run from the surface); ``areape``: other (elevated) warm layers;
    ``areane``: the cold layers below the highest warm layer (the refreezing energy), as a positive number.
    Two passes over the sub-layers (the second needs the height of the highest warm one), each holding only one sub-layer at a time.
    """
    shape = p_hpa.shape[1:]
    surfw = np.zeros(shape)
    areape = np.zeros(shape)
    top_warm = np.full(shape, -1)
    in_sfc = None
    for k, (e, w) in enumerate(_bourgouin_sublayers(p_hpa, t_k)):
        if in_sfc is None:
            in_sfc = w.copy()
        in_sfc = in_sfc & w                                  # the surface-based warm run ends at the first cold sub-layer
        surfw += np.where(w & in_sfc, e, 0.0)
        areape += np.where(w & ~in_sfc, e, 0.0)
        top_warm = np.where(w, k, top_warm)
    areane = np.zeros(shape)
    for k, (e, w) in enumerate(_bourgouin_sublayers(p_hpa, t_k)):
        areane += np.where(~w & (k < top_warm) & (e != 0.0), -e, 0.0)
    return surfw.astype(np.float32), areape.astype(np.float32), areane.astype(np.float32)


def bourgouin_fractions(surfw, areape, areane, t_sfc_k):
    """Precipitation type from the energies, as fractions ``(snow, rain, ice pellets, freezing rain)`` that sum to 1.

    The UPP routine breaks its 50/50 transition zones with a random number; here each zone gives half to each type, so the result is a probability:
    areape < 2:  surfw < 5.6 snow, > 13.2 rain, between: half and half.  Otherwise areane > 66 + 0.66 areape: ice pellets (rain if surfw > 13.2, half and
    half between 5.6 and 13.2); areane < 46 + 0.66 areape: freezing rain if the surface is below freezing, else rain; between the two bounds: half of each.
    """
    sw, pe, ne = (np.asarray(a, np.float64) for a in (surfw, areape, areane))
    sub = np.asarray(t_sfc_k, np.float64) < 273.15
    z = np.zeros(sw.shape)
    # no elevated warm layer: snow or rain by the surface-based melting energy
    sn_a = np.where(sw < 5.6, 1.0, np.where(sw > 13.2, 0.0, 0.5))
    rn_a = 1.0 - sn_a
    # elevated warm layer
    ip_zone_ip = np.where(sw < 5.6, 1.0, np.where(sw > 13.2, 0.0, 0.5))            # ice pellets that survive to the ground, else rain
    zr_zone_zr = np.where(sub, 1.0, 0.0)                                           # freezing rain needs a sub-freezing surface, else rain
    hi, lo = ne > 66.0 + 0.66 * pe, ne < 46.0 + 0.66 * pe
    ip_b = np.where(hi, ip_zone_ip, np.where(lo, 0.0, 0.5 * ip_zone_ip))
    zr_b = np.where(hi, 0.0, np.where(lo, zr_zone_zr, 0.5 * zr_zone_zr))
    rn_b = 1.0 - ip_b - zr_b
    no_warm = pe < 2.0
    sn = np.where(no_warm, sn_a, z)
    rn = np.where(no_warm, rn_a, rn_b)
    ip = np.where(no_warm, z, ip_b)
    zr = np.where(no_warm, z, zr_b)
    bad = ~(np.isfinite(sw) & np.isfinite(pe) & np.isfinite(ne) & np.isfinite(sub))
    return tuple(np.where(bad, np.nan, a).astype(np.float32) for a in (sn, rn, ip, zr))


# ------------------------------------------------------------------------- dynamics on the display raster
def lcc_rotation_deg(lon_deg, lon0=-70.6, phi1=42.0, phi2=46.0):
    """Angle (degrees) by which the display raster's +y axis is turned from true north at each longitude: n * (lon - lon0) for a two-parallel Lambert conformal."""
    p1, p2 = np.radians(phi1), np.radians(phi2)
    n = np.log(np.cos(p1) / np.cos(p2)) / np.log(np.tan(np.pi / 4 + p2 / 2) / np.tan(np.pi / 4 + p1 / 2))
    return n * (np.asarray(lon_deg, np.float64) - lon0)


def rotate_to_grid(u, v, lon_deg):
    """Earth-relative wind components -> components along the raster's +x / +y axes."""
    a = np.radians(lcc_rotation_deg(lon_deg))
    ca, sa = np.cos(a), np.sin(a)
    return (u * ca - v * sa).astype(np.float32), (u * sa + v * ca).astype(np.float32)


def smooth(a, sigma_cells: float = 2.0):
    """Gaussian smoothing of a raster (3 km cells: sigma 2 = 6 km); NaN-safe by filling with the local mean first."""
    from scipy import ndimage
    a = np.asarray(a, np.float64)
    bad = ~np.isfinite(a)
    if bad.any():
        a = np.where(bad, np.nanmean(a) if np.isfinite(a).any() else 0.0, a)
    out = ndimage.gaussian_filter(a, sigma_cells, mode="nearest")
    return np.where(bad, np.nan, out).astype(np.float32)


def theta_k(t_k, p_hpa):
    return np.asarray(t_k, np.float64) * (1000.0 / np.asarray(p_hpa, np.float64)) ** 0.2857


def frontogenesis_k100km3h(theta, ug, vg, dx_m: float = 3000.0, sigma_cells: float = 4.0):
    """Petterssen 2-D frontogenesis of potential temperature, in K per 100 km per 3 h: ``F = -(1/|grad th|) (th_x^2 u_x + th_x th_y (v_x + u_y) + th_y^2 v_y)``.

    ``ug`` / ``vg`` are the wind components along the raster axes (:func:`rotate_to_grid`).  Fields are smoothed first (``sigma_cells`` 3 km cells, 12 km):
    a 3 km derivative field is mostly noise, the banding signal lives at 25-100 km.  Positive = the thermal gradient is sharpening.
    """
    th, u, v = (smooth(a, sigma_cells).astype(np.float64) for a in (theta, ug, vg))
    ty, tx = np.gradient(th, dx_m)
    uy, ux = np.gradient(u, dx_m)
    vy, vx = np.gradient(v, dx_m)
    mag = np.hypot(tx, ty)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = -(tx * tx * ux + tx * ty * (vx + uy) + ty * ty * vy) / mag
    f = np.where(mag > 1e-7, f, 0.0)                      # no thermal gradient: nothing to sharpen
    return (f * 1e5 * 10800.0).astype(np.float32)


def saturation_theta_e_k(t_k, p_hpa):
    """Saturation equivalent potential temperature (K), Bolton (1980) with the mixing ratio of saturation at (T, p)."""
    t = np.asarray(t_k, np.float64)
    p = np.asarray(p_hpa, np.float64)
    tc = t - 273.15
    es = 6.112 * np.exp(17.67 * tc / (tc + 243.5))
    rs = 0.622 * es / np.maximum(p - es, 1.0)              # kg/kg
    th = t * (1000.0 / p) ** (0.2854 * (1.0 - 0.28 * rs))
    return th * np.exp((3.376 / t - 0.00254) * rs * 1000.0 * (1.0 + 0.81 * rs))


def mpv_sat_pvu(t_lo_k, t_hi_k, u_lo, v_lo, u_hi, v_hi, p_lo_hpa, p_hi_hpa, lat_deg, dx_m: float = 3000.0, sigma_cells: float = 4.0):
    """Saturated moist potential vorticity (MPV*) of the layer between two pressure levels, in PVU (1e-6 K m2 kg-1 s-1).

    ``MPV* = -g [ zeta_a d(theta_es)/dp + du/dp d(theta_es)/dy - dv/dp d(theta_es)/dx ]``  (Schultz & Schumacher 1999 / Bennetts & Hoskins 1979).  Negative values
    in a saturated layer mean the layer is unstable to slantwise ascent (conditional symmetric instability, CSI).  ``zeta_a`` is the mean absolute vorticity of the two
    levels, the gradients of theta_es are the mean of the two, and the vertical derivatives are layer differences.  Winds along raster axes; smoothed first.
    """
    th_lo, th_hi = saturation_theta_e_k(t_lo_k, p_lo_hpa), saturation_theta_e_k(t_hi_k, p_hi_hpa)
    dp = (p_hi_hpa - p_lo_hpa) * 100.0                                  # Pa, negative going up
    S = lambda a: smooth(a, sigma_cells).astype(np.float64)
    th_lo, th_hi, u_lo, v_lo, u_hi, v_hi = (S(a) for a in (th_lo, th_hi, u_lo, v_lo, u_hi, v_hi))
    thy_lo, thx_lo = np.gradient(th_lo, dx_m)
    thy_hi, thx_hi = np.gradient(th_hi, dx_m)
    thx, thy = 0.5 * (thx_lo + thx_hi), 0.5 * (thy_lo + thy_hi)
    zeta = 0.0
    for u, v in ((u_lo, v_lo), (u_hi, v_hi)):
        uy = np.gradient(u, dx_m)[0]
        vx = np.gradient(v, dx_m)[1]
        zeta = zeta + 0.5 * (vx - uy)
    f = 2.0 * EARTH_OMEGA * np.sin(np.radians(np.asarray(lat_deg, np.float64)))
    za = zeta + f
    dth_dp = (th_hi - th_lo) / dp
    du_dp, dv_dp = (u_hi - u_lo) / dp, (v_hi - v_lo) / dp
    mpv = -GRAV * (za * dth_dp + du_dp * thy - dv_dp * thx)
    return (mpv * 1e6).astype(np.float32)


# ------------------------------------------------------------------------- timing and joint impacts (per member, from the stored frame series)
def snow_timing(cum_in: np.ndarray, hours, step_h: int = 3, onset_in: float = 0.1, rates_in_per_h=(0.5, 1.0)):
    """Causal timing of snowfall from a (frames, ny, nx) run-total series.

    Each output is "so far" at every frame (the last frame is the answer for the whole run):
    * ``start``  forecast hour of the first frame whose total reaches ``onset_in`` (NaN until then),
    * ``end``    forecast hour of the latest frame whose 3-h increment is at least ``onset_in`` (NaN until snow has started),
    * ``hours``  for each rate: hours of the run so far spent at or above that rate (3-h increments / 3 h, counted in whole 3-h blocks).
    """
    c = np.nan_to_num(np.asarray(cum_in, np.float32), nan=0.0)
    n = c.shape[0]
    inc = np.diff(np.concatenate([np.zeros((1,) + c.shape[1:], np.float32), np.maximum.accumulate(c, axis=0)]), axis=0)
    known = np.isfinite(np.asarray(cum_in)).any(axis=(1, 2)) if n else np.array([], bool)
    start = np.full(c.shape, np.nan, np.float32)
    end = np.full(c.shape, np.nan, np.float32)
    hrs = [np.zeros(c.shape, np.float32) for _ in rates_in_per_h]
    first = np.full(c.shape[1:], np.nan, np.float32)
    last = np.full(c.shape[1:], np.nan, np.float32)
    acc = [np.zeros(c.shape[1:], np.float32) for _ in rates_in_per_h]
    for k in range(n):
        h = float(hours[k])
        run_max = c[k] if k == 0 else np.maximum(run_max, c[k])
        total = run_max
        first = np.where(np.isnan(first) & (total >= onset_in), h, first)
        last = np.where(inc[k] >= onset_in, h, last)
        for j, r in enumerate(rates_in_per_h):
            acc[j] = acc[j] + np.where(inc[k] >= r * step_h, float(step_h), 0.0)
            hrs[j][k] = acc[j]
        start[k], end[k] = first, np.where(np.isnan(first), np.nan, last)
    nodata = ~known
    for a in (start, end, *hrs):
        a[nodata] = np.nan
    return start, end, hrs


def changeover_hours(snow_frac: np.ndarray, wet: np.ndarray, hours):
    """Forecast hour of the first frame where precipitation that had been snow is no longer snow (rain, sleet or freezing rain), "so far" (NaN before it happens).

    ``snow_frac``: (frames, ny, nx) fraction of the precipitation that is snow; ``wet``: boolean, precipitation falling at that frame.
    """
    n = snow_frac.shape[0]
    out = np.full(snow_frac.shape, np.nan, np.float32)
    seen = np.zeros(snow_frac.shape[1:], bool)
    when = np.full(snow_frac.shape[1:], np.nan, np.float32)
    for k in range(n):
        sf = np.nan_to_num(snow_frac[k], nan=-1.0)
        w = wet[k] & (sf >= 0)
        flip = seen & w & (sf < 0.5) & np.isnan(when)
        when = np.where(flip, float(hours[k]), when)
        seen = seen | (w & (sf >= 0.5))
        out[k] = when
    return out


def impact_hours(flag: np.ndarray, step_h: int = 3):
    """Cumulative hours (frames, ny, nx) a boolean condition has held, counted in whole ``step_h`` blocks; NaN where the condition could not be evaluated (flag is NaN)."""
    f = np.asarray(flag, np.float32)
    return np.cumsum(np.nan_to_num(f, nan=0.0), axis=0, dtype=np.float32) * float(step_h)
