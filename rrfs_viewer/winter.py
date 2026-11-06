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


def dgz_metrics(t_c, rh_w, levels, ps_hpa):
    """Dendritic growth zone from the level temperatures (``t_c``: (L, ny, nx) degC, ``levels`` descending hPa).

    Linear interpolation of temperature between adjacent levels gives, per layer, the fraction lying between -12 and -18 degC.
    Returns three (ny, nx) arrays:

    * ``depth``  - DGZ depth, hPa
    * ``sat``    - DGZ depth that is saturated with respect to ice (layer-mean RHi >= 100 %), hPa
    * ``rhi``    - pressure-weighted mean RH w.r.t. ice inside the DGZ, % (NaN where there is no DGZ)

    ``rh_w`` is (L, ny, nx) with NaN where RH was not fetched; unsaturated/unknown layers do not count towards ``sat``.
    """
    p = np.asarray(levels, np.float32)
    above = p.reshape(-1, 1, 1) <= ps_hpa[None]
    T = np.where(above, t_c, np.nan)
    depth = np.zeros(T.shape[1:], np.float32)
    sat = np.zeros_like(depth)
    rhsum = np.zeros_like(depth)
    wsum = np.zeros_like(depth)
    for i in range(len(p) - 1):
        ta, tb, dp = T[i], T[i + 1], float(p[i] - p[i + 1])
        d = tb - ta
        with np.errstate(all="ignore"):
            t1, t2 = (DGZ_WARM_C - ta) / d, (DGZ_COLD_C - ta) / d
            f = np.clip(np.minimum(np.maximum(t1, t2), 1.0) - np.maximum(np.minimum(t1, t2), 0.0), 0.0, 1.0)
        flat = np.abs(d) < 1e-6
        f = np.where(flat, ((ta <= DGZ_WARM_C) & (ta >= DGZ_COLD_C)).astype(np.float32), f)
        f = np.where(np.isnan(ta) | np.isnan(tb), 0.0, f).astype(np.float32)
        depth += f * dp
        ra, rb = rh_ice(rh_w[i], ta), rh_ice(rh_w[i + 1], tb)
        cnt = np.isfinite(ra).astype(np.float32) + np.isfinite(rb)
        known = cnt > 0
        rl = np.where(known, (np.nan_to_num(ra) + np.nan_to_num(rb)) / np.maximum(cnt, 1), 0.0)
        sat += np.where(known & (rl >= 100.0), f * dp, 0.0)
        rhsum += f * dp * rl
        wsum += np.where(known, f * dp, 0.0)
    with np.errstate(all="ignore"):
        mean_rhi = np.where(wsum > 5.0, rhsum / wsum, np.nan).astype(np.float32)
    return depth, sat, mean_rhi


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
