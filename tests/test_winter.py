import numpy as np
import pytest

from rrfs_viewer import winter as W


def a(*v):
    return np.array(v, np.float32)


def test_kuchera_pivot_and_slopes():
    assert W.slr_kuchera(a(271.16))[0] == pytest.approx(12.0)
    assert W.slr_kuchera(a(268.16))[0] == pytest.approx(15.0)      # 3 K colder -> +3
    assert W.slr_kuchera(a(273.16))[0] == pytest.approx(8.0)       # 2 K warmer -> -4 (double slope)
    assert W.slr_kuchera(a(200.0))[0] == 40.0 and W.slr_kuchera(a(290.0))[0] == 3.0      # clipped


def test_max_t_aloft_monotone_and_bounded():
    t = np.arange(-35, 5.1, 0.5)
    s = W.slr_max_t_aloft(t)
    assert s.min() >= W.SLR_MIN and s.max() <= W.SLR_MAX
    assert W.slr_max_t_aloft(a(-6.0))[0] == pytest.approx(np.polyval(W._MAXT_COEF, -6.0), abs=1e-3)
    assert W.slr_max_t_aloft(a(-60.0))[0] == W.slr_max_t_aloft(a(-35.0))[0]       # outside the fitted range is held


def test_column_maxima_ignore_below_ground_levels():
    levels = [1000, 850, 700, 500]
    t = np.array([300.0, 270.0, 268.0, 250.0], np.float32).reshape(4, 1, 1)     # 1000 hPa is below ground (ps 900)
    ps = a(900.0).reshape(1, 1)
    t2 = a(272.0).reshape(1, 1)
    assert W.tmax_column_k(t2, t, levels, ps)[0, 0] == pytest.approx(272.0)       # 300 K ignored; 2 m is the max
    assert W.tmax_aloft_c(t2, t, levels, ps)[0, 0] == pytest.approx(268.0 - 273.15, abs=1e-4)     # 850 hPa is within 70 hPa of the ground
    ps2 = a(1100.0).reshape(1, 1)
    assert W.tmax_aloft_c(t2, t, levels, ps2)[0, 0] == pytest.approx(300.0 - 273.15, abs=1e-4)


def test_warm_nose():
    levels = [1000, 925, 850, 700, 500]
    t = np.array([265.0, 268.0, 275.0, 270.0, 250.0], np.float32).reshape(5, 1, 1)
    assert W.warm_nose_c(t, levels, a(1010.0).reshape(1, 1))[0, 0] == pytest.approx(275.0 - 273.15, abs=1e-4)


def test_rh_ice_exceeds_rh_water_below_freezing():
    assert W.rh_ice(a(100.0), a(0.0))[0] == pytest.approx(100.0, abs=0.5)
    r = W.rh_ice(a(100.0), a(-15.0))[0]
    assert 110 < r < 118                                   # ~115 % at -15 C
    assert W.rh_ice(a(80.0), a(-15.0))[0] > 80


def _column(temps, rh=100.0, ps=1000.0, levels=(1000, 900, 800, 700, 600, 500)):
    t = np.array(temps, np.float32).reshape(-1, 1, 1)
    r = np.full_like(t, rh)
    return t, r, list(levels), a(ps).reshape(1, 1)


def test_dgz_depth_linear_profile():
    # -6 C at 1000 hPa to -26 C at 500 hPa: 4 C per 100 hPa -> -12..-18 C spans exactly 150 hPa
    t, r, lv, ps = _column([-6, -10, -14, -18, -22, -26])
    depth, sat, rhi = W.dgz_metrics(t, r, lv, ps)
    assert depth[0, 0] == pytest.approx(150.0, abs=0.5)
    assert sat[0, 0] == pytest.approx(150.0, abs=0.5)      # RHw 100 % -> RHi > 100 % at these temperatures
    assert rhi[0, 0] > 100


def test_dgz_dry_and_absent():
    t, r, lv, ps = _column([-6, -10, -14, -18, -22, -26], rh=50.0)
    depth, sat, rhi = W.dgz_metrics(t, r, lv, ps)
    assert depth[0, 0] == pytest.approx(150.0, abs=0.5) and sat[0, 0] == 0.0 and rhi[0, 0] < 100
    t, r, lv, ps = _column([5, 3, 1, -1, -3, -5])
    depth, sat, rhi = W.dgz_metrics(t, r, lv, ps)
    assert depth[0, 0] == 0.0 and np.isnan(rhi[0, 0])


def test_dgz_isothermal_layer_and_below_ground():
    t, r, lv, ps = _column([-15, -15, -15, -15, -15, -15])
    assert W.dgz_metrics(t, r, lv, ps)[0][0, 0] == pytest.approx(500.0, abs=0.5)
    t, r, lv, ps = _column([-15, -15, -15, -15, -15, -15], ps=800.0)           # only 800..500 hPa is above ground
    assert W.dgz_metrics(t, r, lv, ps)[0][0, 0] == pytest.approx(300.0, abs=0.5)


def test_wind_chill_nws_table_values():
    assert W.wind_chill_f(a(0.0), a(15.0))[0] == pytest.approx(-19.0, abs=0.6)        # NWS chart: 0 F, 15 mph -> -19 F
    assert W.wind_chill_f(a(-20.0), a(30.0))[0] == pytest.approx(-53.0, abs=0.6)
    assert W.wind_chill_f(a(60.0), a(20.0))[0] == 60.0                                # above 50 F: temperature
    assert W.wind_chill_f(a(10.0), a(2.0))[0] == 10.0                                 # calm: temperature


def test_blizzard_instant():
    f = lambda v, g, w, s: bool(W.blizzard_now(a(v), a(g), a(w), a(s))[0])
    assert f(0.2, 40, 20, 1) and f(0.25, 20, 36, 1)
    assert not f(0.3, 40, 20, 1) and not f(0.2, 30, 30, 1) and not f(0.2, 40, 20, 0)
    assert not f(np.nan, 40, 20, 1)


def test_blizzard_clock_three_hour_rule():
    c = W.BlizzardClock((1, 1), step=3)
    seq = [1, 0, 1, 1, 1, 0, 1]
    hrs = [c.update(np.array([[bool(x)]]))[0, 0] for x in seq]
    assert hrs == [0, 0, 0, 3, 6, 6, 6]            # best run of 3 samples spans 6 h; a single sample is 0 h


def test_precip_fractions_priority():
    sn, ip, zr = W.precip_fractions(a(1, 1, 1, np.nan), a(0, 1, 1, 0), a(0, 0, 1, 0))
    assert list(sn) == [1, 0, 0, 0] and list(ip) == [0, 1, 0, 0] and list(zr) == [0, 0, 1, 0]


def test_layer_thickness_hypsometric():
    levels = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 600, 500]
    t = np.full((len(levels), 1, 1), 273.15, np.float32)                         # isothermal 0 C column
    dz = W.layer_thickness_m(t, levels, 850, 700)[0, 0]
    assert dz == pytest.approx(29.2706 * 273.15 * np.log(850 / 700), rel=1e-4)    # ~1552 m
    assert W.layer_thickness_m(t, levels, 1000, 700)[0, 0] > dz
    assert 1540 < dz < 1560                                                      # the 1540 m rain/snow line is a ~0 C layer mean


def test_cmc_thickness_ratio():
    # cold: 10:1 whatever Z1000-700 is
    assert W.slr_thickness_cmc(a(2800.0), a(1500.0))[0] == pytest.approx(10.0)
    # warm 850-700: Murphy relation 10 * (30.6 - 0.1057 * dam); 2840 m -> 5.8:1, floor 3:1
    assert W.slr_thickness_cmc(a(2840.0), a(1545.0))[0] == pytest.approx(10 * (30.6 - 0.1057 * 284.0), abs=0.01)
    assert W.slr_thickness_cmc(a(2900.0), a(1560.0))[0] == 3.0
    assert W.slr_thickness_cmc(a(2700.0), a(1545.0))[0] == 10.0                  # never above 10:1


# --------------------------------------------------------------------------- literature snow-ratio methods
def test_nbm_thickness_ratio_is_the_published_line():
    z = np.array([1400.0, 1500.0, 1540.0, 1560.0, 1700.0, 1300.0])
    r = W.slr_thickness_nbm(z)
    np.testing.assert_allclose(r[1:3], [-0.16559 * 1500 + 263.35, -0.16559 * 1540 + 263.35], atol=1e-3)
    assert r[1] == pytest.approx(14.965, abs=0.01) and r[2] == pytest.approx(8.341, abs=0.01)
    assert r[4] == 3.0 and r[5] == 40.0                                  # clipped like every other ratio here
    assert np.all(np.diff(W.slr_thickness_nbm(np.linspace(1350, 1600, 60))) <= 0)    # thicker (warmer) layer -> lower ratio


def test_blend_is_the_equal_weight_mean():
    np.testing.assert_allclose(W.slr_blend(np.array([10.0, 20.0]), np.array([14.0, 24.0])), [12.0, 22.0])
    assert np.isnan(W.slr_blend(np.array([10.0, np.nan]), np.array([14.0, 24.0]))[1])


@pytest.mark.parametrize("pr,a,b,c", [(0.2, 18.8, 0.0811, 0.6508), (1.5, 18.8, 0.0811, 0.6508), (2.5, 16.1, 0.2182, 0.5373),
                                      (3.5, 14.9, 0.2295, 0.5174), (4.5, 13.2, 0.2678, 0.5074), (8.0, 11.9, 0.1524, 0.5174)])
def test_byun_uses_the_right_coefficient_set(pr, a, b, c):
    t = np.array([-15.0, b, 3.0])
    r = W.slr_byun(t, np.full(3, pr))
    assert r[0] == pytest.approx(a, abs=0.01)                            # cold limit = a
    assert r[1] == pytest.approx(a / 2, abs=1e-3)                        # half of a exactly at T = b
    assert r[2] < 0.1 * a                                                # nearly nothing well above freezing


def test_byun_shape():
    t = np.linspace(-12, 4, 80)
    r = W.slr_byun(t, np.full_like(t, 3.5))
    assert np.all(np.diff(r) <= 1e-6) and r.max() <= 18.8 and r.min() >= 0
    heavy, light = W.slr_byun(np.array([-5.0]), np.array([9.0])), W.slr_byun(np.array([-5.0]), np.array([1.5]))
    assert heavy[0] < light[0]                                           # heavier precipitation -> denser snow
    assert W.slr_byun(np.array([0.0]), np.array([8.0]))[0] == pytest.approx(6.82, abs=0.02)


def test_baxter_climatology_stays_within_the_published_means():
    lat = np.array([46.87, 41.91, 44.0, 40.87, 43.0])
    lon = np.array([-68.02, -71.07, -72.0, -72.86, -75.0])
    r = W.slr_baxter(lat, lon)
    vals = [v for *_, v in W.BAXTER_CWA]
    assert r.min() >= min(vals) - 1e-4 and r.max() <= max(vals) + 1e-4
    assert r[0] == pytest.approx(12.3, abs=0.2) and r[1] < r[0]          # Caribou ~12.3; the Boston coast is lower (10.7 mean)
    assert r.dtype == np.float32 and W.slr_baxter(np.zeros((3, 4)) + 43, np.zeros((3, 4)) - 71).shape == (3, 4)


# --------------------------------------------------------------------------- Cobb layer ratio, wet bulb, melt
LEV = [925, 900, 850, 800, 750, 700, 600, 500, 400, 300]


def _col(vals):
    return np.asarray(vals, float).reshape(len(vals), 1, 1)


@pytest.mark.parametrize("t,expect", [(-3.0, 7.75), (-15.0, 24.25), (-16.0, 26.0), (-30.0, 7.2), (-45.0, 7.2), (5.0, 0.0), (1.0, 0.0)])
def test_cobb_isothermal_saturated_column_gives_the_table_value(t, expect):
    r = W.slr_cobb(_col([t] * len(LEV)), _col([100.0] * len(LEV)), LEV, np.array([[1000.0]]))
    assert r[0, 0] == pytest.approx(expect, abs=1e-4)


def test_cobb_weights_layers_by_rh_squared():
    # two saturated-vs-dry halves: -16 C (26:1) on the lower half at RH 100, -4 C (8.5:1) on the upper half at RH 50
    n = len(LEV)
    t = _col([-16.0] * 5 + [-4.0] * (n - 5))
    rh = _col([100.0] * 5 + [50.0] * (n - 5))
    r = W.slr_cobb(t, rh, LEV, np.array([[1000.0]]))[0, 0]
    # 25 hPa layers: count the grid layers on each side of the 750/700 hPa break and weight by RH^2
    lo = sum(1 for p in range(925, 299, -25) if p >= 750)
    hi = sum(1 for p in range(925, 299, -25) if p < 750)
    w_lo, w_hi = lo * 100.0 ** 2, hi * 50.0 ** 2
    assert 8.5 < r < 26 and r > (26 * w_lo + 8.5 * w_hi) / (w_lo + w_hi) - 1.5      # transition layers blur the break by a little
    assert r > 0.5 * (26 + 8.5)                                       # the humid layers dominate over an unweighted mean


def test_cobb_ignores_layers_below_ground_and_survives_missing_data():
    t = _col([-15.0] * len(LEV))
    rh = _col([100.0] * len(LEV))
    # a 40:1-looking warm layer underground (925/900) must not count when the surface is at 850 hPa
    t_warm = t.copy()
    t_warm[0], t_warm[1] = 3.0, 3.0
    ground = W.slr_cobb(t_warm, rh, LEV, np.array([[850.0]]))[0, 0]
    assert ground == pytest.approx(24.25, abs=1e-4)
    assert W.slr_cobb(t_warm, rh, LEV, np.array([[1000.0]]))[0, 0] < 24.25     # but it does when the ground is below it
    nan = np.full((len(LEV), 1, 1), np.nan)
    assert np.isnan(W.slr_cobb(nan, nan, LEV, np.array([[1000.0]]))[0, 0])
    # a column that stops at 500 hPa (no 400/300 files) still gives a ratio from the layers it has
    r = W.slr_cobb(t[:8], rh[:8], LEV[:8], np.array([[1000.0]]))
    assert r[0, 0] == pytest.approx(24.25, abs=1e-4)


def test_cobb_interpolates_in_log_pressure():
    # -10 C at 700 hPa to -20 C at 500 hPa: the 600 hPa layer sits at ln(600/700)/ln(500/700) of the way, i.e. -4.6 C from 700 -> about -14.6 C
    lev = [700, 500]
    t = np.array([-10.0, -20.0]).reshape(2, 1, 1)
    rh = np.full((2, 1, 1), 100.0)
    # integrate by hand on the 25 hPa grid between 700 and 500 hPa
    ts = []
    for p in range(700, 499, -25):
        w = np.log(p / 700) / np.log(500 / 700)
        ts.append(np.interp(-10.0 + w * -10.0, W.COBB_T_C, W.COBB_SR))
    r = W.slr_cobb(t, rh, lev, np.array([[1000.0]]))[0, 0]
    assert r == pytest.approx(np.mean(ts), abs=1e-4)


def test_cobb_table_is_the_2011_curve():
    assert W.cobb_layer_sr(np.array([-16.0, -12.0, -8.0, 0.0, 1.0, 10.0])).tolist() == [26.0, 17.5, 9.5, 3.0, 0.0, 0.0]
    assert W.COBB_SR.max() == 26.0 and np.all(np.diff(W.COBB_T_C) > 0)


def test_wet_bulb():
    p = np.full(5, 1000.0)
    tw = W.wet_bulb_c(np.array([2.0, 0.0, 10.0, 5.0, -5.0]), np.array([0.0, -3.0, 5.0, 5.0, -8.0]), p)
    assert tw[3] == pytest.approx(5.0, abs=0.01)                         # saturated: Tw = T
    assert np.all(tw <= np.array([2.0, 0.0, 10.0, 5.0, -5.0]) + 1e-4)
    assert tw[0] == pytest.approx(1.18, abs=0.1) and tw[2] == pytest.approx(7.5, abs=0.15)     # textbook values
    assert W.wet_bulb_c(np.array([3.0]), np.array([8.0]), np.array([1000.0]))[0] == pytest.approx(3.0, abs=0.01)   # Td > T is clamped


@pytest.mark.parametrize("denom,table", [(15.0, [0.01, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.21]),
                                         (10.0, [0.02, 0.05, 0.08, 0.12, 0.16, 0.21, 0.26, 0.30, 0.35])])
def test_melt_rate_reproduces_the_published_tables(denom, table):
    """Leone et al. 2023 (D = 15) and Cobb 2024 (D = 10) tables of hourly melt, 32 F ... 40 F."""
    got = W.melt_rate_in_per_hr(np.arange(32.0, 41.0), denom)
    np.testing.assert_allclose(got, table, atol=0.0075)                  # the slides round (and truncate) to two decimals
    assert W.melt_rate_in_per_hr(np.array([31.0, 20.0, -5.0]), denom).tolist() == [0.0, 0.0, 0.0]
    assert W.melt_rate_in_per_hr(np.array([40.0]), denom)[0] == W.melt_rate_in_per_hr(np.array([55.0]), denom)[0]    # capped at 40 F


def test_melt_factor_worked_example_from_the_slides():
    f = W.melt_factor(np.array([34.0, 38.0, 38.0, 30.0, 34.0]), np.array([0.15, 0.15, 0.05, 0.15, 0.0]), 15.0)
    assert f[0] == pytest.approx(0.667, abs=0.01) and 12 * f[0] == pytest.approx(8.0, abs=0.1)      # 12:1 -> ~8:1
    assert f[1] == 0.0 and f[2] == 0.0                                    # melt >= snow rate: melted as it fell
    assert f[3] == 1.0 and f[4] == 1.0                                    # below onset, or no snow at all
    assert np.all((f >= 0) & (f <= 1))
    heavier = W.melt_factor(np.full(4, 35.0), np.array([0.05, 0.10, 0.20, 0.40]), 15.0)
    assert np.all(np.diff(heavier) >= 0)                                  # heavier snow survives better


# --------------------------------------------------------------------------- Cobb with a vertical-velocity term
GRID = list(range(925, 299, -25))


def _w(vals):
    return np.asarray(vals, float).reshape(len(vals), 1, 1)


def _two_layer_column():
    """-16 C (26:1) in the 600-400 hPa part of the column, -4 C (8.5:1) elsewhere, all saturated."""
    t = _col([-4.0 if p > 600 or p < 400 else -16.0 for p in LEV])
    return t, _col([100.0] * len(LEV))


def test_vv_unit_conversion_and_floor():
    # w = 0.10 m/s at 700 hPa, -10 C: rho = 70000 / (287.05 * 263.15) = 0.927 kg/m3, omega = -0.909 Pa/s = 9.09 microbar/s
    assert W.vv_ubar_per_s(0.10, -10.0, 700.0) == pytest.approx(9.09, abs=0.02)
    assert W.vv_ubar_per_s(-0.5, -10.0, 700.0) == 1.0                     # downward motion counts as 1
    assert W.vv_ubar_per_s(np.nan, -10.0, 700.0) == 1.0                   # so does a missing value
    assert W.vv_ubar_per_s(0.0005, -10.0, 700.0) == 1.0                   # and anything weaker than the floor


def test_cobb_without_lift_equals_zero_lift():
    t, rh = _two_layer_column()
    ps = np.array([[1000.0]])
    base = W.slr_cobb(t, rh, LEV, ps)
    np.testing.assert_allclose(W.slr_cobb(t, rh, LEV, ps, w_ms=np.zeros((26, 1, 1))), base, atol=1e-5)
    np.testing.assert_allclose(W.slr_cobb(t, rh, LEV, ps, w_ms=-np.ones((26, 1, 1))), base, atol=1e-5)
    np.testing.assert_allclose(W.slr_cobb(t, rh, LEV, ps, w_ms=np.full((26, 1, 1), np.nan)), base, atol=1e-5)


def test_cobb_lift_pulls_the_ratio_toward_the_layer_with_the_ascent():
    t, rh = _two_layer_column()
    ps = np.array([[1000.0]])
    cold = np.array([0.30 if 400 <= p <= 600 else 0.0 for p in GRID]).reshape(26, 1, 1)     # ascent inside the cold (26:1) layer
    warm = np.array([0.30 if p >= 800 else 0.0 for p in GRID]).reshape(26, 1, 1)             # ascent in the warm (8.5:1) layer
    base = W.slr_cobb(t, rh, LEV, ps)[0, 0]
    r_cold = W.slr_cobb(t, rh, LEV, ps, w_ms=cold)[0, 0]
    r_warm = W.slr_cobb(t, rh, LEV, ps, w_ms=warm)[0, 0]
    assert r_warm < base < r_cold <= W.COBB_MAX
    assert r_cold > 18 and r_warm < 14


def test_cobb_lift_weight_is_the_square_root_of_vv():
    # a two-level column: -4 C (8.5:1) at 800 hPa and -16 C (26:1) at 500 hPa, both saturated, ascent only at 500 hPa
    lev = [800, 500]
    t, rh = _col([-4.0, -16.0]), _col([100.0, 100.0])
    ps = np.array([[1000.0]])
    w = np.zeros((26, 1, 1))
    w[GRID.index(500)] = 0.04                          # 4 cm/s at 500 hPa only
    base = W.slr_cobb(t, rh, lev, ps)[0, 0]
    assert W.slr_cobb(t, rh, lev, ps, w_ms=w)[0, 0] > base
    # VV at 500 hPa (T=-16 C): rho = 50000/(287.05*257.15) = 0.6774, VV = 10 * 0.6774 * 9.80665 * 0.04 = 2.657: sqrt = 1.63 vs 1
    assert W.vv_ubar_per_s(0.04, -16.0, 500.0) == pytest.approx(2.657, abs=0.01)
