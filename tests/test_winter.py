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
