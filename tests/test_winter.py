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
