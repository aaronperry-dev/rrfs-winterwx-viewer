"""Dendritic-growth-zone products in thousand feet."""
import numpy as np
import pytest

from rrfs_viewer import winter as W
from rrfs_viewer.catalog import BY_KEY

from test_pipeline import lift, member, member_lift, rg  # noqa: F401  (module-scoped fixtures)

a = lambda *v: np.array(v, np.float32).reshape(-1, 1)


# --------------------------------------------------------------------------- DGZ in kft
def _column(temps, rh=100.0, ps=1000.0, levels=(1000, 900, 800, 700, 600, 500)):
    t = np.array(temps, np.float32).reshape(-1, 1, 1)
    return t, np.full_like(t, rh), list(levels), a(ps).reshape(1, 1)


def test_dgz_kft_isothermal_matches_hypsometric_equation():
    t, r, lv, ps = _column([-15] * 6)
    depth, sat, _ = W.dgz_metrics(t, r, lv, ps, units="kft")
    want = W.RD_OVER_G * 258.15 * np.log(1000 / 500) / W.M_PER_KFT          # 17.2 kft
    assert depth[0, 0] == pytest.approx(want, rel=1e-3)
    assert sat[0, 0] == pytest.approx(want, rel=1e-3)                        # saturated (RHw 100 %)


def test_dgz_kft_agrees_with_hpa_depth_and_dry_layers_do_not_count():
    t, r, lv, ps = _column([-6, -10, -14, -18, -22, -26])
    d_hpa = W.dgz_metrics(t, r, lv, ps)[0][0, 0]
    d_kft = W.dgz_metrics(t, r, lv, ps, units="kft")[0][0, 0]
    assert d_hpa == pytest.approx(150.0, abs=0.5)
    assert 3.5 < d_kft < 5.5                                                  # 150 hPa near 650 hPa is about 4.4 kft
    t, r, lv, ps = _column([-6, -10, -14, -18, -22, -26], rh=50.0)
    depth, sat, rhi = W.dgz_metrics(t, r, lv, ps, units="kft")
    assert depth[0, 0] > 3 and sat[0, 0] == 0.0 and rhi[0, 0] < 100
    t, r, lv, ps = _column([5, 3, 1, -1, -3, -5])
    assert W.dgz_metrics(t, r, lv, ps, units="kft")[0][0, 0] == 0.0
    with pytest.raises(ValueError):
        W.dgz_metrics(t, r, lv, ps, units="m")


def test_dgz_with_lift_needs_saturation_and_ascent():
    sat = np.array([3.0, 3.0, 0.0, np.nan], np.float32)
    w = np.array([0.5, 0.02, 0.5, 0.5], np.float32)
    out = W.dgz_with_lift(sat, w)
    assert out[0] == 3.0 and out[1] == 0.0 and out[2] == 0.0 and np.isnan(out[3])
    assert np.isnan(W.dgz_with_lift(np.float32(2.0), np.float32("nan")))


def test_moist_depth_uses_the_nws_90_percent_ice_criterion():
    # RHw 80 % at -15 C is about 95 % RHi (moist enough); RHw 65 % is about 77 % RHi (not)
    for rh, expect in ((80.0, True), (65.0, False)):
        t, r, lv, ps = _column([-15] * 6, rh=rh)
        depth, sat, rhi = W.dgz_metrics(t, r, lv, ps, units="kft")
        assert (sat[0, 0] == pytest.approx(depth[0, 0], rel=1e-3)) if expect else sat[0, 0] == 0.0
    assert W.DGZ_SAT_RHI == 90.0


def test_dgz_ascent_converts_dzdt_to_omega_with_the_ideal_gas_density():
    grid = list(W.COBB_GRID)
    t = np.full((len(grid), 1, 1), -15.0, np.float32)
    ps = np.full((1, 1), 1000.0, np.float32)
    w = np.full((len(grid), 1, 1), 0.10, np.float32)
    w_mean, om = W.dgz_ascent(t, grid, ps, w)
    assert w_mean[0, 0] == pytest.approx(0.10, abs=1e-6)
    rho = 100.0 * np.array(grid) / (W.RD * 258.15)
    assert om[0, 0] == pytest.approx(float(np.mean(10.0 * rho * W.G0 * 0.10)), rel=1e-4)        # about 7-12 microbar/s
    assert 5 < om[0, 0] < 15
    w_mean, om = W.dgz_ascent(t, grid, ps, -w)                                                 # sinking air: negative, not hidden
    assert w_mean[0, 0] < 0 and om[0, 0] < 0
    w_mean, om = W.dgz_ascent(np.full_like(t, -5.0), grid, ps, w)                              # nothing between -12 and -18 C
    assert np.isnan(w_mean[0, 0]) and np.isnan(om[0, 0])
    w_mean, om = W.dgz_ascent(t, grid, np.full((1, 1), 600.0, np.float32), w)                  # only levels above the ground count: still fine
    assert np.isfinite(om[0, 0])
    w_mean, om = W.dgz_ascent(t, grid, ps, np.full_like(w, np.nan))
    assert np.isnan(om[0, 0])


def test_dgz_products_are_in_kft_with_the_reference_ramp():
    for k in ("dgz_depth", "dgz_sat", "dgz_lift"):
        p = BY_KEY[k]
        assert p.unit == "kft" and p.style.levels[0] == 0.5 and p.style.levels[-1] == 20
        assert p.lo + p.step * 254 >= 20                                       # the 8-bit code reaches the top of the ramp
    assert [p.key for p in BY_KEY.values() if p.group == "Snow growth (DGZ)" and not p.key.endswith("_1h")][:4] == ["dgz_depth", "dgz_sat", "dgz_omega", "dgz_lift"]


def test_member_dgz_kft_is_physical_and_lift_is_a_subset_of_saturated_depth(member):  # noqa: F811
    out, _, _ = member
    for k in ("dgz_depth", "dgz_sat", "dgz_lift"):
        a_ = out[k]
        assert np.isfinite(a_).any() and np.nanmin(a_) >= 0 and np.nanmax(a_) <= 25.4, k
    ok = np.isfinite(out["dgz_lift"]) & np.isfinite(out["dgz_sat"])
    assert (out["dgz_lift"][ok] <= out["dgz_sat"][ok] + 1e-5).all() and (out["dgz_sat"][ok] <= out["dgz_depth"][ok] + 1e-5).all()


def test_member_with_deterministic_lift_has_dgz_ascent_and_lift_is_inside_it(member_lift, member):  # noqa: F811
    out, _ = member_lift
    om = out["dgz_omega"]
    assert np.isfinite(om).any() and np.nanmin(om) >= 0 and np.nanmax(om) < 200       # microbar/s
    ok = np.isfinite(out["dgz_lift"]) & np.isfinite(out["dgz_sat"])
    assert (out["dgz_lift"][ok] <= out["dgz_sat"][ok] + 1e-5).all()
    assert not np.isfinite(member[0]["dgz_omega"]).any()                           # without the deterministic file there is no in-layer ascent ...
    assert np.isfinite(member[0]["dgz_lift"]).any()                                # ... and the lift map falls back to the column maximum


def test_dgz_in_the_lowest_layer_is_kept_by_the_surface_node():
    # high ground: surface at 850 hPa, -14 C there; levels 900 (underground), 800 (-20 C), 700 (-26 C).  The -14..-18 C part lies between 850 and 810 hPa.
    t = np.array([-8.0, -20.0, -26.0], np.float32).reshape(-1, 1, 1)
    r = np.full_like(t, 90.0)
    lv, ps = [900, 800, 700], np.full((1, 1), 850.0, np.float32)
    assert W.dgz_metrics(t, r, lv, ps)[0][0, 0] == 0.0                                           # without a surface node that layer is dropped
    depth, sat, rhi = W.dgz_metrics(t, r, lv, ps, t_sfc_c=np.full((1, 1), -14.0, np.float32), rh_sfc=np.full((1, 1), 90.0, np.float32))
    assert depth[0, 0] == pytest.approx(50.0 * 4.0 / 6.0, abs=0.5)
    assert sat[0, 0] == pytest.approx(depth[0, 0], abs=0.5) and rhi[0, 0] > 100
    kft = W.dgz_metrics(t, r, lv, ps, units="kft", t_sfc_c=np.full((1, 1), -14.0, np.float32), rh_sfc=np.full((1, 1), 90.0, np.float32))[0][0, 0]
    assert 0.5 < kft < 1.2                                                                       # 33 hPa near 830 hPa is about 1 kft
    # a surface node must not change a column that the ground does not cut
    t2, r2, lv2, ps2 = _column([-6, -10, -14, -18, -22, -26])
    a_ = W.dgz_metrics(t2, r2, lv2, ps2)
    b_ = W.dgz_metrics(t2, r2, lv2, ps2, t_sfc_c=np.full((1, 1), -4.0, np.float32), rh_sfc=np.full((1, 1), 90.0, np.float32))
    for x, y in zip(a_, b_):
        np.testing.assert_allclose(x, y, rtol=1e-5, equal_nan=True)


def test_dry_air_under_the_old_925_hpa_limit_is_now_seen():
    # DGZ between 975 and 950 hPa: RH exists there (1000/975/950 are fetched for the DGZ) and the layer is moist
    t = np.array([-8.0, -13.0, -19.0, -24.0], np.float32).reshape(-1, 1, 1)
    r = np.full_like(t, 85.0)
    depth, sat, _ = W.dgz_metrics(t, r, [1000, 975, 950, 925], np.full((1, 1), 1010.0, np.float32))
    assert depth[0, 0] > 15 and sat[0, 0] == pytest.approx(depth[0, 0], rel=1e-3)
