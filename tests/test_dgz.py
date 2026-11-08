"""Dendritic-growth-zone products in thousand feet."""
import numpy as np
import pytest

from rrfs_viewer import winter as W
from rrfs_viewer.catalog import BY_KEY

from test_pipeline import member, rg  # noqa: F401  (module-scoped fixtures)

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


def test_dgz_products_are_in_kft_with_the_reference_ramp():
    for k in ("dgz_depth", "dgz_sat", "dgz_lift"):
        p = BY_KEY[k]
        assert p.unit == "kft" and p.style.levels[0] == 0.5 and p.style.levels[-1] == 20
        assert p.lo + p.step * 254 >= 20                                       # the 8-bit code reaches the top of the ramp
    assert [p.key for p in BY_KEY.values() if p.group == "Snow growth (DGZ)" and not p.key.endswith("_1h")][:3] == ["dgz_depth", "dgz_sat", "dgz_lift"]


def test_member_dgz_kft_is_physical_and_lift_is_a_subset_of_saturated_depth(member):  # noqa: F811
    out, _, _ = member
    for k in ("dgz_depth", "dgz_sat", "dgz_lift"):
        a_ = out[k]
        assert np.isfinite(a_).any() and np.nanmin(a_) >= 0 and np.nanmax(a_) <= 25.4, k
    ok = np.isfinite(out["dgz_lift"]) & np.isfinite(out["dgz_sat"])
    assert (out["dgz_lift"][ok] <= out["dgz_sat"][ok] + 1e-5).all() and (out["dgz_sat"][ok] <= out["dgz_depth"][ok] + 1e-5).all()
