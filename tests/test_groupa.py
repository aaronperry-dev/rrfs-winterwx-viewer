"""Group-A winter diagnostics, the HRRR source, and the guarantee that nothing that existed before was removed or changed."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from rrfs_viewer import diag, hrrr, winter as W
from rrfs_viewer.catalog import BY_KEY, GROUPS, PRODUCTS, STORE

HERE = Path(__file__).parent
LEVELS = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 600, 500)


# ----------------------------------------------------------------------------- nothing existing was lost
def test_original_products_unchanged_and_first():
    base = json.loads((HERE / "baseline_catalog.json").read_text())
    now = [[p.key, p.group, p.kind, p.unit, p.dtype, p.lo, p.step, p.pmm, p.sam, p.has1h] for p in PRODUCTS[:len(base["products"])]]
    assert now == base["products"]                                   # same keys, same order, same storage and flags
    assert list(GROUPS[:len(base["groups"])]) == base["groups"]      # the original groups keep their order; new ones come after
    assert set(base["store"]) <= {p.key for p in STORE}
    assert len({p.key for p in STORE}) == len(STORE)


def test_original_page_functions_and_ids_still_present():
    names = json.loads((HERE / "baseline_page_names.json").read_text())
    html = (HERE.parent / "site" / "index.html").read_text()
    have = set(re.findall(r"function\s+(\w+)", html)) | set(re.findall(r"^\s*(?:const|let)\s+(\w+)\s*=", html, re.M))
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    assert not (set(names["names"]) - have), sorted(set(names["names"]) - have)
    assert not (set(names["ids"]) - ids), sorted(set(names["ids"]) - ids)


def test_member_keys_exclude_single_source_products():
    from rrfs_viewer import pipeline
    assert not any(k.startswith("hr_") for k in pipeline.MEMBER_KEYS)
    assert all(p.key in pipeline.MEMBER_KEYS for p in STORE if not p.single)
    assert all(p.single == "HRRR" and not p.pmm and not p.sam for p in PRODUCTS if p.key.startswith("hr_"))


def test_new_products_fit_their_storage():
    from rrfs_viewer import encode
    for p in PRODUCTS:
        top = encode.top_code(p)
        hi = p.lo + p.step * (top - 1)
        assert hi > p.lo
        s = p.style
        assert len(s.colors) == len(s.levels) + 1, p.key
        assert s.levels == sorted(s.levels), p.key


# ----------------------------------------------------------------------------- physics
def _prof(tc, t2c, ps=1010.0, terr=50.0):
    t = np.array([[[tc[l] + 273.15]] for l in LEVELS], np.float32)
    return W.profile_nodes(t, LEVELS, np.array([[t2c + 273.15]]), np.array([[ps]]), np.array([[terr]]))


SNOW = {1000: -4, 975: -5, 950: -6, 925: -7, 900: -8, 850: -10, 800: -12, 750: -15, 700: -17, 600: -24, 500: -32}
RAIN = {1000: 6, 975: 5, 950: 4, 925: 3, 900: 2, 850: 1, 800: -1, 750: -4, 700: -7, 600: -15, 500: -26}
FRZR = {1000: -3, 975: -2, 950: -1, 925: 1.5, 900: 3, 850: 4, 800: 2, 750: -1, 700: -5, 600: -14, 500: -25}
SLEET = {1000: -8, 975: -9, 950: -9, 925: -8, 900: -4, 850: 2.5, 800: 2, 750: -1, 700: -5, 600: -14, 500: -25}


@pytest.mark.parametrize("prof,t2,want", [(SNOW, -3.5, 0), (RAIN, 6.5, 1), (SLEET, -8.0, 2), (FRZR, -3.5, 3)])
def test_bourgouin_types(prof, t2, want):          # snow, rain, ice pellets, freezing rain
    n = _prof(prof, t2)
    fr = W.bourgouin_fractions(*W.bourgouin_energies(n["p"], n["t"]), n["t"][0])
    got = [float(a[0, 0]) for a in fr]
    assert got[want] == 1.0 and abs(sum(got) - 1) < 1e-6, got


def test_bourgouin_energies_have_the_right_signs_and_scale():
    n = _prof(RAIN, 6.5)
    sw, pe, ne = (float(a[0, 0]) for a in W.bourgouin_energies(n["p"], n["t"]))
    assert sw > 100 and pe == 0 and ne == 0                     # a deep surface-based warm layer: all of it "surface" energy
    n = _prof(FRZR, -3.5)
    sw, pe, ne = (float(a[0, 0]) for a in W.bourgouin_energies(n["p"], n["t"]))
    assert sw == 0 and pe > 100 and ne > 10                     # warm nose aloft over a cold layer
    # a layer 5 K warm and 100 hPa deep: Rd * Tm * ln(p1/p2) * ln(Tm/273.15), as in UPP
    p = np.array([[[1000.0]], [[900.0]]], np.float32)
    t = np.full((2, 1, 1), 278.15, np.float32)
    sw, _, _ = W.bourgouin_energies(p, t)
    assert float(sw[0, 0]) == pytest.approx(287.04 * 278.15 * np.log(1000 / 900) * np.log(278.15 / 273.15), rel=1e-3)


def test_bourgouin_transition_zones_split_half_and_half():
    f = W.bourgouin_fractions(np.array([[9.0]]), np.array([[0.0]]), np.array([[0.0]]), np.array([[274.0]]))     # surface melting energy between 5.6 and 13.2, no warm layer aloft
    assert [float(a[0, 0]) for a in f] == [0.5, 0.5, 0.0, 0.0]
    f = W.bourgouin_fractions(np.array([[0.0]]), np.array([[50.0]]), np.array([[46 + 0.66 * 50 + 5]]), np.array([[270.0]]))   # between the freezing-rain and ice-pellet bounds
    assert [round(float(a[0, 0]), 3) for a in f] == [0.0, 0.0, 0.5, 0.5]


def test_level_heights_snow_level_and_freezing_level():
    n = _prof(RAIN, 6.5)
    lvl, agl = W.level_heights(n["t"] - 273.15, n["z"])
    assert 1500 < float(lvl[0, 0]) < 2000 and float(agl[0, 0]) == pytest.approx(float(lvl[0, 0]) - 50.0, abs=1)    # crossing between 800 and 750 hPa
    n = _prof(SNOW, -3.5)
    lvl, agl = W.level_heights(n["t"] - 273.15, n["z"])
    assert float(agl[0, 0]) == 0.0 and float(lvl[0, 0]) == pytest.approx(50.0)                                        # cold to the ground: the level is the ground
    warm = {k: 8.0 - 0.01 * (1000 - k) for k in LEVELS}                                                               # warm everywhere: no level
    n = _prof(warm, 9.0)
    lvl, agl = W.level_heights(n["t"] - 273.15, n["z"])
    assert np.isnan(lvl[0, 0]) and float(agl[0, 0]) == 3000.0


def test_levels_below_ground_are_left_out():
    n = _prof(SNOW, -3.5, ps=900.0, terr=1000.0)                 # surface pressure 900 hPa: the 1000..900 levels are underground
    assert np.isnan(n["t"][1:6, 0, 0]).all() or np.isfinite(n["t"][6:, 0, 0]).all()
    assert np.isnan(n["t"][1, 0, 0]) and np.isfinite(n["t"][-1, 0, 0])
    assert float(n["z"][6, 0, 0]) > 1000.0                         # first level above ground is above the terrain


def test_wet_bulb_between_dew_point_and_temperature():
    tw = float(W.wet_bulb_c(np.array([2.0]), np.array([-6.0]), np.array([1000.0]))[0])
    assert -6.0 < tw < 2.0 and tw == pytest.approx(-1.5, abs=1.0)
    td = float(W.td_from_rh_c(np.array([10.0]), np.array([100.0]))[0])
    assert td == pytest.approx(10.0, abs=0.05)
    td = float(W.td_from_rh_c(np.array([10.0]), np.array([50.0]))[0])
    assert td == pytest.approx(0.1, abs=0.4)


def test_inversion_strength():
    n = _prof(FRZR, -3.5)
    inv = float(W.inversion_strength_c(n["t"], n["p"])[0, 0])
    assert inv == pytest.approx(4 + 3.5, abs=0.6)              # warmest node below 850 hPa is the 850 hPa value: 4 C minus -3.5 C
    n = _prof(SNOW, -3.5)
    assert float(W.inversion_strength_c(n["t"], n["p"])[0, 0]) == 0.0


def test_rotation_matches_the_projection():
    """The raster's axes are turned from east / north by the Lambert convergence angle: compare with the direction of true north found by projecting two nearby points."""
    import cartopy.crs as ccrs
    from rrfs_viewer.basemap import PROJ
    pc = ccrs.PlateCarree()
    for lon, lat in ((-76.0, 43.0), (-70.6, 44.0), (-66.0, 45.0), (-72.0, 42.5)):
        x0, y0 = PROJ.transform_point(lon, lat, pc)
        x1, y1 = PROJ.transform_point(lon, lat + 0.01, pc)
        n = np.array([x1 - x0, y1 - y0]) / np.hypot(x1 - x0, y1 - y0)         # true north in raster axes
        e = np.array([n[1], -n[0]])                                          # true east
        u, v = W.rotate_to_grid(np.array([5.0]), np.array([3.0]), np.array([lon]))
        want = 5.0 * e + 3.0 * n
        assert float(u[0]) == pytest.approx(want[0], abs=0.01) and float(v[0]) == pytest.approx(want[1], abs=0.01), (lon, lat)


def test_frontogenesis_sign_and_magnitude():
    ny, nx, dx = 100, 120, 3000.0
    y = np.arange(ny)[:, None] * dx * np.ones((1, nx))
    x = np.arange(nx)[None, :] * dx * np.ones((ny, 1))
    g, c = 5.0 / 1e5, 1e-4                                      # 5 K per 100 km, convergence 1e-4 /s
    th = 300 + g * (y - y.mean())
    f = W.frontogenesis_k100km3h(th, np.zeros_like(y), -c * (y - y.mean()), dx)
    assert f[50, 60] == pytest.approx(g * c * 1e5 * 10800, rel=0.02)             # frontogenesis: positive
    assert W.frontogenesis_k100km3h(th, np.zeros_like(y), c * (y - y.mean()), dx)[50, 60] == pytest.approx(-g * c * 1e5 * 10800, rel=0.02)
    w = 1e-4                                                      # solid-body rotation does not change |grad theta|
    assert abs(W.frontogenesis_k100km3h(th, -w * (y - y.mean()), w * (x - x.mean()), dx)[50, 60]) < 0.05
    assert W.frontogenesis_k100km3h(np.full_like(y, 300.0), -w * y, w * x, dx).max() == 0.0     # no thermal gradient: no frontogenesis


def test_mpv_stable_positive_and_slantwise_unstable_negative():
    ny, nx = 60, 60
    lat = np.full((ny, nx), 44.0)
    z = np.zeros((ny, nx), np.float32)
    tl, th = np.full((ny, nx), 263.15, np.float32), np.full((ny, nx), 248.15, np.float32)
    assert float(W.mpv_sat_pvu(tl, th, z, z, z, z, 700.0, 500.0, lat)[30, 30]) > 0.1            # stable, f-dominated
    # baroclinic zone (temperature falling to the north) with westerlies increasing with height, as thermal wind requires: the shear-gradient term
    # lowers MPV* below the static-stability value, and a strong enough shear drives it negative (slantwise instability)
    y = np.arange(ny)[:, None] * 3000.0 * np.ones((1, nx))
    tl2 = (263.15 - 3.0e-5 * (y - y.mean())).astype(np.float32)
    calm = float(W.mpv_sat_pvu(tl2, tl2 - 13.0, z, z, z, z, 700.0, 500.0, lat)[30, 30])
    mid = float(W.mpv_sat_pvu(tl2, tl2 - 13.0, z, z, np.full((ny, nx), 20.0, np.float32), z, 700.0, 500.0, lat)[30, 30])
    strong = float(W.mpv_sat_pvu(tl2, tl2 - 13.0, z, z, np.full((ny, nx), 40.0, np.float32), z, 700.0, 500.0, lat)[30, 30])
    assert calm > mid > strong and calm > 0 > strong


def test_timing_products():
    hours = [3, 6, 9, 12]
    cum = np.zeros((4, 1, 3), np.float32)
    cum[:, 0, 0] = [0.0, 0.05, 0.2, 3.5]          # start at 9 h (reaches 0.1 in), last significant block at 12 h; 3.3 in in the last block: >= 3 in (1 in/h)
    cum[:, 0, 1] = [0, 0, 0, 0]                    # never snows
    cum[:, 0, 2] = [2.0, 4.0, 4.0, 4.0]            # 2 in in the first 3 h block, 2 in the second, then nothing
    start, end, (h05, h10) = W.snow_timing(cum, hours)
    assert start[3, 0, 0] == 9 and np.isnan(start[1, 0, 0]) and np.isnan(start[:, 0, 1]).all()
    assert end[3, 0, 0] == 12 and end[3, 0, 2] == 6 and start[3, 0, 2] == 3
    assert h05[3, 0, 0] == 3 and h10[3, 0, 0] == 3 and h05[3, 0, 2] == 6 and h10[3, 0, 2] == 0     # blocks of >= 1.5 in / >= 3 in
    assert (np.diff(h05, axis=0) >= 0).all()                                                        # "so far": never decreases


def test_changeover_and_impact_hours():
    sf = np.array([[[1.0]], [[1.0]], [[0.0]], [[1.0]]], np.float32)
    wet = np.ones((4, 1, 1), bool)
    ch = W.changeover_hours(sf, wet, [3, 6, 9, 12])
    assert np.isnan(ch[1, 0, 0]) and ch[2, 0, 0] == 9 and ch[3, 0, 0] == 9                          # first flip stays the answer
    dry_start = W.changeover_hours(np.array([[[0.0]], [[0.0]]], np.float32), np.ones((2, 1, 1), bool), [3, 6])
    assert np.isnan(dry_start).all()                                                                 # rain from the start is not a changeover
    flag = np.array([[[1.0]], [[0.0]], [[1.0]]], np.float32)
    assert list(W.impact_hours(flag)[:, 0, 0]) == [3.0, 3.0, 6.0]


# ----------------------------------------------------------------------------- on the demo storm
NOW = datetime(2026, 11, 5, 19, 20, tzinfo=timezone.utc)
CYC = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def rg(tmp_path_factory):
    from rrfs_viewer import build
    from rrfs_viewer.s3 import S3Source
    from rrfs_viewer.testing import FakeS3
    return build.ensure_regrid(S3Source(FakeS3(NOW)), CYC, tmp_path_factory.mktemp("cache"), 3)


@pytest.fixture(scope="module")
def member(rg):
    from rrfs_viewer import pipeline
    from rrfs_viewer.s3 import S3Source
    from rrfs_viewer.testing import FakeS3
    return pipeline.process_member(S3Source(FakeS3(NOW)), CYC, "m001", rg, 9)


GROUP_A = ("tw2", "snowlvl", "snowlvl_agl", "frzlvl", "mslp", "thk_1000_850", "thk_850_700", "t925", "t850", "w925", "w850", "inversion", "fronto700", "band_pot", "csi",
           "bsnow", "brain", "bsleet", "bfrzr", "bagree", "t_start", "t_end", "t_rate05", "t_rate10", "imp_wetwind", "imp_ice_wind")


def test_member_produces_group_a_fields(member):
    out, info = member
    for k in ("tw2", "snowlvl", "snowlvl_agl", "frzlvl", "mslp", "thk_1000_850", "thk_850_700", "t925", "t850", "w925", "w850", "inversion", "fronto700", "bsnow", "brain"):
        assert np.isfinite(out[k]).any(), k
    # plausibility of the invented storm
    assert 940 < np.nanmin(out["mslp"]) and np.nanmax(out["mslp"]) < 1060
    assert 100 < np.nanmin(out["thk_1000_850"]) and np.nanmax(out["thk_1000_850"]) < 160
    tw, t2 = out["tw2"][0], out["t2"][0]
    assert np.nanmax(tw - t2) <= 0.5                                     # wet bulb never above the temperature
    assert np.nanmin(out["snowlvl_agl"]) >= 0
    for k in ("bsnow", "brain", "bsleet", "bfrzr"):
        v = out[k][np.isfinite(out[k])]
        assert v.min() >= 0 and v.max() <= 100
    tot = sum(np.nan_to_num(out[k][0]) for k in ("bsnow", "brain", "bsleet", "bfrzr"))
    ok = np.isfinite(out["bsnow"][0])
    assert np.allclose(tot[ok], 100.0, atol=0.1)                           # the four profile types partition the wet area
    assert np.isfinite(out["fronto700"]).any() and np.abs(out["fronto700"][np.isfinite(out["fronto700"])]).max() < 200


def test_existing_member_products_are_unchanged_by_the_new_code(member, rg):
    """The same fake storm run through the baseline pipeline code gives identical existing products (the baseline copy is the pre-change source tree)."""
    base = Path("/tmp/baseline/rrfs_viewer")
    if not base.exists():
        pytest.skip("baseline copy not available")
    import importlib.util, subprocess, sys, tempfile, os, pickle
    out, _ = member
    code = r'''
import sys, pickle
sys.path.insert(0, "/tmp/baseline")
from datetime import datetime, timezone
from rrfs_viewer import build, pipeline
from rrfs_viewer.s3 import S3Source
from rrfs_viewer.testing import FakeS3
import tempfile, pathlib
NOW = datetime(2026, 11, 5, 19, 20, tzinfo=timezone.utc); CYC = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)
rg = build.ensure_regrid(S3Source(FakeS3(NOW)), CYC, pathlib.Path(tempfile.mkdtemp()), 3)
o, i = pipeline.process_member(S3Source(FakeS3(NOW)), CYC, "m001", rg, 9)
pickle.dump(o, open(sys.argv[1], "wb"))
'''
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "o.pkl")
        r = subprocess.run([sys.executable, "-c", code, f], capture_output=True, text=True, timeout=900, cwd="/tmp")
        assert r.returncode == 0, r.stderr[-2000:]
        old = pickle.load(open(f, "rb"))
    for k, a in old.items():
        assert k in out, k
        assert np.array_equal(np.isfinite(a), np.isfinite(out[k])), k
        m = np.isfinite(a)
        assert np.allclose(a[m], out[k][m], atol=1e-4, rtol=1e-5), k


# ----------------------------------------------------------------------------- HRRR
@pytest.fixture(scope="module")
def hrrr_run(rg, tmp_path_factory):
    from rrfs_viewer.s3 import S3Source
    from rrfs_viewer.testing import FakeS3
    src = S3Source(FakeS3(NOW), bucket=hrrr.HRRR_BUCKET)
    return hrrr.process_hrrr(src, CYC, rg, 12)


def test_hrrr_products(hrrr_run):
    out, info = hrrr_run
    assert info["frames"] == 4 and set(out) == set(hrrr.HRRR_KEYS)
    for k in ("hr_snow", "hr_snow_1h", "hr_qpf", "hr_qpf_1h", "hr_snod", "hr_snod_chg", "hr_slr", "hr_refc", "hr_ptype", "hr_cpofp", "hr_t2", "hr_tw2", "hr_gust", "hr_vis",
              "hr_mslp", "hr_frzlvl", "hr_dgz", "hr_dgz_lift", "hr_dgz_omega", "hr_slr_cobb"):
        assert np.isfinite(out[k]).any(), k
    assert np.nanmax(out["hr_snow"][3]) >= np.nanmax(out["hr_snow"][1])                      # an accumulation only grows
    assert np.nanmax(np.abs(out["hr_snod_chg"][0])) < 100
    t2, tw = out["hr_t2"][0], out["hr_tw2"][0]
    assert np.nanmax(tw - t2) <= 0.1                                                          # wet bulb at or below the temperature
    assert np.nanmax(out["hr_cpofp"]) <= 100 and np.nanmin(out["hr_cpofp"]) >= 0


def test_hrrr_helpers():
    # omega -> w: rising air (omega < 0) gives positive w; 1 Pa/s at 700 hPa, 260 K is about 11 cm/s (rho = 0.94 kg/m3)
    w = hrrr.omega_to_w(np.array([-1.0]), np.array([260.0]), 700.0)
    assert float(w[0]) == pytest.approx(0.1087, rel=0.01)
    v = {p: np.full((2, 2), float(p), np.float32) for p in (1000, 900, 800)}
    out = hrrr.interp_levels(v, (1000, 900, 800), (950, 850, 1000, 700))
    assert [float(o[0, 0]) for o in out] == [950.0, 850.0, 1000.0, 800.0]


def test_hrrr_idx_names_match_the_real_inventories():
    """Field and level names used for HRRR requests exist in the real HRRR .idx files (hrrr.t00z.wrfsfcf06 / wrfprsf45)."""
    from rrfs_viewer.s3 import parse_idx
    sfc, prs = HERE / "data" / "hrrr.t00z.wrfsfcf06.grib2.idx.txt", HERE / "data" / "hrrr.t00z.wrfprsf45.grib2.idx.txt"
    if not (sfc.exists() and prs.exists()):
        pytest.skip("real HRRR idx samples not present")
    es, ep = parse_idx(sfc.read_text()), parse_idx(prs.read_text())
    ks = {(e.var, e.level) for e in es}
    for var, lev in (("ASNOW", "surface"), ("APCP", "surface"), ("WEASD", "surface"), ("FRZR", "surface"), ("FROZR", "surface"), ("SNOD", "surface"), ("CSNOW", "surface"),
                     ("CICEP", "surface"), ("CFRZR", "surface"), ("CRAIN", "surface"), ("CPOFP", "surface"), ("TMP", "2 m above ground"), ("DPT", "2 m above ground"),
                     ("RH", "2 m above ground"), ("PRES", "surface"), ("GUST", "surface"), ("VIS", "surface"), ("MSLMA", "mean sea level"), ("HGT", "0C isotherm"),
                     ("LAYTH", "261 K level - 256 K level")):
        assert (var, lev) in ks, (var, lev)
    assert any(v == "REFC" and l.startswith("entire atmosphere") for v, l in ks)
    kp = {(e.var, e.level) for e in ep}
    for p in hrrr.HRRR_PRS_T:
        assert ("TMP", f"{p} mb") in kp and ("RH", f"{p} mb") in kp
    from rrfs_viewer.config import LIFT_LEVELS
    for p in LIFT_LEVELS:
        assert ("VVEL", f"{p} mb") in kp
    assert any(e.var == "ASNOW" and e.info.startswith("0-6 hour acc") for e in es)


def test_hrrr_keys_match_the_bucket_layout():
    from rrfs_viewer.config import FILE_HPRS, FILE_HSFC
    c = datetime(2026, 11, 8, 0, tzinfo=timezone.utc)
    assert FILE_HSFC.key_for("hrrr", c, 6) == "hrrr.20261108/conus/hrrr.t00z.wrfsfcf06.grib2"
    assert FILE_HPRS.key_for("hrrr", c, 45) == "hrrr.20261108/conus/hrrr.t00z.wrfprsf45.grib2"
