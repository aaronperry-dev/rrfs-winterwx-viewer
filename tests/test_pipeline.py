"""End-to-end data path against the offline FakeS3 (real GRIB2 bytes, real .idx files, real Range semantics)."""
from datetime import datetime, timezone

import numpy as np
import pytest

from rrfs_viewer import build, pipeline
from rrfs_viewer import winter as W
from rrfs_viewer.catalog import BY_KEY
from rrfs_viewer.s3 import S3Source
from rrfs_viewer.testing import FakeS3

NOW = datetime(2026, 11, 5, 19, 20, tzinfo=timezone.utc)
CYC = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def rg(tmp_path_factory):
    src = S3Source(FakeS3(NOW))
    return build.ensure_regrid(src, CYC, tmp_path_factory.mktemp("cache"), 3)


@pytest.fixture(scope="module")
def member(rg):
    fake = FakeS3(NOW)
    out, info = pipeline.process_member(S3Source(fake), CYC, "m001", rg, 6)
    return out, info, fake


@pytest.fixture(scope="module")
def lift(rg):
    arr, info = pipeline.process_lift(S3Source(FakeS3(NOW)), CYC, rg, 6)
    return arr, info


@pytest.fixture(scope="module")
def member_lift(rg, lift):
    out, info = pipeline.process_member(S3Source(FakeS3(NOW)), CYC, "m001", rg, 6, lift=lift[0])
    return out, info


def test_find_cycle_picks_newest_complete():
    assert pipeline.find_cycle(S3Source(FakeS3(NOW)), NOW, 6) == CYC
    early = datetime(2026, 11, 5, 15, 0, tzinfo=timezone.utc)
    assert pipeline.find_cycle(S3Source(FakeS3(early)), early, 6) == datetime(2026, 11, 5, 6, tzinfo=timezone.utc)
    assert pipeline.find_cycle(S3Source(FakeS3(datetime(2026, 11, 5, 1, tzinfo=timezone.utc))), datetime(2026, 11, 5, 1, tzinfo=timezone.utc), 6, lookback=1) is None


def test_member_products_complete_and_physical(member, rg):
    out, info, _ = member
    assert set(out) == set(pipeline.MEMBER_KEYS) and set(out) <= set(BY_KEY)
    ny, nx = rg.shape
    for k, a in out.items():
        assert a.shape == (2, ny, nx), k
        if "cobblift" in k:                       # needs the deterministic run's DZDT: checked in test_cobb_with_lift_products
            continue
        assert np.isfinite(a).any(), f"{k} is empty"
    assert info["missing"] == {}
    n = lambda k: np.nan_to_num(out[k])
    q, s12, sk, sm, st = (n(k) for k in ("qpf", "snow_12", "snow_kuchera", "snow_maxt", "snow_thick"))
    assert (np.diff(q, axis=0) >= -1e-4).all()                       # accumulations never decrease
    assert (s12 <= 12 * q + 1e-3).all()                              # snow liquid <= total liquid
    liq = s12 / 12.0
    for arr in (sk, sm, st):
        assert (arr >= 3 * liq - 1e-3).all() and (arr <= 40 * liq + 1e-3).all()      # ratios clipped 3..40
    assert (st <= 10 * liq + 1e-3).all()                             # the thickness ratio never exceeds 10:1
    assert (n("frzr") + n("sleet") <= q + 1e-3).all()
    pt = sum(n(k) for k in ("psnow", "psleet", "pfrzr", "prain"))
    assert pt.max() <= 100.0 + 1e-3
    for k in ("slr_kuchera", "slr_maxt", "slr_thick"):
        assert out[k].min() >= 3 and out[k].max() <= 40, k
    assert np.all(out["slr_12"] == 12.0)
    assert out["dgz_depth"].min() >= 0 and out["dgz_depth"].max() <= 1000
    # the multi-method mean lies between the lowest and highest method
    meth = np.stack([out[m] for m in pipeline.SNOW_METHODS])
    lo, hi = np.nanmin(meth, axis=0), np.nanmax(meth, axis=0)
    ok = np.isfinite(out["snow_mean"])
    assert (out["snow_mean"][ok] >= lo[ok] - 1e-3).all() and (out["snow_mean"][ok] <= hi[ok] + 1e-3).all()


def test_literature_methods(member, rg):
    out, _, _ = member
    n = lambda k: np.nan_to_num(out[k])
    liq = n("snow_12") / 12.0
    # ratios stay inside their documented ranges
    assert out["slr_nbmthick"].min() >= 3 and out["slr_nbmthick"].max() <= 40
    assert np.nanmax(out["slr_byun"]) <= 18.8 + 1e-3 and np.nanmin(out["slr_byun"]) >= 0
    assert 10.6 - 1e-3 <= out["slr_clim"].min() and out["slr_clim"].max() <= 13.4 + 1e-3
    assert np.all(out["slr_clim"][0] == out["slr_clim"][1])                       # constant in time
    np.testing.assert_allclose(out["slr_clim"][0], out["slr_clim"][1])
    assert np.isfinite(out["slr_nbmthick"]).any() and np.isfinite(out["slr_byun"]).any()
    # snowfall = snow liquid x ratio (all methods share the same snow liquid, so totals are bracketed by their ratio limits)
    assert (n("snow_nbmthick") >= 3 * liq - 1e-3).all() and (n("snow_nbmthick") <= 40 * liq + 1e-3).all()
    assert (n("snow_byun") <= 18.8 * liq + 1e-3).all()
    assert (n("snow_clim") >= 10.6 * liq - 1e-3).all() and (n("snow_clim") <= 13.4 * liq + 1e-3).all()
    # the spread is max - min over the forecast methods and never negative
    meth = np.stack([out[m] for m in pipeline.SPREAD_METHODS])
    ok = np.isfinite(out["snow_spread"])
    assert ok.any() and (out["snow_spread"][ok] >= 0).all()
    np.testing.assert_allclose(out["snow_spread"][ok], (np.nanmax(meth, axis=0) - np.nanmin(meth, axis=0))[ok], atol=1e-4)
    # the multi-method mean is unchanged: still the five original methods only
    assert pipeline.SNOW_METHODS == ("snow_model", "snow_12", "snow_kuchera", "snow_maxt", "snow_thick")
    assert "snow_byun" not in pipeline.SPREAD_METHODS and "snow_clim" not in pipeline.SPREAD_METHODS


def test_cobb_product_and_the_products_left_out(member):
    """Cobb is in (without a lift term); its melt variant, the NBM blends and the melt-loss map stay out."""
    out, _, _ = member
    gone = ("snow_cobbmelt", "snow_nbm", "snow_nbmmelt", "slr_nbm", "melt_loss")
    assert not [k for k in gone if k in out or k in pipeline.MEMBER_KEYS or k + "_1h" in out]
    assert not hasattr(pipeline, "MELT_METHODS")
    assert "snow_cobb" in pipeline.SPREAD_METHODS and "snow_cobb" not in pipeline.SNOW_METHODS     # the five-method mean is unchanged
    liq = np.nan_to_num(out["snow_12"]) / 12.0
    assert np.isfinite(out["slr_cobb"]).any()
    assert np.nanmin(out["slr_cobb"]) >= 0 and np.nanmax(out["slr_cobb"]) <= 30 + 1e-3
    assert (np.nan_to_num(out["snow_cobb"]) <= 30 * liq + 1e-3).all()
    assert np.isfinite(out["snow_cobb"]).any() and np.isfinite(out["snow_cobb_1h"]).any()


def test_byun_ratio_is_recomputable_from_stored_fields(member):
    """Frame 0 is forecast hour 3, so its 3-hour precipitation is the run-total QPF: the stored ratio must follow from T2 and QPF."""
    out, _, _ = member
    t_c = (out["t2"][0].astype(np.float64) - 32.0) / 1.8
    expect = W.slr_byun(t_c, out["qpf"][0].astype(np.float64) * 25.4)
    pr = out["qpf"][0].astype(np.float64) * 25.4
    near_edge = np.min(np.abs(pr[..., None] - np.array([2.0, 3.0, 4.0, 5.0])), axis=-1) < 1e-4    # float rounding picks the other set
    ok = np.isfinite(out["slr_byun"][0]) & np.isfinite(expect) & ~near_edge
    assert ok.sum() > 1000
    np.testing.assert_allclose(out["slr_byun"][0][ok], expect[ok], atol=0.05)
    # heavier 3-h precipitation selects a coefficient set with a lower cold limit
    heavy = ok & (out["qpf"][0] * 25.4 >= 5) & (t_c < -5)
    light = ok & (out["qpf"][0] * 25.4 < 2) & (t_c < -5)
    if heavy.any() and light.any():
        assert out["slr_byun"][0][heavy].max() <= 11.9 + 1e-3 < out["slr_byun"][0][light].min()


def test_one_hour_fields_are_consistent_with_totals(member):
    out, _, _ = member
    for base in ("qpf", "snow_12", "snow_kuchera", "snow_maxt", "snow_thick", "snow_mean", "snow_model",
                 "snow_cobb", "snow_nbmthick", "snow_byun", "snow_clim"):
        one, tot = np.nan_to_num(out[base + "_1h"]), np.nan_to_num(out[base])
        assert one.min() >= -1e-4, base
        assert (one[1] <= tot[1] - tot[0] + 0.02).all(), base          # the last hour of a 3-h block <= the 3-h window
        assert (one[0] <= tot[0] + 0.02).all(), base                   # ... and <= the run total at the first frame
    assert out["qpf_1h"].max() > 0.02


def test_thickness_from_heights_matches_hypsometric(rg):
    """The FakeS3 heights are hypsometric, so the height path and the temperature fallback must agree closely."""
    fake = FakeS3(NOW, missing={("HGT", 3), ("HGT", 6)})
    out_t, info = pipeline.process_member(S3Source(fake), CYC, "m001", rg, 3)
    out_z, _ = pipeline.process_member(S3Source(FakeS3(NOW)), CYC, "m001", rg, 3)
    assert "HGT:1000 mb" in info["missing"]
    d = np.abs(out_t["slr_thick"] - out_z["slr_thick"])
    assert np.nanmax(d) < 1.5 and np.isfinite(out_t["slr_thick"]).any()


def test_totals_only_members_give_the_same_qpf(rg, member):
    """Members that publish only run-total APCP (no 1-hour windows) are handled by differencing the totals."""
    out, _ = pipeline.process_member(S3Source(FakeS3(NOW, member_hourly_apcp=False)), CYC, "m001", rg, 3)
    ref, _, _ = member
    np.testing.assert_allclose(out["qpf"][0], ref["qpf"][0], atol=0.02)
    np.testing.assert_allclose(out["qpf_1h"][0], ref["qpf_1h"][0], atol=0.02)


def test_member_fetches_are_lean(member):
    _, info, fake = member
    assert info["requests"] > 0
    assert fake.range_requests and all(rng and rng.startswith("bytes=") for _, rng in fake.range_requests)   # always byte ranges
    whole = sum(len(blob) for blob, _ in fake._files.values())
    assert info["bytes"] < whole and info["bytes"] > 0


@pytest.mark.parametrize("totals", [False, True])
def test_refs_accumulation_modes_agree(rg, totals):
    out, info = pipeline.process_refs(S3Source(FakeS3(NOW, refs_totals=totals)), CYC, rg, 6)
    assert info["mode"]["APCP"] == ("total" if totals else "sum")
    assert set(out) == set(pipeline.REFS_KEYS)
    for k in ("qpf", "qpf_1h", "snow_12", "snow_12_1h", "snow_model", "snow_model_1h"):
        assert np.isfinite(out[k]).any(), k
    globals().setdefault("_REFS", {})[totals] = out
    if len(_REFS) == 2:
        for k in ("qpf", "qpf_1h", "snow_12", "snow_12_1h", "snow_model", "snow_model_1h"):
            np.testing.assert_allclose(_REFS[False][k], _REFS[True][k], atol=0.03, equal_nan=True, err_msg=k)


def test_missing_message_is_not_fatal(rg):
    fake = FakeS3(NOW, missing={("VIS", 3), ("RH", 3)})
    out, info = pipeline.process_member(S3Source(fake), CYC, "m002", rg, 6)
    assert not np.isfinite(out["vis"][0]).any() and np.isfinite(out["vis"][1]).any()
    assert any(k.startswith("VIS") for k in info["missing"])
    assert "VIS:surface" in info["hints"]                  # the log says what the file does contain for that field
    assert np.isfinite(out["qpf"]).any() and np.isfinite(out["t2"]).any()


# ---------------------------------------------------------------- graceful degradation: use whatever is there
@pytest.fixture(scope="module")
def clean12(rg):
    return pipeline.process_member(S3Source(FakeS3(NOW)), CYC, "m003", rg, 12)


def test_hourly_precip_hole_is_repaired_from_the_run_total(rg):
    fake = FakeS3(NOW, missing={("APCP", 2)})
    out, info = pipeline.process_member(S3Source(fake), CYC, "m003", rg, 6)
    ref, _ = pipeline.process_member(S3Source(FakeS3(NOW)), CYC, "m003", rg, 6)
    assert np.isfinite(out["qpf"]).all(axis=(1, 2)).all()                      # nothing is withheld
    assert np.allclose(out["qpf"], ref["qpf"], atol=1e-3)                      # the run total supplies the hour that is missing
    assert np.isfinite(out["snow_mean"]).any() and np.isfinite(out["snow_kuchera"]).any()
    assert info["quality"][0] & pipeline.Q_ESTIMATED and not info["quality"][0] & pipeline.Q_PARTIAL
    assert not info["quality"][1] & pipeline.Q_PARTIAL                       # the repaired hours are not counted missing later on
    assert np.nanmax(np.abs(out["snow_mean"] - ref["snow_mean"])) < 0.5 * np.nanmax(ref["snow_mean"])      # the type of the lost hour is estimated, not invented from nothing
    assert np.isfinite(out["gust"]).any() and np.isfinite(out["slr_kuchera"]).any()


def test_unrecoverable_hours_give_a_lower_bound_not_a_blank(rg):
    fake = FakeS3(NOW, missing={("APCP", 2), ("APCP", 3)})                     # the run total of F003 is gone too
    out, info = pipeline.process_member(S3Source(fake), CYC, "m003", rg, 6)
    ref, _ = pipeline.process_member(S3Source(FakeS3(NOW)), CYC, "m003", rg, 6)
    assert np.isfinite(out["qpf"][0]).any() and np.isfinite(out["snow_mean"][0]).any()
    assert np.nansum(out["qpf"][0]) < np.nansum(ref["qpf"][0])                 # F003 is what the one good hour adds up to ...
    assert info["quality"][0] & pipeline.Q_PARTIAL                              # ... and says so
    assert np.allclose(out["qpf"][1], ref["qpf"][1], atol=1e-3)                # F006 has its run total again: exact
    assert not info["quality"][1] & pipeline.Q_PARTIAL


def test_missing_type_flags_use_the_previous_hours_type(rg):
    fake = FakeS3(NOW, missing={("CSNOW", 5), ("CICEP", 5), ("CFRZR", 5)})
    out, info = pipeline.process_member(S3Source(fake), CYC, "m001", rg, 12)
    ref, _ = pipeline.process_member(S3Source(FakeS3(NOW)), CYC, "m001", rg, 12)
    assert np.isfinite(out["snow_kuchera"]).all(axis=(1, 2)).all()
    assert np.allclose(out["qpf"], ref["qpf"], atol=1e-3)                      # liquid is untouched
    assert info["quality"][1] & pipeline.Q_ESTIMATED and not info["quality"][0] & pipeline.Q_ESTIMATED
    assert not any(q & pipeline.Q_ESTIMATED for q in info["quality"][2:])              # only the frame whose own block was estimated is flagged


def test_missing_model_snowfall_total_is_bridged(rg, clean12):
    ref, _ = clean12
    fake = FakeS3(NOW, missing={("ASNOW", 6)})
    out, info = pipeline.process_member(S3Source(fake), CYC, "m003", rg, 12)
    assert np.isfinite(out["snow_model"][1]).any()                              # F006 is not blank ...
    assert info["quality"][1] & pipeline.Q_INTERP                               # ... it is marked as interpolated
    assert not info["quality"][1] & pipeline.Q_PARTIAL                          # (the lower bound that was set aside is not what is shown)
    assert np.allclose(out["snow_model"][1], 0.5 * (out["snow_model"][0] + out["snow_model"][2]), atol=1e-4)
    assert np.allclose(out["snow_model"][2], ref["snow_model"][2], atol=1e-4)  # the next run total is exact again
    assert not info["quality"][2] & pipeline.Q_INTERP


def test_model_snowfall_from_hourly_windows_only(rg, clean12):
    ref, _ = clean12
    fake = FakeS3(NOW, member_asnow_totals=False)
    out, info = pipeline.process_member(S3Source(fake), CYC, "m003", rg, 12)
    assert np.allclose(out["snow_model"], ref["snow_model"], rtol=0.01, atol=0.02, equal_nan=True)     # summed from the 1-hour windows (each one packed with its own rounding)
    assert not any(q & (pipeline.Q_PARTIAL | pipeline.Q_INTERP) for q in info["quality"])


def test_model_snowfall_with_a_missing_hour_is_a_lower_bound(rg):
    fake = FakeS3(NOW, member_asnow_totals=False, missing={("ASNOW", 2)})
    out, info = pipeline.process_member(S3Source(fake), CYC, "m003", rg, 6)
    ref, _ = pipeline.process_member(S3Source(FakeS3(NOW, member_asnow_totals=False)), CYC, "m003", rg, 6)
    assert np.isfinite(out["snow_model"]).any(axis=(1, 2)).all()
    assert np.nansum(out["snow_model"][1]) <= np.nansum(ref["snow_model"][1]) + 1e-6
    assert info["quality"][0] & pipeline.Q_PARTIAL and info["quality"][1] & pipeline.Q_PARTIAL      # it never recovers without a run total


def test_fill_gaps():
    n, shape = 7, (2, 3)
    a = np.full((n,) + shape, np.nan, np.float32)
    for k, v in {1: 1.0, 4: 4.0}.items():
        a[k] = v
    out = {"qpf": a.copy(), "snow_model": a.copy(), "gust": np.full_like(a, np.nan)}
    q = [0] * n
    pipeline.fill_gaps(out, q, max_gap=2)
    assert [float(out["qpf"][k][0, 0]) for k in range(5)] == pytest.approx([0.5, 1.0, 2.0, 3.0, 4.0])    # leading hole from zero, inner hole linear
    assert not np.isfinite(out["qpf"][5:]).any()                               # trailing frames stay empty
    assert [bool(x & pipeline.Q_INTERP) for x in q] == [True, False, True, True, False, False, False]
    assert all(q[k] & pipeline.Q_EMPTY for k in (5, 6))
    long = np.full((6,) + shape, np.nan, np.float32)
    long[0] = 1.0
    long[5] = 6.0
    o2 = {"qpf": long}
    pipeline.fill_gaps(o2, [0] * 6, max_gap=2)
    assert not np.isfinite(o2["qpf"][1:5]).any()                               # a 4-frame hole is too long to guess


def test_fewer_methods_still_make_a_mean():
    a, b = np.full((2, 2), 2.0, np.float32), np.full((2, 2), 4.0, np.float32)
    assert np.allclose(pipeline._method_mean([a, None, None, None, None]), 2.0)
    assert np.allclose(pipeline._method_mean([a, b, None, None, None]), 3.0)
    assert not np.isfinite(pipeline._method_range([a, None, None])).any()      # a range needs two
    assert np.allclose(pipeline._method_range([a, b, None]), 2.0)


def test_deterministic_lift_is_fetched_once_for_all_levels(lift, rg):
    arr, info = lift
    ny, nx = rg.shape
    assert arr is not None and arr.shape == (2, 26, ny, nx) and arr.dtype == np.float16
    assert info["frames_ok"] == 2 and not info.get("problem")
    assert np.isfinite(arr).any(axis=(0, 2, 3)).all()                       # every level was filled
    assert np.nanmax(arr) > 0.05 and abs(float(np.nanmax(arr))) < 20       # m/s, plausible


def test_lift_is_unavailable_when_the_deterministic_file_is_missing(rg):
    class NoDet(FakeS3):
        def head_object(self, Bucket, Key, **kw):
            if Key.startswith("rrfs."):
                raise KeyError(Key)
            return super().head_object(Bucket, Key, **kw)

        def get_object(self, Bucket, Key, **kw):
            if Key.startswith("rrfs."):
                raise KeyError(Key)
            return super().get_object(Bucket, Key, **kw)

    arr, info = pipeline.process_lift(S3Source(NoDet(NOW)), CYC, rg, 6)
    assert arr is None and "no deterministic file" in info["problem"]


def test_grid_check_rejects_a_shifted_grid(rg):
    lat = np.zeros(rg.native_shape)
    lon = np.zeros(rg.native_shape)
    ok = rg.idx >= 0
    lat.ravel()[rg.idx[ok]] = rg.lat[ok]
    lon.ravel()[rg.idx[ok]] = rg.lon[ok]
    assert pipeline.grid_matches(rg, lat, lon)
    assert not pipeline.grid_matches(rg, lat + 0.5, lon)
    assert not pipeline.grid_matches(rg, lat[:-1], lon[:-1])


def test_cobb_with_lift_products(member, member_lift):
    out, _ = member_lift
    assert np.isfinite(out["slr_cobblift"]).any() and np.isfinite(out["snow_cobblift"]).any() and np.isfinite(out["snow_cobblift_1h"]).any()
    assert np.nanmin(out["slr_cobblift"]) >= 0 and np.nanmax(out["slr_cobblift"]) <= 30 + 1e-3
    liq = np.nan_to_num(out["snow_12"]) / 12.0
    assert (np.nan_to_num(out["snow_cobblift"]) <= 30 * liq + 1e-3).all()
    # the lift changes the ratio where there is ascent, and nothing else about the member
    base, plain = member[0], out["slr_cobb"]
    np.testing.assert_allclose(base["slr_cobb"], plain, equal_nan=True)
    assert np.nanmax(np.abs(out["slr_cobblift"] - plain)) > 0.2
    for k in ("snow_12", "snow_kuchera", "snow_cobb", "qpf"):
        np.testing.assert_allclose(out[k], base[k], equal_nan=True, atol=1e-5)
    # it never feeds the spread or the mean
    assert "snow_cobblift" not in pipeline.SPREAD_METHODS and "snow_cobblift" not in pipeline.SNOW_METHODS


def test_without_lift_the_lift_products_stay_empty(member):
    out, _, _ = member
    assert not np.isfinite(out["slr_cobblift"]).any() and not np.isfinite(out["snow_cobblift"]).any()


class _TwoMembersDown(FakeS3):
    def _match(self, Key):
        if "/m004/" in Key or "/m005/" in Key:
            raise KeyError(Key)
        return super()._match(Key)


def test_incomplete_cycle_is_used_when_enough_members_finished():
    src = S3Source(_TwoMembersDown(NOW))
    assert pipeline.find_cycle(src, NOW, 6) is None                              # strict: two members never publish, so no cycle qualifies ...
    assert pipeline.find_cycle(src, NOW, 6, min_members=3) == CYC                # ... but three complete members are enough
    assert pipeline.find_cycle(src, NOW, 6, min_members=4) is None
    assert pipeline.complete_sources(src, CYC, 6) == ["refs", "m001", "m002", "m003"]
    assert pipeline.find_cycle(S3Source(FakeS3(NOW)), NOW, 6, min_members=3) == CYC


def test_a_failing_source_does_not_stop_the_build(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("member exploded")
    monkeypatch.setattr(build, "_run_source", boom)
    s, blobs, info = build.run_source(("m002", CYC.isoformat(), 6, str(tmp_path / "x"), None, str(tmp_path)))
    assert s == "m002" and blobs == {} and "member exploded" in info["error"]


def test_up_to_date_waits_for_late_members():
    m = {"cycle": "2026-11-05T12:00:00Z", "code": "c", "complete": ["refs", "m001", "m002", "m003"]}
    assert build.up_to_date(m, "2026-11-05T12:00:00Z", "c", ["m003", "m002", "m001", "refs"])
    assert not build.up_to_date(m, "2026-11-05T12:00:00Z", "c", ["refs", "m001", "m002", "m003", "m004"])     # m004 has finished since
    assert build.up_to_date({**m, "complete": None}, "2026-11-05T12:00:00Z", "c", ["refs"])                    # older sites without the field
