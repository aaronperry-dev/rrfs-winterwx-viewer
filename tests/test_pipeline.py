"""End-to-end data path against the offline FakeS3 (real GRIB2 bytes, real .idx files, real Range semantics)."""
from datetime import datetime, timezone

import numpy as np
import pytest

from rrfs_viewer import build, pipeline
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


def test_one_hour_fields_are_consistent_with_totals(member):
    out, _, _ = member
    for base in ("qpf", "snow_12", "snow_kuchera", "snow_maxt", "snow_thick", "snow_mean", "snow_model"):
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


def test_missing_hour_poisons_only_totals(rg):
    fake = FakeS3(NOW, missing={("APCP", 2)})
    out, info = pipeline.process_member(S3Source(fake), CYC, "m003", rg, 6)
    assert not np.isfinite(out["qpf"]).any()            # an accumulation with a hole is withheld, never under-reported
    assert np.isfinite(out["gust"]).any() and np.isfinite(out["slr_kuchera"]).any()
