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
    q, s10, sk, sm = (np.nan_to_num(out[k]) for k in ("qpf", "snow_10", "snow_kuchera", "snow_maxt"))
    assert (np.diff(q, axis=0) >= -1e-4).all()                       # accumulations never decrease
    assert (s10 <= 10 * q + 1e-3).all()                              # snow liquid <= total liquid
    assert (sk >= 3 * s10 / 10 - 1e-3).all() and (sk <= 40 * s10 / 10 + 1e-3).all()   # ratios clipped 3..40
    assert (np.nan_to_num(out["frzr"]) + np.nan_to_num(out["sleet"]) <= q + 1e-3).all()
    pt = sum(np.nan_to_num(out[k]) for k in ("psnow", "psleet", "pfrzr", "prain"))
    assert pt.max() <= 100.0 + 1e-3
    assert out["slr_kuchera"].min() >= 3 and out["slr_kuchera"].max() <= 40
    assert out["dgz_depth"].min() >= 0 and out["dgz_depth"].max() <= 1000


def test_member_fetches_are_lean(member):
    _, info, fake = member
    assert info["requests"] > 0
    assert fake.range_requests and all(rng and rng.startswith("bytes=") for _, rng in fake.range_requests)   # always byte ranges
    whole = sum(len(blob) for blob, _ in fake._files.values())
    assert info["bytes"] < whole and info["bytes"] > 0


@pytest.mark.parametrize("totals", [False, True])
def test_refs_accumulation_modes_agree(rg, totals):
    out, info = pipeline.process_refs(S3Source(FakeS3(NOW, refs_totals=totals)), CYC, rg, 6)
    assert info["mode"]["APCP"] == ("total" if totals else "hourly-sum")
    assert set(out) == set(pipeline.REFS_KEYS)
    assert np.isfinite(out["qpf"]).any() and np.isfinite(out["snow_10"]).any()
    globals().setdefault("_REFS", {})[totals] = out["qpf"]
    if len(_REFS) == 2:
        np.testing.assert_allclose(_REFS[False], _REFS[True], atol=0.02, equal_nan=True)


def test_missing_message_is_not_fatal(rg):
    fake = FakeS3(NOW, missing={("VIS", 3), ("RH", 3)})
    out, info = pipeline.process_member(S3Source(fake), CYC, "m002", rg, 6)
    assert not np.isfinite(out["vis"][0]).any() and np.isfinite(out["vis"][1]).any()
    assert any(k.startswith("VIS") for k in info["missing"])
    assert np.isfinite(out["qpf"]).any() and np.isfinite(out["t2"]).any()


def test_missing_hour_poisons_only_totals(rg):
    fake = FakeS3(NOW, missing={("APCP", 2)})
    out, info = pipeline.process_member(S3Source(fake), CYC, "m003", rg, 6)
    assert not np.isfinite(out["qpf"]).any()            # an accumulation with a hole is withheld, never under-reported
    assert np.isfinite(out["gust"]).any() and np.isfinite(out["slr_kuchera"]).any()
