"""Which cycle is built, when a build counts as up to date, and the code fingerprint (no GRIB data needed: a stand-in S3 client that only knows which keys exist)."""
import json
import urllib.request
from datetime import datetime, timedelta, timezone

from rrfs_viewer import build, pipeline
from rrfs_viewer.config import FILE_2D, FILE_PRS, MEMBERS
from rrfs_viewer.s3 import S3Source

UTC = timezone.utc
at = lambda day, hour, minute=0: datetime(2026, 11, day, hour, minute, tzinfo=UTC)


class Keys:
    """Just enough of a boto3 client for ``S3Source.exists``."""

    def __init__(self, keys):
        self.keys = set(keys)

    def head_object(self, Bucket, Key):
        if Key not in self.keys:
            raise KeyError(Key)


def published(cycle, members=MEMBERS, refs=True, fhr=60):
    keys = [FILE_2D.key_for("refs", cycle, fhr) + ".idx"] if refs else []
    for m in members:
        keys += [FILE_PRS.key_for(m, cycle, fhr) + ".idx", FILE_2D.key_for(m, cycle, fhr) + ".idx"]
    return keys


def src_with(*per_cycle):
    keys = []
    for cyc, kw in per_cycle:
        keys += published(cyc, **kw)
    return S3Source(Keys(keys))


# ------------------------------------------------------------------------------------------------ cycle choice
def test_complete_newest_cycle_wins():
    src = src_with((at(8, 18), {}), (at(9, 0), {}))
    assert pipeline.choose_cycle(src, at(9, 3, 50), 60) == (at(9, 0), False)


def test_slow_cycle_keeps_the_previous_one_until_the_partial_deadline():
    """06z has only four members.  The site stays on the complete 00z cycle while 06z may still finish (before +5 h) ..."""
    four = ("m001", "m002", "m004", "m005")
    src = src_with((at(9, 0), {}), (at(9, 6), {"members": four}))
    assert pipeline.choose_cycle(src, at(9, 9, 50), 60) == (at(9, 0), False)         # +3 h 50
    assert pipeline.choose_cycle(src, at(9, 10, 30), 60) == (at(9, 0), False)        # +4 h 30
    # ... and after the last retry (+5 h 30) the newer cycle goes out with the members it has, instead of staying on the 00z cycle until 12z + 3 h 50
    assert pipeline.choose_cycle(src, at(9, 11, 30), 60) == (at(9, 6), True)
    assert pipeline.choose_cycle(src, at(9, 14, 0), 60) == (at(9, 6), True)


def test_too_few_members_never_make_a_partial_cycle():
    src = src_with((at(9, 0), {}), (at(9, 6), {"members": ("m001", "m002")}))
    assert pipeline.choose_cycle(src, at(9, 12, 30), 60) == (at(9, 0), False)


def test_without_any_complete_cycle_the_partial_one_is_used_at_any_age():
    src = src_with((at(9, 6), {"members": ("m001", "m002", "m003")}))
    assert pipeline.choose_cycle(src, at(9, 9, 50), 60) == (at(9, 6), True)


def test_nothing_published():
    assert pipeline.choose_cycle(S3Source(Keys([])), at(9, 9, 50), 60) == (None, False)


def test_late_member_completes_the_cycle():
    src = src_with((at(9, 0), {}), (at(9, 6), {}))                                   # the fifth member has arrived after all
    assert pipeline.choose_cycle(src, at(9, 12, 30), 60) == (at(9, 6), False)


# ------------------------------------------------------------------------------------------------ up to date
def test_failed_member_is_retried_by_the_next_run():
    m = {"cycle": "2026-11-09T06:00:00Z", "code": "c", "complete": ["refs", *MEMBERS], "errors": {"m003": "OSError: boom"}}
    assert not build.up_to_date(m, "2026-11-09T06:00:00Z", "c", ["refs", *MEMBERS])
    assert build.up_to_date({**m, "errors": {}}, "2026-11-09T06:00:00Z", "c", ["refs", *MEMBERS])
    assert build.up_to_date({**m, "errors": {"hrrr": "no file"}}, "2026-11-09T06:00:00Z", "c", ["refs", *MEMBERS])     # HRRR is optional


# ------------------------------------------------------------------------------------------------ fingerprint and manifest fetch
def test_code_version_changes_with_every_data_table(monkeypatch, tmp_path):
    base = build.code_version()
    data = build.Path(build.__file__).resolve().parent / "data"
    seen = []
    real = build.Path.read_bytes

    def spy(self):
        seen.append(self.name)
        return real(self)

    monkeypatch.setattr(build.Path, "read_bytes", spy)
    assert build.code_version() == base
    assert {p.name for p in data.glob("*.csv")} <= set(seen)                          # places.csv, peaks.csv, ski_areas.csv, stations.csv all count


def test_deployed_manifest_bypasses_the_cdn_cache(monkeypatch):
    got = {}

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return json.dumps({"cycle": "x"}).encode()

    def fake_urlopen(req, timeout=0):
        got["url"], got["headers"] = req.full_url, {k.lower(): v for k, v in req.header_items()}
        return R()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert build.deployed_manifest("https://o.github.io/r/") == {"cycle": "x"}
    assert got["url"].startswith("https://o.github.io/r/data.json?nocache=")
    assert got["headers"]["cache-control"] == "no-cache"
