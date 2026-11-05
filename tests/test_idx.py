from datetime import datetime, timezone

from rrfs_viewer.config import FILE_2D, FILE_PRS, frames
from rrfs_viewer.s3 import find_entry, parse_idx

IDX = """\
1:0:d=2026110512:PRES:surface:3 hour fcst:
2:1000:d=2026110512:TMP:2 m above ground:3 hour fcst:
3:2500:d=2026110512:APCP:surface:2-3 hour acc fcst:
4:3100:d=2026110512:APCP:surface:0-3 hour acc fcst:
5:4000:d=2026110512:ASNOW:surface:0-3 hour acc fcst:
6:5200:d=2026110512:MAXUVV:100-1000 mb:2-3 hour max fcst:
"""


def test_parse_ranges():
    e = parse_idx(IDX)
    assert [x.var for x in e] == ["PRES", "TMP", "APCP", "APCP", "ASNOW", "MAXUVV"]
    assert (e[0].start, e[0].end) == (0, 999)
    assert e[-1].end is None                      # last message runs to the end of the file


def test_find_entry_by_time_descriptor():
    e = parse_idx(IDX)
    hourly = find_entry(e, "APCP", "surface", r"^2-3 hour acc")
    total = find_entry(e, "APCP", "surface", r"^0-3 hour acc")
    assert hourly.num == 3 and total.num == 4
    assert find_entry(e, "APCP", "surface").num == 3          # no regex: first match
    assert find_entry(e, "VIS", "surface") is None
    assert find_entry(e, "MAXUVV", "100-1000 mb", r"^2-3 hour max").start == 5200


def test_garbage_lines_ignored():
    assert parse_idx("\n<html>nope</html>\nabc:def:\n") == []


def test_keys():
    cyc = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)
    assert FILE_2D.key_for("m003", cyc, 6) == "rrfsens.20261105/12/m003/rrfs.t12z.m003.2dfldnomads.3km.f006.conus.grib2"
    assert FILE_PRS.key_for("m001", cyc, 60).endswith("rrfs.t12z.m001.prslevnomads.3km.f060.conus.grib2")
    assert FILE_2D.key_for("refs", cyc, 9) == "refs.20261105/12/ensprod/refs.t12z.mean.f09.conus.grib2"


def test_frames():
    assert frames(60)[0] == 3 and frames(60)[-1] == 60 and len(frames(60)) == 20
    assert frames(9) == (3, 6, 9)


def test_skip_logic():
    from rrfs_viewer.build import code_version, up_to_date
    code = code_version()
    m = {"cycle": "2026-11-05T12:00:00Z", "code": code, "demo": False}
    assert up_to_date(m, "2026-11-05T12:00:00Z", code)
    assert not up_to_date(m, "2026-11-05T18:00:00Z", code)                 # newer cycle
    assert not up_to_date({**m, "code": "oldcode"}, "2026-11-05T12:00:00Z", code)   # new code
    assert not up_to_date({k: v for k, v in m.items() if k != "code"}, "2026-11-05T12:00:00Z", code)   # pre-fingerprint site
    assert not up_to_date({**m, "demo": True}, "2026-11-05T12:00:00Z", code)
    assert not up_to_date(None, "2026-11-05T12:00:00Z", code)
