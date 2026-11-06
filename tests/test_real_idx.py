"""Every message the pipeline asks for must exist in real RRFS index files (member m001, 2026-11-05 18Z, F006, NOMADS layout)."""
import types
from datetime import datetime, timezone
from pathlib import Path

from rrfs_viewer import pipeline as P
from rrfs_viewer import s3
from rrfs_viewer.config import COBB_UPPER, FILE_2D, FILE_PRS, RH_LEVELS, T_LEVELS, frames

DATA = Path(__file__).parent / "data"
E2D = s3.parse_idx((DATA / "rrfs.t18z.m001.2dfldnomads.3km.f006.idx.txt").read_text())
EPRS = s3.parse_idx((DATA / "rrfs.t18z.m001.prslevnomads.3km.f006.idx.txt").read_text())


class _Src:
    def locate(self, kind, source, cycle, h):
        return types.SimpleNamespace(entries=E2D if kind is FILE_2D else EPRS) if h == 6 else None


def _plan():
    cyc = datetime(2026, 11, 5, 18, tzinfo=timezone.utc)
    acc_p, acc_s = (P.Acc(_Src(), "m001", cyc, FILE_2D, v) for v in ("APCP", "ASNOW"))
    F = frames(60)
    return acc_p, acc_s, P._plan_member(F, F.index(6), acc_p, acc_s)


def test_index_parses_despite_the_trailing_ens_field():
    assert len(E2D) == 56 and len(EPRS) == 112
    assert all(e.end is None or e.end >= e.start for e in E2D + EPRS)


def test_every_request_at_f006_is_in_the_real_inventories():
    _, _, (R, plans, reqs) = _plan()
    missing = [(r.kind.key, r.var, r.level, r.info) for r in dict.fromkeys(reqs)
               if r.fhr == 6 and not s3.find_entry(E2D if r.kind is FILE_2D else EPRS, r.var, r.level, r.info)]
    assert missing == []
    assert sum(1 for r in dict.fromkeys(reqs) if r.fhr == 6) >= 30


def test_cobb_and_melt_inputs_exist():
    for p in tuple(RH_LEVELS) + tuple(COBB_UPPER):
        assert s3.find_entry(EPRS, "TMP", f"{p} mb") and s3.find_entry(EPRS, "RH", f"{p} mb")
    for p in T_LEVELS:
        assert s3.find_entry(EPRS, "TMP", f"{p} mb")
    for var in ("TMP", "DPT"):
        assert s3.find_entry(E2D, var, "2 m above ground")
    assert s3.find_entry(E2D, "PRES", "surface")


def test_accumulation_windows_as_published():
    acc_p, acc_s, _ = _plan()
    assert acc_p.windows(6) == {(0, 6), (5, 6)}          # APCP: a stored 1-hour window and a run total
    assert acc_s.windows(6) == {(0, 6)}                  # ASNOW: run total only, so its 1-hour map is a difference of totals
    assert acc_p.inc1(6) is not None and acc_s.total(6) is not None


def test_no_pressure_level_omega_in_the_files():
    """Why Cobb is out of the dashboard: no VVEL / DZDT on pressure levels (only the column MAXUVV / MAXDVV in the 2D file)."""
    assert not [e for e in EPRS if e.var in ("VVEL", "DZDT", "VVELSIG")]
    assert {e.var for e in E2D} >= {"MAXUVV", "MAXDVV"}


# ---- REFS ensemble mean (refs.t18z.mean.f53.conus.grib2, 2026-11-05 18Z) ------------------------------------------
REFS = s3.parse_idx((DATA / "refs.t18z.mean.f53.conus.grib2.idx.txt").read_text())


def test_refs_mean_requests_exist():
    """process_refs asks only for these; they must be in the real mean file."""
    for var, level in (("TMP", "2 m above ground"), ("WIND", "10 m above ground"), ("VIS", "surface"), ("CSNOW", "surface"),
                       ("CICEP", "surface"), ("CFRZR", "surface"), ("CRAIN", "surface")):
        assert s3.find_entry(REFS, var, level), (var, level)
    for var in ("APCP", "WEASD", "ASNOW"):
        e = [x for x in REFS if x.var == var]
        assert e and "52-53 hour acc" in e[0].info          # 1-hour windows, not run totals


def test_refs_mean_omega_is_one_level_only():
    """The only vertical motion anywhere in the files seen: DZDT (geometric w, m/s) at 700 mb and a 700-500 mb layer, in the *mean*.
    Too sparse to give Cobb a profile of its own (T on 4 levels, RH on 1), so at most it could stand in for the lift term."""
    assert s3.find_entry(REFS, "DZDT", "700 mb") and s3.find_entry(REFS, "DZDT", "700-500 mb above ground")
    assert {e.level for e in REFS if e.var == "DZDT"} == {"700 mb", "700-500 mb above ground"}
    assert [e.level for e in REFS if e.var == "RH"] == ["700 mb"]
    assert {e.level for e in REFS if e.var == "TMP" and e.level.endswith(" mb")} == {"250 mb", "500 mb", "700 mb", "850 mb", "925 mb"}


# --------------------------------------------------------------------------- deterministic RRFS (full prslev, F003, 00Z 2026-11-06)
EDET = s3.parse_idx((DATA / "rrfs.t00z.prslev.3km.f003.conus.idx.txt").read_text())


def test_deterministic_file_has_dzdt_at_every_lift_level():
    from rrfs_viewer.config import LIFT_LEVELS
    missing = [p for p in LIFT_LEVELS if not s3.find_entry(EDET, "DZDT", f"{p} mb", r"^3 hour fcst")]
    assert missing == [] and len(LIFT_LEVELS) == 26
    # each of those messages is small enough to fetch one by one: well under 2 MB
    sizes = [s3.find_entry(EDET, "DZDT", f"{p} mb").end - s3.find_entry(EDET, "DZDT", f"{p} mb").start for p in LIFT_LEVELS]
    assert max(sizes) < 2_000_000


def test_deterministic_key_layout():
    from rrfs_viewer.config import FILE_PRS
    cyc = datetime(2026, 11, 6, 6, tzinfo=timezone.utc)
    assert FILE_PRS.key_for("det", cyc, 3) == "rrfs.20261106/06/rrfs.t06z.prslev.3km.f003.conus.grib2"
