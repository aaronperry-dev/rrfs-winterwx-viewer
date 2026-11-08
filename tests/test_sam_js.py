"""SAM (spatially aligned mean) is implemented in the page's JavaScript; these tests run that very source under Node on synthetic fields."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HTML = Path(__file__).resolve().parents[1] / "site" / "index.html"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

DRIVER = r"""
const NX = 455, NY = 293;
function blob(cx, cy, sx, sy, a) { const o = new Float32Array(NX * NY); for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) o[y * NX + x] = a * Math.exp(-((x - cx) ** 2 / (2 * sx * sx) + (y - cy) ** 2 / (2 * sy * sy))); return o; }
const add = (a, b) => a.map((v, i) => v + b[i]);
const base = (dx, dy, a) => add(blob(220 + dx, 140 + dy, 40, 18, a), blob(300 + dx, 100 + dy, 15, 15, 0.6 * a));
function centroid(a) { let s = 0, sx = 0, sy = 0; for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) { const v = a[y * NX + x]; s += v; sx += v * x; sy += v * y; } return [sx / s, sy / s]; }
const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
const out = {}, mx = a => { let m = -Infinity; for (const v of a) if (v > m) m = v; return m; };
const tgt = base(0, 0, 1);
out.moves = [[0, 0], [30, 0], [-25, 15], [0, -40]].map(([dx, dy]) => { const m = base(dx, dy, 1.3), al = samAlign(m, tgt, NX, NY); return [dist(centroid(m), centroid(tgt)), dist(centroid(al), centroid(tgt))]; });
// distribution kept: the sorted values of the aligned member equal those of the original (peaks are not lost to the interpolation)
const m = base(30, 10, 1.3), al = samAlign(m, tgt, NX, NY), a = Float32Array.from(m).sort(), b = Float32Array.from(al).sort();
let md = 0; for (let i = 0; i < a.length; i++) md = Math.max(md, Math.abs(a[i] - b[i])); out.maxDistDiff = md;
// five members, displaced differently: the aligned mean keeps a sharper, higher peak than the plain mean
const shifts = [[-30, 5], [20, -10], [35, 15], [-10, -25], [0, 0]], mem = shifts.map(([dx, dy], i) => base(dx, dy, 1 + 0.1 * i));
const mean = arrs => { const o = new Float32Array(NX * NY); for (const r of arrs) for (let i = 0; i < o.length; i++) o[i] += r[i] / arrs.length; return o; };
const plain = mean(mem), pm = pmmField(mem, 0, NX, NY), sam = mean(mem.map(r => samAlign(r, pm, NX, NY)));
out.peaks = [mx(plain), mx(sam)];
out.spread = [dist(centroid(plain), centroid(sam))];
// missing data stays missing, and a dry member stays dry
const nanm = base(10, 0, 1); for (let x = 0; x < 50; x++) nanm[x] = NaN;
const aln = samAlign(nanm, tgt, NX, NY); out.nanKept = Number.isNaN(aln[0]) || Number.isNaN(aln[10]);
out.dry = mx(samAlign(new Float32Array(NX * NY), tgt, NX, NY));
console.log(JSON.stringify(out));
"""


def extract():
    s = HTML.read_text()
    a = s.index("const SAM = {")
    b = s.index("function work(m)")
    b = s.index("\n", b)
    pmm0 = s.index("function lowerBound")
    pmm1 = a
    reduce0 = s.index("function percentile(")
    reduce1 = s.index("// Probability matching (Ebert 2001)")
    return 'const PCT_MODES = {p05:.05,p10:.1,p25:.25,p50:.5,p75:.75,p90:.9,p95:.95};\n' + s[reduce0:reduce1] + s[pmm0:pmm1] + s[a:b]


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    f = tmp_path_factory.mktemp("sam") / "t.js"
    f.write_text(extract() + "\n" + DRIVER)
    r = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_members_are_moved_onto_the_target(result):
    for before, after in result["moves"]:
        assert after < 1.0 and (before < 1e-3 or after < 0.1 * before)


def test_aligned_member_keeps_its_own_distribution(result):
    assert result["maxDistDiff"] < 2e-3


def test_aligned_mean_is_sharper_than_plain_mean(result):
    plain, sam = result["peaks"]
    assert sam > 1.15 * plain


def test_missing_data_and_dry_members(result):
    assert result["nanKept"] and result["dry"] == 0


# ---- the other means: checked against independent NumPy implementations on random ensembles with ties and gaps
EXTRACT_REDUCE = r"""
const NX = 90, NY = 60, M = 5;
let seed = 12345; const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
const members = Array.from({ length: M }, () => Float32Array.from({ length: NX * NY }, () => { const r = rnd(); return r < 0.4 ? 0 : Math.round(r * r * 400) / 20; }));   // dry cells, ties (0.05 steps), a heavy tail
members[1][7] = NaN;
const out = { members: members.map(a => Array.from(a)), NX, NY };
out.mean = Array.from(reduce(members, "mean")); out.p50 = Array.from(reduce(members, "p50")); out.p25 = Array.from(reduce(members, "p25")); out.p90 = Array.from(reduce(members, "p90"));
out.pmm = Array.from(pmmField(members, 0, NX, NY)); out.lpmm = Array.from(pmmField(members, 120, NX, NY));
console.log(JSON.stringify(out));
"""


def test_mean_percentiles_pmm_lpmm_match_independent_numpy(tmp_path):
    np = pytest.importorskip("numpy")
    f = tmp_path / "r.js"
    f.write_text(extract() + "\n" + EXTRACT_REDUCE)
    r = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout.strip().splitlines()[-1])
    nx, ny = o["NX"], o["NY"]
    M = np.array(o["members"], dtype=np.float32).reshape(-1, ny, nx)
    M[1].flat[7] = np.nan
    mean = np.nanmean(M.astype(np.float64), axis=0).astype(np.float32)
    got = lambda k: np.array(o[k], dtype=np.float32).reshape(ny, nx)
    assert np.allclose(got("mean"), mean, atol=1e-6)
    for k, q in (("p50", 50), ("p25", 25), ("p90", 90)):                      # linear interpolation between sorted members (NumPy's default)
        assert np.allclose(got(k), np.nanpercentile(M.astype(np.float64), q, axis=0), atol=1e-4), k

    def pmm(pool, m):                                                           # Ebert (2001): value of equal rank from the pooled members
        K, P = m.size, pool.size
        sm, sp = np.sort(m.ravel()), np.sort(pool[np.isfinite(pool)])
        r_ = (np.searchsorted(sm, m.ravel(), "left") + np.searchsorted(sm, m.ravel(), "right") - 1) / 2
        return sp[np.minimum(sp.size - 1, np.floor((r_ + 0.5) * sp.size / K).astype(int))].reshape(m.shape)

    assert np.allclose(got("pmm"), pmm(M, mean), atol=1e-6)
    ps = max(4, round(120 / 9))
    ref = np.full((ny, nx), np.nan, dtype=np.float32)                          # Snook et al. (2020): patches receive values from a 3x3-patch neighbourhood
    for py in range(0, ny, ps):
        for px in range(0, nx, ps):
            x0, x1, y0, y1 = max(0, px - ps), min(nx, px + 2 * ps), max(0, py - ps), min(ny, py + 2 * ps)
            pool, mv = M[:, y0:y1, x0:x1], mean[y0:y1, x0:x1]
            sub = mean[py:min(ny, py + ps), px:min(nx, px + ps)]
            if np.nanmin(pool) == np.nanmax(pool):
                ref[py:py + sub.shape[0], px:px + sub.shape[1]] = np.nanmin(pool)
                continue
            sm, sp = np.sort(mv.ravel()), np.sort(pool[np.isfinite(pool)])
            r_ = (np.searchsorted(sm, sub, "left") + np.searchsorted(sm, sub, "right") - 1) / 2
            ref[py:py + sub.shape[0], px:px + sub.shape[1]] = sp[np.minimum(sp.size - 1, np.floor((r_ + 0.5) * sp.size / mv.size).astype(int))]
    assert np.allclose(got("lpmm"), ref, atol=1e-6)
    assert np.nanmax(got("pmm")) <= np.nanmax(M) and np.nanmin(got("pmm")) >= 0


SAM_PROPS = r"""
const NX = 455, NY = 293;
function blob(cx, cy, sx, sy, a) { const o = new Float32Array(NX * NY); for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) o[y * NX + x] = a * Math.exp(-((x - cx) ** 2 / (2 * sx * sx) + (y - cy) ** 2 / (2 * sy * sy))); return o; }
const mean = arrs => { const o = new Float32Array(NX * NY); for (const r of arrs) for (let i = 0; i < o.length; i++) o[i] += r[i] / arrs.length; return o; };
const sum = a => { let s = 0; for (const v of a) s += v; return s; }, mx = a => { let m = -Infinity; for (const v of a) if (v > m) m = v; return m; };
const band = (dx, dy, a) => blob(160 + dx, 100 + dy, 14, 10, a), res = {};
const sets = { displaced: [[-20, -15, 1], [18, 20, 1.3], [-10, 25, 0.8], [25, -18, 1.15], [0, 5, 0.9]], inplace: [[0, 0, 1], [0, 0, 1.3], [0, 0, 0.8], [0, 0, 1.15], [0, 0, 0.9]], bimodal: [[0, 0, 1], [0, 0, 1], [0, 0, 1], [150, 90, 1], [150, 90, 1]] };
for (const [k, spec] of Object.entries(sets)) {
  const ms = spec.map(([dx, dy, a]) => band(dx, dy, a)), pm = pmmField(ms, 0, NX, NY), al = ms.map(a => samAlign(a, pm, NX, NY)), sam = mean(al), pl = mean(ms), lp = pmmField(al, 180, NX, NY);
  let md = 0; for (let i = 0; i < pl.length; i++) md = Math.max(md, Math.abs(sam[i] - pl[i]));
  res[k] = { sumRatio: sum(sam) / sum(pl), peakSam: mx(sam), peakMean: mx(pl), maxDiff: md, lpmmMax: mx(lp), memberMax: Math.max(...ms.map(mx)) };
}
console.log(JSON.stringify(res));
"""


def test_sam_conserves_total_sharpens_displaced_and_does_nothing_in_place(tmp_path):
    f = tmp_path / "s.js"
    f.write_text(extract() + "\n" + SAM_PROPS)
    r = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout.strip().splitlines()[-1])
    for k, v in o.items():
        assert abs(v["sumRatio"] - 1) < 2e-3, (k, v)                           # every aligned member keeps its own distribution, so the domain total is the mean's
        assert v["lpmmMax"] <= v["memberMax"] * (1 + 1e-5), k                    # no value larger than any member produced
    assert o["displaced"]["peakSam"] > 1.3 * o["displaced"]["peakMean"]        # members that disagree on position: the aligned mean keeps the peak
    assert o["inplace"]["maxDiff"] < 0.02 * o["inplace"]["peakMean"]           # members already in place: SAM is the mean
    assert o["bimodal"]["peakSam"] < 1.1 * o["bimodal"]["peakMean"]            # two clusters beyond the search range: no invented structure
