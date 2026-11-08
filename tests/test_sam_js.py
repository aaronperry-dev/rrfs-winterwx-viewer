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
