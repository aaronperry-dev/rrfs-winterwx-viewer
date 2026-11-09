"""The significance-line (thickness) additions: catalog entries, the manifest field, and the page's isoline routine (run in node)."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from rrfs_viewer.catalog import PRODUCTS

HTML = (Path(__file__).resolve().parent.parent / "site" / "index.html").read_text()


def test_thickness_lines_in_catalog():
    by = {p.key: p for p in PRODUCTS}
    assert by["thk_1000_850"].lines == ((130.0, "130 dam: rain/snow line"),)
    assert by["thk_850_700"].lines == ((154.0, "154 dam: snow line aloft"),)
    assert by["thk_1000_500"].lines == ((540.0, "540 dam: traditional rain/snow line"),)
    assert [p.key for p in PRODUCTS if p.lines] == ["thk_1000_850", "thk_850_700", "thk_1000_500"]       # no other product draws a line


def _function_source(name: str) -> str:
    i = HTML.index(f"function {name}(")
    depth, j = 0, HTML.index("{", i)
    for k in range(j, len(HTML)):
        depth += HTML[k] == "{"
        depth -= HTML[k] == "}"
        if depth == 0:
            return HTML[i:k + 1]
    raise AssertionError


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_isoline_follows_the_level_and_handles_gaps():
    js = _function_source("isoPolylines") + """
    const nx = 20, ny = 10, v = new Float32Array(nx * ny);
    for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) v[r * nx + c] = c;            // a ramp in x: the level 7.5 is the vertical line c = 7.5
    const a = isoPolylines(v, nx, ny, 7.5);
    const w = v.slice(); for (let c = 0; c < nx; c++) w[4 * nx + c] = NaN;                    // a row of missing data cuts the line in two
    const b = isoPolylines(w, nx, ny, 7.5);
    const none = isoPolylines(v, nx, ny, 99);
    // a closed ring: a bump in the middle
    const ring = new Float32Array(nx * ny); for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) ring[r * nx + c] = -Math.hypot(c - 10, r - 5);
    const rg = isoPolylines(ring, nx, ny, -3);
    console.log(JSON.stringify({ n: a.length, xs: a[0].map(p => p[0]), len: a[0].length, nb: b.length, none: none.length, ringN: rg.length,
      closed: rg.length === 1 && Math.hypot(rg[0][0][0] - rg[0][rg[0].length - 1][0], rg[0][0][1] - rg[0][rg[0].length - 1][1]) < 1e-6 }));
    """
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert out["n"] == 1 and out["len"] == 10 and all(abs(x - 7.5) < 1e-6 for x in out["xs"])
    assert out["nb"] == 2 and out["none"] == 0
    assert out["ringN"] == 1 and out["closed"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_freezing_line_smoothing_and_specs():
    js = "const D = { grid: { nx: 6, ny: 5 } };\nconst fzSmoothed = new WeakMap();\n" + _function_source("fzSmooth") + """
    const a = new Float32Array(30).fill(10); a[14] = 40; a[0] = NaN;
    const o = fzSmooth(a);
    console.log(JSON.stringify({ same: fzSmooth(a) === o, centre: o[14], nan: Number.isNaN(o[0]), far: o[29] }));
    """
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert out["same"] and out["nan"] and out["far"] == 10
    assert abs(out["centre"] - (40 + 8 * 10) / 9) < 1e-4                    # a lone warm cell is spread over its 3 x 3 neighbourhood
    # every map gets the line: the 2 m temperature (or HRRR's) is the fallback, the temperature maps use their own field
    assert 'const FZ_T2 = [32, "32 °F (0 °C) 2 m temperature"' in HTML
    for k in ("t2", "hr_t2", "tw2", "hr_tw2", "t925", "t850"):
        assert re.search(rf"\b{k}: \[(32|0),", HTML)
    assert '["z", "0 °C line"]' in HTML and "z: true," in HTML                 # a Map Layers switch, on by default
