"""Probability maps take a ramp from the product: every product with members maps to a theme and every theme has 11 colours."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from rrfs_viewer.catalog import PRODUCTS

HTML = (Path(__file__).resolve().parent.parent / "site" / "index.html").read_text()


def _block(start: str, end: str) -> str:
    i = HTML.index(start)
    return HTML[i:HTML.index(end, i)]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_every_probability_product_has_a_theme_and_a_full_ramp():
    src = HTML[HTML.index("const PROB_RAMPS"):HTML.index("const POS = {")]
    prods = [{"key": p.key, "thr_dir": p.thr_dir} for p in PRODUCTS if not p.single or True]
    code = "const PROB = { levels: [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95], ticks: [10, 30, 50, 70, 90] };\n" + src + f"""
    const prods = {json.dumps(prods)}, used = {{}};
    for (const p of prods) {{ const t = probTheme(p); used[t] = (used[t] || 0) + 1; }}
    const st = probStyle({{ key: "snow_kuchera" }});
    console.log(JSON.stringify({{ used, sizes: Object.fromEntries(Object.entries(PROB_RAMPS).map(([k, v]) => [k, v.length])),
      hex: Object.values(PROB_RAMPS).every(r => r.every(c => /^#[0-9a-f]{{6}}$/.test(c))), len: st.colors.length, lv: st.levels.length, same: probStyle({{ key: "slr_12" }}) === st,
      t2: probTheme({{ key: "t2", thr_dir: "le" }}) + "/" + probTheme({{ key: "t2", thr_dir: "ge" }}), q: probTheme({{ key: "qpf" }}), f: probTheme({{ key: "frzr" }}), r: probTheme({{ key: "refc" }}) }}));
    """
    out = json.loads(subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout)
    assert all(n == 11 for n in out["sizes"].values()) and out["hex"]
    assert out["len"] == out["lv"] + 1 and out["same"]                               # colours: below the first level + one per bin + above the last
    assert set(out["used"]) <= set(out["sizes"])                                      # no product falls through to a missing ramp
    assert out["t2"] == "cold/warm" and out["q"] == "rain" and out["f"] == "ice" and out["r"] == "radar"
    assert {"snow", "cold", "rain", "ice", "wind", "radar"} <= set(out["used"])
