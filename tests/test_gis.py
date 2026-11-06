"""Highway / city layers (needs the Natural Earth files; skipped offline) and the SLR colour ramp."""
import gzip
import json

import pytest

from rrfs_viewer import basemap, colors, gis


@pytest.fixture(scope="module")
def g():
    try:
        return gis.build()
    except Exception as exc:                                  # no network / no cached shapefiles
        pytest.skip(f"Natural Earth data not available: {exc}")


def test_roads_cover_all_three_classes_and_stay_near_the_frame(g):
    assert g["units"] == basemap.LINE_UNITS and set(g["roads"]) == {"interstate", "us", "state"}
    lo, hi = -basemap.LINE_MARGIN - 1e-3, 1 + basemap.LINE_MARGIN + 1e-3
    for name, polys in g["roads"].items():
        assert len(polys) > 50, name
        for p in polys:
            assert len(p) >= 4 and len(p) % 2 == 0
            xs, ys = [sum(p[0::2][:i + 1]) / g["units"] for i in range(len(p) // 2)], [sum(p[1::2][:i + 1]) / g["units"] for i in range(len(p) // 2)]
            assert lo <= min(xs) and max(xs) <= hi and lo <= min(ys) and max(ys) <= hi, name


def test_labels_are_well_formed_and_sorted(g):
    labels = g["labels"]
    assert {l[0] for l in labels if l[1] == 0} >= {"I-95", "I-90", "I-91", "I-93", "I-89"}
    assert any(l[0] == "US 1" for l in labels) and any(l[0].startswith("MA ") for l in labels)
    assert all(l[0].startswith("I-") for l in labels if l[1] == 0) and all(l[0].startswith("US ") for l in labels if l[1] == 1)
    assert all(0 <= l[2] <= 1 and 0 <= l[3] <= 1 for l in labels)
    assert [l[4] for l in labels] == sorted(l[4] for l in labels)          # fewest-zoom first, so the page can stop early
    assert min(l[4] for l in labels if l[1] == 0) == 1.0 and min(l[4] for l in labels if l[1] == 2) > 2             # interstates at the full view, state routes only zoomed in


def test_each_new_england_state_has_cities_and_a_capital(g):
    names = {c[0] for c in g["cities"]}
    assert {"Boston", "Providence", "Hartford", "Concord", "Montpelier", "Augusta", "Portland", "Burlington", "Manchester", "Worcester"} <= names
    caps = [c[0] for c in g["cities"] if c[3]]
    assert sorted(caps) == ["Augusta", "Boston", "Concord", "Hartford", "Montpelier", "Providence"]
    assert [c[3] for c in g["cities"]][:6] == [1] * 6                     # capitals first
    assert all(0 <= c[1] <= 1 and 0 <= c[2] <= 1 for c in g["cities"])


def test_file_is_json_serialisable_and_modest(g):
    blob = gzip.compress(json.dumps(g, separators=(",", ":")).encode(), 9)
    assert 50_000 < len(blob) < 900_000


def test_slr_ramp_is_colorbrewer_rdylbu_reversed():
    st = colors.SLR
    assert st.levels == [4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30] and len(st.colors) == len(st.levels) + 1
    assert st.colors[:11] == ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee090", "#ffffbf", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4", "#313695"]
    assert st.colors[-1] == st.colors[-2]
    rgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
    import colorsys
    hue = [colorsys.rgb_to_hsv(*(v / 255 for v in rgb(c)))[0] for c in st.colors[1:]]   # hue runs steadily from red (4-6:1) through yellow to blue (25:1+)
    assert hue == sorted(hue) and hue[0] < 0.05 and hue[-1] > 0.6
    assert rgb(st.colors[0])[0] > 150 and rgb(st.colors[0])[2] < 60 and rgb(st.colors[-1])[2] > 120 > rgb(st.colors[-1])[0]
    assert st.colors[0] != "#ffffff" and all(c.lower() != "#ffffff" for c in st.colors)
