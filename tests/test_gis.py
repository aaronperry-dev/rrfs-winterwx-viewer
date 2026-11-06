"""Highway / city / town / peak layers (needs the Natural Earth files; skipped offline) and the SLR colour ramp."""
import csv
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


def test_towns_are_new_england_places_smaller_than_the_cities_and_zoom_tiered(g):
    towns, cities = g["towns"], g["cities"]
    assert 800 < len(towns) < 1200
    assert all(0 <= t[1] <= 1 and 0 <= t[2] <= 1 for t in towns)
    assert not any(t[0] == c[0] and abs(t[1] - c[1]) < 0.012 and abs(t[2] - c[2]) < 0.012 for t in towns for c in cities)      # no town sits on top of a city of its name
    assert sum(t[0] == "Newport" for t in towns) == 3                                                                    # the NH, VT and ME Newports stay (the RI one is a city)
    names = {t[0] for t in towns}
    assert {"West Hartford", "Amherst", "Keene", "Brattleboro", "Stowe", "Kennebunk"} <= names           # smaller places in every state, incl. the "West Hartford" kind
    assert not names & {"Dorchester", "South Boston", "Jamaica Plain", "North Stamford"}                              # neighbourhoods / census pieces are cleaned out
    zooms = [t[3] for t in towns]
    assert zooms == sorted(zooms) and zooms[0] == 1.6 and zooms[-1] == 5.6                                          # biggest first, fewest-zoom first
    assert gis.ZMAX_TOWN <= 12


def test_peaks_have_name_elevation_position_and_tier_zoom(g):
    peaks = g["peaks"]
    by = {p[0]: p for p in peaks}
    assert len(peaks) >= 40 and by["Mount Washington"][1] == 6288 and by["Katahdin"][1] == 5269 and by["Mount Mansfield"][1] == 4395
    assert all(0 <= p[2] <= 1 and 0 <= p[3] <= 1 and p[4] in gis.PEAK_ZOOM.values() for p in peaks)
    assert [p[4] for p in peaks] == sorted(p[4] for p in peaks)                      # most important first
    assert by["Mount Washington"][4] == 1.0 and by["Jerimoth Hill"][4] == max(gis.PEAK_ZOOM.values())
    assert by["Mount Washington"][2] < by["Katahdin"][2] and by["Katahdin"][3] < by["Mount Washington"][3]            # Katahdin is north-east of Washington (y runs down)


def test_peak_and_place_files_sit_inside_their_states():
    import cartopy.io.shapereader as sr
    from shapely.geometry import Point
    from rrfs_viewer import boundaries as bnd
    try:
        recs = [r for r in sr.Reader(bnd.natural_earth_path("cultural", "admin_1_states_provinces_lakes")).records() if r.attributes.get("adm0_a3") == "USA"]
    except Exception as exc:
        pytest.skip(f"Natural Earth data not available: {exc}")
    st = {r.attributes["postal"]: r.geometry for r in recs if r.attributes["postal"] in {"ME", "NH", "VT", "MA", "CT", "RI", "NY"}}
    for path in (gis.DATA_DIR / "peaks.csv", gis.DATA_DIR / "places.csv"):
        for r in csv.DictReader(open(path)):
            pt = Point(float(r["lon"]), float(r["lat"]))
            near = min(st, key=lambda k: st[k].distance(pt))            # (the generalised coast and borders can put a point a couple of km off its state, but never nearer another one)
            assert near == r["state"] or st[r["state"]].distance(pt) < 0.03, (r["name"], r["state"], near)


def test_file_is_json_serialisable_and_modest(g):
    blob = gzip.compress(json.dumps(g, separators=(",", ":")).encode(), 9)
    assert 50_000 < len(blob) < 900_000 and len(g["towns"]) and len(g["peaks"])


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
