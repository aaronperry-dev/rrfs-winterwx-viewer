"""The vector boundary lines the page draws when zoomed in, and the full station list (needs the Natural Earth files; skipped offline)."""
import numpy as np
import pytest

from rrfs_viewer import basemap


@pytest.fixture(scope="module")
def lines():
    try:
        from rrfs_viewer import boundaries
        layers = boundaries.load(basemap.PROJ)
    except Exception as exc:                                  # no network / no cached shapefiles
        pytest.skip(f"Natural Earth data not available: {exc}")
    return basemap.overlay_lines(layers)


def test_lines_cover_all_layers_and_stay_near_the_frame(lines):
    assert lines["order"] == ["counties", "lakes", "states", "coast"] and lines["units"] == basemap.LINE_UNITS
    lo, hi = -basemap.LINE_MARGIN - 1e-3, 1 + basemap.LINE_MARGIN + 1e-3
    for name, polys in lines["lines"].items():
        assert polys, name
        for p in polys:
            assert len(p) >= 4 and len(p) % 2 == 0
            x, y = np.cumsum(p[0::2]) / lines["units"], np.cumsum(p[1::2]) / lines["units"]
            assert lo <= x.min() and x.max() <= hi and lo <= y.min() and y.max() <= hi, name


def test_line_widths_match_the_png_renderer(lines):
    for name, (lw, alpha) in basemap.LINE_STYLES.items():
        assert lines["styles"][name] == [round(lw * basemap.DPI / 72.0, 4), alpha]
    assert lines["frame"] == pytest.approx(basemap.FRAME_LW_PT * basemap.DPI / 72.0, abs=1e-3)


def test_station_list_is_complete_and_in_priority_order():
    rg_x = np.arange(basemap.EXTENT[0], basemap.EXTENT[1], 3000.0)
    rg_y = np.arange(basemap.EXTENT[2], basemap.EXTENT[3], 3000.0)
    allst = basemap.station_all(rg_x, rg_y)
    base = basemap.station_layout(rg_x, rg_y)
    assert len(allst) >= len(base) > 20
    assert [s["p"] for s in allst] == sorted(s["p"] for s in allst)
    assert {s["id"] for s in base} <= {s["id"] for s in allst}
    assert all(0 <= s["fx"] <= 1 and 0 <= s["fy"] <= 1 for s in allst)


def test_counties_are_light_gray_and_their_own_overlay(lines, tmp_path):
    assert lines["counties"] == ["counties"] and lines["colors"]["counties"] == "#b4b4b4"
    assert all(lines["colors"][k] == "#000000" for k in ("lakes", "states", "coast"))               # every other boundary stays black ...
    assert [lines["styles"][k] for k in ("lakes", "states", "coast")] == [[round(w * basemap.DPI / 72.0, 4), a] for w, a in (basemap.LINE_STYLES[k] for k in ("lakes", "states", "coast"))]
    assert basemap.LINE_STYLES["lakes"] == (1.0, 1.0) and basemap.LINE_STYLES["states"] == (1.8, 1.0) and basemap.LINE_STYLES["coast"] == (1.2, 1.0)      # ... at the same widths


def test_overlay_png_splits_counties_from_the_other_boundaries(tmp_path):
    try:
        from rrfs_viewer import boundaries
        layers = boundaries.load(basemap.PROJ)
    except Exception as exc:
        pytest.skip(f"Natural Earth data not available: {exc}")
    from PIL import Image
    basemap.render_overlay(layers, tmp_path / "main.png")
    basemap.render_overlay(layers, tmp_path / "counties.png", names=basemap.COUNTY_LAYERS, frame=False)
    main, cty = (np.asarray(Image.open(tmp_path / f).convert("RGBA")) for f in ("main.png", "counties.png"))
    assert main.shape == cty.shape == (742, 1150, 4)
    ink = main[main[..., 3] > 200][:, :3]
    assert ink.max() < 40                                              # main overlay: black lines and frame only, no gray county lines
    gray = cty[cty[..., 3] > 0]
    assert len(gray) > 5000 and abs(int(np.median(gray[:, 0])) - 0xb4) < 6 and np.ptp(gray[:, :3].astype(int), axis=1).max() < 12      # counties: gray only
    assert (main[0, :, 3] > 200).mean() > 0.9 and (cty[0, :, 3] > 100).mean() < 0.2                  # the black frame is in the main overlay only (county lines just touch the edge)
