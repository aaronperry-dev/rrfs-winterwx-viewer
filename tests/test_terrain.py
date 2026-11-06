"""Terrain / water basemap: tile maths, hillshade, colour ramps, the two rendered images and the cache.  No network except the Natural Earth
files that the other overlay tests need as well (those tests are skipped when they are not available)."""
import json
import math

import numpy as np
import pytest
import shapely.geometry as sg
from PIL import Image

from rrfs_viewer import basemap, terrain
from rrfs_viewer.colors import srgb_to_oklab


# ------------------------------------------------------------------------------------------------------------ tile maths
def test_terrarium_round_trip():
    z = np.array([[-4200.5, -12.25, 0.0], [3.5, 1917.0, 8848.0]], np.float32)
    back = terrain.decode_terrarium(terrain.encode_terrarium(np.pad(z, ((0, 254), (0, 253)))))[:2, :3]
    assert np.abs(back - z).max() <= 1 / 256 + 1e-6


def test_tile_coords_known_points():
    x, y = terrain.tile_coords([0.0, -180.0, 180.0], [0.0, 0.0, 0.0], 1)
    assert np.allclose(x, [1.0, 0.0, 2.0]) and np.allclose(y, [1.0, 1.0, 1.0])
    lon, lat, z = -71.3033, 44.2706, 9                                      # Mount Washington; the textbook formulas, written out independently
    phi = math.radians(lat)
    x, y = terrain.tile_coords(lon, lat, z)
    assert x == pytest.approx((lon + 180) / 360 * 2 ** z)
    assert y == pytest.approx((1 - math.log(math.tan(phi) + 1 / math.cos(phi)) / math.pi) / 2 * 2 ** z)
    assert (int(x), int(y)) == (154, 185)


def test_dem_sample_hits_pixel_centres_and_interpolates():
    zoom, rows, cols = 2, 512, 512                                         # a 2 x 2 tile mosaic whose top-left tile is (0, 0)
    r, c = np.mgrid[0:rows, 0:cols]
    dem = terrain.Dem((r * 10.0 + c).astype(np.float32), 0, 0, zoom)

    def lonlat(px, py):                                                    # of the centre of mosaic pixel (px, py)
        n = 2 ** zoom * 256
        lon = (px + 0.5) / n * 360 - 180
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (py + 0.5) / n))))
        return lon, lat
    lon, lat = lonlat(np.array([5.0, 6.0]), np.array([7.0, 7.0]))
    got = dem.sample(lon[None, :], lat[None, :])[0]
    assert np.allclose(got, [75.0, 76.0], atol=1e-3)
    lon_m, lat_m = lonlat(5.5, 7.0)                                       # half way between pixel 5 and 6 along x
    assert dem.sample(np.array([lon_m]), np.array([lat_m]))[0] == pytest.approx(75.5, abs=0.01)


# ------------------------------------------------------------------------------------------------------------ hillshade
def test_hillshade_flat_is_one_and_faces_the_sun():
    n = 60
    flat = np.full((n, n), 300.0, np.float32)
    assert np.allclose(terrain.hillshade(flat, 400.0), 1.0, atol=1e-5)
    r, c = np.mgrid[0:n, 0:n].astype(np.float32)
    rising_se = 30.0 * (r + c)                                             # higher to the south-east: the slope faces north-west, towards the sun
    rising_nw = -rising_se
    lit, dark = terrain.hillshade(rising_se, 400.0)[10:-10, 10:-10], terrain.hillshade(rising_nw, 400.0)[10:-10, 10:-10]
    assert lit.min() > 1.0 > dark.max()
    rising_n = 30.0 * (n - r)                                              # higher to the north: faces south, away from a north-western sun
    rising_s = 30.0 * r
    assert terrain.hillshade(rising_s, 400.0)[10:-10, 10:-10].mean() > terrain.hillshade(rising_n, 400.0)[10:-10, 10:-10].mean()


# ------------------------------------------------------------------------------------------------------------ colours
def test_ramps_are_pale_and_get_darker_with_elevation_and_depth():
    for ramp, top in ((terrain.LAND_RAMP, terrain.LAND_TOP_M), (terrain.OCEAN_RAMP, terrain.OCEAN_TOP_M), (terrain.LAKE_RAMP, terrain.LAKE_TOP_M)):
        rgb = ramp(np.linspace(0, top, 50))
        light = srgb_to_oklab(rgb)[:, 0]
        assert np.all(np.diff(light) <= 1e-6), "lightness must not increase with elevation / depth"
        assert light[0] > 0.9 and light[-1] > 0.65, "pale enough not to compete with the data colours"
        chroma = np.hypot(*srgb_to_oklab(rgb)[:, 1:].T)
        assert chroma.max() < 0.05, "low chroma: the data ramps must stay the loudest thing on the map"


def test_legend_lists_sampled_ramps():
    lg = terrain.legend()
    assert set(lg) == {"land", "ocean", "lake"}
    for k, v in lg.items():
        assert len(v["colors"]) >= 8 and all(c.startswith("#") and len(c) == 7 for c in v["colors"]) and v["max_m"] > 0


def _scene(h=40, w=60):
    """Left third ocean (deep in the far left), a land block with a lit and a shadowed slope, a lake."""
    land = np.zeros((h, w), np.float32)
    land[:, 20:] = 1
    lake = np.zeros((h, w), np.float32)
    lake[25:35, 40:55] = 1
    elev = np.zeros((h, w), np.float32)
    elev[:, 20:] = 120.0
    ocean_depth = np.zeros((h, w), np.float32)
    ocean_depth[:, :20] = np.linspace(2500, 10, 20)
    lake_depth = np.zeros((h, w), np.float32)
    lake_depth[25:35, 40:55] = 80.0
    shade = np.ones((h, w), np.float32)
    shade[5:10, 25:30] = 0.55                                              # shadowed slope
    shade[5:10, 32:37] = 1.25                                              # lit slope
    return terrain.Scene(w, h, land, lake, elev, ocean_depth, lake_depth, shade)


def test_basemap_colours_land_ocean_and_lakes_distinctly():
    sc = _scene()
    rgb = terrain.render_basemap(sc)
    assert rgb.shape == (40, 60, 3) and rgb.dtype == np.uint8
    ocean, land, lake = rgb[20, 3].astype(int), rgb[20, 45].astype(int), rgb[30, 47].astype(int)
    assert ocean[2] > ocean[0] + 10 and lake[2] > lake[0] + 10            # water is bluish
    assert abs(land[2] - land[0]) < 14                                     # land is nearly neutral
    assert rgb[20, 2].sum() < rgb[20, 18].sum(), "deeper water is darker"
    assert rgb[7, 27].sum() < land.sum() < rgb[7, 34].sum() + 1, "shadowed slope darker than flat ground, lit slope not darker"
    assert rgb.min() >= 130, "the whole basemap stays pale, even in shadow"


def test_relief_overlay_is_black_white_on_land_and_blue_on_water():
    sc = _scene()
    rel = terrain.render_relief(sc)
    assert rel.shape == (40, 60, 4)
    assert rel[20, 45, 3] == 0, "flat land adds nothing over the data"
    assert rel[7, 27, 3] > 20 and rel[7, 27, :3].max() < 10, "a shadow is black with alpha"
    assert rel[7, 34, 3] > 20 and rel[7, 34, :3].min() > 245, "a lit slope is white with alpha"
    assert rel[20, 3, 3] > rel[20, 17, 3] > 0, "the ocean always shows a faint tint, stronger where it is deeper"
    assert rel[20, 3, 2] > rel[20, 3, 0], "ocean tint is blue"
    assert rel[30, 47, 3] > 0 and rel[30, 47, 1] > rel[30, 47, 0], "lakes are tinted too"
    assert rel[..., 3].max() <= 0.6 * 255 + 1, "never strong enough to bury a legend colour at full strength"


# ------------------------------------------------------------------------------------------------------------ scene from polygons and a DEM
class _SteppedDem:
    """Duck-typed Dem: flat 130 m land to the east, a sea floor that falls away to the west, and an 80 m hole (50 m deep) in the lake."""

    def sample(self, lon, lat):
        z = np.where(lon > -71.4, 130.0, -300.0 + 200.0 * (lon + 71.4))
        hole = (np.abs(lon + 70.6) < 0.2) & (np.abs(lat - 44.2) < 0.15)
        return np.where(hole, 80.0, z).astype(np.float32)


def test_build_scene_classifies_water_and_measures_depth():
    x0, x1, y0, y1 = basemap.EXTENT
    xm = (x0 + x1) / 2
    cx, cy = basemap.PROJ.transform_point(-70.6, 44.2, basemap.PC)
    land = [sg.box(xm - 1e3, y0 - 1e4, x1 + 1e4, y1 + 1e4)]
    lake = [sg.box(cx - 25e3, cy - 25e3, cx + 25e3, cy + 25e3)]
    w, h = 230, 148
    sc = terrain.build_scene(w, h, land, lake, _SteppedDem())
    assert sc.land[h // 2, w - 5] == 1 and sc.land[h // 2, 5] == 0
    assert sc.ocean_depth[h // 2, 5] > 0 and sc.ocean_depth[h // 2, w - 5] == 0
    assert sc.elev[h // 2, w - 5] > 100 and sc.elev[h // 2, 5] == 0
    assert 45 < sc.lake_depth.max() <= 55, "the deepest part of the lake is 50 m below its shore level"
    assert sc.lake_depth[h // 2, w - 5] == 0, "no lake depth on land"
    assert np.allclose(sc.shade[h // 2, 5], 1.0, atol=1e-3), "the sea is flat"


# ------------------------------------------------------------------------------------------------------------ assets and cache
@pytest.fixture(scope="module")
def water():
    try:
        return terrain.load_water_geometry()
    except Exception as exc:
        pytest.skip(f"Natural Earth data not available: {exc}")


def _flat_tile(z, x, y):
    return terrain.encode_terrarium(np.full((256, 256), 150.0))


def test_polygon_mask_matches_the_frame(water):
    land, lakes = water
    m = terrain.polygon_mask(land, (230, 148))
    assert m.shape == (148, 230) and 0.05 < m.mean() < 0.95
    ll = terrain.frame_lonlat(230, 148)
    assert ll[0].shape == (148, 230) and ll[1][0, 0] > ll[1][-1, 0], "row 0 is the north edge"
    gulf = m[130, 200]                                                      # south-east corner: Atlantic
    assert gulf < 0.5
    assert m[74, 100] > 0.5                                                 # the middle of the frame is land (New England / Quebec / Maine)


def test_ensure_renders_once_then_reuses_the_cache(tmp_path, water):
    calls = []

    def fetch(z, x, y):
        calls.append((z, x, y))
        return _flat_tile(z, x, y)

    kw = dict(base_scale=1, relief_scale=1, zoom=6)
    out1 = tmp_path / "o1"
    block = terrain.ensure(out1, tmp_path / "cache", fetch=fetch, synthetic=True, **kw)
    assert block and not block["flat"] and calls
    for key in ("img", "relief", "legend", "credit", "px", "relief_px"):
        assert key in block
    base, rel = Image.open(out1 / block["img"]), Image.open(out1 / block["relief"])
    assert base.size == tuple(block["px"]) == basemap.OVERLAY_PX and base.mode == "RGB"
    assert rel.size == tuple(block["relief_px"]) and rel.mode == "RGBA"
    n = len(calls)
    out2 = tmp_path / "o2"
    block2 = terrain.ensure(out2, tmp_path / "cache", fetch=fetch, synthetic=True, **kw)
    assert block2 == block and len(calls) == n, "second build: no tiles fetched, same assets"
    assert (out2 / block["img"]).read_bytes() == (out1 / block["img"]).read_bytes()


def test_tiles_are_cached_on_disk_and_bad_downloads_are_not(tmp_path, water):
    cache = tmp_path / "tiles"
    lon, lat = terrain.frame_lonlat(40, 26)
    calls = []
    terrain.load_dem(lon, lat, 6, lambda z, x, y: (calls.append(1), _flat_tile(z, x, y))[1], cache)
    n = len(calls)
    terrain.load_dem(lon, lat, 6, lambda z, x, y: pytest.fail("cached tiles must not be fetched again"), cache)
    assert n > 0 and len(list(cache.rglob("*.png"))) == n
    bad = tmp_path / "bad"
    with pytest.raises(Exception):
        terrain.load_dem(lon, lat, 6, lambda z, x, y: b"not a png", bad)
    assert not list(bad.rglob("*.png")), "a corrupt tile must not be cached"


def test_flat_fallback_when_tiles_cannot_be_fetched(tmp_path, water):
    def boom(z, x, y):
        raise RuntimeError("offline")

    out = tmp_path / "o"
    block = terrain.ensure(out, tmp_path / "cache", fetch=boom, synthetic=True, base_scale=1, relief_scale=1, zoom=6)
    assert block["flat"] and (out / block["img"]).exists()
    assert not list((tmp_path / "cache" / "terrain").rglob("meta.json")), "a flat render must not be cached: the next build tries the tiles again"
    rgb = np.asarray(Image.open(out / block["img"]))
    assert rgb.min() >= 150 and len(np.unique(rgb.reshape(-1, 3), axis=0)) > 20     # still coloured: ocean, land and lakes differ


def test_manifest_block_is_json_serialisable(tmp_path, water):
    block = terrain.ensure(tmp_path / "o", tmp_path / "c", fetch=_flat_tile, synthetic=True, base_scale=1, relief_scale=1, zoom=6)
    json.dumps(block)
    assert block["img"].startswith("terrain/basemap.") and block["img"].endswith(".webp")
