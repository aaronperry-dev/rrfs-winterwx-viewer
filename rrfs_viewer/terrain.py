"""Terrain and water basemap: hypsometric-tinted, hill-shaded land and depth-shaded oceans and lakes.

Two static images are rendered once (and cached), in the same Lambert frame as the boundary overlay, so the page can lie them under
and over every product:

``basemap``  opaque RGB.  Land = elevation tint x multi-directional hillshade.  Oceans and lakes = a depth ramp.  It is drawn first, so it is
             what you see wherever a product has no colour (below its first level, or no data).
``relief``   RGBA, drawn *onto the painted field* (canvas ``source-atop``) so the terrain and the water stay visible through products that
             cover the whole map (temperature, wind ...): black / white alpha for the hillshade on land, a faint blue alpha on water.  The
             page scales it with a three-step control, so the legend colours of the data hardly move.

The colours are deliberately pale and low in chroma: the data ramps (greens, blues, yellows, reds, purples) must stay the loudest thing on the map.

Elevation and bathymetry come from the public AWS Terrain Tiles (Mapzen "terrarium" encoding: SRTM, USGS NED / 3DEP, GMTED, ETOPO1, GEBCO ...),
fetched once, cached on disk, and sampled onto the map frame.  Land / ocean / lake outlines are Natural Earth 1:10m, the same data as the
coast and lake lines of the overlay, so fills and lines agree.  If the tiles cannot be fetched the basemap is still produced, flat (no hillshade,
no depth), instead of failing the build.

    python -m rrfs_viewer.terrain --out rrfs_viewer/data/terrain      # pre-render the assets (optional; the build does it by itself)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.colors import to_hex
from PIL import Image
from scipy import ndimage

from . import basemap as bm
from .colors import Ramp

log = logging.getLogger(__name__)

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ATTRIBUTION = "Terrain: Mapzen / AWS Terrain Tiles (USGS, NOAA, GEBCO, SRTM)"
COMMITTED_DIR = Path(__file__).resolve().parent / "data" / "terrain"      # optional pre-rendered assets, used when their key matches

# ---------------------------------------------------------------------------------------------------------------- parameters
DEM_ZOOM = 9                  # web-mercator zoom of the elevation tiles: ~220 m per pixel at 44 N (~400 tiles, ~40 MB, downloaded once)
BASE_SCALE = 3                # basemap pixels per overlay pixel: 3450 x 2226.  The page allows 12x zoom, so the basemap is soft past ~3x
RELIEF_SCALE = 1.5           # the through-data relief is smooth and faint, so it gets less resolution (1725 x 1113)
RELIEF_BLUR_PX = 0.8         # and is smoothed a little: fine hillshade texture is what makes it big to download
SMOOTH_PX = 0.8              # Gaussian smoothing (render pixels) of the elevation before slopes are taken: removes DEM terracing
VERT_EXAG = 3.0              # vertical exaggeration for the hillshade; New England relief is low, a true-scale shade would be almost flat
SUN_ALT = 40.0               # degrees above the horizon
SUN_AZ = ((225, 0.10), (270, 0.35), (315, 0.40), (360, 0.15))     # multi-directional hillshade: (compass azimuth, weight); mostly from the NW
WEBP_BASE_Q, WEBP_RELIEF_Q = 86, 60

# Hillshade -> basemap lightness.  `shade` is 1 on flat ground, < 1 in shadow, > 1 on lit slopes.
BASE_DARK, BASE_LIGHT, BASE_MIN, BASE_MAX = 1.00, 0.45, 0.62, 1.08
# Hillshade -> relief alpha at "strong" (the page scales it by 0.35 / 0.65 / 1): black where shade < 1, white where shade > 1.
RELIEF_DARK, RELIEF_LIGHT, RELIEF_DARK_MAX, RELIEF_LIGHT_MAX = 1.50, 0.85, 0.60, 0.40
# Water seen through the data: colour and alpha at "strong"; alpha grows with depth (sqrt scale to the reference depth).
OCEAN_TINT, OCEAN_ALPHA, OCEAN_ALPHA_DEEP, OCEAN_REF_M = (58, 104, 158), 0.10, 0.32, 2500.0
LAKE_TINT, LAKE_ALPHA, LAKE_ALPHA_DEEP, LAKE_REF_M = (44, 128, 158), 0.18, 0.14, 150.0

# Colour ramps (metres -> colour; OKLab interpolation).  Pale and low in chroma on purpose.
LAND_RAMP = Ramp([(0, "#e9eee3"), (60, "#e4e9dc"), (200, "#dbdfd0"), (400, "#d0d2c2"), (700, "#c3c1b1"),
                  (1000, "#b6b2a2"), (1400, "#a9a395"), (1950, "#9d9689")])
LAND_TOP_M = 1950.0
OCEAN_RAMP = Ramp([(0, "#dde7ee"), (20, "#d5e0ea"), (100, "#c8d7e4"), (300, "#b9cbdc"), (800, "#a9bcd0"), (2000, "#98acc4"), (4000, "#899eb7")])
OCEAN_TOP_M = 4000.0
LAKE_RAMP = Ramp([(0, "#d6e9f0"), (15, "#cde4ed"), (60, "#bcd8e6"), (150, "#acccdd"), (300, "#9cc0d3")])
LAKE_TOP_M = 300.0
LUT_N = 2048                  # colour look-up size for the ramps


def _lut(ramp: Ramp, top: float, sqrt_scale: bool = False) -> np.ndarray:
    """(LUT_N, 3) float32 colours of a ramp over 0..top metres; water ramps are indexed by sqrt(depth) so shallow depths get the resolution."""
    v = np.linspace(0, 1, LUT_N)
    return ramp((v ** 2 if sqrt_scale else v) * top).astype(np.float32)


def _lookup(lut: np.ndarray, v: np.ndarray, top: float, sqrt_scale: bool = False) -> np.ndarray:
    x = np.clip(v / top, 0, 1)
    if sqrt_scale:
        x = np.sqrt(x)
    return lut[(x * (LUT_N - 1) + 0.5).astype(np.int32)]


# ---------------------------------------------------------------------------------------------------------------- tiles
def tile_coords(lon, lat, z: int):
    """Fractional web-mercator tile coordinates (x east, y south) of lon / lat (degrees) at zoom z."""
    n = 2.0 ** z
    x = (np.asarray(lon, float) + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(np.radians(np.asarray(lat, float)))) / np.pi) / 2.0 * n
    return x, y


def decode_terrarium(png: bytes) -> np.ndarray:
    """Terrarium PNG -> elevation in metres, float32 (256, 256): R * 256 + G + B / 256 - 32768."""
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), np.float32)
    return a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0


def encode_terrarium(elev: np.ndarray) -> bytes:
    v = np.clip(np.asarray(elev, np.float64) + 32768.0, 0, 65535.99)
    r, g = np.floor(v / 256.0), np.floor(v) % 256
    b = np.floor((v - np.floor(v)) * 256.0)
    buf = io.BytesIO()
    Image.fromarray(np.stack([r, g, b], -1).astype(np.uint8), "RGB").save(buf, format="PNG")
    return buf.getvalue()


def http_tile(z: int, x: int, y: int, tries: int = 4) -> bytes:
    req = urllib.request.Request(TILE_URL.format(z=z, x=x, y=y), headers={"User-Agent": "rrfs-winterwx-viewer terrain build"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as exc:
            if k == tries - 1:
                raise RuntimeError(f"terrain tile {z}/{x}/{y} could not be fetched: {exc}") from exc
            time.sleep(1.5 * (k + 1))
    raise AssertionError("unreachable")


@dataclass
class Dem:
    """Elevation mosaic in web-mercator pixels: ``z`` is (rows, cols) metres; (x0, y0) the tile index of its top-left tile; zoom the tile zoom."""
    z: np.ndarray
    x0: int
    y0: int
    zoom: int

    def sample(self, lon, lat) -> np.ndarray:
        """Bilinear elevation at lon / lat (arrays of one shape), float32."""
        tx, ty = tile_coords(lon, lat, self.zoom)
        fx, fy = tx * 256.0 - 0.5 - self.x0 * 256, ty * 256.0 - 0.5 - self.y0 * 256     # pixel centres are at integer coordinates
        out = ndimage.map_coordinates(self.z, [fy.ravel(), fx.ravel()], order=1, mode="nearest", prefilter=False)
        return out.reshape(np.shape(lon)).astype(np.float32)


def load_dem(lon: np.ndarray, lat: np.ndarray, zoom: int, fetch, cache: Path | None, workers: int = 16) -> Dem:
    """Fetch (and cache on disk) the elevation tiles that cover the lon / lat samples, and stitch them into a Dem."""
    tx, ty = tile_coords(lon[::7, ::7], lat[::7, ::7], zoom)
    n = 2 ** zoom
    pad = 0.04                                           # a few tile-pixels beyond every sample (the sampling grid is thinned, and the interpolation reaches a pixel further)
    codes = np.unique(np.concatenate([np.clip(np.floor(tx + dx), 0, n - 1).astype(np.int64).ravel() * n + np.clip(np.floor(ty + dy), 0, n - 1).astype(np.int64).ravel()
                                      for dx in (-pad, pad) for dy in (-pad, pad)]))
    want = [(int(c // n), int(c % n)) for c in codes]    # only the tiles the frame touches (it is a tilted rectangle in web-mercator), not their bounding box
    x0, x1 = min(x for x, _ in want), max(x for x, _ in want)
    y0, y1 = min(y for _, y in want), max(y for _, y in want)
    log.info("terrain: %d elevation tiles at zoom %d", len(want), zoom)

    def one(xy):
        x, y = xy
        path = cache / str(zoom) / str(x) / f"{y}.png" if cache else None
        if path is not None and path.exists():
            try:
                return xy, decode_terrarium(path.read_bytes())
            except Exception:
                path.unlink(missing_ok=True)
        data = fetch(zoom, x, y)
        a = decode_terrarium(data)                       # decode before caching: a bad download must not be cached
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return xy, a

    mosaic = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), np.float32)
    with ThreadPoolExecutor(workers) as ex:
        for (x, y), a in ex.map(one, want):
            mosaic[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = a
    return Dem(mosaic, x0, y0, zoom)


# ---------------------------------------------------------------------------------------------------------------- geometry
def load_water_geometry():
    """Projected Natural Earth land and lake polygons around the map frame: (land, lakes)."""
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import box

    from . import boundaries as bnd
    from .config import DATA_PAD_DEG, DISPLAY_BBOX

    lon0, lon1, lat0, lat1 = DISPLAY_BBOX
    p = DATA_PAD_DEG + 3.0
    clip_box = box(lon0 - p, lat0 - p, lon1 + p, lat1 + p)

    def recs(name):
        return list(shpreader.Reader(bnd.natural_earth_path("physical", name)).records())

    land = bnd._project(bnd._clip(recs("land"), clip_box), bm.PROJ)
    lakes = bnd._project(bnd._clip(recs("lakes"), clip_box), bm.PROJ)
    try:                                                 # more (smaller) lakes; optional
        lakes += bnd._project(bnd._clip(recs("lakes_north_america"), clip_box), bm.PROJ)
    except Exception as exc:
        log.warning("lakes_north_america unavailable (%s); using the main lake layer only", exc)
    return land, lakes


def polygon_mask(geoms, size: tuple[int, int]) -> np.ndarray:
    """Anti-aliased coverage (0..1, float32) of projected polygons over the map frame, ``size`` = (width, height) px, row 0 = north."""
    from cartopy.feature import ShapelyFeature
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    w, h = size
    fig = Figure(figsize=(w / bm.DPI, h / bm.DPI), dpi=bm.DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], projection=bm.PROJ)
    ax.set_extent(list(bm.EXTENT), crs=bm.PROJ)
    ax.patch.set_facecolor("black")
    ax.spines["geo"].set_visible(False)
    if geoms:
        ax.add_feature(ShapelyFeature(geoms, bm.PROJ, facecolor="white", edgecolor="none", linewidth=0))
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    assert rgba.shape[:2] == (h, w), (rgba.shape, size)
    return rgba[..., 0].astype(np.float32) / 255.0


def frame_lonlat(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """lon / lat (float64, shape (h, w)) of the pixel centres of the map frame."""
    x0, x1, y0, y1 = bm.EXTENT
    x = x0 + (np.arange(w) + 0.5) / w * (x1 - x0)
    y = y1 - (np.arange(h) + 0.5) / h * (y1 - y0)
    ll = bm.PC.transform_points(bm.PROJ, *(a.ravel() for a in np.meshgrid(x, y)), np.zeros(w * h))
    return ll[:, 0].reshape(h, w), ll[:, 1].reshape(h, w)


# ---------------------------------------------------------------------------------------------------------------- shading
def hillshade(z: np.ndarray, cell_m: float, exag: float = VERT_EXAG, alt: float = SUN_ALT, suns=SUN_AZ) -> np.ndarray:
    """Multi-directional hillshade of z (rows run north -> south): 1 on flat ground, < 1 in shadow, > 1 on slopes facing the sun."""
    gy, gx = np.gradient(z.astype(np.float32) * np.float32(exag), np.float32(cell_m))
    p, q = gx, -gy                                           # dz/dx (east) and dz/dy (north)
    norm = 1.0 / np.sqrt(1.0 + p * p + q * q)
    ca, sa = np.cos(np.radians(alt)), np.sin(np.radians(alt))
    out = np.zeros_like(z, np.float32)
    for az, wgt in suns:
        s, c = np.sin(np.radians(az)), np.cos(np.radians(az))
        out += np.float32(wgt) * np.clip((sa - p * (s * ca) - q * (c * ca)) * norm, 0, None)
    return out / np.float32(sa * sum(w for _, w in suns))


@dataclass
class Scene:
    """Everything the two images are made of, on a w x h grid of the map frame."""
    w: int
    h: int
    land: np.ndarray            # polygon coverage 0..1
    lake: np.ndarray            # polygon coverage 0..1 (lakes win over land and ocean)
    elev: np.ndarray            # metres above sea level on land (>= 0); lake surface inside lakes
    ocean_depth: np.ndarray     # metres, >= 0
    lake_depth: np.ndarray      # metres, >= 0
    shade: np.ndarray           # hillshade, 1 = flat


def build_scene(w: int, h: int, land_geoms, lake_geoms, dem: Dem | None) -> Scene:
    land, lake = polygon_mask(land_geoms, (w, h)), polygon_mask(lake_geoms, (w, h))
    if dem is None:                                          # flat fallback: no relief, no depth
        flat = np.full((h, w), 250.0, np.float32)
        return Scene(w, h, land, lake, flat, np.full((h, w), 60.0, np.float32), np.zeros((h, w), np.float32), np.ones((h, w), np.float32))
    lon, lat = frame_lonlat(w, h)
    raw = dem.sample(lon, lat)
    del lon, lat
    lake_core = lake > 0.5
    lab, n = ndimage.label(lake_core)
    lake_depth = np.zeros((h, w), np.float32)
    surface = raw.copy()
    if n:                                                    # a lake's surface = the 90th percentile of the elevation inside it: its shore level
        surf = ndimage.labeled_comprehension(raw, lab, np.arange(1, n + 1), lambda a: np.percentile(a, 90), float, np.nan)
        s = np.concatenate([[np.nan], surf]).astype(np.float32)[lab]
        inside = lab > 0
        lake_depth[inside] = np.clip(s[inside] - raw[inside], 0, LAKE_TOP_M)
        surface[inside] = s[inside]                          # flat inside lakes, so the shore does not cast a shadow into the water
    ocean_core = (land < 0.5) & ~lake_core
    ocean_depth = np.where(ocean_core, np.clip(-raw, 0, None), 0).astype(np.float32)
    elev = np.clip(np.where(ocean_core, 0, surface), 0, None).astype(np.float32)
    cell = (bm.EXTENT[1] - bm.EXTENT[0]) / w
    smooth = ndimage.gaussian_filter(elev, SMOOTH_PX) if SMOOTH_PX else elev
    return Scene(w, h, land, lake, elev, ocean_depth, lake_depth, hillshade(smooth, cell))


def _mix(under: np.ndarray, over: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return under + (over - under) * alpha[..., None]


def render_basemap(sc: Scene) -> np.ndarray:
    """Opaque RGB uint8 (h, w, 3): land tint x hillshade, depth-shaded ocean, depth-shaded lakes on top."""
    land_lut, ocean_lut, lake_lut = _lut(LAND_RAMP, LAND_TOP_M), _lut(OCEAN_RAMP, OCEAN_TOP_M, True), _lut(LAKE_RAMP, LAKE_TOP_M, True)
    sh = sc.shade - 1.0
    k = np.clip(1.0 + np.where(sh < 0, BASE_DARK, BASE_LIGHT) * sh, BASE_MIN, BASE_MAX)[..., None]
    rgb = _lookup(ocean_lut, sc.ocean_depth, OCEAN_TOP_M, True)
    rgb = _mix(rgb, np.clip(_lookup(land_lut, sc.elev, LAND_TOP_M) * k, 0, 1), sc.land)
    rgb = _mix(rgb, _lookup(lake_lut, sc.lake_depth, LAKE_TOP_M, True), sc.lake)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def render_relief(sc: Scene) -> np.ndarray:
    """RGBA uint8 (h, w, 4) for drawing onto the painted field: hillshade as black / white alpha on land, a faint depth-scaled blue on water."""
    sh = sc.shade - 1.0
    dark = np.clip(-sh * RELIEF_DARK, 0, RELIEF_DARK_MAX)
    light = np.clip(sh * RELIEF_LIGHT, 0, RELIEF_LIGHT_MAX)
    land_a = np.maximum(dark, light)
    land_c = np.where((sh >= 0)[..., None], 1.0, 0.0).astype(np.float32) * np.ones(3, np.float32)
    ocean_a = OCEAN_ALPHA + (OCEAN_ALPHA_DEEP - OCEAN_ALPHA) * np.sqrt(np.clip(sc.ocean_depth / OCEAN_REF_M, 0, 1))
    lake_a = LAKE_ALPHA + LAKE_ALPHA_DEEP * np.sqrt(np.clip(sc.lake_depth / LAKE_REF_M, 0, 1))
    ocean_c, lake_c = np.array(OCEAN_TINT, np.float32) / 255, np.array(LAKE_TINT, np.float32) / 255
    # coverage weights: lakes over land over ocean (same order as the basemap), then premultiplied mixing so anti-aliased shores blend properly
    wl = sc.lake
    wd = (1 - wl) * sc.land
    wo = (1 - wl) * (1 - sc.land)
    a = wl * lake_a + wd * land_a + wo * ocean_a
    pm = wl[..., None] * lake_a[..., None] * lake_c + wd[..., None] * land_a[..., None] * land_c + wo[..., None] * ocean_a[..., None] * ocean_c
    rgb = np.where(a[..., None] > 1e-4, pm / np.maximum(a[..., None], 1e-4), 0)
    return (np.clip(np.concatenate([rgb, a[..., None]], -1), 0, 1) * 255 + 0.5).astype(np.uint8)


def legend() -> dict:
    """The ramps for the sidebar key: sampled colours from shallow / low to deep / high."""
    def sample(ramp, top, sqrt_scale=False, n=14):
        v = np.linspace(0, 1, n)
        return [to_hex(c) for c in ramp((v ** 2 if sqrt_scale else v) * top)]
    return {"land": {"max_m": LAND_TOP_M, "colors": sample(LAND_RAMP, LAND_TOP_M)},
            "ocean": {"max_m": OCEAN_TOP_M, "colors": sample(OCEAN_RAMP, OCEAN_TOP_M, True)},
            "lake": {"max_m": LAKE_TOP_M, "colors": sample(LAKE_RAMP, LAKE_TOP_M, True)}}


def _webp(arr: np.ndarray, path: Path, quality: int, alpha: bool) -> int:
    img = Image.fromarray(arr, "RGBA" if alpha else "RGB")
    img.save(path, format="WEBP", quality=quality, method=6, **({"alpha_quality": 60, "exact": False} if alpha else {}))
    return path.stat().st_size


# ---------------------------------------------------------------------------------------------------------------- build
def render_assets(out: Path, dem_fetch, cache: Path | None, base_scale: int = BASE_SCALE, relief_scale: int = RELIEF_SCALE, zoom: int = DEM_ZOOM) -> dict:
    """Render ``basemap`` and ``relief`` into ``out`` and return the manifest block (file names carry a content hash, so browsers keep them between builds)."""
    t0 = time.time()
    land_geoms, lake_geoms = load_water_geometry()
    wb, hb = round(bm.OVERLAY_PX[0] * base_scale), round(bm.OVERLAY_PX[1] * base_scale)
    wr, hr = round(bm.OVERLAY_PX[0] * relief_scale), round(bm.OVERLAY_PX[1] * relief_scale)
    dem, flat = None, False
    try:
        lon, lat = frame_lonlat(wr, hr)                      # (the tiles needed are the same for both resolutions)
        dem = load_dem(lon, lat, zoom, dem_fetch, cache)
        del lon, lat
    except Exception as exc:
        log.warning("terrain: elevation tiles unavailable (%s); rendering the flat basemap", exc)
        flat = True
    log.info("terrain: data ready (%.0fs)", time.time() - t0)
    out.mkdir(parents=True, exist_ok=True)
    sc = build_scene(wb, hb, land_geoms, lake_geoms, dem)
    base = render_basemap(sc)
    del sc
    sc = build_scene(wr, hr, land_geoms, lake_geoms, dem)
    if RELIEF_BLUR_PX:
        sc.shade = ndimage.gaussian_filter(sc.shade, RELIEF_BLUR_PX)
    rel = render_relief(sc)
    del sc, dem
    tmp_b, tmp_r = out / "basemap.tmp.webp", out / "relief.tmp.webp"
    nb, nr = _webp(base, tmp_b, WEBP_BASE_Q, False), _webp(rel, tmp_r, WEBP_RELIEF_Q, True)
    tag = hashlib.sha1(tmp_b.read_bytes() + tmp_r.read_bytes()).hexdigest()[:10]
    fb, fr = f"basemap.{tag}.webp", f"relief.{tag}.webp"
    tmp_b.replace(out / fb)
    tmp_r.replace(out / fr)
    log.info("terrain: basemap %.0f kB (%dx%d), relief %.0f kB (%dx%d) in %.0fs%s", nb / 1e3, wb, hb, nr / 1e3, wr, hr, time.time() - t0, "  [flat: no elevation data]" if flat else "")
    return {"img": f"terrain/{fb}", "relief": f"terrain/{fr}", "px": [wb, hb], "relief_px": [wr, hr], "flat": flat,
            "legend": legend(), "credit": ATTRIBUTION}


def source_key(synthetic: bool = False) -> str:
    """Identifies the rendering code and parameters (this file), the frame and whether the elevation is synthetic: assets are re-rendered when it changes."""
    h = hashlib.sha1(Path(__file__).read_bytes())
    h.update(repr((bm.EXTENT, bm.OVERLAY_PX, synthetic)).encode())
    return h.hexdigest()[:12]


def ensure(out: Path, cache: Path, fetch=None, synthetic: bool = False, reuse: bool = True, **kw) -> dict | None:
    """Put the basemap assets into ``out/terrain`` and return the manifest block, or None if they cannot be made.

    Order: pre-rendered assets committed in ``rrfs_viewer/data/terrain`` (when their key matches), the render cache, then a fresh render
    (downloads the elevation tiles once).  A flat fallback render is not cached, so the next build tries the tiles again.
    """
    key, dest = source_key(synthetic), out / "terrain"
    for root in ((COMMITTED_DIR, cache / "terrain" / key) if reuse else ()):
        meta = root / "meta.json"
        if meta.exists():
            try:
                block = json.loads(meta.read_text())
                if block.get("key") == key and (root / Path(block["img"]).name).exists() and (root / Path(block["relief"]).name).exists():
                    dest.mkdir(parents=True, exist_ok=True)
                    for f in (block["img"], block["relief"]):
                        shutil.copy(root / Path(f).name, dest / Path(f).name)
                    log.info("terrain: using %s", root)
                    return {k: v for k, v in block.items() if k != "key"}
            except Exception as exc:
                log.warning("terrain: ignoring unreadable assets in %s (%s)", root, exc)
    work = cache / "terrain" / key
    shutil.rmtree(work, ignore_errors=True)
    block = render_assets(work, fetch or http_tile, cache / "terrain_tiles" / ("synthetic" if synthetic else "terrarium"), **kw)
    dest.mkdir(parents=True, exist_ok=True)
    for f in (block["img"], block["relief"]):
        shutil.copy(work / Path(f).name, dest / Path(f).name)
    if block["flat"]:
        shutil.rmtree(work, ignore_errors=True)              # (not cached: try the tiles again next time)
    else:
        (work / "meta.json").write_text(json.dumps({**block, "key": key}))
    return block


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(COMMITTED_DIR), help="directory for the pre-rendered assets (default: rrfs_viewer/data/terrain)")
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--synthetic", action="store_true", help="invented terrain instead of the AWS tiles (offline preview)")
    ap.add_argument("--scale", type=int, default=BASE_SCALE)
    ap.add_argument("--relief-scale", type=float, default=RELIEF_SCALE)
    ap.add_argument("--zoom", type=int, default=DEM_ZOOM)
    ap.add_argument("--force", action="store_true", help="render again even if matching assets exist")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    fetch = None
    if args.synthetic:
        from .testing import SyntheticTerrain
        fetch = SyntheticTerrain()
    tmp = Path(args.cache) / "terrain_cli"
    shutil.rmtree(tmp, ignore_errors=True)
    block = ensure(tmp, Path(args.cache), fetch=fetch, synthetic=args.synthetic, reuse=not args.force, base_scale=args.scale, relief_scale=args.relief_scale, zoom=args.zoom)
    if block["flat"]:
        log.warning("the elevation tiles could not be fetched: these assets are FLAT (no hillshade, no depth); run again with network access")
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for f in (block["img"], block["relief"]):
        shutil.copy(tmp / "terrain" / Path(f).name, out / Path(f).name)
    (out / "meta.json").write_text(json.dumps({**block, "key": source_key(args.synthetic)}))
    shutil.rmtree(tmp, ignore_errors=True)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
