"""Map frame, boundary overlay PNG, station label positions and the land mask.

The fields themselves are drawn in the browser (canvas); cartopy/matplotlib draw everything that is *geographic and static*:
the thick state lines over thin county lines (Natural Earth 1:10m), the coast, lakes and the black map frame, rendered once as
a transparent palette PNG that lies on top of the field.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from cartopy.feature import ShapelyFeature
from PIL import Image

from . import boundaries as bnd
from . import stations as stn
from .config import (DISPLAY_BBOX, EDGE_M, HALF_H_M, MAP_CENTER, MAP_H_IN, MAP_W_IN, MIN_SEP_M, OUT_DX_M)

log = logging.getLogger(__name__)

PC = ccrs.PlateCarree()
PROJ = ccrs.LambertConformal(central_longitude=-70.6, central_latitude=44.2, standard_parallels=(42.0, 46.0))
DPI = 100
OVERLAY_PX = (round(MAP_W_IN * DPI), round(MAP_H_IN * DPI))       # (1150, 742)


def _extent():
    cx, cy = PROJ.transform_point(*MAP_CENTER, PC)
    hw = HALF_H_M * MAP_W_IN / MAP_H_IN
    return cx - hw, cx + hw, cy - HALF_H_M, cy + HALF_H_M


EXTENT = _extent()


def check_frame_inside_bbox():
    x0, x1, y0, y1 = EXTENT
    t = np.linspace(0, 1, 100)
    xs = np.concatenate([x0 + (x1 - x0) * t, np.full(100, x1), x1 - (x1 - x0) * t, np.full(100, x0)])
    ys = np.concatenate([np.full(100, y0), y0 + (y1 - y0) * t, np.full(100, y1), y1 - (y1 - y0) * t])
    ll = PC.transform_points(PROJ, xs, ys)
    lon0, lon1, lat0, lat1 = DISPLAY_BBOX
    if ll[:, 0].min() < lon0 or ll[:, 0].max() > lon1 or ll[:, 1].min() < lat0 or ll[:, 1].max() > lat1:
        raise ValueError("map frame spills outside DISPLAY_BBOX")


check_frame_inside_bbox()


def render_overlay(layers: bnd.Boundaries, out: Path) -> int:
    """Transparent boundary overlay exactly the size of the map frame (1150 x 742 px)."""
    fig = plt.figure(figsize=(MAP_W_IN, MAP_H_IN), dpi=DPI)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1], projection=PROJ)
    ax.patch.set_alpha(0)
    ax.set_extent(list(EXTENT), crs=PROJ)
    ax.spines["geo"].set_linewidth(1.6)
    ax.spines["geo"].set_edgecolor("black")

    def add(geoms, **kw):
        if geoms:
            ax.add_feature(ShapelyFeature(geoms, PROJ, facecolor="none", **kw))

    add(layers.counties, edgecolor="black", linewidth=0.45, alpha=0.85, zorder=2)
    add(layers.lakes, edgecolor="black", linewidth=1.0, zorder=3)
    add(layers.states, edgecolor="black", linewidth=1.8, zorder=4)
    add(layers.coast, edgecolor="black", linewidth=1.2, zorder=4)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    assert rgba.shape[1::-1] == OVERLAY_PX, rgba.shape
    img = Image.fromarray(rgba, "RGBA").quantize(colors=48, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG", optimize=True)
    try:
        import oxipng
        oxipng.optimize(str(out), level=4)
    except Exception:
        pass
    return out.stat().st_size


def land_mask(layers: bnd.Boundaries, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Display cells inside the six New England states."""
    return shapely.contains_xy(layers.ne_land, lon, lat)


def station_layout(rg_x: np.ndarray, rg_y: np.ndarray) -> list[dict]:
    """METAR stations to label: fractional position in the frame (0..1, origin top-left) + nearest display cell."""
    sts = stn.load()
    sxy = PROJ.transform_points(PC, np.array([s.lon for s in sts]), np.array([s.lat for s in sts]))[:, :2]
    x0, x1, y0, y1 = EXTENT
    out = []
    for i in stn.declutter(sts, sxy, EXTENT, MIN_SEP_M, EDGE_M):
        px, py = sxy[i]
        col = int(np.argmin(np.abs(rg_x - px)))
        row = int(np.argmin(np.abs(rg_y - py)))
        if abs(rg_x[col] - px) > OUT_DX_M or abs(rg_y[row] - py) > OUT_DX_M:
            continue
        out.append({"id": sts[i].icao, "fx": round((px - x0) / (x1 - x0), 5), "fy": round((y1 - py) / (y1 - y0), 5),
                    "r": row, "c": col})
    return out
