"""Optional GIS layers for the map - highways with route labels and cities - as vectors in frame units.

Source: Natural Earth 1:10m ``roads_north_america`` and ``populated_places_simple`` (public domain), the same download path as the
boundaries.  The page draws these on request (toggles in the "Map Layers" panel), at any zoom, and in the PNG export.  The roads are
generalised (about 1:10 million), so at high zoom they follow the real road only roughly; the city list is short (28 places).

``data/gis.bin`` is gzip JSON::

    {"units": 32768,
     "roads":  {"interstate": [[x0, y0, dx1, dy1, ...], ...], "us": [...], "state": [...]},   # as the boundary lines in lines.bin
     "labels": [[text, class, fx, fy, min_zoom], ...],       # class 0 interstate, 1 US route, 2 state route; sorted by (min_zoom, longest route first)
     "cities": [[name, fx, fy, is_capital], ...]}            # capitals first, then by population

``fx``/``fy`` are fractions of the map frame (0..1, y downward).
"""
from __future__ import annotations

import logging

import numpy as np
import shapely
from shapely.geometry import box

from . import basemap
from . import boundaries as bnd
from .config import DATA_PAD_DEG, DISPLAY_BBOX

log = logging.getLogger(__name__)

CLASSES = {"Interstate": "interstate", "Federal": "us", "State": "state"}       # Natural Earth ``class`` -> layer; "Other", ferries and the rest are left out
CLASS_CODE = {"interstate": 0, "us": 1, "state": 2}
SKIP_TYPES = {("State", "Other Paved")}          # minor state roads (Natural Earth scale rank 11): hundreds of short pieces, no use at this scale
ABBR = {"Maine": "ME", "New Hampshire": "NH", "Vermont": "VT", "Massachusetts": "MA", "Connecticut": "CT", "Rhode Island": "RI",
        "New York": "NY", "New Jersey": "NJ", "Pennsylvania": "PA"}
NE_NAMES = {"Maine", "New Hampshire", "Vermont", "Massachusetts", "Connecticut", "Rhode Island"}

LABEL_SPACING_M = 70_000.0          # further labels of one route are this far apart along it
MIN_PIECE_M = 8_000.0               # shorter pieces of a route get no label
LABEL_ZOOM = {"interstate": (1.0, 1.7, 2.6, 3.6, 4.8), "us": (1.5, 2.2, 3.1, 4.1, 5.2), "state": (2.4, 3.2, 4.2, 5.2, 6.2)}   # min zoom of a route's 1st, 2nd... label
LABEL_EDGE = 0.02                   # labels keep this far (frame fraction) from the frame edge


def _project(geom):
    return shapely.transform(geom, lambda c: basemap.PROJ.transform_points(basemap.PC, c[:, 0], c[:, 1])[:, :2])


def _anchors(line, spacing: float = LABEL_SPACING_M):
    """(distance along the line, tier) of the label sites of one piece: its middle first, then outwards in both directions."""
    n, half = line.length, line.length / 2
    out, j = [(half, 0)], 1
    while j <= 8:
        off = ((j + 1) // 2) * spacing * (1 if j % 2 else -1)
        if abs(off) <= 0.4 * n:
            out.append((half + off, min((j + 1) // 2, 4)))
        j += 1
    return out


def load_roads(roads_shp: str | None = None):
    """Records (class layer, label text, projected geometry) of the highways in and around the map, from Natural Earth."""
    import cartopy.io.shapereader as shpreader
    lon0, lon1, lat0, lat1 = DISPLAY_BBOX
    pad = DATA_PAD_DEG + 1.0
    bbox = box(lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad)
    path = roads_shp or bnd.natural_earth_path("cultural", "roads_north_america")
    out = []
    for rec in shpreader.Reader(path).records():
        a, g = rec.attributes, rec.geometry
        layer = CLASSES.get(a.get("class"))
        if layer is None or g is None or a.get("country") != "United States" or (a.get("class"), a.get("type")) in SKIP_TYPES or not g.intersects(bbox):
            continue
        num = str(a.get("number") or "").strip()
        st = ABBR.get(a.get("state"), "")
        text = (f"I-{num}" if layer == "interstate" else f"US {num}" if layer == "us" else f"{st} {num}".strip()) if num else ""
        out.append((layer, text, _project(g.intersection(bbox))))
    return out


def road_layers(records) -> tuple[dict, list]:
    """Polylines per layer (frame units) and the route labels from :func:`load_roads` records."""
    x0, x1, y0, y1 = basemap.EXTENT
    w, h = x1 - x0, y1 - y0
    m = basemap.LINE_MARGIN
    clip = box(x0 - m * w, y0 - m * h, x1 + m * w, y1 + m * h)
    inner = box(x0 + LABEL_EDGE * w, y0 + LABEL_EDGE * h, x1 - LABEL_EDGE * w, y1 - LABEL_EDGE * h)
    roads: dict[str, list] = {k: [] for k in CLASS_CODE}
    routes: dict[tuple, list] = {}
    for layer, text, g in records:
        g = g.intersection(clip)
        if g.is_empty:
            continue
        for c in basemap._line_parts(g):
            if len(c) >= 2:
                p = basemap.frame_polyline(c)
                if p is not None:
                    roads[layer].append(p)
        if text:
            routes.setdefault((layer, text), []).append(g.intersection(inner))
    labels = []
    for (layer, text), parts in routes.items():
        merged = shapely.line_merge(shapely.union_all(parts))
        pieces = [q for q in (merged.geoms if hasattr(merged, "geoms") else [merged]) if q.geom_type == "LineString" and q.length >= MIN_PIECE_M]
        pieces.sort(key=lambda q: -q.length)
        for q in pieces:
            for d, tier in _anchors(q):
                pt = q.interpolate(d)
                labels.append([text, CLASS_CODE[layer], round(float((pt.x - x0) / w), 5), round(float((y1 - pt.y) / h), 5), LABEL_ZOOM[layer][tier], q.length])
    labels.sort(key=lambda r: (r[4], -r[5]))
    return roads, [r[:5] for r in labels]


def load_cities(places_shp: str | None = None) -> list:
    import cartopy.io.shapereader as shpreader
    x0, x1, y0, y1 = basemap.EXTENT
    path = places_shp or bnd.natural_earth_path("cultural", "populated_places_simple")
    rows = []
    for rec in shpreader.Reader(path).records():
        a = rec.attributes
        if a.get("adm0_a3") != "USA" or a.get("adm1name") not in NE_NAMES:
            continue
        px, py = basemap.PROJ.transform_point(rec.geometry.x, rec.geometry.y, basemap.PC)
        if not (x0 <= px <= x1 and y0 <= py <= y1):
            continue
        cap = 1 if a.get("featurecla") == "Admin-1 capital" else 0
        rows.append((-cap, -int(a.get("pop_max") or 0), [a["name"], round(float((px - x0) / (x1 - x0)), 5), round(float((y1 - py) / (y1 - y0)), 5), cap]))
    rows.sort(key=lambda r: r[:2])
    return [r[2] for r in rows]


def build(roads_shp: str | None = None, places_shp: str | None = None) -> dict:
    roads, labels = road_layers(load_roads(roads_shp))
    return {"units": basemap.LINE_UNITS, "roads": roads, "labels": labels, "cities": load_cities(places_shp)}
