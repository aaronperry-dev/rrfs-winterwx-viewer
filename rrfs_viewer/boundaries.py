"""High-resolution (1:10m Natural Earth) boundaries, clipped to New England and pre-projected once."""
from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import numpy as np
import shapely
from shapely.geometry import box

from .config import DATA_PAD_DEG, DISPLAY_BBOX, NEW_ENGLAND_STATES

log = logging.getLogger(__name__)

_MIRROR = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/10m_{category}/ne_10m_{name}.{ext}"


def natural_earth_path(category: str, name: str) -> str:
    """cartopy's own downloader first (naturalearth S3); fall back to the public GitHub mirror."""
    try:
        return shpreader.natural_earth(resolution="10m", category=category, name=name)
    except Exception as exc:
        log.warning("cartopy download of %s failed (%s); trying GitHub mirror", name, exc)
    folder = Path(cartopy.config["data_dir"]) / "shapefiles" / "natural_earth" / category
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("shp", "dbf", "shx"):
        dest = folder / f"ne_10m_{name}.{ext}"
        if not dest.exists():
            urllib.request.urlretrieve(_MIRROR.format(category=category, name=name, ext=ext), dest)
    return str(folder / f"ne_10m_{name}.shp")


@dataclass
class Boundaries:
    counties: list
    states: list
    coast: list
    lakes: list
    ne_land: object          # shapely geometry of the six New England states (lon/lat)


def _clip(records, clip_box, keep=lambda rec: True):
    out = []
    for rec in records:
        g = rec.geometry
        if g is None or not keep(rec) or not g.intersects(clip_box):
            continue
        g = g.intersection(clip_box)
        parts = list(g.geoms) if hasattr(g, "geoms") else [g]
        out.extend(p for p in parts if not p.is_empty and p.geom_type in
                   ("Polygon", "LineString", "MultiPolygon", "MultiLineString"))
    return out


def _project(geoms, proj):
    out = []
    for g in geoms:
        try:
            out.append(proj.project_geometry(g, ccrs.PlateCarree()))
        except Exception:
            continue
    return out


def load(proj) -> Boundaries:
    lon0, lon1, lat0, lat1 = DISPLAY_BBOX
    p = DATA_PAD_DEG + 3.0  # clip well outside the map so the clip edge is never drawn
    clip_box = box(lon0 - p, lat0 - p, lon1 + p, lat1 + p)

    def recs(category, name):
        return list(shpreader.Reader(natural_earth_path(category, name)).records())

    states_recs = recs("cultural", "admin_1_states_provinces_lakes")
    ne_polys = [r.geometry for r in states_recs
                if r.attributes.get("adm0_a3") == "USA" and r.attributes.get("postal") in NEW_ENGLAND_STATES]
    ne_land = shapely.union_all(ne_polys)

    # Counties: sub-state detail.  Optional layer - skip quietly if it cannot be fetched.
    try:
        counties = _clip(recs("cultural", "admin_2_counties"), clip_box)
    except Exception as exc:
        log.warning("county layer unavailable: %s", exc)
        counties = []

    return Boundaries(
        counties=_project(counties, proj),
        states=_project(_clip(states_recs, clip_box), proj),
        coast=_project(_clip(recs("physical", "coastline"), clip_box), proj),
        lakes=_project(_clip(recs("physical", "lakes"), clip_box), proj),
        ne_land=ne_land,
    )
