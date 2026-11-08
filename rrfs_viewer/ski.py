"""New England ski areas for the point-forecast table: nearest display cell of each base-area position (approximate, about +/- 1-2 km, not summits)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SKI_CSV = Path(__file__).resolve().parent / "data" / "ski_areas.csv"


def load(path: Path = SKI_CSV) -> list[dict]:
    with open(path, newline="") as f:
        return [{"id": r["id"], "name": r["name"], "st": r["state"], "lat": float(r["lat"]), "lon": float(r["lon"])} for r in csv.DictReader(f)]


def ski_all(rg_x: np.ndarray, rg_y: np.ndarray, proj, pc, extent, dx: float) -> list[dict]:
    """Ski areas inside the display raster: ``{id, name, st, r, c}`` (row / column of the nearest display cell)."""
    areas = load()
    xy = proj.transform_points(pc, np.array([a["lon"] for a in areas]), np.array([a["lat"] for a in areas]))[:, :2]
    x0, x1, y0, y1 = extent
    out = []
    for a, (px, py) in zip(areas, xy):
        if not (x0 <= px <= x1 and y0 <= py <= y1):
            continue
        col, row = int(np.argmin(np.abs(rg_x - px))), int(np.argmin(np.abs(rg_y - py)))
        if abs(rg_x[col] - px) > dx or abs(rg_y[row] - py) > dx:
            continue
        out.append({"id": a["id"], "name": a["name"], "st": a["st"], "r": row, "c": col})
    return out
