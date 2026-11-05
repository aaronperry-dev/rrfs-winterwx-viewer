"""New England METAR stations (airports + the Mount Washington summit observatory, KMWN)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STATIONS_CSV = Path(__file__).resolve().parent / "data" / "stations.csv"


@dataclass(frozen=True)
class Station:
    icao: str
    name: str
    state: str
    lat: float
    lon: float
    elev_m: float
    priority: int          # 0 = always labelled, 1 = labelled first, 2 = filled in where there is room


def load(path: Path = STATIONS_CSV) -> list[Station]:
    with open(path, newline="") as f:
        rows = [Station(r["icao"], r["name"], r["state"], float(r["lat"]), float(r["lon"]), float(r["elev_m"]),
                        int(r["priority"])) for r in csv.DictReader(f)]
    return sorted(rows, key=lambda s: (s.priority, s.icao))


def declutter(stations: list[Station], xy: np.ndarray, extent: tuple[float, float, float, float],
              min_sep: float, edge: float) -> list[int]:
    """Indices of stations to label: inside the frame (minus ``edge``), and at least ``min_sep`` apart.

    Stations are visited in priority order (the list is pre-sorted), so important sites win a crowded area.
    ``xy`` is the (n, 2) projected position of each station, in the same units as ``extent``/``min_sep``.
    """
    x0, x1, y0, y1 = extent
    kept: list[int] = []
    for i, (x, y) in enumerate(xy):
        if not (x0 + edge <= x <= x1 - edge and y0 + edge <= y <= y1 - edge):
            continue
        if stations[i].priority > 0 and any((x - xy[k, 0]) ** 2 + (y - xy[k, 1]) ** 2 < min_sep ** 2 for k in kept):
            continue
        kept.append(i)
    return kept
