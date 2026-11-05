"""GRIB2 decoding (eccodes, straight from memory) and the one-off mapping of the 3 km native grid to the display raster.

The display raster is a regular grid in the *map projection* (a Lambert conformal frame centred on New England,
see ``basemap.py``) at the native 3 km spacing.  Each display cell stores the flat index of its nearest native cell,
so regridding a decoded message is a single fancy-index (``vals.ravel()[idx]``) - no interpolation library at run time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def decode(msg: bytes, want_geometry: bool = False):
    """Decode one GRIB2 message -> ``(values[ny, nx] float32, NaN where missing, (lat, lon) | None)``."""
    import eccodes as ec

    gid = ec.codes_new_from_message(msg)
    try:
        try:
            nx, ny = ec.codes_get(gid, "Nx"), ec.codes_get(gid, "Ny")
        except Exception:
            nx, ny = ec.codes_get(gid, "Ni"), ec.codes_get(gid, "Nj")
        vals = ec.codes_get_array(gid, "values")
        missing = ec.codes_get(gid, "missingValue")
        vals = np.where(vals == missing, np.nan, vals).astype(np.float32)
        geo = None
        if want_geometry:
            lat = ec.codes_get_array(gid, "latitudes").reshape(ny, nx)
            lon = ec.codes_get_array(gid, "longitudes").reshape(ny, nx)
            geo = (lat, np.where(lon > 180, lon - 360, lon))
        return vals.reshape(ny, nx), geo
    finally:
        ec.codes_release(gid)


@dataclass
class Regrid:
    """Native grid -> display raster."""
    idx: np.ndarray          # (ny, nx) int64 flat index into the native grid, -1 outside the model domain
    lon: np.ndarray          # (ny, nx) float32 display-cell longitude
    lat: np.ndarray
    x: np.ndarray            # (nx,) projected coordinates of the display cell centres (map projection, m)
    y: np.ndarray            # (ny,)
    native_shape: tuple[int, int]

    @property
    def shape(self):
        return self.idx.shape

    def __call__(self, native: np.ndarray) -> np.ndarray:
        if native.shape != self.native_shape:
            raise ValueError(f"unexpected grid shape {native.shape}, expected {self.native_shape}")
        out = native.ravel()[np.where(self.idx < 0, 0, self.idx)]
        out[self.idx < 0] = np.nan
        return out

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, idx=self.idx, lon=self.lon, lat=self.lat, x=self.x, y=self.y,
                            native_shape=np.array(self.native_shape))

    @classmethod
    def load(cls, path: Path) -> "Regrid | None":
        if not path.exists():
            return None
        try:
            z = np.load(path)
            return cls(z["idx"], z["lon"], z["lat"], z["x"], z["y"], tuple(int(v) for v in z["native_shape"]))
        except Exception:
            return None


def build_regrid(lat: np.ndarray, lon: np.ndarray, proj, extent, dx: float) -> Regrid:
    """Nearest-native-cell lookup for a regular display raster covering ``extent`` (x0, x1, y0, y1 in ``proj`` metres)."""
    import cartopy.crs as ccrs
    from scipy.spatial import cKDTree

    pc = ccrs.PlateCarree()
    x0, x1, y0, y1 = extent
    nx, ny = int(round((x1 - x0) / dx)), int(round((y1 - y0) / dx))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    X, Y = np.meshgrid(xs, ys)
    dll = pc.transform_points(proj, X, Y)
    dlon, dlat = dll[..., 0], dll[..., 1]

    # only native cells near the display raster go into the tree
    m = ((lat >= dlat.min() - 1.0) & (lat <= dlat.max() + 1.0) & (lon >= dlon.min() - 1.5) & (lon <= dlon.max() + 1.5))
    rr, cc = np.nonzero(m)
    if rr.size == 0:
        raise ValueError("model grid does not overlap the display frame")
    nxy = proj.transform_points(pc, lon[rr, cc].astype(float), lat[rr, cc].astype(float))[:, :2]
    dist, k = cKDTree(nxy).query(np.column_stack([X.ravel(), Y.ravel()]))
    flat = rr[k] * lat.shape[1] + cc[k]
    flat = np.where(dist <= 2.0 * dx, flat, -1).reshape(ny, nx)
    return Regrid(flat.astype(np.int64), dlon.astype(np.float32), dlat.astype(np.float32), xs, ys, tuple(lat.shape))
