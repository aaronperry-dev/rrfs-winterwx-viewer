"""Compact on-disk format for the browser: one gzip file per (product, source) holding every frame.

* values are quantised to uint8 / uint16 with a per-product offset and step (the catalogue says which); the top code = missing,
* the gzip payload is one mode byte followed by the data.  Mode bit 0 = 2-D delta (along x, then along y), bit 1 = delta along the
  frame axis; all arithmetic is modulo 2**8 / 2**16.  The writer stores "spatial" or "spatial + temporal", whichever a cheap bit-cost
  estimate says is smaller: for running totals (QPF, snowfall) the increment between frames is far smaller and sparser than the total,
  for instantaneous fields it is usually not,
* 16-bit data is then byte-shuffled (all high bytes, then all low bytes), which makes smooth fields compress far better,
* the container is plain gzip, so ``fetch(url).body.pipeThrough(new DecompressionStream("gzip"))`` decodes it natively.
"""
from __future__ import annotations

import gzip

import numpy as np

from .catalog import Product

SPATIAL, TEMPORAL = 1, 2


def top_code(p: Product) -> int:
    return 255 if p.dtype == "u8" else 65535


def quantize(a: np.ndarray, p: Product) -> np.ndarray:
    top = top_code(p)
    q = np.rint((a - p.lo) / p.step)
    q = np.clip(q, 0, top - 1)
    q = np.where(np.isfinite(a), q, top)
    return q.astype(np.uint8 if p.dtype == "u8" else np.uint16)


def dequantize(q: np.ndarray, p: Product) -> np.ndarray:
    out = p.lo + p.step * q.astype(np.float32)
    out[q == top_code(p)] = np.nan
    return out


def _spatial(q: np.ndarray) -> np.ndarray:
    """Difference along x, then along y (unsigned wrap-around is intended)."""
    a = q.copy()
    a[..., :, 1:] -= q[..., :, :-1]
    b = a.copy()
    b[..., 1:, :] -= a[..., :-1, :]
    return b


def _temporal(a: np.ndarray) -> np.ndarray:
    b = a.copy()
    b[1:] -= a[:-1]
    return b


def _bits(a: np.ndarray) -> float:
    """Rough coded size: sum of log2(1 + |delta|) with the deltas read as signed numbers (zeros are free)."""
    s = a.view(np.int16 if a.dtype == np.uint16 else np.int8)
    return float(np.log2(1.0 + np.abs(s.astype(np.int32))).sum())


def pack(q: np.ndarray) -> bytes:
    s = _spatial(q)
    st = _temporal(s)
    mode, a = (SPATIAL, s) if _bits(s) <= _bits(st) else (SPATIAL | TEMPORAL, st)
    if a.dtype == np.uint16:
        body = np.concatenate([(a >> 8).astype(np.uint8).ravel(), (a & 0xFF).astype(np.uint8).ravel()]).tobytes()
    else:
        body = a.tobytes()
    return gzip.compress(bytes([mode]) + body, compresslevel=9, mtime=0)


def unpack(blob: bytes, p: Product, shape: tuple[int, int, int]) -> np.ndarray:
    raw = np.frombuffer(gzip.decompress(blob), np.uint8)
    mode, n = int(raw[0]), int(np.prod(shape))
    if p.dtype == "u16":
        a = ((raw[1:1 + n].astype(np.uint16) << 8) | raw[1 + n:1 + 2 * n].astype(np.uint16)).reshape(shape)
    else:
        a = raw[1:1 + n].reshape(shape)
    if mode & TEMPORAL:
        a = np.cumsum(a, axis=0, dtype=a.dtype)
    if mode & SPATIAL:
        a = np.cumsum(np.cumsum(a, axis=-2, dtype=a.dtype), axis=-1, dtype=a.dtype)
    return a


def pack_mask(mask: np.ndarray) -> bytes:
    return gzip.compress(np.packbits(mask.astype(np.uint8).ravel()).tobytes(), compresslevel=9, mtime=0)
