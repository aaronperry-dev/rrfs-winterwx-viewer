"""Compact on-disk format for the browser: one gzip file per (product, source) holding every frame.

* values are quantised to uint8 / uint16 with a per-product offset and step (the catalogue says which); the top code = missing,
* 8-bit data is stored as a 2-D delta (difference along x, then along y, modulo 256): ~40-50 % smaller after gzip,
* 16-bit data is byte-shuffled (all high bytes, then all low bytes), which makes smooth fields compress far better,
* the container is plain gzip, so ``fetch(url).body.pipeThrough(new DecompressionStream("gzip"))`` decodes it natively.
"""
from __future__ import annotations

import gzip

import numpy as np

from .catalog import Product


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


def pack(q: np.ndarray) -> bytes:
    if q.dtype == np.uint16:
        raw = np.concatenate([(q >> 8).astype(np.uint8).ravel(), (q & 0xFF).astype(np.uint8).ravel()]).tobytes()
    else:
        dx = np.diff(q.astype(np.int16), axis=-1, prepend=0)
        dxy = np.diff(dx, axis=-2, prepend=0)
        raw = dxy.astype(np.uint8).tobytes()
    return gzip.compress(raw, compresslevel=9, mtime=0)


def unpack(blob: bytes, p: Product, shape: tuple[int, int, int]) -> np.ndarray:
    raw = np.frombuffer(gzip.decompress(blob), np.uint8)
    n = int(np.prod(shape))
    if p.dtype == "u16":
        return ((raw[:n].astype(np.uint16) << 8) | raw[n:2 * n].astype(np.uint16)).reshape(shape)
    dxy = raw[:n].reshape(shape)
    dx = np.cumsum(dxy, axis=-2, dtype=np.uint8)
    return np.cumsum(dx, axis=-1, dtype=np.uint8)


def pack_mask(mask: np.ndarray) -> bytes:
    return gzip.compress(np.packbits(mask.astype(np.uint8).ravel()).tobytes(), compresslevel=9, mtime=0)
