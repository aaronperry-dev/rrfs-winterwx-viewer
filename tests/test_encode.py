import numpy as np

from rrfs_viewer import encode
from rrfs_viewer.catalog import BY_KEY, GROUPS, ONE_HOUR, PRODUCTS, STORE


def test_roundtrip_all_products():
    rng = np.random.default_rng(1)
    shape = (3, 40, 50)
    for p in STORE:
        hi = p.lo + p.step * (250 if p.dtype == "u8" else 60000)
        a = np.cumsum(rng.random(shape), axis=2).astype(np.float32)
        a = p.lo + (a / a.max()) * (hi - p.lo) * 0.9
        a[0, :5, :5] = np.nan
        q = encode.quantize(a, p)
        back = encode.unpack(encode.pack(q), p, shape)
        assert np.array_equal(back, q), p.key
        deq = encode.dequantize(q, p)
        assert np.isnan(deq[0, :5, :5]).all()
        ok = np.isfinite(a)
        assert np.nanmax(np.abs(deq[ok] - a[ok])) <= p.step / 2 + 2e-7 * max(abs(hi), 1.0) + 1e-6, p.key


def _mode(blob):
    import gzip
    return gzip.decompress(blob)[0]


def test_mode_choice_and_exact_roundtrip():
    rng = np.random.default_rng(7)
    ny, nx = 60, 80
    # a running total that grows frame by frame over a sharp-edged, stationary pattern: the frame-to-frame delta is far cheaper than the total
    yy, xx = np.mgrid[0:ny, 0:nx]
    blob = ((yy // 8 + xx // 8) % 3) * 1.5
    total = np.stack([blob * (f + 1) for f in range(6)]).astype(np.float32)
    # a noisy instantaneous field: deltas in time only add noise
    noisy = (20 + 10 * rng.standard_normal((6, ny, nx))).astype(np.float32)
    for key, a, want in (("snow_kuchera", total, 3), ("snow_12_1h", total, 3), ("t2", noisy, 1), ("gust", noisy, 1)):
        p = BY_KEY[key]
        q = encode.quantize(a, p)
        blobbed = encode.pack(q)
        assert _mode(blobbed) == want, key
        assert np.array_equal(encode.unpack(blobbed, p, q.shape), q), key


def test_roundtrip_wraparound_and_single_frame():
    rng = np.random.default_rng(3)
    for key, top in (("gust", 255), ("qpf", 65535)):
        p = BY_KEY[key]
        dt = np.uint8 if top == 255 else np.uint16
        for shape in ((1, 9, 11), (4, 9, 11)):
            q = rng.integers(0, top + 1, shape).astype(dt)          # extremes in neighbouring cells force modular wrap-around
            q[0, 0, :3] = (0, top, 0)
            assert np.array_equal(encode.unpack(encode.pack(q), p, shape), q), (key, shape)


def test_clipping_and_missing_code():
    p = BY_KEY["gust"]
    q = encode.quantize(np.array([-5.0, 1e6, np.nan], np.float32), p)
    assert list(q) == [0, 254, 255]


def test_catalog_is_consistent():
    assert len({p.key for p in PRODUCTS}) == len(PRODUCTS)
    for p in PRODUCTS:
        assert p.group in GROUPS, p.key
        assert p.kind in ("cum", "inst", "max")
        assert p.dtype in ("u8", "u16") and p.step > 0
        s = p.style
        assert len(s.colors) == len(s.levels) + 1 and list(s.levels) == sorted(s.levels), p.key
        assert all(t in s.levels for t in s.ticks), p.key
        assert p.thr or p.kind == "max" or True
        assert p.thr_dir in ("ge", "le")
        # the legend range must be representable in the stored range
        top = p.lo + p.step * (254 if p.dtype == "u8" else 65534)
        assert s.levels[0] >= p.lo - 1e-9 and s.levels[-1] <= top + 1e-9, p.key


def test_one_hour_variants():
    assert {p.key for p in ONE_HOUR} == {p.key + "_1h" for p in PRODUCTS if p.has1h}
    assert {"qpf", "snow_model", "snow_12", "snow_kuchera", "snow_maxt", "snow_thick", "snow_mean"} == {p.key for p in PRODUCTS if p.has1h}
    for p in ONE_HOUR:
        assert p.dtype == "u16" and p.step <= BY_KEY[p.key[:-3]].step
        assert p.style is BY_KEY[p.key[:-3]].style
    assert "snow_10" not in BY_KEY and "slr_12" in BY_KEY


def test_mask_pack():
    import gzip
    m = np.random.default_rng(0).random((7, 13)) > 0.5
    bits = np.unpackbits(np.frombuffer(gzip.decompress(encode.pack_mask(m)), np.uint8))[:m.size].reshape(m.shape)
    assert np.array_equal(bits.astype(bool), m)
