"""Offline stand-in for the NOAA RRFS/REFS bucket (``--demo`` mode and the test-suite).

It serves *real* GRIB2 messages on an HRRR-style Lambert-conformal 3 km grid (smaller extent than the real CONUS grid),
a correct ``.idx`` sidecar for every file, and honours ``Range:`` requests, so the production fetch / decode / derive / encode
code runs unchanged.  The meteorology is a synthetic east-moving winter storm with a warm nose on its southeast side
(snow -> sleet -> freezing rain -> rain), strong winds and a heavy-snow band, perturbed per member.  It exists to exercise the
pipeline and the dashboard - it is not weather.
"""
from __future__ import annotations

import io
import re
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

import numpy as np

R_EARTH = 6371229.0
DX = 3000.0
CENTRE = (-71.4, 44.2)
LEVELS_ALL = (250, 300, 400, 500, 600, 700, 750, 800, 850, 900, 925, 950, 975, 1000)
T_LEV = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 600, 500, 400, 300)
RH_LEV = (925, 900, 850, 800, 750, 700, 600, 500, 400, 300)
HGT_LEV = (1000, 850, 700, 500)
LIFT_LEV = tuple(range(925, 299, -25))


def _noise(shape, seed, sigma):
    n = np.random.default_rng(seed).standard_normal(shape)
    f = np.fft.rfft2(n)
    ky, kx = np.fft.fftfreq(shape[0])[:, None], np.fft.rfftfreq(shape[1])[None, :]
    out = np.fft.irfft2(f * np.exp(-2 * (np.pi * sigma) ** 2 * (kx ** 2 + ky ** 2)), s=shape)
    return (out / out.std()).astype(np.float32)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class DemoGrid:
    def __init__(self, nx=600, ny=480):
        import pyproj
        self.nx, self.ny = nx, ny
        self.proj = pyproj.Proj(proj="lcc", lat_1=38.5, lat_2=38.5, lat_0=38.5, lon_0=-97.5, a=R_EARTH, b=R_EARTH)
        xc, yc = self.proj(*CENTRE)
        self.x0, self.y0 = xc - nx / 2 * DX, yc - ny / 2 * DX
        X, Y = np.meshgrid(self.x0 + np.arange(nx) * DX, self.y0 + np.arange(ny) * DX)
        self.lon, self.lat = self.proj(X, Y, inverse=True)
        self.lon, self.lat = self.lon.astype(np.float32), self.lat.astype(np.float32)
        self.first = self.proj(self.x0, self.y0, inverse=True)
        self.a = ((self.lon - CENTRE[0]) * np.cos(np.radians(self.lat)) * 111.0).astype(np.float32)   # km east of centre
        self.b = ((self.lat - CENTRE[1]) * 111.0).astype(np.float32)                                   # km north
        shape = (ny, nx)
        self.n = [_noise(shape, s, sg) for s, sg in ((1, 8), (2, 14), (3, 6), (4, 5))]
        self.elev = (700 * np.exp(-((self.a - 20) ** 2 + (self.b - 120) ** 2) / (2 * 70 ** 2))
                     + 300 * np.exp(-((self.a + 150) ** 2 + (self.b - 40) ** 2) / (2 * 90 ** 2))).astype(np.float32)

    def message(self, values: np.ndarray, cycle: datetime) -> bytes:
        import eccodes as ec
        g = ec.codes_grib_new_from_samples("GRIB2")
        ec.codes_set(g, "gridDefinitionTemplateNumber", 30)
        keys = dict(shapeOfTheEarth=6, Nx=self.nx, Ny=self.ny,
                    latitudeOfFirstGridPointInDegrees=self.first[1], longitudeOfFirstGridPointInDegrees=self.first[0] % 360,
                    LoVInDegrees=262.5, LaDInDegrees=38.5, Latin1InDegrees=38.5, Latin2InDegrees=38.5,
                    DxInMetres=DX, DyInMetres=DX, latitudeOfSouthernPoleInDegrees=-90.0, longitudeOfSouthernPoleInDegrees=0.0,
                    iScansNegatively=0, jScansPositively=1, dataDate=int(f"{cycle:%Y%m%d}"), dataTime=cycle.hour * 100,
                    packingType="grid_simple", bitsPerValue=14)
        for k, v in keys.items():
            ec.codes_set(g, k, v)
        ec.codes_set_values(g, np.nan_to_num(values, nan=9999.0).ravel().astype(np.float64))
        ec.codes_set(g, "bitmapPresent", 0)
        try:
            return ec.codes_get_message(g)
        finally:
            ec.codes_release(g)


class Storm:
    """One ensemble member's synthetic weather, hour by hour (SI units, like the GRIB files)."""

    def __init__(self, grid: DemoGrid, member: int, cycle: datetime):
        self.g, self.m = grid, member
        rng = np.random.default_rng(100 + member)
        self.shift_x, self.shift_y = rng.normal(0, 45), rng.normal(0, 35)
        self.amp = 1.0 + 0.18 * rng.standard_normal()
        self.cold = 0.9 * rng.standard_normal()
        self.nose = 1.0 + 0.5 * rng.standard_normal()
        self.cycle = cycle
        self._hour: dict[int, dict] = {}
        self._cum_snow_m: dict[int, np.ndarray] = {0: np.zeros_like(grid.a)}
        self._cum_rate: dict[int, np.ndarray] = {}

    def hour(self, h: int) -> dict[str, np.ndarray]:
        if h in self._hour:
            return self._hour[h]
        g = self.g
        a, b = g.a, g.b
        cx, cy = -700 + 28 * h + self.shift_x, -120 + 4 * h + self.shift_y
        d2 = (a - cx) ** 2 + (b - cy - 0.3 * (a - cx)) ** 2
        env = np.exp(-d2 / (2 * 170 ** 2))
        rate = np.clip(self.amp * (5.0 * env + 1.0 * np.exp(-d2 / (2 * 330 ** 2))) * (0.75 + 0.25 * g.n[3]) - 0.15, 0, None)
        sector = _sig((a - 190 + 0.7 * b) / 70.0)                                   # southeast side: milder, warm nose
        diurnal = 1.2 * np.cos(2 * np.pi * (((self.cycle.hour + h) - 5) % 24 - 15) / 24)
        t2 = (-4.5 + self.cold - 0.012 * b + 6.5 * sector + 1.1 * g.n[0] + diurnal - 0.0065 * g.elev).astype(np.float32)   # degC
        ps = (1010.0 - 0.115 * g.elev).astype(np.float32)                                                               # hPa
        wind = (6 + 20 * np.exp(-d2 / (2 * 240 ** 2)) + 3 * g.n[2]).clip(1, None)                                      # m/s
        gust = wind * (1.35 + 0.08 * np.abs(g.n[3])) + 2
        ang = np.radians(40 + 30 * g.n[2])
        u, v = -wind * np.sin(ang), -wind * np.cos(ang)

        T, RH = {}, {}
        env_c = np.clip(env * 1.2 + 0.25, 0, 1)
        for p in LEVELS_ALL:
            z = 44330.0 * (1 - (p / 1013.25) ** 0.1903) / 1000.0
            t = t2 + 273.15 - 6.2 * z + 4.2 * self.nose * sector * np.exp(-((p - 850.0) / 55.0) ** 2) + 0.4 * g.n[1]
            T[p] = t.astype(np.float32)
            RH[p] = np.clip(55 + 45 * env_c + 8 * g.n[1], 5, 100).astype(np.float32)
        nose_c = np.max([T[p] for p in (925, 900, 850, 800, 750, 700)], axis=0) - 273.15

        wet = rate >= 0.1
        zr = wet & (t2 < -0.3) & (nose_c > 2.2)
        ip = wet & ~zr & (t2 < 0.5) & (nose_c > 0.4)
        rain = wet & ~zr & ~ip & (t2 > 1.2)
        sn = wet & ~zr & ~ip & ~rain
        slr = np.clip(11 + 1.1 * np.clip(-t2, 0, 14) - 6 * np.clip(t2, 0, 3), 4, 25)
        snow_m_h = np.where(sn, rate * slr / 1000.0, 0.0) + np.where(ip, 0.35 * rate * 8 / 1000.0, 0.0)
        prev = self._cum_snow_m.get(h - 1)
        if prev is None:
            prev = self.asnow(h - 1)
        self._cum_snow_m[h] = (prev + snow_m_h).astype(np.float32)
        km = np.where(sn, 16.0 / (1 + 9 * rate ** 1.25), np.where(wet, 16.0 / (1 + 1.5 * rate), 16.0))
        out = dict(rate=rate.astype(np.float32), t2=t2, ps=ps, u=u.astype(np.float32), v=v.astype(np.float32),
                   wind=wind.astype(np.float32), gust=gust.astype(np.float32), T=T, RH=RH, sn=sn, ip=ip, zr=zr, rain=rain,
                   vis=(km * 1000).astype(np.float32), uvv=(0.04 + 0.5 * env * (0.8 + 0.2 * g.n[0])).clip(0, None).astype(np.float32),
                   asnow=self._cum_snow_m[h])
        self._hour[h] = out
        if len(self._hour) > 8:
            self._hour.pop(min(self._hour))
        return out

    def cum_rate(self, h: int) -> np.ndarray:
        if h not in self._cum_rate:
            prev = self.cum_rate(h - 1) if h > 1 else 0.0
            self._cum_rate[h] = (prev + self.hour(h)["rate"]).astype(np.float32)
        return self._cum_rate[h]

    def asnow(self, h: int) -> np.ndarray:
        if h <= 0:
            return self._cum_snow_m[0]
        self.hour(h)
        return self._cum_snow_m[h]


def _dbz(rate) -> np.ndarray:
    """Reflectivity (dBZ) of a rain rate in mm/h by Z = 200 R^1.6; -10 (the model's kind of no-echo fill, below 0) where it is dry."""
    r = np.asarray(rate, np.float32)
    return np.where(r > 0.03, 10.0 * np.log10(200.0 * np.maximum(r, 0.03) ** 1.6), -10.0).astype(np.float32)


def _info(kind: str, h: int) -> str:
    return {"inst": f"{h} hour fcst", "acc1": f"{h - 1}-{h} hour acc fcst", "acc0": f"0-{h} hour acc fcst",
            "max": f"{h - 1}-{h} hour max fcst"}[kind]


class FakeS3:
    """Implements the boto3 calls the pipeline uses: head_object and get_object (with ``Range``)."""
    _KEY = re.compile(
        r"^(?:refs\.(?P<d1>\d{8})/(?P<c1>\d{2})/ensprod/refs\.t(?P=c1)z\.(?P<t1>mean|pmmn)\.f(?P<h1>\d{2})\.conus\.grib2"
        r"|rrfs\.(?P<d3>\d{8})/(?P<c3>\d{2})/rrfs\.t(?P=c3)z\.prslev\.3km\.f(?P<h3>\d{3})\.conus\.grib2"
        r"|rrfsens\.(?P<d2>\d{8})/(?P<c2>\d{2})/(?P<m>m00[1-5])/rrfs\.t(?P=c2)z\.(?P=m)\.(?P<k>2dfld|prslev)nomads\.3km\.f(?P<h2>\d{3})\.conus\.grib2)"
        r"(?P<idx>\.idx)?$")

    def __init__(self, now: datetime, lag_hours: int = 4, refs_totals: bool = False, grid: DemoGrid | None = None,
                 missing: set[tuple[str, int]] | None = None, member_hourly_apcp: bool = True, member_asnow_totals: bool = True):
        self.now = now.astimezone(timezone.utc)
        self.lag = lag_hours
        self.refs_totals = refs_totals
        self.member_hourly_apcp = member_hourly_apcp      # False: members publish run-total APCP only (no 1-hour windows)
        self.member_asnow_totals = member_asnow_totals    # False: members publish 1-hour ASNOW windows only (no run totals)
        self.missing = missing or set()          # {(var, forecast_hour)} messages that are "not in the file" (for every source); (var, hour, member) for one member
        self.grid = grid or DemoGrid()
        self._storms: dict = {}
        self._files: OrderedDict = OrderedDict()
        self.range_requests: list[tuple[str, str | None]] = []

    # ------------------------------------------------------------------ availability
    def _published(self, cycle: datetime, h: int) -> bool:
        return cycle + timedelta(hours=self.lag + 0.04 * h) <= self.now and cycle.hour % 6 == 0

    def _storm(self, member: int, cycle: datetime) -> Storm:
        k = (member, cycle)
        if k not in self._storms:
            self._storms[k] = Storm(self.grid, member, cycle)
            while len(self._storms) > 6:
                self._storms.pop(next(iter(self._storms)))
        return self._storms[k]

    # ------------------------------------------------------------------ file synthesis
    def _messages(self, key: str, m) -> list[tuple[str, str, str, np.ndarray]]:
        """(var, level, info, values) for one file."""
        cyc = datetime.strptime((m["d1"] or m["d2"] or m["d3"]) + (m["c1"] or m["c2"] or m["c3"]), "%Y%m%d%H").replace(tzinfo=timezone.utc)
        h = int(m["h1"] or m["h2"] or m["h3"])
        K, MPH = 273.15, 1.0
        msgs: list[tuple[str, str, str, np.ndarray]] = []
        if m["h3"]:                                       # ---- deterministic run: DZDT [m/s] every 25 hPa, ascent peaking near 600 hPa
            x = self._storm(1, cyc).hour(h)
            up = np.asarray(x["uvv"], np.float32)
            return [("DZDT", f"{p} mb", _info("inst", h), (up * np.exp(-(((p - 600.0) / 220.0) ** 2))).astype(np.float32))
                    for p in LIFT_LEV if ("DZDT", h) not in self.missing]
        if m["h1"] and m["t1"] == "pmmn":                 # ---- REFS probability-matched mean: composite and 1 km reflectivity (invented from the members' rates)
            dbz = np.stack([_dbz(self._storm(i, cyc).hour(h)["rate"]) for i in range(1, 6)])
            pmm = np.sort(dbz, axis=0)[-2]                # a high-ish pick of the members: PMM keeps the peaks a plain mean would smear out
            msgs = [("REFC", "entire atmosphere", _info("inst", h), pmm),
                    ("REFD", "1000 m above ground", _info("inst", h), np.where(pmm > 0, pmm - 4.0, pmm).astype(np.float32))]
            return [t for t in msgs if (t[0], h) not in self.missing]
        if m["h1"]:                                       # ---- REFS mean (members' mean; flags -> fractions)
            S = [self._storm(i, cyc) for i in range(1, 6)]
            H = [s.hour(h) for s in S]
            avg = lambda f: np.mean([f(x) for x in H], axis=0)
            hourly = lambda f, hh: np.mean([f(s.hour(hh)) for s in S], axis=0)
            acc1 = hourly(lambda x: x["rate"], h)
            msgs += [("TMP", "2 m above ground", _info("inst", h), avg(lambda x: x["t2"]) + K),
                     ("WIND", "10 m above ground", _info("inst", h), avg(lambda x: x["wind"])),
                     ("VIS", "surface", _info("inst", h), avg(lambda x: x["vis"])),
                     ("APCP", "surface", _info("acc1", h), acc1),
                     ("WEASD", "surface", _info("acc1", h), hourly(lambda x: x["rate"] * (x["sn"] | x["ip"]), h)),
                     ("ASNOW", "surface", _info("acc1", h), np.mean([s.asnow(h) - s.asnow(h - 1) for s in S], axis=0)),
                     ("ASNOW", "surface", _info("acc0", h), np.mean([s.asnow(h) for s in S], axis=0)),
                     ("CRAIN", "surface", _info("inst", h), avg(lambda x: x["rain"].astype(np.float32))),
                     ("CFRZR", "surface", _info("inst", h), avg(lambda x: x["zr"].astype(np.float32))),
                     ("CICEP", "surface", _info("inst", h), avg(lambda x: x["ip"].astype(np.float32))),
                     ("CSNOW", "surface", _info("inst", h), avg(lambda x: x["sn"].astype(np.float32)))]
            if self.refs_totals:
                msgs += [("APCP", "surface", _info("acc0", h), np.mean([s.cum_rate(h) for s in S], axis=0))]
            return msgs
        s = self._storm(int(m["m"][-1]), cyc)
        x = s.hour(h)
        if m["k"] == "2dfld":
            f32 = lambda a: np.asarray(a, np.float32)
            msgs = [("REFC", "entire atmosphere (considered as a single layer)", _info("inst", h), _dbz(x["rate"])),
                    ("VIS", "surface", _info("inst", h), x["vis"]),
                    ("GUST", "surface", _info("inst", h), x["gust"]),
                    ("MAXUVV", "100-1000 mb", _info("max", h), x["uvv"]),
                    ("PRES", "surface", _info("inst", h), x["ps"] * 100),
                    ("TMP", "2 m above ground", _info("inst", h), x["t2"] + K),
                    ("DPT", "2 m above ground", _info("inst", h), x["t2"] + K - 1.5),
                    ("UGRD", "10 m above ground", _info("inst", h), x["u"]),
                    ("VGRD", "10 m above ground", _info("inst", h), x["v"]),
                    ("CSNOW", "surface", _info("inst", h), f32(x["sn"])),
                    ("CICEP", "surface", _info("inst", h), f32(x["ip"])),
                    ("CFRZR", "surface", _info("inst", h), f32(x["zr"])),
                    ("CRAIN", "surface", _info("inst", h), f32(x["rain"])),
                    ("APCP", "surface", _info("acc0", h), s.cum_rate(h))]
            if self.member_hourly_apcp:
                msgs.append(("APCP", "surface", _info("acc1", h), x["rate"]))
            if self.member_asnow_totals:
                msgs.append(("ASNOW", "surface", _info("acc0", h), x["asnow"]))
            else:
                msgs.append(("ASNOW", "surface", _info("acc1", h), np.maximum(s.asnow(h) - s.asnow(h - 1), 0)))
        else:
            msgs = []
            z = np.full_like(x["t2"], 100.0, dtype=np.float64)                    # heights from the hypsometric equation, 1000 hPa up
            for lo, hi in zip(HGT_LEV[:-1], HGT_LEV[1:]):
                z = z + 29.2706 * 0.5 * (x["T"][lo].astype(np.float64) + x["T"][hi]) * np.log(lo / hi)
                msgs.append(("HGT", f"{hi} mb", _info("inst", h), z.astype(np.float32)))
            msgs.insert(0, ("HGT", "1000 mb", _info("inst", h), np.full_like(x["t2"], 100.0)))
            msgs += [("TMP", f"{p} mb", _info("inst", h), x["T"][p]) for p in T_LEV]
            msgs += [("RH", f"{p} mb", _info("inst", h), x["RH"][p]) for p in RH_LEV]
        mem = m["m"]
        return [t for t in msgs if (t[0], h) not in self.missing and (t[0], h, mem) not in self.missing]

    def _file(self, key: str, m):
        if key not in self._files:
            cyc = datetime.strptime((m["d1"] or m["d2"] or m["d3"]) + (m["c1"] or m["c2"] or m["c3"]), "%Y%m%d%H").replace(tzinfo=timezone.utc)
            blob, idx, off = b"", [], 0
            for i, (var, level, info, vals) in enumerate(self._messages(key, m), 1):
                msg = self.grid.message(np.asarray(vals, np.float32), cyc)
                idx.append(f"{i}:{off}:d={cyc:%Y%m%d%H}:{var}:{level}:{info}:")
                blob += msg
                off += len(msg)
            self._files[key] = (blob, "\n".join(idx) + "\n")
            while len(self._files) > 8:
                self._files.popitem(last=False)
        return self._files[key]

    # ------------------------------------------------------------------ boto3 surface
    def _match(self, Key):
        m = self._KEY.match(Key)
        if not m:
            raise KeyError(Key)
        cyc = datetime.strptime((m["d1"] or m["d2"] or m["d3"]) + (m["c1"] or m["c2"] or m["c3"]), "%Y%m%d%H").replace(tzinfo=timezone.utc)
        if not self._published(cyc, int(m["h1"] or m["h2"] or m["h3"])):
            raise KeyError(Key)
        return m

    def head_object(self, Bucket, Key, **_):
        self._match(Key)
        return {}

    def get_object(self, Bucket, Key, Range=None, **_):
        m = self._match(Key)
        base = Key[:-4] if m["idx"] else Key
        blob, idx = self._file(base, m)
        data = idx.encode() if m["idx"] else blob
        if not m["idx"]:
            self.range_requests.append((Key, Range))
        if Range:
            a, b = re.match(r"bytes=(\d+)-(\d*)", Range).groups()
            data = data[int(a): (int(b) + 1 if b else None)]
        return {"Body": io.BytesIO(data)}


# ============================================================================== invented terrain (demo build and tests)
class SyntheticTerrain:
    """Offline stand-in for the AWS terrain tiles: ``SyntheticTerrain()(z, x, y)`` returns a "terrarium" PNG of an invented landscape.

    Land and sea follow the real Natural Earth coast; the land rises inland, with the real summits of ``data/peaks.csv`` and a broad ridged
    texture on top, and the sea floor drops away from the coast.  It exists to exercise the terrain code and to preview the colours without
    network access.  It is not geography.
    """

    def __init__(self):
        import cartopy.io.shapereader as shpreader
        import shapely
        from scipy import ndimage

        from . import boundaries
        from .config import DISPLAY_BBOX

        lon0, lon1, lat0, lat1 = DISPLAY_BBOX
        pad = 4.0
        box = shapely.geometry.box(lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad)
        recs = [r.geometry for r in shpreader.Reader(boundaries.natural_earth_path("physical", "land")).records()]
        land = shapely.union_all([g.intersection(box) for g in recs if g is not None and g.intersects(box)])
        self.step = 0.04
        self.lon_g = np.arange(lon0 - pad, lon1 + pad + 1e-9, self.step)
        self.lat_g = np.arange(lat0 - pad, lat1 + pad + 1e-9, self.step)
        LON, LAT = np.meshgrid(self.lon_g, self.lat_g)
        mask = shapely.contains_xy(land, LON, LAT)
        km = (self.step * 111.0 * np.cos(np.radians(44.0)), self.step * 111.0)           # km per grid step east, north
        self.inland = ndimage.distance_transform_edt(mask, sampling=(km[1], km[0])).astype(np.float32)
        self.offshore = ndimage.distance_transform_edt(~mask, sampling=(km[1], km[0])).astype(np.float32)
        rng = np.random.default_rng(7)
        k = 28
        wave = 1.0 / np.geomspace(2.5, 45.0, k)                                          # wavelengths 45 km .. 2.5 km, in cycles per km
        ang = rng.uniform(0, 2 * np.pi, k)
        self._k = np.stack([wave * np.cos(ang), wave * np.sin(ang)], 1)
        self._phase = rng.uniform(0, 2 * np.pi, k)
        self._amp = (wave / wave.min()) ** -0.9
        self._amp /= np.sqrt((self._amp ** 2).sum() / 2)
        import csv
        from pathlib import Path
        with open(Path(__file__).parent / "data" / "peaks.csv", newline="") as f:
            self.peaks = [(float(r["lon"]), float(r["lat"]), float(r["feet"]) * 0.3048) for r in csv.DictReader(f)]

    def _grid(self, field, lon, lat):
        from scipy import ndimage
        return ndimage.map_coordinates(field, [(lat - self.lat_g[0]) / self.step, (lon - self.lon_g[0]) / self.step], order=1, mode="nearest")

    def elevation(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        x = (lon - (-71.4)) * 111.0 * np.cos(np.radians(44.0))                           # km east / north of the map centre
        y = (lat - 44.2) * 111.0
        wav = np.zeros(lon.shape)
        for (kx, ky), ph, a in zip(self._k, self._phase, self._amp):
            wav += a * np.sin(2 * np.pi * (kx * x + ky * y) + ph)
        ridged = 1.0 - np.abs(wav) / 2.0                                                  # 0..1, sharp crests
        inland, offshore = self._grid(self.inland, lon, lat), self._grid(self.offshore, lon, lat)
        land = 6 + 1.4 * np.minimum(inland, 110) + (25 + 2.2 * np.minimum(inland, 100)) * ridged + 18 * wav
        for plon, plat, m in self.peaks:
            d2 = ((lon - plon) * 111.0 * np.cos(np.radians(plat))) ** 2 + ((lat - plat) * 111.0) ** 2
            land += 0.30 * m * np.exp(-d2 / (2 * 38.0 ** 2)) + 0.62 * m * np.exp(-d2 / (2 * 9.0 ** 2))
        d2 = ((lon + 74.0) * 111.0 * np.cos(np.radians(44.0))) ** 2 + ((lat - 44.1) * 111.0) ** 2
        land += 520 * np.exp(-d2 / (2 * 55.0 ** 2)) * (0.6 + 0.4 * ridged)             # an Adirondack-like dome
        sea = -(6 + 2.4 * offshore ** 1.08) * (1 + 0.04 * wav)
        return np.where(inland > 0, np.maximum(land, 1.0), np.maximum(sea, -4200.0)).astype(np.float32)

    def __call__(self, z: int, x: int, y: int) -> bytes:
        from .terrain import encode_terrarium, tile_coords
        n = 2.0 ** z
        px = (x + (np.arange(256) + 0.5) / 256.0) / n
        py = (y + (np.arange(256) + 0.5) / 256.0) / n
        lon = np.broadcast_to(px * 360.0 - 180.0, (256, 256))
        lat = np.broadcast_to(np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * py)))).reshape(256, 1), (256, 256))
        return encode_terrarium(self.elevation(np.array(lon), np.array(lat)))
