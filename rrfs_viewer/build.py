"""Build the static site: fetch -> derive -> encode -> data.json + overlay + index.html.

    python -m rrfs_viewer.build --out public --cache .cache
    python -m rrfs_viewer.build --demo --max-fhr 12        # offline, synthetic storm, shortened run
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import encode, grib, hrrr as hrrr_mod, pipeline
from .catalog import BY_KEY, GROUPS, PRODUCTS, STORE
from .config import EDGE_M, FILE_2D, FILE_HSFC, MEMBERS, MIN_SEP_M, OUT_DX_M, frames
from .s3 import S3Source

log = logging.getLogger("rrfs_viewer")
SITE_DIR = Path(__file__).resolve().parent.parent / "site"
iso = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
SOURCE_LABELS = {"refs": "RRFS Mean (REFS)", **{m: m for m in MEMBERS}}


# ============================================================================== worker (runs in a child process)
def _client(demo_now: str | None):
    if demo_now:
        from .testing import FakeS3
        # RRFS_DEMO_HOLES="m003:ASNOW:48,m004:APCP:23": messages missing from single members, to try the fallbacks on the demo site
        holes = {(v, int(h), m) for m, v, h in (x.split(":") for x in os.environ.get("RRFS_DEMO_HOLES", "").split(",") if x)}
        return FakeS3(datetime.fromisoformat(demo_now), missing=holes)
    from .s3 import make_client
    return make_client()


LIFT_FILE = "lift.npy"          # deterministic DZDT on the display raster (float16), written once per build, read by every member


def run_source(args) -> tuple[str, dict[str, bytes], dict]:
    """One source, never fatal: if it fails outright the build goes on with the others and the failure is reported."""
    try:
        return _run_source(args)
    except Exception as exc:
        log.exception("%s failed - continuing without it", args[0])
        return args[0], {}, {"bytes": 0, "requests": 0, "seconds": 0, "missing": {}, "hints": {}, "error": f"{type(exc).__name__}: {exc}"}


def _run_source(args) -> tuple[str, dict[str, bytes], dict]:
    source, cycle_iso, max_fhr, regrid_path, demo_now, lift_dir = args
    t0 = time.time()
    cycle = datetime.fromisoformat(cycle_iso)
    rg = grib.Regrid.load(Path(regrid_path))
    src = S3Source(_client(demo_now), **({"bucket": hrrr_mod.HRRR_BUCKET} if source == "hrrr" else {}))
    tick = lambda s, fr: log.info("%s F%03d  (%.0fs)", s, fr, time.time() - t0)
    if source == "det":                                   # not a displayed source: it only feeds the members' Cobb-with-lift products
        lift, info = pipeline.process_lift(src, cycle, rg, max_fhr, tick)
        if lift is not None:
            np.save(Path(lift_dir) / LIFT_FILE, lift)
        info["seconds"] = round(time.time() - t0, 1)
        return source, {}, info
    if source == "hrrr":                                  # one deterministic run, its own native grid; hourly files to F048 (frames beyond stay empty)
        arrays, info = hrrr_mod.process_hrrr(src, cycle, rg, max_fhr, tick)
    elif source == "refs":
        arrays, info = pipeline.process_refs(src, cycle, rg, max_fhr, tick)
    else:
        path = Path(lift_dir) / LIFT_FILE
        lift = np.load(path, mmap_mode="r") if path.exists() else None
        arrays, info = pipeline.process_member(src, cycle, source, rg, max_fhr, tick, lift=lift)
    blobs = {}
    for key, a in arrays.items():
        if np.isfinite(a).any():
            blobs[key] = encode.pack(encode.quantize(a, BY_KEY[key]))
    info["seconds"] = round(time.time() - t0, 1)
    return source, blobs, info


# ============================================================================== main
def _report(s: str, blobs: dict, info: dict) -> None:
    log.info("%s done in %.0fs: %d products, %.0f MB downloaded, %d requests, missing=%s", s, info["seconds"],
             len(blobs), info["bytes"] / 1e6, info["requests"], info.get("missing") or "-")
    for k, h in sorted((info.get("hints") or {}).items()):          # what the files really contain for fields that were not found
        log.warning("%s: field not found %s -> %s", s, k, h)
    q = info.get("quality") or []
    for bit, text in ((1, "estimated hours"), (2, "lower bounds"), (4, "interpolated frames"), (8, "empty frames")):
        frames_ = [f"F{(i + 1) * 3:03d}" for i, v in enumerate(q) if v & bit]
        if frames_:
            log.warning("%s: %s in %d frame(s): %s", s, text, len(frames_), " ".join(frames_[:12]) + (" ..." if len(frames_) > 12 else ""))
    if info.get("error"):
        log.error("%s: not published (%s)", s, info["error"])


def code_version() -> str:
    """Fingerprint of everything that shapes the output (python package + dashboard page), so a code change forces a rebuild."""
    import hashlib
    h = hashlib.sha1()
    root = Path(__file__).resolve().parent
    for f in sorted([*root.glob("*.py"), root / "data" / "stations.csv", SITE_DIR / "index.html"]):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def deployed_manifest(url: str) -> dict | None:
    """The live site's data.json, or None when it cannot be read - then we simply build."""
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/data.json", timeout=20) as r:
            return json.load(r)
    except Exception as e:                       # first run, Pages not enabled yet, network hiccup ...
        log.info("could not read the deployed manifest (%s)", e)
        return None


def up_to_date(manifest: dict | None, cycle_iso: str, code: str, complete: list[str] | None = None) -> bool:
    """Skip the build only when the live site shows this cycle, was built by this version of the code, and is not demo data.

    ``complete``: the sources that have finished publishing; a cycle built while a member was still missing is rebuilt once it is there.
    """
    return (bool(manifest) and not manifest.get("demo") and manifest.get("cycle") == cycle_iso and manifest.get("code") == code
            and (complete is None or manifest.get("complete") is None or sorted(manifest["complete"]) == sorted(complete)))


MIN_MEMBERS = 3                                              # fewest complete members for which an incomplete cycle is still published


def ensure_regrid(src: S3Source, cycle: datetime, cache: Path, first_fhr: int) -> grib.Regrid:
    from .basemap import EXTENT, PROJ
    from .config import OUT_DX_M

    path = cache / "regrid.npz"
    rg = grib.Regrid.load(path)
    # one message with geometry: the surface-pressure field of the first member that has a file
    loc = next((l for l in (src.locate(FILE_2D, m, cycle, first_fhr) for m in MEMBERS) if l is not None), None)       # any member will do
    if loc is None:
        raise RuntimeError("cannot read the grid definition: no member file for the chosen cycle")
    data = src.fetch_message(loc, "PRES", "surface")
    vals, geo = grib.decode(data, want_geometry=True)
    if rg is not None and rg.native_shape == vals.shape:
        return rg
    log.info("building native -> display grid mapping (%s)", vals.shape)
    rg = grib.build_regrid(geo[0], geo[1], PROJ, EXTENT, OUT_DX_M)
    rg.save(path)
    return rg


def ensure_regrid_hrrr(src: S3Source, cycle: datetime, cache: Path, fhr: int) -> grib.Regrid:
    """HRRR native grid (1799 x 1059 Lambert conformal, 3 km) -> the same display raster as the RRFS products; cached like the RRFS mapping."""
    from .basemap import EXTENT, PROJ
    from .config import OUT_DX_M

    path = cache / "regrid_hrrr.npz"
    loc = src.locate(FILE_HSFC, "hrrr", cycle, fhr)
    if loc is None:
        raise RuntimeError(f"no HRRR surface file for {cycle:%Y%m%d %H}z F{fhr:02d}")
    vals, geo = grib.decode(src.fetch_message(loc, "PRES", "surface"), want_geometry=True)
    rg = grib.Regrid.load(path)
    if rg is not None and rg.native_shape == vals.shape:
        return rg
    log.info("building HRRR native -> display grid mapping (%s)", vals.shape)
    rg = grib.build_regrid(geo[0], geo[1], PROJ, EXTENT, OUT_DX_M)
    rg.save(path)
    return rg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="public")
    ap.add_argument("--cache", default=".cache", help="small cache (grid mapping); never committed")
    ap.add_argument("--demo", action="store_true", help="offline synthetic data via a fake S3")
    ap.add_argument("--now", help="override 'now' (ISO UTC)")
    ap.add_argument("--max-fhr", type=int, default=60)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--sources", default="refs," + ",".join(MEMBERS))
    ap.add_argument("--no-lift", action="store_true", help="skip the deterministic-RRFS DZDT fetch (the Cobb-with-lift products stay empty)")
    ap.add_argument("--no-hrrr", action="store_true", help="skip the HRRR products (a separate bucket and grid; the HRRR group stays empty)")
    ap.add_argument("--no-terrain", action="store_true", help="skip the terrain / water basemap (the map background stays white)")
    ap.add_argument("--cycle", help="force a cycle (YYYYMMDDHH) instead of the newest complete one")
    ap.add_argument("--skip-if-deployed", metavar="URL", help="exit early (code 0, no output directory) when the site at URL already shows this cycle built by this code version")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("botocore", "boto3", "urllib3", "matplotlib", "PIL", "fiona", "shapely"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    t0 = time.time()
    now = datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now else datetime.now(timezone.utc)
    now = now.astimezone(timezone.utc)
    demo_now = iso(now) if args.demo else None
    src = S3Source(_client(demo_now))
    F = frames(args.max_fhr)

    partial = False
    if args.cycle:
        cycle = datetime.strptime(args.cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    else:
        cycle = pipeline.find_cycle(src, now, args.max_fhr)
        if cycle is None:                                  # no cycle is complete everywhere: take the newest one most members have finished
            cycle = pipeline.find_cycle(src, now, args.max_fhr, min_members=MIN_MEMBERS)
            partial = cycle is not None
    if cycle is None:
        log.error("no usable RRFS/REFS cycle (through F%03d, at least %d members) found in the last 48 h", args.max_fhr, MIN_MEMBERS)
        return 1
    complete = pipeline.complete_sources(src, cycle, args.max_fhr)
    log.info("cycle %s (F%03d), %d frames; complete sources: %s", cycle, args.max_fhr, len(F), ",".join(complete) or "-")
    code = code_version()
    if args.skip_if_deployed and up_to_date(deployed_manifest(args.skip_if_deployed), iso(cycle), code, complete):
        log.info("the deployed site already shows this cycle and this code version (%s) - nothing to do", code)
        return 0

    cache, out = Path(args.cache), Path(args.out)
    cache.mkdir(parents=True, exist_ok=True)
    rg = ensure_regrid(src, cycle, cache, F[0])
    ny, nx = rg.shape
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)

    # ---------------------------------------------------------------- the ensemble sources, in parallel processes
    sources = [s for s in args.sources.split(",") if s]
    if partial:                                            # whatever has begun to publish is used: missing frames simply stay empty
        began = pipeline.complete_sources(src, cycle, F[0])
        log.warning("cycle %s is not complete for every source: using %s (complete through F%03d: %s)", cycle, ",".join(s for s in sources if s in began),
                    args.max_fhr, ",".join(complete))
        sources = [s for s in sources if s in began] or sources
    lift_dir = tempfile.mkdtemp(prefix="rrfs_lift_")                  # ~300 MB at F060: scratch space, removed below, never the cache
    use_hrrr = not args.no_hrrr
    if use_hrrr:                                                      # HRRR is optional: without its files or grid the rest of the site is built as before
        try:
            ensure_regrid_hrrr(S3Source(_client(demo_now), bucket=hrrr_mod.HRRR_BUCKET), cycle, cache, F[0])
        except Exception as exc:
            log.warning("HRRR products unavailable for this cycle: %s", exc)
            use_hrrr = False
    mk = lambda s: (s, iso(cycle), args.max_fhr, str(cache / ("regrid_hrrr.npz" if s == "hrrr" else "regrid.npz")), demo_now, lift_dir)
    results: dict[str, tuple[dict, dict]] = {}
    lift_info: dict = {}

    def collect(s, blobs, info):
        if s == "det":
            lift_info.update(info)
            log.info("deterministic lift: %s frames, %.0f MB downloaded, %d requests%s", info.get("frames_ok", 0), info.get("bytes", 0) / 1e6,
                      info.get("requests", 0), f"  -> {info['problem']}" if info.get("problem") else "")
            for k, h in sorted((info.get("hints") or {}).items()):
                log.warning("det: field not found %s -> %s", k, h)
        else:
            results[s] = (blobs, info)
            _report(s, blobs, info)

    # the deterministic lift first (the members read it), in parallel with the REFS mean, then the members
    first = [] if args.no_lift else ["det"]
    first += [s for s in sources if s == "refs"]
    first += ["hrrr"] if use_hrrr else []
    rest = [s for s in sources if s != "refs"]
    try:
        if args.workers <= 1:
            for s in first + rest:
                collect(*run_source(mk(s)))
        else:
            with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
                for out_ in pool.map(run_source, [mk(s) for s in first]):
                    collect(*out_)
                for out_ in pool.map(run_source, [mk(s) for s in rest]):
                    collect(*out_)
    finally:
        shutil.rmtree(lift_dir, ignore_errors=True)

    # ---------------------------------------------------------------- write data files + manifest
    total = 0
    avail: dict[str, dict[str, bool]] = {p.key: {} for p in STORE}
    for s, (blobs, info) in results.items():
        slot = "refs" if s == "hrrr" else s                # HRRR products are single-source: they use the single-source slot, shown on the page under their own name
        for key, blob in blobs.items():
            (out / "data" / f"{key}.{slot}.bin").write_bytes(blob)
            total += len(blob)
            avail[key][slot] = True
    from . import basemap, boundaries
    layers = boundaries.load(basemap.PROJ)
    overlay_bytes = basemap.render_overlay(layers, out / "overlay.png")
    counties_bytes = basemap.render_overlay(layers, out / "counties.png", names=basemap.COUNTY_LAYERS, frame=False)      # light-gray county lines: their own layer, toggled on the page
    (out / "data" / "land.bin").write_bytes(encode.pack_mask(basemap.land_mask(layers, rg.lon, rg.lat)))
    lines = basemap.overlay_lines(layers)                   # the same boundaries as vectors: the page draws these instead of the PNG once zoomed in
    (out / "data" / "lines.bin").write_bytes(gzip.compress(json.dumps(lines, separators=(",", ":")).encode(), 9, mtime=0))

    try:                                                    # optional layers (toggled on the page); the site works without them
        from . import gis
        (out / "data" / "gis.bin").write_bytes(gzip.compress(json.dumps(gis.build(), separators=(",", ":")).encode(), 9, mtime=0))
        gis_file = "data/gis.bin"
    except Exception as exc:
        log.warning("highway / city layers unavailable: %s", exc)
        gis_file = None

    basemap_block = None                                    # terrain + water basemap: rendered once and cached (the elevation tiles are fetched the first time only)
    if not args.no_terrain:
        try:
            from . import terrain
            fetch = None
            if args.demo:                                   # offline: invented terrain, like the invented storm
                from .testing import SyntheticTerrain
                fetch = SyntheticTerrain()
            basemap_block = terrain.ensure(out, cache, fetch=fetch, synthetic=args.demo)
        except Exception as exc:                            # the site works without it (white background, as before)
            log.warning("terrain basemap unavailable: %s", exc)

    manifest = {
        "generated": iso(now), "version": int(now.timestamp()), "demo": args.demo, "code": code,
        "cycle": iso(cycle), "max_fhr": args.max_fhr, "frames": list(F), "complete": complete,
        # per member and frame: bit 1 some hours estimated, 2 lower bound (hours missing), 4 frame interpolated, 8 nothing (see pipeline.fill_gaps)
        "quality": {s: results[s][1]["quality"] for s in MEMBERS if s in results and results[s][1].get("quality")},
        "errors": {s: results[s][1]["error"] for s in results if results[s][1].get("error")},
        "valid": [iso(cycle + timedelta(hours=h)) for h in F],
        "grid": {"nx": nx, "ny": ny}, "overlay": "overlay.png", "counties": "counties.png", "overlay_px": list(basemap.OVERLAY_PX),
        "sources": {s: {"label": SOURCE_LABELS[s], "ok": bool(results.get(s, ({}, {}))[0])} for s in ["refs", *MEMBERS]},
        "members": list(MEMBERS), "groups": list(GROUPS),
        "hrrr": {"ok": bool(results.get("hrrr", ({}, {}))[0]), "frames": results.get("hrrr", ({}, {}))[1].get("frames", 0), "max_fhr": hrrr_mod.HRRR_MAX_FHR},
        "basemap": basemap_block, "stations": basemap.station_layout(rg.x, rg.y), "stations_all": basemap.station_all(rg.x, rg.y), "lines": "data/lines.bin", "gis": gis_file,
        "station_px": {"sep": round(MIN_SEP_M / ((basemap.EXTENT[1] - basemap.EXTENT[0]) / basemap.OVERLAY_PX[0]), 3),
                       "edge": round(EDGE_M / ((basemap.EXTENT[1] - basemap.EXTENT[0]) / basemap.OVERLAY_PX[0]), 3)},
        "products": [{
            "key": p.key, "label": p.label, "title": p.map_title, "group": p.group, "unit": p.unit, "kind": p.kind,
            "desc": p.desc, "decimals": p.decimals, "dtype": p.dtype, "lo": p.lo, "step": p.step, "refs": p.refs,
            "levels": p.style.levels, "colors": p.style.colors, "ticks": p.style.ticks, "equal": p.style.equal_width, "classes": p.style.classes,
            "thr": list(p.thr), "thr_dir": p.thr_dir, "available": avail[p.key],
            "pmm": p.pmm, "sam": p.sam, "single": p.single, "lines": [list(l) for l in p.lines], "cat": p.cat, "has1h": p.has1h, "step1h": (p.step1h or p.step) if p.has1h else None,
            "available_1h": avail.get(p.key + "_1h", {}) if p.has1h else {},
        } for p in PRODUCTS],
        "stats": {"data_bytes": total, "overlay_bytes": overlay_bytes, "counties_bytes": counties_bytes,
                  "downloaded_bytes": {s: results[s][1]["bytes"] for s in results},
                  "s3_requests": {s: results[s][1]["requests"] for s in results},
                  "refs_accumulation": results.get("refs", ({}, {}))[1].get("mode"),
                  "det_lift": {k: lift_info[k] for k in ("frames_ok", "bytes", "requests", "problem") if k in lift_info}},
    }
    (out / "data.json").write_text(json.dumps(manifest, separators=(",", ":")))
    shutil.copy(SITE_DIR / "index.html", out / "index.html")
    (out / ".nojekyll").touch()
    ok = [s for s in sources if results[s][0]]
    log.info("done in %.0fs: %d sources ok, data %.1f MB, overlay %.0f kB", time.time() - t0, len(ok), total / 1e6, overlay_bytes / 1e3)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
