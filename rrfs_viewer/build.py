"""Build the static site: fetch -> derive -> encode -> data.json + overlay + index.html.

    python -m rrfs_viewer.build --out public --cache .cache
    python -m rrfs_viewer.build --demo --max-fhr 12        # offline, synthetic storm, shortened run
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import encode, grib, pipeline
from .catalog import BY_KEY, GROUPS, PRODUCTS, STORE
from .config import FILE_2D, MEMBERS, frames
from .s3 import S3Source

log = logging.getLogger("rrfs_viewer")
SITE_DIR = Path(__file__).resolve().parent.parent / "site"
iso = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
SOURCE_LABELS = {"refs": "RRFS Mean (REFS)", **{m: m for m in MEMBERS}}


# ============================================================================== worker (runs in a child process)
def _client(demo_now: str | None):
    if demo_now:
        from .testing import FakeS3
        return FakeS3(datetime.fromisoformat(demo_now))
    from .s3 import make_client
    return make_client()


def run_source(args) -> tuple[str, dict[str, bytes], dict]:
    source, cycle_iso, max_fhr, regrid_path, demo_now = args
    t0 = time.time()
    cycle = datetime.fromisoformat(cycle_iso)
    rg = grib.Regrid.load(Path(regrid_path))
    src = S3Source(_client(demo_now))
    tick = lambda s, fr: log.info("%s F%03d  (%.0fs)", s, fr, time.time() - t0)
    arrays, info = (pipeline.process_refs(src, cycle, rg, max_fhr, tick) if source == "refs"
                    else pipeline.process_member(src, cycle, source, rg, max_fhr, tick))
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


def up_to_date(manifest: dict | None, cycle_iso: str, code: str) -> bool:
    """Skip the build only when the live site shows this cycle, was built by this version of the code, and is not demo data."""
    return bool(manifest) and not manifest.get("demo") and manifest.get("cycle") == cycle_iso and manifest.get("code") == code


def ensure_regrid(src: S3Source, cycle: datetime, cache: Path, first_fhr: int) -> grib.Regrid:
    from .basemap import EXTENT, PROJ
    from .config import OUT_DX_M

    path = cache / "regrid.npz"
    rg = grib.Regrid.load(path)
    # one message with geometry: the surface-pressure field of member m001
    loc = src.locate(FILE_2D, MEMBERS[0], cycle, first_fhr)
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="public")
    ap.add_argument("--cache", default=".cache", help="small cache (grid mapping); never committed")
    ap.add_argument("--demo", action="store_true", help="offline synthetic data via a fake S3")
    ap.add_argument("--now", help="override 'now' (ISO UTC)")
    ap.add_argument("--max-fhr", type=int, default=60)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--sources", default="refs," + ",".join(MEMBERS))
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

    cycle = (datetime.strptime(args.cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc) if args.cycle
             else pipeline.find_cycle(src, now, args.max_fhr))
    if cycle is None:
        log.error("no complete RRFS/REFS cycle (through F%03d) found in the last 48 h", args.max_fhr)
        return 1
    log.info("cycle %s (F%03d), %d frames", cycle, args.max_fhr, len(F))
    code = code_version()
    if args.skip_if_deployed and up_to_date(deployed_manifest(args.skip_if_deployed), iso(cycle), code):
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
    jobs = [(s, iso(cycle), args.max_fhr, str(cache / "regrid.npz"), demo_now) for s in sources]
    results: dict[str, tuple[dict, dict]] = {}
    if args.workers <= 1:
        for j in jobs:
            s, blobs, info = run_source(j)
            results[s] = (blobs, info)
            _report(s, blobs, info)
    else:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
            for s, blobs, info in pool.map(run_source, jobs):
                results[s] = (blobs, info)
                _report(s, blobs, info)

    # ---------------------------------------------------------------- write data files + manifest
    total = 0
    avail: dict[str, dict[str, bool]] = {p.key: {} for p in STORE}
    for s, (blobs, info) in results.items():
        for key, blob in blobs.items():
            (out / "data" / f"{key}.{s}.bin").write_bytes(blob)
            total += len(blob)
            avail[key][s] = True
    from . import basemap, boundaries
    layers = boundaries.load(basemap.PROJ)
    overlay_bytes = basemap.render_overlay(layers, out / "overlay.png")
    (out / "data" / "land.bin").write_bytes(encode.pack_mask(basemap.land_mask(layers, rg.lon, rg.lat)))

    manifest = {
        "generated": iso(now), "version": int(now.timestamp()), "demo": args.demo, "code": code,
        "cycle": iso(cycle), "max_fhr": args.max_fhr, "frames": list(F),
        "valid": [iso(cycle + timedelta(hours=h)) for h in F],
        "grid": {"nx": nx, "ny": ny}, "overlay": "overlay.png", "overlay_px": list(basemap.OVERLAY_PX),
        "sources": {s: {"label": SOURCE_LABELS[s], "ok": bool(results.get(s, ({}, {}))[0])} for s in ["refs", *MEMBERS]},
        "members": list(MEMBERS), "groups": list(GROUPS),
        "stations": basemap.station_layout(rg.x, rg.y),
        "products": [{
            "key": p.key, "label": p.label, "title": p.map_title, "group": p.group, "unit": p.unit, "kind": p.kind,
            "desc": p.desc, "decimals": p.decimals, "dtype": p.dtype, "lo": p.lo, "step": p.step, "refs": p.refs,
            "levels": p.style.levels, "colors": p.style.colors, "ticks": p.style.ticks, "equal": p.style.equal_width,
            "thr": list(p.thr), "thr_dir": p.thr_dir, "available": avail[p.key],
            "has1h": p.has1h, "step1h": (p.step1h or p.step) if p.has1h else None,
            "available_1h": avail.get(p.key + "_1h", {}) if p.has1h else {},
        } for p in PRODUCTS],
        "stats": {"data_bytes": total, "overlay_bytes": overlay_bytes,
                  "downloaded_bytes": {s: results[s][1]["bytes"] for s in results},
                  "s3_requests": {s: results[s][1]["requests"] for s in results},
                  "refs_accumulation": results.get("refs", ({}, {}))[1].get("mode")},
    }
    (out / "data.json").write_text(json.dumps(manifest, separators=(",", ":")))
    shutil.copy(SITE_DIR / "index.html", out / "index.html")
    (out / ".nojekyll").touch()
    ok = [s for s in sources if results[s][0]]
    log.info("done in %.0fs: %d sources ok, data %.1f MB, overlay %.0f kB", time.time() - t0, len(ok), total / 1e6, overlay_bytes / 1e3)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
