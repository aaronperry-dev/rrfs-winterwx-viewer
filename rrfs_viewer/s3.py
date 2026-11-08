"""Anonymous access to the NOAA RRFS / REFS open-data bucket, one HTTP byte-range request per GRIB2 message.

Every GRIB2 file has a sidecar ``.idx`` that lists the start byte of each message, so a single field is a
single ``Range:`` request (~1-2 MB) rather than the whole file (hundreds of MB).
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime

from .config import BUCKET, FileKind

log = logging.getLogger(__name__)


def make_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client(
        "s3", region_name="us-east-1",
        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 6, "mode": "standard"},
                      max_pool_connections=32, connect_timeout=15, read_timeout=90),
    )


# ------------------------------------------------------------------------- .idx parsing
@dataclass(frozen=True)
class IdxEntry:
    num: int
    start: int
    end: int | None      # inclusive; None = to end of file
    var: str
    level: str
    info: str            # e.g. "19 hour fcst", "18-19 hour acc fcst", "0-19 hour acc fcst", "18-19 hour max fcst"


def parse_idx(text: str) -> list[IdxEntry]:
    """Parse wgrib2-style inventory lines: ``num:start:d=YYYYMMDDHH:VAR:level:info:`` (extra ':' fields ignored)."""
    rows = []
    for line in text.splitlines():
        p = line.strip().split(":")
        if len(p) < 6 or not p[0].isdigit() or not p[1].isdigit():
            continue
        rows.append((int(p[0]), int(p[1]), p[3], p[4], p[5]))
    out = []
    for i, (num, start, var, level, info) in enumerate(rows):
        end = rows[i + 1][1] - 1 if i + 1 < len(rows) else None
        out.append(IdxEntry(num, start, end, var, level, info))
    return out


def find_entry(entries: list[IdxEntry], var: str, level: str, info: str | None = None) -> IdxEntry | None:
    """First record with this variable and level; ``info`` is a regex that must match the start of the time descriptor."""
    rx = re.compile(info) if info else None
    lv = re.compile(level[1:]) if level.startswith("~") else None      # "~regex": the level descriptor only has to start like this (for fields whose wording is not certain)
    for e in entries:
        if e.var == var and (lv.match(e.level) if lv else e.level == level) and (rx is None or rx.match(e.info)):
            return e
    return None


# ------------------------------------------------------------------------- source
@dataclass
class Located:
    key: str
    entries: list[IdxEntry]


class S3Source:
    def __init__(self, client, bucket: str = BUCKET):
        self.client, self.bucket = client, bucket
        self.bytes_downloaded = 0
        self.requests = 0
        self._idx: dict[str, Located | None] = {}
        self._lock = threading.Lock()

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def _get(self, key: str, byte_range: str | None = None) -> bytes:
        kw = {"Bucket": self.bucket, "Key": key}
        if byte_range:
            kw["Range"] = byte_range
        data = self.client.get_object(**kw)["Body"].read()
        with self._lock:
            self.bytes_downloaded += len(data)
            self.requests += 1
        return data

    def locate(self, kind: FileKind, source: str, cycle: datetime, fhr: int) -> Located | None:
        """The GRIB2 file (+ parsed idx) for one source / cycle / forecast hour, or None if it is not published."""
        key = kind.key_for(source, cycle, fhr)
        with self._lock:
            if key in self._idx:
                return self._idx[key]
        loc = None
        try:
            entries = parse_idx(self._get(key + ".idx").decode("utf-8", "replace"))
            if entries:
                loc = Located(key, entries)
        except Exception as exc:
            log.debug("no idx for %s (%s)", key, exc)
        with self._lock:
            self._idx[key] = loc
        return loc

    def fetch_message(self, loc: Located, var: str, level: str, info: str | None = None) -> bytes | None:
        """Download exactly one GRIB2 message with an HTTP byte-range request (None if absent from the idx)."""
        e = find_entry(loc.entries, var, level, info)
        if e is None:
            return None
        rng = f"bytes={e.start}-{e.end}" if e.end is not None else f"bytes={e.start}-"
        data = self._get(loc.key, rng)
        if data[:4] != b"GRIB":
            raise ValueError(f"byte range {rng} of {loc.key} did not return a GRIB message")
        return data
