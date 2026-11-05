"""Regenerate rtma_viewer/data/stations.csv (New England METAR-reporting airports) from OurAirports.

    python scripts/make_stations.py [path/to/airports.csv]

OurAirports (public domain, https://ourairports.com/data/) supplies coordinates and elevation; the list of
ICAO identifiers below is the curated set of New England airports that issue METARs (ASOS/AWOS), plus the
Mount Washington summit observatory (KMWN), which is not an airport and is added by hand.
priority 0 = always labelled (KMWN); 1 = labelled first when the map is crowded (primary/commercial or regionally
important sites); 2 = filled in where there is room.
"""
import csv, io, sys, urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/airports.csv"
P1 = """KBOS KPVD KBDL KBTV KPWM KBGR KCON KMHT KORH KCAR KHYA KACK KBDR KHVN KGON KHFD KPSM KLEB KEEN KMPV KRUT
KPQI KHUL KAUG KLEW KRKD KBHB KMLT KFVE KBML KHIE KEFK KMVL KBAF KPSF KAQW KFIT KOWD KMVY KFMH KPVC KCQX KBID KUUU
KWST KIZG KGNR KMVM KDDH KVSF KFSO KCEF KLCI KAFN""".split()
P2 = """KBED KBVY KLWM KGHG KPYM KEWB KTAN KORE KGBR KOQU KDXR KOXC KMMK KIJD KSNC KASH KCNH KDAW KSFM KBXM KWVL
KOLD KBST KIWI KRUM KPNN KSLK KDKK""".split()
MANUAL = [  # not in OurAirports
    ("KMWN", "Mount Washington Observatory", "NH", 44.2706, -71.3033, 1917.0, 0),
]
NE = {"US-CT", "US-MA", "US-ME", "US-NH", "US-RI", "US-VT"}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    text = open(src, encoding="utf-8").read() if src else urllib.request.urlopen(URL, timeout=120).read().decode()
    by_id = {}
    for r in csv.DictReader(io.StringIO(text)):
        for key in (r["gps_code"], r["ident"], r["icao_code"]):
            if key:
                by_id.setdefault(key, r)
    rows, missing = [], []
    for prio, ids in ((1, P1), (2, P2)):
        for icao in ids:
            r = by_id.get(icao)
            if not r or r["iso_region"] not in NE or r["type"] == "closed":
                missing.append(icao)
                continue
            rows.append((icao, r["name"], r["iso_region"][-2:], round(float(r["latitude_deg"]), 4),
                         round(float(r["longitude_deg"]), 4), round(float(r["elevation_ft"] or 0) * 0.3048, 0), prio))
    rows += MANUAL
    out = Path(__file__).resolve().parent.parent / "rtma_viewer" / "data" / "stations.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["icao", "name", "state", "lat", "lon", "elev_m", "priority"])
        w.writerows(sorted(rows))
    print(f"wrote {len(rows)} stations to {out}")
    if missing:
        print("not found / not in New England (dropped):", " ".join(missing))


if __name__ == "__main__":
    main()
