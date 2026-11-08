"""Central configuration: bucket layout, cycles, forecast frames, region, vertical levels."""
from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- data source (NOAA Open Data, anonymous)
BUCKET = "noaa-rrfs-ops-pds"            # operational RRFS v1 / REFS (the older noaa-rrfs-pds holds retro/prototype data only)
CYCLES = (0, 6, 12, 18)                 # RRFS ensemble + REFS products: 4x/day, hourly to F060
MAX_FHR = 60
FRAME_STEP = 3                          # maps are produced every 3 h; accumulations are exact for any 3-h multiple window
MEMBERS = ("m001", "m002", "m003", "m004", "m005")   # the five RRFS ensemble members published on S3
LOOKBACK_CYCLES = 8                     # how many 6-hourly cycles back to look for a complete run


def frames(max_fhr: int = MAX_FHR) -> tuple[int, ...]:
    return tuple(range(FRAME_STEP, max_fhr + 1, FRAME_STEP))


@dataclass(frozen=True)
class FileKind:
    """One family of GRIB2 files on the bucket."""
    key: str

    def key_for(self, source: str, cycle, fhr: int) -> str:
        d, cc = f"{cycle:%Y%m%d}", f"{cycle:%H}"
        if source == "det":                          # deterministic RRFS, full pressure-level file (has DZDT every 25 hPa)
            return f"rrfs.{d}/{cc}/rrfs.t{cc}z.prslev.3km.f{fhr:03d}.conus.grib2"
        if source == "refs":                         # official REFS ensemble mean (CONUS 3 km), or its probability-matched mean (the file with REFC / REFD)
            return f"refs.{d}/{cc}/ensprod/refs.t{cc}z.{'pmmn' if self.key == 'pmmn' else 'mean'}.f{fhr:02d}.conus.grib2"
        if self.key in ("hsfc", "hprs"):             # HRRR, bucket noaa-hrrr-bdp-pds: surface (wrfsfc) and pressure-level (wrfprs) files, hourly
            return f"hrrr.{d}/conus/hrrr.t{cc}z.wrf{self.key[1:]}f{fhr:02d}.grib2"
        suffix = {"2dfld": "2dfldnomads", "prslev": "prslevnomads"}[self.key]
        return f"rrfsens.{d}/{cc}/{source}/rrfs.t{cc}z.{source}.{suffix}.3km.f{fhr:03d}.conus.grib2"


FILE_2D = FileKind("2dfld")
FILE_PRS = FileKind("prslev")
FILE_HSFC = FileKind("hsfc")                                           # HRRR surface file (source "hrrr")
FILE_HPRS = FileKind("hprs")                                           # HRRR pressure-level file
FILE_PMMN = FileKind("pmmn")                                            # REFS probability-matched mean: carries composite (REFC) and 1 km (REFD) reflectivity
LIFT_LEVELS = tuple(range(925, 299, -25))                               # deterministic DZDT levels (hPa) = winter.COBB_GRID

# --------------------------------------------------------------------------- vertical levels fetched from the member prslev files
T_LEVELS = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 600, 500)     # TMP: Kuchera / max-T-aloft / DGZ / warm nose
RH_LEVELS = (925, 900, 850, 800, 750, 700, 600, 500)                    # RH: saturation test inside the DGZ
DGZ_RH_LOW = (1000, 975, 950)           # extra RH levels for the DGZ only (the Cobb method starts at 925): in shallow cold air the DGZ sits in the lowest 100 hPa
COBB_UPPER = (400, 300)                 # extra TMP + RH levels so the Cobb integration reaches NBM's 300 hPa top; kept apart from T_LEVELS
                                        # so Kuchera / max-T-aloft / DGZ / warm nose do not change

# --------------------------------------------------------------------------- region / display grid
# lon_min, lon_max, lat_min, lat_max that bounds the map frame (checked in basemap.py)
DISPLAY_BBOX = (-80.7, -62.2, 39.8, 48.3)
DATA_PAD_DEG = 0.6
NEW_ENGLAND_STATES = {"CT", "ME", "MA", "NH", "RI", "VT"}
OUT_DX_M = 3000.0                       # display raster spacing (RRFS native 3 km)
DISPLAY_TZ = "America/New_York"

# map frame: centred on New England, 880 km tall; width follows from the aspect ratio
FIG_W, MAP_W_IN, MAP_H_IN = 12.0, 11.5, 7.42      # map frame 1150 x 742 px at 100 dpi
MAP_CENTER, HALF_H_M = (-71.4, 44.2), 440e3
MIN_SEP_M, EDGE_M = 40e3, 22e3          # station label spacing / keep-out near the frame edge
