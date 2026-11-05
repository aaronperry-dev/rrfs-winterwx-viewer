"""The product catalogue: every map the dashboard can show.

``kind``:  ``cum``  - accumulated since initialisation (any 3-h multiple window = difference of two stored totals)
           ``inst`` - valid at the frame time
           ``max``  - running maximum / longest run since initialisation (no windows)
``refs``:  True when the official REFS mean file carries the field (else only the members, and the browser-side member mean,
           can show it).
``quant``: storage of the field in the data files: dtype, offset, step  (value = lo + step * q; the top code means "missing").
``thr``:   thresholds offered for probability maps (``dir`` = ``ge`` or ``le``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import colors as C


@dataclass(frozen=True)
class Product:
    key: str
    label: str
    group: str
    unit: str
    kind: str
    style: C.Style
    dtype: str
    lo: float
    step: float
    decimals: int
    refs: bool
    desc: str
    thr: tuple[float, ...] = ()
    thr_dir: str = "ge"
    title: str = ""             # map title (defaults to label)

    @property
    def map_title(self) -> str:
        return self.title or self.label


_P = Product
G_PRECIP, G_SNOW, G_MIX, G_DGZ, G_WIND, G_TEMP = "Precipitation", "Snowfall", "Ice & p-type", "Snow growth (DGZ)", "Wind & blizzard", "Temperature"

PRODUCTS: tuple[Product, ...] = (
    # ------------------------------------------------------------------ precipitation
    _P("qpf", "QPF", G_PRECIP, "in", "cum", C.QPF, "u16", 0.0, 0.005, 2, True,
       "Total liquid-equivalent precipitation (all types).", (0.1, 0.25, 0.5, 1, 2), "ge", "Liquid Precipitation (QPF)"),
    # ------------------------------------------------------------------ snowfall, four ratio methods
    _P("snow_model", "Model snowfall", G_SNOW, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, True,
       "RRFS physical snow accumulation (variable snow-liquid ratio; snow + sleet). The model's own answer, not a post-processed ratio.",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - RRFS model (ASNOW)"),
    _P("snow_10", "10:1 snowfall", G_SNOW, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, True,
       "Snow-type liquid equivalent x 10. Members: hourly QPF x snow flag; REFS mean: its WEASD field.",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - fixed 10:1"),
    _P("snow_kuchera", "Kuchera snowfall", G_SNOW, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, False,
       "Snow-type liquid equivalent x Kuchera ratio (12 + 2/1 x (271.16 K - warmest temperature, surface to 500 hPa)).",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - Kuchera ratio"),
    _P("snow_maxt", "Max-T-aloft snowfall", G_SNOW, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, False,
       "Snow-type liquid equivalent x max-temperature-aloft polynomial ratio (Veals et al. 2025; max T from ~610 m AGL to 500 hPa).",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - max-T-aloft ratio"),
    _P("slr_kuchera", "Kuchera ratio", G_SNOW, ":1", "inst", C.SLR, "u8", 0.0, 0.2, 0, False,
       "Kuchera snow-to-liquid ratio from the warmest temperature between the surface and 500 hPa (clipped 3:1 - 40:1).",
       (10, 12, 15, 20), "ge", "Snow-to-Liquid Ratio - Kuchera"),
    _P("slr_maxt", "Max-T-aloft ratio", G_SNOW, ":1", "inst", C.SLR, "u8", 0.0, 0.2, 0, False,
       "Snow-to-liquid ratio from the max-temperature-aloft polynomial (clipped 3:1 - 40:1).",
       (10, 12, 15, 20), "ge", "Snow-to-Liquid Ratio - max T aloft"),
    # ------------------------------------------------------------------ ice and precipitation type
    _P("frzr", "Freezing rain", G_MIX, "in", "cum", C.ICE, "u16", 0.0, 0.0025, 2, False,
       "Liquid-equivalent freezing-rain accumulation (hourly QPF x freezing-rain flag). Flat-surface liquid, not radial ice.",
       (0.01, 0.1, 0.25), "ge", "Freezing Rain (liquid equivalent)"),
    _P("sleet", "Sleet", G_MIX, "in", "cum", C.SLEET, "u16", 0.0, 0.0025, 2, False,
       "Liquid-equivalent ice-pellet accumulation (hourly QPF x ice-pellet flag).", (0.01, 0.1, 0.25), "ge", "Sleet (liquid equivalent)"),
    _P("psnow", "Snow", G_MIX, "%", "inst", C.PCT_SNOW, "u8", 0.0, 0.4, 0, True,
       "Percent of members with snow as the precipitation type (single member: 0 or 100).", (50,), "ge", "Precipitation Type - Snow"),
    _P("psleet", "Sleet (type)", G_MIX, "%", "inst", C.PCT_SLEET, "u8", 0.0, 0.4, 0, True,
       "Percent of members with ice pellets.", (50,), "ge", "Precipitation Type - Sleet"),
    _P("pfrzr", "Freezing rain (type)", G_MIX, "%", "inst", C.PCT_FRZR, "u8", 0.0, 0.4, 0, True,
       "Percent of members with freezing rain.", (50,), "ge", "Precipitation Type - Freezing Rain"),
    _P("prain", "Rain", G_MIX, "%", "inst", C.PCT_RAIN, "u8", 0.0, 0.4, 0, True,
       "Percent of members with rain.", (50,), "ge", "Precipitation Type - Rain"),
    _P("warmnose", "Warm nose", G_MIX, "°C", "inst", C.WARM_NOSE, "u8", -40.0, 0.25, 1, False,
       "Warmest temperature between 925 and 700 hPa. Above 0 °C over sub-freezing surface air means sleet or freezing rain.",
       (0, 1, 3), "ge", "Elevated Warm Layer (max T 925-700 mb)"),
    # ------------------------------------------------------------------ dendritic growth zone
    _P("dgz_depth", "DGZ depth", G_DGZ, "hPa", "inst", C.DGZ, "u8", 0.0, 1.0, 0, False,
       "Depth of the layer between the -12 and -18 °C isotherms (dendritic growth zone), in hPa.", (25, 50, 100), "ge",
       "Dendritic Growth Zone Depth"),
    _P("dgz_sat", "Saturated DGZ depth", G_DGZ, "hPa", "inst", C.DGZ, "u8", 0.0, 1.0, 0, False,
       "Part of the DGZ saturated with respect to ice (layer-mean RHi >= 100 %): the depth where dendrites can actually grow.",
       (25, 50, 100), "ge", "Ice-Saturated DGZ Depth"),
    _P("dgz_rhi", "DGZ mean RH (ice)", G_DGZ, "%", "inst", C.RHI, "u8", 0.0, 0.6, 0, False,
       "Pressure-weighted mean relative humidity with respect to ice inside the DGZ.", (90, 100), "ge", "Mean RH w.r.t. Ice in the DGZ"),
    _P("uvv", "Peak upward motion", G_DGZ, "m/s", "inst", C.UVV, "u8", 0.0, 0.01, 2, False,
       "Hourly-maximum upward vertical velocity (100-1000 hPa). RRFS does not publish omega on pressure levels, so this column "
       "maximum is the lift proxy: look for it overlapping a saturated DGZ.", (0.1, 0.25, 0.5), "ge", "Peak Upward Vertical Velocity"),
    # ------------------------------------------------------------------ wind and blizzard
    _P("gust", "Wind gust", G_WIND, "mph", "inst", C.wind_style(), "u8", 0.0, 0.6, 0, False,
       "10 m wind gust.", (30, 40, 50, 58), "ge", "10m Wind Gust"),
    _P("wind10", "Wind speed", G_WIND, "mph", "inst", C.wind_style(), "u8", 0.0, 0.6, 0, True,
       "10 m wind speed.", (20, 25, 30, 35), "ge", "10m Wind Speed"),
    _P("vis", "Visibility", G_WIND, "mi", "inst", C.VIS, "u8", 0.0, 0.05, 2, True,
       "Surface visibility (model diagnostic from falling hydrometeors; blowing snow is not modelled). Capped at 12 mi.",
       (0.25, 0.5, 1), "le", "Surface Visibility"),
    _P("blizzard_now", "Blizzard conditions", G_WIND, "%", "inst", C.PCT_RED, "u8", 0.0, 0.4, 0, False,
       "All three NWS criteria at the valid time: wind or gust >= 35 mph, visibility <= 1/4 mi, and snow or sleet falling. "
       "As a member mean, the percent of members meeting them.", (50,), "ge", "Blizzard Conditions (instantaneous)"),
    _P("blizzard_dur", "Blizzard duration", G_WIND, "h", "max", C.BLIZ_HOURS, "u8", 0.0, 0.25, 0, False,
       "Longest run so far of blizzard conditions (3-hourly samples). NWS blizzard criteria need >= 3 h.", (3, 6, 12), "ge",
       "Longest Blizzard-Criteria Run (h)"),
    # ------------------------------------------------------------------ temperature
    _P("t2", "2 m temperature", G_TEMP, "°F", "inst", C.temp_style(), "u8", -50.0, 0.6, 0, True,
       "2 m air temperature.", (32, 20, 10, 0), "le", "2m Temperature"),
    _P("windchill", "Wind chill", G_TEMP, "°F", "inst", C.wind_chill_style(), "u8", -70.0, 0.75, 0, True,
       "NWS wind chill (T <= 50 F and wind > 3 mph), otherwise the air temperature.", (0, -10, -20, -30), "le", "Wind Chill"),
)
BY_KEY = {p.key: p for p in PRODUCTS}
GROUPS = tuple(dict.fromkeys(p.group for p in PRODUCTS))
