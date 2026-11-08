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

from dataclasses import dataclass, replace

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
    has1h: bool = False         # also stored as a separate "1-hour accumulation ending at the valid time" field (key + "_1h")
    step1h: float = 0.0         # quantisation step of that 1-hour field (0 = same as ``step``)
    cat: bool = False           # categorical code field (reflectivity masked by precipitation type): never interpolated, the page decodes it
    pmm: bool = False           # offers PMM / LPMM: set below for the precipitation-amount products (liquid, snow and ice accumulations)

    @property
    def map_title(self) -> str:
        return self.title or self.label


_P = Product
G_PRECIP, G_RADAR, G_SNOW, G_SLR, G_MIX, G_DGZ, G_WIND, G_TEMP = ("Precipitation", "Reflectivity", "Snowfall", "Snow ratio (SLR)", "Ice & p-type",
                                                                  "Snow growth (DGZ)", "Wind & blizzard", "Temperature")

PRODUCTS: tuple[Product, ...] = (
    # ------------------------------------------------------------------ precipitation
    _P("qpf", "QPF", G_PRECIP, "in", "cum", C.QPF, "u16", 0.0, 0.005, 2, True,
       "Liquid-equivalent precipitation, all types. Choose a 1, 3, 6, 12 or 24 hour window or the run total.",
       (0.1, 0.25, 0.5, 1, 2), "ge", "Liquid Precipitation (QPF)", has1h=True),
    # ------------------------------------------------------------------ radar reflectivity (simulated)
    _P("refc", "Composite reflectivity", G_RADAR, "dBZ", "inst", C.REFL, "u8", -20.0, 0.4, 0, True,
       "Simulated composite reflectivity: the strongest echo in the column. Members publish it; the REFS file carries its probability-matched "
       "mean (PMM), the official ensemble picture, which keeps realistic peak values that a plain average would smear out. The member mean "
       "here is taken in linear Z, not in dBZ.",
       (20, 30, 40, 50), "ge", "Composite Reflectivity"),
    _P("refd", "Base reflectivity (1 km)", G_RADAR, "dBZ", "inst", C.REFL, "u8", -20.0, 0.4, 0, True,
       "Simulated reflectivity 1 km above ground, the model's stand-in for a radar's lowest-tilt base reflectivity. The RRFS member files do not "
       "publish it, so only the official REFS probability-matched mean (PMM) is shown, with no individual members, percentiles or probabilities.",
       (), "ge", "Base Reflectivity (1 km AGL)"),
    _P("refl_ptype", "Reflectivity by precip type", G_RADAR, "dBZ", "inst", C.REFL_PTYPE, "u8", 0.0, 1.0, 0, True,
       "Composite reflectivity coloured by the model's precipitation type: green-red rain, blue snow, violet sleet, pink freezing rain (darker = stronger echo). "
       "Rain uses the same colours as the reflectivity maps. Each member uses its own type flags. The member view picks the type most members agree on "
       "(wintry types win ties) and the linear-Z mean echo; the maximum view shows the strongest member and its type. The REFS view pairs the PMM echo with the REFS mean type.",
       (), "ge", "Reflectivity by Precipitation Type", cat=True),
    # ------------------------------------------------------------------ snowfall: model + five ratio methods + their mean
    _P("snow_model", "Model snowfall", G_SNOW, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, True,
       "RRFS physical snow accumulation (variable snow-liquid ratio; snow + sleet). The model's own answer, not a post-processed ratio.",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - RRFS model (ASNOW)", has1h=True, step1h=0.02),
    _P("snow_12", "12:1 snowfall", G_SNOW, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, True,
       "Snow-type liquid equivalent x 12, a ratio closer to New England climatology than 10:1. Members: hourly QPF x snow flag; "
       "REFS mean: its WEASD field.",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - fixed 12:1", has1h=True, step1h=0.02),
    _P("snow_kuchera", "Kuchera snowfall", G_SNOW, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, False,
       "Snow-type liquid equivalent x Kuchera ratio (12 + 2 x (271.16 K - warmest temperature from the surface to 500 hPa) when that "
       "temperature is above 271.16 K, 12 + (271.16 K - T) otherwise).",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - Kuchera ratio", has1h=True, step1h=0.02),
    _P("snow_maxt", "Max-T-aloft snowfall", G_SNOW, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, False,
       "Snow-type liquid equivalent x max-temperature-aloft polynomial ratio (Veals et al. 2025; max T from ~610 m = 2000 ft AGL to "
       "500 hPa; NBM uses 400 hPa as the top, 500 hPa is the highest level fetched here). Experimental: fitted to Alaska data.",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - max-T-aloft ratio", has1h=True, step1h=0.02),
    _P("snow_thick", "Thickness snowfall", G_SNOW, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, False,
       "Snow-type liquid equivalent x the CMC thickness ratio: 10:1, reduced by the Murphy (Ontario) relation "
       "10 x (30.6 - 0.1057 x Z1000-700 [dam]) wherever the 850-700 hPa thickness exceeds 1540 m (marginal, wet snow).",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - 850-700 mb thickness ratio", has1h=True, step1h=0.02),
    _P("snow_mean", "Multi-method mean", G_SNOW, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, False,
       "Average of the five snowfall estimates above (model, 12:1, Kuchera, max-T-aloft, CMC thickness): the middle of the method spread. "
       "The literature methods below are not part of it.",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - mean of methods", has1h=True, step1h=0.02),
    # ------------------------------------------------------------------ snowfall: literature methods (not part of the mean)
    _P("snow_cobb", "Cobb snowfall", G_SNOW, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, False,
       "Snow-type liquid equivalent x the Cobb & Waldstreicher cloud-layer ratio as NBM applies it: layer ratios from the 2011 temperature "
       "curve (peak 26:1 near -16 C) averaged over 925-300 hPa on a 25 hPa grid, weighted by (RH/80)^2. NBM's lift weight is set to its "
       "minimum everywhere because the RRFS member files carry no pressure-level vertical velocity, so this is a humidity-weighted "
       "layer mean, not the full method. Capped at 30:1.",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - Cobb layer ratio (no lift term)", has1h=True, step1h=0.02),
    _P("snow_cobblift", "Cobb snowfall (with lift)", G_SNOW, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, False,
       "As the Cobb snowfall above, with the layers also weighted by sqrt(VV), the upward vertical velocity. The RRFS member files have "
       "none, so VV comes from the deterministic RRFS run's DZDT (every 25 hPa, 925-300 hPa; omega = -rho g w, in microbar/s, at least 1) "
       "while temperature and humidity are the member's own. The deterministic run's lift is not the member's lift: where the two "
       "disagree on where the ascent is, this ratio is too. Empty where the deterministic file is missing.",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - Cobb layer ratio (deterministic lift)", has1h=True, step1h=0.02),
    _P("snow_nbmthick", "NBM thickness snowfall", G_SNOW, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, False,
       "Snow-type liquid equivalent x the National Blend of Models thickness ratio, a straight line in the 850-700 hPa thickness: "
       "-0.16559 x Z[m] + 263.35 (8:1 at 1540 m, 15:1 at 1500 m). One of the methods NBM blends operationally.",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - NBM 850-700 mb thickness ratio", has1h=True, step1h=0.02),
    _P("snow_byun", "Byun snowfall", G_SNOW, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, False,
       "Snow-type liquid equivalent x the Byun et al. (2008) logistic ratio of 2 m temperature, with coefficients chosen by 3-hour "
       "precipitation. Fitted to Korean data and capped near 19:1: it is a wet-snow-sensitive lower estimate, and runs low in cold powder. "
       "Uses the 2 m temperature at the valid time, not a 3-hour mean.",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - Byun et al. temperature / precipitation-rate ratio", has1h=True, step1h=0.02),
    _P("snow_clim", "Climatological-ratio snowfall", G_SNOW, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, False,
       "Snow-type liquid equivalent x the regional mean ratio from Baxter et al. (2005): 10.6-13.4 across the NWS offices around New England, "
       "interpolated between them. A reference for how much the weather-dependent methods differ from a typical winter, not a forecast.",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - regional climatological ratio (Baxter et al.)", has1h=True, step1h=0.02),
    _P("snow_spread", "Method spread", G_SNOW, "in", "inst", C.SPREAD, "u8", 0.0, 0.1, 1, False,
       "Run-total snowfall: highest minus lowest of the model, 12:1, Kuchera, max-T-aloft, CMC thickness, Cobb and NBM thickness estimates. "
       "Large where the choice of ratio method matters most. Valid at the frame time (run total), so no accumulation windows.",
       (1, 2, 4, 8), "ge", "Snowfall - spread between ratio methods (run total)"),
    # ------------------------------------------------------------------ snow-to-liquid ratios
    _P("slr_12", "12:1 ratio", G_SLR, ":1", "inst", C.SLR, "u8", 0.0, 0.2, 0, False,
       "The fixed 12:1 ratio (a constant field, shown for comparison with the others).", (10, 12, 15, 20), "ge", "Snow-to-Liquid Ratio - fixed 12:1"),
    _P("slr_kuchera", "Kuchera ratio", G_SLR, ":1", "inst", C.SLR, "u8", 0.0, 0.2, 0, False,
       "Kuchera snow-to-liquid ratio from the warmest temperature between the surface and 500 hPa (clipped 3:1 - 40:1).",
       (10, 12, 15, 20), "ge", "Snow-to-Liquid Ratio - Kuchera"),
    _P("slr_maxt", "Max-T-aloft ratio", G_SLR, ":1", "inst", C.SLR, "u8", 0.0, 0.2, 0, False,
       "Snow-to-liquid ratio from the max-temperature-aloft polynomial (clipped 3:1 - 40:1).",
       (10, 12, 15, 20), "ge", "Snow-to-Liquid Ratio - max T aloft"),
    _P("slr_thick", "Thickness ratio", G_SLR, ":1", "inst", C.SLR, "u8", 0.0, 0.2, 0, False,
       "CMC thickness ratio: 10:1 unless the 850-700 hPa thickness is above 1540 m, then 10 x (30.6 - 0.1057 x Z1000-700 [dam]) (min 3:1).",
       (6, 8, 10), "le", "Snow-to-Liquid Ratio - 850-700 mb thickness"),
    _P("slr_cobb", "Cobb ratio", G_SLR, ":1", "inst", C.SLR, "u8", 0.0, 0.2, 0, False,
       "Cobb & Waldstreicher cloud-layer ratio (2011 curve) averaged over 925-300 hPa with (RH/80)^2 weights and no lift term, capped at "
       "30:1. Where the column is dry the weights are small and the ratio is set by whatever moist layers there are.",
       (10, 12, 15, 20), "ge", "Snow-to-Liquid Ratio - Cobb layer (no lift term)"),
    _P("slr_cobblift", "Cobb ratio (with lift)", G_SLR, ":1", "inst", C.SLR, "u8", 0.0, 0.2, 0, False,
       "Cobb layer ratio as above, with each layer also weighted by sqrt(VV) from the deterministic RRFS run's DZDT (member temperature "
       "and humidity). Capped at 30:1.",
       (10, 12, 15, 20), "ge", "Snow-to-Liquid Ratio - Cobb layer (deterministic lift)"),
    _P("slr_nbmthick", "NBM thickness ratio", G_SLR, ":1", "inst", C.SLR, "u8", 0.0, 0.2, 0, False,
       "NBM linear 850-700 hPa thickness ratio, -0.16559 x Z[m] + 263.35 (clipped 3:1 - 40:1).",
       (10, 12, 15, 20), "ge", "Snow-to-Liquid Ratio - NBM 850-700 mb thickness"),
    _P("slr_byun", "Byun ratio", G_SLR, ":1", "inst", C.SLR, "u8", 0.0, 0.2, 0, False,
       "Byun et al. (2008) logistic ratio of 2 m temperature and 3-hour precipitation. Near 0 above about +1 C, half of its limit "
       "(11.9-18.8) at 0 C, and never above that limit.", (6, 10, 12, 15), "ge", "Snow-to-Liquid Ratio - Byun et al."),
    _P("slr_clim", "Climatological ratio", G_SLR, ":1", "inst", C.SLR, "u8", 0.0, 0.2, 0, False,
       "Regional mean ratio from Baxter et al. (2005), 1971-2000 observations at the NWS offices around New England, interpolated. "
       "Constant in time; shown as the reference for the others.", (11, 12, 13), "ge", "Snow-to-Liquid Ratio - regional climatology"),
    _P("slr_model", "Model-implied ratio", G_SLR, ":1", "inst", C.SLR, "u8", 0.0, 0.2, 0, False,
       "Run-total ASNOW divided by the run-total snow-type liquid (shown where at least 0.02 in of snow liquid has fallen). "
       "The two come from different model diagnostics, so treat it as a diagnostic, not a measurement.",
       (10, 12, 15, 20), "ge", "Model-Implied Snow-to-Liquid Ratio"),
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
# Probability-matched means only make sense for intermittent, skewed precipitation amounts, where averaging members smears out the maxima they
# restore.  Ratios, temperatures, winds, p-type probabilities and the like keep the plain mean.  (The 1-hour variants below inherit the flag.)
PRODUCTS = tuple(replace(p, pmm=(p.kind == "cum" and p.unit == "in")) for p in PRODUCTS)
# Stored series: every product, plus the 1-hour variant of those that have one (the browser derives 3/6/12/24-h windows from totals).
ONE_HOUR = tuple(replace(p, key=p.key + "_1h", label=p.label + " (1 h)", has1h=False, step1h=0.0, kind="inst",
                         step=p.step1h or p.step) for p in PRODUCTS if p.has1h)
STORE = PRODUCTS + ONE_HOUR
BY_KEY = {p.key: p for p in STORE}
GROUPS = tuple(dict.fromkeys(p.group for p in PRODUCTS))
