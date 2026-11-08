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
    sam: bool = False           # offers SAM / SAM-LPM (experimental spatially aligned means): liquid and snowfall amounts only
    pmm: bool = False           # offers PMM / LPMM: set below for the precipitation-amount products (liquid, snow and ice accumulations)
    single: str = ""            # a product that has ONE source that is not the REFS mean or an RRFS member ("HRRR"): it is stored in the single-source slot and shown under this name

    @property
    def map_title(self) -> str:
        return self.title or self.label


_P = Product
G_PRECIP, G_RADAR, G_SNOW, G_SLR, G_MIX, G_DGZ, G_WIND, G_TEMP = ("Precipitation", "Reflectivity", "Snowfall", "Snow ratio (SLR)", "Ice & p-type",
                                                                  "Snow growth (DGZ)", "Wind & blizzard", "Temperature")
# groups added after the originals (the original groups keep their order and position)
G_LEVEL, G_LOW, G_FORCE, G_PROF, G_TIME, G_IMPACT, G_HRRR = ("Snow level & wet bulb", "Low-level cold air", "Forcing & banding", "P-type from profiles",
                                                          "Timing", "Joint impacts", "HRRR (hi-res, to 48 h)")

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
       "RRFS physical snow accumulation (variable snow-liquid ratio; snow + sleet). The model's own answer, not a post-processed ratio. "
       "This is also the closest thing RRFS publishes to snow depth change: the public files carry no snow depth (SNOD), and ASNOW counts "
       "the snow that reaches the ground without taking off later settling or melt.",
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
       "Warmest temperature between 925 and 700 hPa. Above 0 °C over sub-freezing surface air means sleet or freezing rain. A warm layer that nears 0 °C "
       "also keeps snow ratios low (WPC snow forecasting guide), and one that warms the DGZ out of its -12 to -18 °C window ends dendrite growth: read it with the DGZ maps.",
       (0, 1, 3), "ge", "Elevated Warm Layer (max T 925-700 mb)"),
    # ------------------------------------------------------------------ dendritic growth zone
    # Dendrites need three things together: air between -12 and -18 C (where vapour excess over ice and growth rates peak, Bergeron-Findeisen), air close to ice
    # saturation, and ascent to keep supplying vapour.  The products go in that order, then the humidity they are built on.  Sources: Cobb & Waldstreicher
    # (2005); NWS MDL / NBM Cobb method (2011 ratio curve, >90 % RH w.r.t. ice and rising motion); Roebber et al. (2003); Baxter et al. (2005);
    # Libbrecht (2012) after Nakaya (1954); Korolev & Mazin (2003).  Full list in the README.
    _P("dgz_depth", "DGZ depth", G_DGZ, "kft", "inst", C.DGZ, "u8", 0.0, 0.1, 1, False,
       "Thickness of the layer between the -12 and -18 °C isotherms, in thousands of feet of height, whether or not it is moist. Dendrites and sectored "
       "plates grow fastest here, so a deeper layer gives crystals more time to grow as they fall (the DGZ-depth maps of NWS and College of DuPage). "
       "Under about 1 kft is a thin sliver, 2-4 kft a solid layer, 5 kft or more deep. Blank where no air in the column is that cold or the layer is below ground. "
       "Temperature only: read it with the saturated depth and the ascent.",
       (1, 2, 4, 6), "ge", "Dendritic Growth Zone Depth (kft)"),
    _P("dgz_sat", "Moist DGZ depth", G_DGZ, "kft", "inst", C.DGZ, "u8", 0.0, 0.1, 1, False,
       "The part of the growth zone that is also moist enough, in kft: layer-mean relative humidity with respect to ice of 90 % or more, the criterion the NWS "
       "MDL Cobb method uses (about 75-78 % RH over water at these temperatures). This is the depth in which dendrites can really grow, so it is the one to read "
       "for fluffy snow. A deep zone with little moist depth means dry air in the zone: sparse flakes, or plates and needles. Uses 925-500 hPa humidity, so "
       "zone below 925 hPa counts as dry.",
       (1, 2, 4, 6), "ge", "Moist (RHi >= 90%) DGZ Depth (kft)"),
    _P("dgz_omega", "DGZ ascent", G_DGZ, "µbar/s", "inst", C.OMEGA, "u8", 0.0, 0.2, 0, False,
       "Mean upward motion inside the -12 to -18 °C layer, as a pressure vertical velocity in microbar/s, the omega unit of BUFKIT and the original Cobb & "
       "Waldstreicher work. Converted from the geometric vertical velocity DZDT with omega = -rho g w (ideal gas, hydrostatic). The RRFS members publish no "
       "vertical velocity on pressure levels, so DZDT is the deterministic RRFS run's, on this member's own temperatures: where the two disagree on where "
       "the ascent is, this map does too. Strong ascent inside a moist DGZ is what makes heavy snow bursts and bands. Empty where the deterministic file is missing.",
       (5, 10, 20), "ge", "Ascent in the DGZ (microbar/s)"),
    _P("dgz_lift", "Moist DGZ with lift", G_DGZ, "kft", "inst", C.DGZ, "u8", 0.0, 0.1, 1, False,
       "Moist growth-zone depth (kft) only where that layer also has ascent of 0.1 m/s or more (about 7 microbar/s near 700 hPa; Korolev & Mazin (2003) need "
       "roughly 10 cm/s to keep air ice-supersaturated for dendrite growth). All three ingredients are present where it is coloured: the setup for the heaviest, "
       "fluffiest snow bands. Ascent is the mean inside the DGZ from the deterministic run's DZDT; where that file is missing, the hourly column-maximum "
       "upward motion stands in, which only shows that ascent exists somewhere above the point.",
       (1, 2, 4), "ge", "Moist DGZ Depth With Ascent (kft)"),
    _P("dgz_rhi", "DGZ mean RH (ice)", G_DGZ, "%", "inst", C.RHI, "u8", 0.0, 0.6, 0, False,
       "Pressure-weighted mean relative humidity with respect to ice inside the growth zone. 100 % is ice saturation; the 90 % line is the NWS moisture "
       "criterion. Blues are moist (dendrites grow), tans are dry (growth stalls, flakes sublimate). Humidity is the secondary control on snow density; "
       "temperature matters more (Roebber et al. 2003).", (90, 100), "ge", "Mean RH w.r.t. Ice in the DGZ"),
    _P("uvv", "Peak upward motion", G_DGZ, "m/s", "inst", C.UVV, "u8", 0.0, 0.01, 2, False,
       "Hourly-maximum upward vertical velocity (100-1000 hPa), published by every member. It is the column maximum, not the ascent inside the DGZ, so use "
       "it where the deterministic DZDT (DGZ ascent) is missing.", (0.1, 0.25, 0.5), "ge", "Peak Upward Vertical Velocity"),
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
# ============================================================================== added products (group A + HRRR); the products above are unchanged
_ADDED: tuple[Product, ...] = (
    # ------------------------------------------------------------------ snow level and wet bulb
    _P("tw2", "2 m wet-bulb temperature", G_LEVEL, "°F", "inst", C.temp_style(), "u8", -50.0, 0.6, 0, False,
       "Wet-bulb temperature of the air at 2 m (from the model's 2 m temperature, humidity and surface pressure). Falling snow cools the air it falls "
       "through toward the wet bulb, so Tw, not T, says whether it keeps its form: snow survives at the surface up to about Tw 33-34 °F, and the 32 °F "
       "line (green on this ramp) is the usual rain-snow line when the air is dry. Compare it with the 2 m temperature: the gap is the evaporative cooling.",
       (34, 32, 30), "le", "2m Wet-Bulb Temperature"),
    _P("snowlvl", "Snow level (0 °C wet bulb)", G_LEVEL, "kft", "inst", C.HEIGHT_KFT, "u8", 0.0, 0.05, 1, False,
       "Height above sea level of the 0 °C wet-bulb level, the first one found going up from the ground: snow falls above it and melts toward rain below it. "
       "Where the wet bulb is already at or below 0 °C at the ground it is the ground itself (snow to the surface, so it follows the terrain). Blank where the "
       "whole column to 500 hPa is above 0 °C wet bulb. Profile on 11 pressure levels with heights from the hypsometric equation: a 25-50 hPa grid, "
       "so thin layers (a shallow warm nose) are not resolved. A rain-snow transition often sits 1000-1500 ft lower than this level, where the melting finishes.",
       (1, 2, 3), "le", "Snow Level (0 °C wet-bulb height, MSL)"),
    _P("snowlvl_agl", "Snow level above ground", G_LEVEL, "ft", "inst", C.AGL_FT, "u8", 0.0, 40.0, 0, False,
       "How far above the ground the 0 °C wet-bulb level is, in feet. Blank = zero: snow (if any is falling) reaches the ground, the wet bulb is at or below 0 °C there. "
       "Yellow: the melting layer is shallow, snow arrives as wet snow or mixes. Red: a deep warm layer, the precipitation is rain at the ground. Capped at 10,000 ft, "
       "which is also what a column that is above 0 °C wet bulb everywhere shows. Terrain-following, so on a mountain the same air gives a smaller number than in the valley.",
       (1000, 2000), "ge", "Snow Level Above the Ground (0 °C wet bulb)"),
    _P("frzlvl", "Freezing level (0 °C)", G_LEVEL, "kft", "inst", C.HEIGHT_KFT, "u8", 0.0, 0.05, 1, False,
       "Height above sea level of the 0 °C air-temperature level, the first one going up from the ground (the ground itself where it is at or below freezing). "
       "The dry-bulb counterpart of the snow level, which sits lower in dry air (compare the two).",
       (1, 2, 3), "le", "Freezing Level (0 °C, MSL)"),
    # ------------------------------------------------------------------ low-level cold air
    _P("mslp", "Mean sea-level pressure", G_LOW, "hPa", "inst", C.MSLP, "u16", 940.0, 0.05, 1, False,
       "Pressure reduced to sea level (the model's MSLET). In New England winter events a ridge of high pressure to the north (a 1030 hPa+ high over Quebec or the "
       "Maritimes) with isobars bending along the mountains marks cold-air damming; a coastal low is a sea-level pressure trough just off the coast.",
       (1000, 1010, 1020, 1030), "ge", "Mean Sea-Level Pressure"),
    _P("thk_1000_850", "1000-850 hPa thickness", G_LOW, "dam", "inst", C.THK_1000_850, "u8", 100.0, 0.25, 1, False,
       "Thickness between 1000 and 850 hPa in decameters (heights from the model, else the hypsometric equation). Below about 130 dam the lowest 1.5 km is cold "
       "enough for snow; above about 132 dam it is rain or mixed (the usual low-level rain-snow rule of thumb for the Northeast). The 850-700 hPa layer is the next map.",
       (126, 130, 134), "le", "1000-850 mb Thickness (dam)"),
    _P("thk_850_700", "850-700 hPa thickness", G_LOW, "dam", "inst", C.THK_850_700, "u8", 135.0, 0.15, 1, False,
       "Thickness between 850 and 700 hPa in decameters. About 154 dam (1540 m) is the traditional marginal line for snow aloft; the snow-ratio maps use the same "
       "layer. Read it with the 1000-850 thickness: a cold 1000-850 layer under a warm 850-700 layer is a warm nose (sleet, freezing rain).",
       (151, 154, 157), "le", "850-700 mb Thickness (dam)"),
    _P("t925", "925 hPa temperature", G_LOW, "°C", "inst", C.temp_c_style(), "u8", -40.0, 0.3, 1, False,
       "Temperature at 925 hPa (about 2500 ft). Where the terrain is higher than that level the map is left blank (the model's value there would be a below-ground extrapolation). "
       "925 hPa 0 °C is the classic low-level rain-snow guide for the coast; with 850 hPa it brackets the warm layer that decides snow versus sleet or freezing rain.",
       (0, -2, -5), "le", "925 mb Temperature"),
    _P("t850", "850 hPa temperature", G_LOW, "°C", "inst", C.temp_c_style(), "u8", -40.0, 0.3, 1, False,
       "Temperature at 850 hPa (about 5000 ft). The -2 to 0 °C isotherms mark where warm advection turns snow to mixed precipitation, the forecaster's first look in a coastal storm.",
       (0, -2, -5), "le", "850 mb Temperature"),
    _P("w925", "925 hPa wind speed", G_LOW, "mph", "inst", C.wind_style(), "u8", 0.0, 0.8, 0, False,
       "Wind speed at 925 hPa. A strong onshore low-level jet (50 mph or more from the east or southeast) pumps warm ocean air and moisture inland, the setup for a coastal changeover; "
       "it also shows where gusts at the surface can mix down.", (30, 40, 50, 60), "ge", "925 mb Wind Speed"),
    _P("w850", "850 hPa wind speed", G_LOW, "mph", "inst", C.wind_style(), "u8", 0.0, 0.8, 0, False,
       "Wind speed at 850 hPa (about 5000 ft), the level of the low-level jet in most New England storms.", (30, 40, 50, 60), "ge", "850 mb Wind Speed"),
    _P("inversion", "Low-level inversion", G_LOW, "°C", "inst", C.INVERSION, "u8", 0.0, 0.1, 1, False,
       "Largest amount, in °C, by which the air between the ground and 850 hPa is warmer than the 2 m air. 0 = no inversion (blank). A deep, strong inversion with the surface "
       "below freezing is cold air trapped under warm air: cold-air damming and coastal fronts, where freezing rain and sleet persist after the warm air has taken over aloft.",
       (2, 4, 8), "ge", "Low-Level Inversion Strength (sfc to 850 mb)"),
    # ------------------------------------------------------------------ forcing and banding
    _P("fronto700", "700 hPa frontogenesis", G_FORCE, "K/100km/3h", "inst", C.FRONTO, "u8", -10.0, 0.3, 1, False,
       "Petterssen frontogenesis of the 700 hPa potential temperature, in K per 100 km per 3 h: how fast the temperature contrast is sharpening. Positive and large "
       "(10 or more) marks a front-like zone of forced ascent on its warm side, the engine of narrow heavy snow bands; values of 15-30 are strong. Computed from the "
       "model's 3 km fields smoothed over about 12 km (3 km derivatives are noise), so features under about 30 km wide are lost.",
       (5, 10, 20), "ge", "700 mb Frontogenesis (K/100 km/3 h)"),
    _P("band_pot", "Banding potential", G_FORCE, "K/100km/3h", "inst", C.BAND, "u8", 0.0, 0.15, 1, False,
       "A composite, not a model field: the 700 hPa frontogenesis where three things are true together (frontogenesis at least 2, 700 hPa relative humidity at least 80 %, "
       "and a moist DGZ at least 1 kft deep), 0 elsewhere. Heavy bands need forcing, saturation and snow-growth temperatures in the same place; this map shows only where all three are.",
       (5, 10, 20), "ge", "Banding Potential (frontogenesis where saturated, in the DGZ)"),
    _P("csi", "Slantwise instability", G_FORCE, "PVU", "inst", C.CSI, "u8", 0.0, 0.01, 2, False,
       "Negative saturated moist potential vorticity (MPV*) of the 700-500 hPa layer, in PVU (1e-6 K m2 kg-1 s-1), shown where the layer is also nearly saturated (relative "
       "humidity of at least 80 % at 700 and 600 hPa); blank where it is drier. Zero = stable. Colours mark conditional symmetric instability (MPV* below 0): ascent along sloping "
       "surfaces that releases energy, which supports banding, often with a bit of thundersnow. Layer differences of a 3 km model: treat as a flag, not a magnitude.",
       (0.05, 0.25), "ge", "Slantwise Instability (-MPV*, 700-500 mb)"),
    # ------------------------------------------------------------------ precipitation type from the model profile
    _P("bsnow", "Snow (profile)", G_PROF, "%", "inst", C.PCT_SNOW, "u8", 0.0, 0.4, 0, False,
       "Snow as diagnosed by the Bourgouin energy method from the member's temperature profile (the method NCEP's UPP post-processor uses), percent of members. "
       "It is a second opinion: the model's own precipitation-type flags (Ice & p-type group) come from the microphysics.", (50,), "ge", "Precipitation Type from the Profile - Snow"),
    _P("brain", "Rain (profile)", G_PROF, "%", "inst", C.PCT_RAIN, "u8", 0.0, 0.4, 0, False,
       "Rain by the Bourgouin energy method, percent of members.", (50,), "ge", "Precipitation Type from the Profile - Rain"),
    _P("bsleet", "Sleet (profile)", G_PROF, "%", "inst", C.PCT_SLEET, "u8", 0.0, 0.4, 0, False,
       "Ice pellets by the Bourgouin energy method (a deep, cold layer under the warm nose, enough to refreeze), percent of members.", (50,), "ge",
       "Precipitation Type from the Profile - Sleet"),
    _P("bfrzr", "Freezing rain (profile)", G_PROF, "%", "inst", C.PCT_FRZR, "u8", 0.0, 0.4, 0, False,
       "Freezing rain by the Bourgouin energy method (warm nose over a not-deep cold layer, with the ground below freezing), percent of members.", (50,), "ge",
       "Precipitation Type from the Profile - Freezing Rain"),
    _P("bagree", "Profile vs model p-type", G_PROF, "%", "inst", C.PCT_SNOW, "u8", 0.0, 0.4, 0, False,
       "Agreement between the profile type and the model's own type flags where precipitation is falling: 100 % = same type, 0 % = different. For each member the overlap "
       "of the two type distributions (the profile type can be split, e.g. 50/50 rain-snow in the transition band), averaged over members. Low values mark where the type "
       "forecast is fragile: look at the sounding.", (50, 80), "le", "Agreement: Bourgouin Profile Type vs Model Type"),
    # ------------------------------------------------------------------ timing (so far, each frame; the last frame is the whole run)
    _P("t_start", "Snow start", G_TIME, "h", "max", C.TIME_H, "u8", 0.0, 0.25, 0, False,
       "Forecast hour at which the multi-method snowfall total first reached 0.1 in (3-hourly resolution: the end of the 3-h block in which it did). Blank until snow "
       "has started at that point. Each frame shows what is known up to that frame, so the last frame is the answer for the whole run. The mean and percentiles use only "
       "the members in which snow starts at that point.", (), "le", "Snow Start (forecast hour)"),
    _P("t_end", "Latest significant snow", G_TIME, "h", "max", C.TIME_H, "u8", 0.0, 0.25, 0, False,
       "Forecast hour of the latest 3-h block so far with at least 0.1 in of snow. At the last frame, it is when the snow ends (or the run does: a value at the last "
       "forecast hour means it is still going). 3-hourly resolution.", (), "le", "End of Significant Snow (forecast hour)"),
    _P("t_change", "Changeover to rain/mix", G_TIME, "h", "max", C.TIME_H, "u8", 0.0, 0.25, 0, False,
       "Forecast hour of the first frame at which precipitation that had been snow (at least half of it by the model's own type flags) is no longer snow: rain, sleet or "
       "freezing rain. Blank where snow never changes over (or never fell). 3-hourly resolution, so the changeover is dated to within 3 h.", (), "le",
       "Changeover Time, Snow to Rain or Mix (forecast hour)"),
    _P("t_rate05", "Hours at 0.5 in/h or more", G_TIME, "h", "max", C.IMPACT_H, "u8", 0.0, 0.25, 0, False,
       "Hours so far with snowfall at or above 0.5 in per hour, counted in 3-hour blocks (a block counts when its 3-hour multi-method snowfall is at least 1.5 in). "
       "A rate that stays that high for several hours is what produces a heavy storm total.", (3, 6, 9), "ge", "Hours of Snowfall at 0.5 in/h or More"),
    _P("t_rate10", "Hours at 1 in/h or more", G_TIME, "h", "max", C.IMPACT_H, "u8", 0.0, 0.25, 0, False,
       "Hours so far with snowfall at or above 1 in per hour (3-hour blocks with at least 3 in of multi-method snowfall). Rates this high are rarer and cripple travel.",
       (3, 6), "ge", "Hours of Snowfall at 1 in/h or More"),
    # ------------------------------------------------------------------ joint impacts, evaluated member by member
    _P("imp_wetwind", "Heavy wet snow with wind", G_IMPACT, "h", "max", C.IMPACT_H, "u8", 0.0, 0.25, 0, False,
       "Hours so far (3-hour blocks) in which three things happen together in the same member: at least 1 in of multi-method snow in the block, a Kuchera snow-to-liquid ratio "
       "below 9:1 (heavy, wet snow: the kind that loads trees and power lines) and a gust of 35 mph or more at the valid time. Because the test is applied member by "
       "member, the probability map is the chance that all three come together, not three separate chances multiplied.", (3, 6), "ge", "Heavy Wet Snow + Gusts 35 mph (hours)"),
    _P("imp_ice_wind", "Ice with wind", G_IMPACT, "h", "max", C.IMPACT_H, "u8", 0.0, 0.25, 0, False,
       "Hours so far (3-hour blocks) with at least 0.02 in of freezing rain in the block and a gust of 30 mph or more at the valid time, in the same member: glaze on lines and "
       "trees that is also being blown around. Computed member by member, so the probability map is the chance that both happen together.", (3, 6), "ge",
       "Freezing Rain + Gusts 30 mph (hours)"),
    # ------------------------------------------------------------------ HRRR (single source: shown under the name HRRR; its own regrid; to 48 h, hourly)
    _P("hr_snow", "HRRR snowfall", G_HRRR, "in", "cum", C.SNOW, "u16", 0.0, 0.05, 1, False,
       "HRRR physical snow accumulation (ASNOW: variable snow-liquid ratio). A single deterministic run, hourly; the forecast stops at F048 (cycles 00, 06, 12, 18 UTC).",
       (1, 3, 6, 12, 18, 24), "ge", "Snowfall - HRRR (ASNOW)", has1h=True, step1h=0.02, single="HRRR"),
    _P("hr_qpf", "HRRR QPF", G_HRRR, "in", "cum", C.QPF, "u16", 0.0, 0.005, 2, False,
       "HRRR liquid-equivalent precipitation, all types.", (0.1, 0.25, 0.5, 1, 2), "ge", "Liquid Precipitation - HRRR", has1h=True, single="HRRR"),
    _P("hr_frzr", "HRRR freezing rain", G_HRRR, "in", "cum", C.ICE, "u16", 0.0, 0.0025, 2, False,
       "HRRR accumulated freezing rain (FRZR), liquid equivalent on a flat surface (not radial ice).", (0.01, 0.1, 0.25), "ge", "Freezing Rain - HRRR (liquid equivalent)", single="HRRR"),
    _P("hr_sleet", "HRRR frozen rain / sleet", G_HRRR, "in", "cum", C.SLEET, "u16", 0.0, 0.0025, 2, False,
       "HRRR accumulated frozen rain (FROZR: ice pellets and refrozen drops), liquid equivalent.", (0.01, 0.1, 0.25), "ge", "Sleet - HRRR (FROZR, liquid equivalent)", single="HRRR"),
    _P("hr_snod", "HRRR snow depth", G_HRRR, "in", "inst", C.SNOW, "u16", 0.0, 0.05, 1, False,
       "HRRR snow depth on the ground (SNOD) at the valid time, in inches: what is lying there, including snow on the ground before the run and net of compaction and melt. "
       "RRFS publishes no snow depth.", (1, 6, 12, 24), "ge", "Snow Depth - HRRR", single="HRRR"),
    _P("hr_snod_chg", "HRRR snow depth change", G_HRRR, "in", "inst", C.SNOD_CHG, "u16", -30.0, 0.025, 1, False,
       "Change in HRRR snow depth since the start of the run (SNOD now minus SNOD at F000). Unlike an accumulated-snowfall map it takes settling and melting off, so it "
       "is the closest thing to what a ruler would read; browns mean the depth went down.", (1, 3, 6, 12), "ge", "Snow Depth Change since Run Start - HRRR", single="HRRR"),
    _P("hr_slr", "HRRR model snow ratio", G_HRRR, ":1", "inst", C.SLR, "u8", 0.0, 0.2, 0, False,
       "Run-total ASNOW (snow depth) divided by the run-total liquid equivalent of snow (WEASD), shown where at least 0.5 mm of snow liquid has fallen: the ratio HRRR itself used.",
       (10, 12, 15, 20), "ge", "Snow-to-Liquid Ratio - HRRR (model)", single="HRRR"),
    _P("hr_refc", "HRRR composite reflectivity", G_HRRR, "dBZ", "inst", C.REFL, "u8", -20.0, 0.4, 0, False,
       "HRRR simulated composite reflectivity (REFC).", (20, 30, 40, 50), "ge", "Composite Reflectivity - HRRR", single="HRRR"),
    _P("hr_ptype", "HRRR reflectivity by precip type", G_HRRR, "dBZ", "inst", C.REFL_PTYPE, "u8", 0.0, 1.0, 0, False,
       "HRRR composite reflectivity coloured by HRRR's own precipitation-type flags: green-red rain, blue snow, violet sleet, pink freezing rain.", (), "ge",
       "Reflectivity by Precipitation Type - HRRR", cat=True, single="HRRR"),
    _P("hr_cpofp", "HRRR percent frozen", G_HRRR, "%", "inst", C.PCT_SNOW, "u8", 0.0, 0.4, 0, False,
       "HRRR's percent of frozen precipitation (CPOFP): 100 = all snow or ice, 0 = all liquid; the transition band (20-80) is where rain-snow lines run.", (50,), "ge",
       "Percent of Precipitation Frozen - HRRR", single="HRRR"),
    _P("hr_t2", "HRRR 2 m temperature", G_HRRR, "°F", "inst", C.temp_style(), "u8", -50.0, 0.6, 0, False, "HRRR 2 m air temperature.", (32, 20, 10, 0), "le",
       "2m Temperature - HRRR", single="HRRR"),
    _P("hr_tw2", "HRRR 2 m wet-bulb", G_HRRR, "°F", "inst", C.temp_style(), "u8", -50.0, 0.6, 0, False,
       "2 m wet-bulb temperature from HRRR's 2 m temperature, dew point and surface pressure.", (34, 32, 30), "le", "2m Wet-Bulb Temperature - HRRR", single="HRRR"),
    _P("hr_gust", "HRRR wind gust", G_HRRR, "mph", "inst", C.wind_style(), "u8", 0.0, 0.6, 0, False, "HRRR 10 m wind gust.", (30, 40, 50, 58), "ge",
       "10m Wind Gust - HRRR", single="HRRR"),
    _P("hr_vis", "HRRR visibility", G_HRRR, "mi", "inst", C.VIS, "u8", 0.0, 0.05, 2, False, "HRRR surface visibility (capped at 12 mi).", (0.25, 0.5, 1), "le",
       "Surface Visibility - HRRR", single="HRRR"),
    _P("hr_mslp", "HRRR sea-level pressure", G_HRRR, "hPa", "inst", C.MSLP, "u16", 940.0, 0.05, 1, False,
       "HRRR mean sea-level pressure (MSLMA, the MAPS system reduction).", (1000, 1010, 1020, 1030), "ge", "Mean Sea-Level Pressure - HRRR", single="HRRR"),
    _P("hr_frzlvl", "HRRR freezing level", G_HRRR, "kft", "inst", C.HEIGHT_KFT, "u8", 0.0, 0.05, 1, False,
       "Height of HRRR's 0 °C isotherm (HGT:0C isotherm) above sea level, as HRRR computes it.", (1, 2, 3), "le", "Freezing Level - HRRR (0 °C isotherm)", single="HRRR"),
    _P("hr_dgz", "HRRR DGZ depth", G_HRRR, "kft", "inst", C.DGZ, "u8", 0.0, 0.1, 1, False,
       "Thickness of the layer between HRRR's 261 K and 256 K levels (-12.15 and -17.15 °C), the dendritic growth zone, in kft (HRRR's LAYTH field). Temperature only: "
       "the moist-and-lifted versions are the next two maps.", (1, 2, 4, 6), "ge", "DGZ Depth - HRRR (261-256 K layer)", single="HRRR"),
    _P("hr_dgz_lift", "HRRR moist DGZ with lift", G_HRRR, "kft", "inst", C.DGZ, "u8", 0.0, 0.1, 1, False,
       "Depth of the -12 to -18 °C layer that is also at least 90 % saturated with respect to ice and rising at 0.1 m/s or more, from HRRR's own temperature, humidity "
       "and vertical velocity (the RRFS version has to borrow the deterministic run's lift). Computed on the 25 hPa levels from 925 to 300 hPa.", (1, 2, 4), "ge",
       "Moist DGZ Depth with Ascent - HRRR (kft)", single="HRRR"),
    _P("hr_dgz_omega", "HRRR DGZ ascent", G_HRRR, "µbar/s", "inst", C.OMEGA, "u8", 0.0, 0.2, 0, False,
       "Mean upward motion inside HRRR's -12 to -18 °C layer, in microbar/s, from HRRR's pressure-level vertical velocity.", (5, 10, 20), "ge",
       "Ascent in the DGZ - HRRR (microbar/s)", single="HRRR"),
    _P("hr_slr_cobb", "HRRR Cobb ratio (with lift)", G_HRRR, ":1", "inst", C.SLR, "u8", 0.0, 0.2, 0, False,
       "Cobb & Waldstreicher cloud-layer snow-to-liquid ratio on HRRR's own temperature, humidity and vertical velocity (925-300 hPa, RH and sqrt(VV) weights, capped at 30:1).",
       (10, 12, 15, 20), "ge", "Snow-to-Liquid Ratio - Cobb layer on HRRR (with lift)", single="HRRR"),
)
PRODUCTS = PRODUCTS + _ADDED

# Probability-matched means only make sense for intermittent, skewed precipitation amounts, where averaging members smears out the maxima they
# restore.  Ratios, temperatures, winds, p-type probabilities and the like keep the plain mean.  (The 1-hour variants below inherit the flag.)
PRODUCTS = tuple(replace(p, pmm=(p.kind == "cum" and p.unit == "in")) for p in PRODUCTS)
PRODUCTS = tuple(replace(p, sam=(p.pmm and p.group in (G_PRECIP, G_SNOW))) for p in PRODUCTS)
PRODUCTS = tuple(replace(p, pmm=False, sam=False) if p.single else p for p in PRODUCTS)          # single-source (HRRR) products have no members to match
# Stored series: every product, plus the 1-hour variant of those that have one (the browser derives 3/6/12/24-h windows from totals).
ONE_HOUR = tuple(replace(p, key=p.key + "_1h", label=p.label + " (1 h)", has1h=False, step1h=0.0, kind="inst",
                         step=p.step1h or p.step) for p in PRODUCTS if p.has1h)
STORE = PRODUCTS + ONE_HOUR
BY_KEY = {p.key: p for p in STORE}
GROUPS = tuple(dict.fromkeys(p.group for p in PRODUCTS))
