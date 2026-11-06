"""Colour ramps for every winter product.

* OKLab-interpolated, fixed physical levels (a colour always means the same value, in every frame and for every member).
* Wind / gust, wind chill and temperature use the same palettes as the RTMA dashboard (NWS legends / classic -40..120 F ramp).
* QPF and snowfall use the NWS Gray/Portland legends (sampled from the NWS maps); ice, sleet, probabilities, visibility, SLR, DGZ, warm nose, lift and blizzard ramps are new, in the same spirit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import to_hex, to_rgb

# --------------------------------------------------------------------------- OKLab
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M1I, _M2I = np.linalg.inv(_M1), np.linalg.inv(_M2)


def srgb_to_oklab(rgb) -> np.ndarray:
    c = np.asarray(rgb, float)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.cbrt(lin @ _M1.T) @ _M2.T


def oklab_to_srgb(lab) -> np.ndarray:
    lin = (np.asarray(lab, float) @ _M2I.T) ** 3 @ _M1I.T
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055).clip(0, 1)


class Ramp:
    """Piece-wise ramp defined by (value, hex) anchors, interpolated in OKLab."""

    def __init__(self, stops: list[tuple[float, str]]):
        stops = sorted(stops)
        self.values = np.array([v for v, _ in stops], float)
        self.lab = np.array([srgb_to_oklab(to_rgb(h)) for _, h in stops])

    def __call__(self, v) -> np.ndarray:
        v = np.atleast_1d(np.asarray(v, float))
        lab = np.stack([np.interp(v, self.values, self.lab[:, i]) for i in range(3)], axis=-1)
        return oklab_to_srgb(lab)

    def lightness(self, n: int = 200) -> np.ndarray:
        return np.stack([np.interp(np.linspace(self.values[0], self.values[-1], n), self.values, self.lab[:, 0])])[0]



TEMP_F = Ramp([  # classic -40..120 F operational ramp: magenta -> purple -> blue | green (32 F) -> yellow -> orange -> red -> pink
    (-40, "#ffa4f0"), (-35, "#fd81f0"), (-30, "#fc53f1"), (-25, "#fa30f1"), (-20, "#e31eea"), (-15, "#ce14e3"),
    (-10, "#b208da"), (-5, "#9e0bdc"), (0, "#8517e3"), (5, "#711fe8"), (10, "#5339e7"), (15, "#3957e1"),
    (20, "#197ed8"), (25, "#1d91dc"), (30, "#23a9e1"),
    (31.9, "#24aee2"),                       # <32 F: blue ...
    (32.1, "#25b84a"),                       # ... >=32 F: green - the freezing line is unmistakable
    (35, "#2ac045"), (40, "#64d318"), (45, "#8ee100"), (50, "#baef00"), (55, "#ddfb00"), (60, "#e5e002"),
    (65, "#e0be04"), (70, "#da9306"), (75, "#d56e08"), (80, "#cf3e0a"), (85, "#ca190c"), (90, "#d10f24"),
    (95, "#e12146"), (100, "#f53871"), (105, "#fa3e88"), (110, "#ea288b"), (115, "#dd188e"), (120, "#d20991"),
])


# --------------------------------------------------------------------------- NWS legends (sampled from NWS Gray/Portland maps)
# Wind speed / gust (mph): NWS "Maximum Wind Gust" legend.  Edges are the whole-number bin starts of the legend labels
# (0-5, 6-10, 11-15, ... 81-100, 101-120, 121-140, 141-150).  Same colours for sustained wind and gusts.
_NWS_WIND_LEVELS = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 140, 150], float)
_NWS_WIND_COLORS = ["#103f78", "#225ea8", "#1d91c0", "#41b6c4", "#7fcdbb", "#b4d79e", "#dfff9e", "#ffffa6",
                    "#ffe873", "#ffc400", "#ffaa00", "#ff5900", "#ff0000", "#a80000", "#6e0000", "#ffbee8", "#ff73df"]

# Apparent temperature (°F): NWS "24-Hour Minimum Wind Chill Index Temperature" legend (<-60 .. 35+, 5 °F bins) joined to the
# NWS "24-Hour Maximum Heat Index Temperature" legend (75-79 .. 120+, 5 °F bins).  Neither legend covers 35-75 °F (wind chill is
# flat 35+, heat index is a single grey below 75), so that span is a smooth OKLab bridge from the wind-chill teal to the
# first heat-index yellow.
_WC_COLORS = ["#a51453",                                                      # < -60
              "#e2266a", "#fb3d9e", "#f379c4", "#ff87f3", "#ffd2fc",          # -60 .. -35
              "#ffffff", "#eeeeff", "#d0d1f0", "#b2aedc", "#897fc5", "#683ba3", "#211491",   # -35 .. 0
              "#2151b0", "#147ad6", "#3db2ff", "#5edbff", "#87ebff", "#c1ffff",              # 0 .. 30
              "#44e3d6"]                                                      # 30 .. 35
_WC_TEAL = "#14adaa"                                                          # 35+
_HI_COLORS = ["#ffffb4", "#ffed8a", "#ffc23e", "#ffa150", "#ff623e", "#f72e30", "#c5143a", "#94143a", "#6d1438"]  # 75 .. 120
_HI_TOP = "#3c143c"                                                           # 120+


def _apparent_colors() -> tuple[np.ndarray, list[str]]:
    levels = np.arange(-60, 121, 5, dtype=float)                              # 36 bins
    bridge = Ramp([(37.5, _WC_TEAL), (77.5, _HI_COLORS[0])])
    mid = [to_hex(c) for c in bridge(np.arange(40, 75, 5) + 2.5)]            # 40-45 .. 70-75
    cols = _WC_COLORS + [_WC_TEAL] + mid + _HI_COLORS + [_HI_TOP]             # under + 36 bins + over
    assert len(cols) == len(levels) + 1
    return levels, cols




# --------------------------------------------------------------------------- style container
@dataclass
class Style:
    levels: list[float]                 # bin edges (ascending)
    colors: list[str | None]            # len(levels) + 1: [below first edge, bin 1 .. bin n-1, above last edge]; None = transparent
    ticks: list[float]
    equal_width: bool = False           # legend with equal-width blocks (uneven levels)


def _bins(ramp: Ramp, levels, under: str | None, over: str | None) -> list[str | None]:
    mids = [(levels[i] + levels[i + 1]) / 2 for i in range(len(levels) - 1)]
    return [under] + [to_hex(c) for c in ramp(mids)] + [over]


def ramp_style(anchors, levels, under=None, over="same", ticks=None, equal_width=True) -> Style:
    r = Ramp(anchors)
    top = to_hex(r(levels[-1] + 1e-6)[0]) if over == "same" else over
    return Style(list(levels), _bins(r, levels, under, top), list(ticks if ticks is not None else levels), equal_width)


# --------------------------------------------------------------------------- winter ramps
# QPF: NWS "24-Hour Observed Precipitation" legend (sampled from the NWS Gray/Portland map).  Bins 0.01-0.09, 0.10-0.24,
# 0.25-0.49, 0.50-0.99, 1.00-1.49, 1.50-1.99, 2.00-2.99, 3.00-3.99, 4.00-5.99, 6.00-7.99, 8.00-9.99, 10.00-14.99, 15.00+.
_QPF_LEVELS = [0.01, 0.10, 0.25, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00, 8.00, 10.00, 15.00]
_QPF_COLORS = ["#c7e9c0", "#a1d99b", "#74c476", "#31a353", "#006d2c", "#fffa8a", "#ffcc4f", "#fe8d3c", "#fc4e2a",
               "#d61a1c", "#ad0026", "#700026", "#3b0030"]
QPF = Style(list(_QPF_LEVELS), [None, *_QPF_COLORS], list(_QPF_LEVELS), True)

# Snowfall: NWS "Total Snow Forecast" legend.  Bins <1 (> trace), 1-2, 2-3, 3-4, 4-6, 6-8, 8-12, 12-18, 18-24, 24-30, 30-36, 36+.
_SNOW_LEVELS = [0.1, 1, 2, 3, 4, 6, 8, 12, 18, 24, 30, 36]
_SNOW_COLORS = ["#bdd7e7", "#6baed6", "#3182bd", "#08519c", "#082694", "#ffff96", "#ffc400", "#ff8700", "#db1400",
                "#9e0000", "#690000", "#360000"]
SNOW = Style(list(_SNOW_LEVELS), [None, *_SNOW_COLORS], list(_SNOW_LEVELS), True)

ICE = ramp_style([(0.01, "#ffd1ec"), (0.1, "#ff7ac8"), (0.25, "#e0309f"), (0.5, "#9a1f9a"), (1, "#4b1a6e")],
                 [0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1])
SLEET = ramp_style([(0.01, "#ffe9c2"), (0.1, "#ffb347"), (0.25, "#f07a1a"), (0.5, "#c14a00"), (1, "#7a2b00")],
                   [0.01, 0.05, 0.1, 0.25, 0.5, 1])
_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
PCT_SNOW = ramp_style([(5, "#e3f0fc"), (50, "#4a92e0"), (100, "#1d2f9a")], _PCT, ticks=[10, 30, 50, 70, 90])
PCT_RAIN = ramp_style([(5, "#e3f5df"), (50, "#58b868"), (100, "#0f6b3f")], _PCT, ticks=[10, 30, 50, 70, 90])
PCT_FRZR = ramp_style([(5, "#fde0f1"), (50, "#e8499f"), (100, "#6a1b7a")], _PCT, ticks=[10, 30, 50, 70, 90])
PCT_SLEET = ramp_style([(5, "#fff0d6"), (50, "#f59a2c"), (100, "#7a2b00")], _PCT, ticks=[10, 30, 50, 70, 90])
PCT_RED = ramp_style([(5, "#ffe0e6"), (50, "#ee4a68"), (100, "#6a0f3a")], _PCT, ticks=[10, 30, 50, 70, 90])
VIS = ramp_style(
    [(0, "#3b0a56"), (0.25, "#8a1c8f"), (0.5, "#d03a7a"), (1, "#f2703b"), (2, "#f6b83f"), (3, "#f5e27a"),
     (5, "#cfe6c2"), (7, "#a8d3e6"), (10, "#e9f4fb")],
    [0.1, 0.25, 0.5, 1, 2, 3, 5, 7, 10], under="#2a0640", over=None)
# Snow-to-liquid ratio: ColorBrewer RdYlBu (11 classes), reversed so warm colours are LOW ratios (dense, wet snow) and cool
# colours are HIGH ratios (light, fluffy snow).  One class per bin: <4 (darkest red), 4-6, 6-8, 8-10, 10-12, 12-14, 14-16,
# 16-18, 18-20, 20-25, 25-30; ratios above 30 keep the top blue.  The midpoint class (12-14) is the pale yellow #ffffbf, which stays distinct from the white no-data fill.
_SLR_LEVELS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30]
_SLR_COLORS = ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee090", "#ffffbf", "#e0f3f8", "#abd9e9", "#74add1",
               "#4575b4", "#313695"]
SLR = Style(list(_SLR_LEVELS), [*_SLR_COLORS, _SLR_COLORS[-1]], list(_SLR_LEVELS), True)
SPREAD = ramp_style([(0.5, "#eeeaf7"), (2, "#b9aee3"), (4, "#7d56c4"), (8, "#c4306f"), (14, "#6e0d2c")],
                    [0.5, 1, 2, 3, 4, 6, 8, 12])
DGZ = ramp_style([(5, "#eaf0fc"), (25, "#9cc0ee"), (50, "#4a82d8"), (100, "#4a2fa8"), (200, "#a02a9e")],
                 [5, 10, 25, 50, 75, 100, 150, 200])
RHI = ramp_style([(60, "#c8a46e"), (80, "#e8e2c8"), (100, "#7cc0e8"), (110, "#2b56c0")],
                 [70, 80, 90, 95, 100, 105, 110], under="#b58a52")
WARM_NOSE = ramp_style([(-12, "#2a4fb0"), (-3, "#9fc4ee"), (0, "#ffffff"), (2, "#ffc27a"), (5, "#ef6a3b"), (8, "#9b1c2f")],
                       [-10, -6, -3, -1, 0, 1, 2, 3, 5, 8], under="#2a4fb0")
UVV = ramp_style([(0.05, "#e6e6ee"), (0.3, "#9a8fd0"), (0.75, "#7a2fa6"), (2, "#d8438f")],
                 [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2])
BLIZ_HOURS = ramp_style([(3, "#ffb3c1"), (6, "#ff5f7e"), (12, "#c4163f"), (24, "#6a1b9a"), (36, "#2a0a4a")],
                        [3, 6, 9, 12, 18, 24, 36])


def _from_edges(levels, color_list, under, over, ticks=None, equal_width=True) -> Style:
    return Style(list(levels), [under] + list(color_list) + [over], list(ticks if ticks is not None else levels), equal_width)


def wind_style() -> Style:
    return _from_edges(_NWS_WIND_LEVELS, _NWS_WIND_COLORS, None, _NWS_WIND_COLORS[-1])


def temp_style() -> Style:
    """2 m temperature: classic -40..120 F ramp, 2 F bins over the range that matters in winter."""
    lv = np.arange(-40, 102, 2, dtype=float)
    mids = (lv[:-1] + lv[1:]) / 2
    cols = [to_hex(c) for c in TEMP_F(mids)]
    return Style(lv.tolist(), [to_hex(TEMP_F(-40.0001)[0])] + cols + [to_hex(TEMP_F(120)[0])], np.arange(-40, 101, 10).tolist())


def wind_chill_style() -> Style:
    levels, cols = _apparent_colors()
    return Style(levels.tolist(), list(cols), np.arange(-60, 121, 10).tolist())
