"""Snow production levels, the family grouping of the menu, and the page's families row / class-name legend."""
import re
from pathlib import Path

import numpy as np

from rrfs_viewer import diag
from rrfs_viewer.catalog import FAMILIES, GROUPS, PRODUCTS

HTML = (Path(__file__).resolve().parent.parent / "site" / "index.html").read_text()


def _out(n=4, shape=(2, 3)):
    z = lambda v=0.0: np.full((n,) + shape, v, np.float32)
    return {"snow_mean": z(), "snow_model": z(), "dgz_sat": z(), "dgz_lift": z(), "band_pot": z()}


def test_levels_follow_the_ingredients():
    o = _out()
    # cell (0,0): moist growth zone only; (0,1): growth zone with lift, nothing falling; (0,2): snowing, no growth zone;
    # (1,0): snowing with lift; (1,1): snowing, lift and banding; (1,2): nothing
    for f in range(4):
        o["dgz_sat"][f, 0, 0] = 1.5
        o["dgz_sat"][f, 0, 1] = 2.0
        o["dgz_lift"][f, 0, 1] = 1.0
        o["dgz_sat"][f, 1, 0] = 2.0
        o["dgz_lift"][f, 1, 0] = 1.0
        o["dgz_sat"][f, 1, 1] = 2.0
        o["dgz_lift"][f, 1, 1] = 1.0
        o["band_pot"][f, 1, 1] = 5.0
    o["snow_mean"][:, 0, 2] = np.arange(1, 5) * 0.2              # 0.2 in per 3-hour block
    o["snow_mean"][:, 1, 0] = np.arange(1, 5) * 0.2
    o["snow_mean"][:, 1, 1] = np.arange(1, 5) * 0.2
    r = diag.series_products(o, [3, 6, 9, 12])
    lv = r["snow_prod"]
    assert lv[3].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert r["snow_prod_h"][3, 1, 0] == 12 and r["snow_prod_h"][3, 1, 1] == 12 and r["snow_prod_h"][3, 0, 2] == 0   # only levels 4 and 5 count as efficient hours
    assert not np.isnan(lv).any()


def test_unknown_stays_blank_and_missing_lift_is_handled():
    o = _out()
    o["dgz_sat"][:] = np.nan
    o["snow_mean"][1:] = np.nan                                  # frames after the first have no snow total (gap)
    o["dgz_lift"][:] = np.nan
    r = diag.series_products(o, [3, 6, 9, 12])
    assert np.isfinite(r["snow_prod"][0]).all()                  # the first frame has a snow total
    o2 = _out()
    o2["dgz_lift"][:] = np.nan                                   # no deterministic ascent file: levels 2, 4 and 5 cannot occur
    o2["dgz_sat"][:] = 2.0
    o2["snow_mean"][:] = np.arange(1, 5).reshape(4, 1, 1) * 0.3
    lv = diag.series_products(o2, [3, 6, 9, 12])["snow_prod"]
    assert set(np.unique(lv)) <= {1.0, 3.0}


def test_catalog_entries():
    by = {p.key: p for p in PRODUCTS}
    assert by["snow_prod"].group == "Snow production" and len(by["snow_prod"].names) == 5
    st = by["snow_prod"].style
    assert len(st.levels) - 1 == len(by["snow_prod"].names)          # one class name per colour block of the legend
    assert by["snow_prod_h"].group == "Snow production" and by["snow_prod"].kind == "inst" and by["snow_prod_h"].kind == "max"


def test_every_group_is_in_exactly_one_family():
    seen = [g for _, groups in FAMILIES for g in groups]
    assert sorted(seen) == sorted(GROUPS) and len(seen) == len(set(seen))
    assert len({n for n, _ in FAMILIES}) == len(FAMILIES)


def test_page_uses_families_and_names():
    assert "D.families" in HTML and "style.names" in HTML
