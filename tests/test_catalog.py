from rrfs_viewer.catalog import BY_KEY, PRODUCTS, STORE


def test_pmm_only_for_precipitation_amounts():
    on = {p.key for p in PRODUCTS if p.pmm}
    assert on == {"qpf", "snow_model", "snow_12", "snow_kuchera", "snow_maxt", "snow_thick", "snow_mean", "snow_cobb", "snow_cobblift",
                  "snow_nbmthick", "snow_byun", "snow_clim", "frzr", "sleet"}
    for k in ("slr_kuchera", "slr_cobb", "psnow", "prain", "t2", "windchill", "gust", "wind10", "vis", "blizzard_now", "uvv", "dgz_depth", "warmnose", "snow_spread"):
        assert not BY_KEY[k].pmm, k


def test_sam_only_for_liquid_and_snowfall_amounts():
    on = {p.key for p in PRODUCTS if p.sam}
    assert on == {"qpf", "snow_model", "snow_12", "snow_kuchera", "snow_maxt", "snow_thick", "snow_mean", "snow_cobb", "snow_cobblift",
                  "snow_nbmthick", "snow_byun", "snow_clim"}
    for k in ("frzr", "sleet", "slr_kuchera", "t2", "uvv", "dgz_depth", "snow_spread"):
        assert not BY_KEY[k].sam, k


def test_one_hour_variants_follow_their_parent():
    for p in STORE:
        if p.key.endswith("_1h"):
            assert p.pmm == BY_KEY[p.key[:-3]].pmm, p.key
            assert p.sam == BY_KEY[p.key[:-3]].sam, p.key
