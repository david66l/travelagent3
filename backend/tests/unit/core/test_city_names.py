from core.city_names import canonical_city_name


def test_english_and_suffix_city_aliases_are_canonicalized():
    assert canonical_city_name("Shanghai") == "上海"
    assert canonical_city_name("西安市") == "西安"
