from core.services.pn_canonical import canonicalize_part_number, canonicalize_pn


def test_canonicalize_pn_keeps_plain_code_without_fake_suffix_split():
    assert canonicalize_pn("E0029472") == "E0029472"


def test_canonicalize_pn_splits_suffix_with_explicit_separator():
    assert canonicalize_pn("E0029472 01") == "E0029472-01"


def test_canonicalize_pn_splits_compact_even_without_matching_rev():
    assert canonicalize_pn("E002947201", rev="01") == "E0029472-01"
    assert canonicalize_pn("E002947201", rev="03") == "E0029472-03"


def test_canonicalize_part_number_aligns_compact_and_spaced_forms():
    assert canonicalize_part_number("E002947301") == canonicalize_part_number("E0029473 01")


def test_canonicalize_part_number_does_not_duplicate_suffix_when_already_present():
    assert canonicalize_part_number("166104001-04", suffix="04") == "166104001-04"
