from core.services.pn_canonical import canonicalize_pn


def test_canonicalize_pn_keeps_plain_code_without_fake_suffix_split():
    assert canonicalize_pn("E0029472") == "E0029472"


def test_canonicalize_pn_splits_suffix_with_explicit_separator():
    assert canonicalize_pn("E0029472 01") == "E0029472-01"


def test_canonicalize_pn_splits_compact_only_when_rev_matches():
    assert canonicalize_pn("E002947201", rev="01") == "E0029472-01"
    assert canonicalize_pn("E002947201", rev="03") == "E002947201"
