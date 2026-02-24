import sys
import types

sys.modules.setdefault("pdfplumber", types.SimpleNamespace())

from core.use_case.analyze_folder_pdf import (
    _select_canonical_root_code,
    _select_header_code_effective,
    build_base_to_full_alias_from_headers,
)


def test_select_canonical_root_code_prefers_root_code_and_has_no_suffix():
    header = {
        "title": "E0029472 01-03",
        "code": "E0029472 01",
        "root_code": "E0029472",
    }
    assert _select_canonical_root_code(header) == "E0029472"


def test_select_canonical_root_code_fallback_from_title_and_code():
    assert _select_canonical_root_code({"title": "E0181296 01-06", "code": ""}) == "E0181296"
    assert _select_canonical_root_code({"title": "", "code": "E0181296 01"}) == "E0181296"


def test_select_canonical_root_code_discards_malformed_pattern():
    assert _select_canonical_root_code({"title": "E01813-31", "code": ""}) == ""


def test_select_header_code_effective_prefers_header_code_over_root_code():
    header = {"code": "E0181296 01", "root_code": "E0181296", "title": ""}
    assert _select_header_code_effective(header) == "E0181296 01"


def test_build_base_to_full_alias_from_headers_only_uses_dash_suffix_format():
    alias = build_base_to_full_alias_from_headers({"E0181296", "E0254438-01", "E0254438"})
    assert "E01812" not in alias
    assert alias.get("E0254438") == "E0254438-01"
