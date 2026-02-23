import sys
import types

sys.modules.setdefault("pdfplumber", types.SimpleNamespace())

from core.use_case.analyze_folder_pdf import _select_canonical_root_code


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
