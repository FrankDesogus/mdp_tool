import pytest

pytest.importorskip('pdfplumber')

from core.parsers.bom_pdf import _compare_parser_outputs, _find_table_header_band


def _w(text: str, x0: float, top: float):
    return {"text": text, "x0": x0, "x1": x0 + 10, "top": top, "bottom": top + 6, "height": 6}


def test_find_table_header_band_returns_diag_on_missing_header():
    hb, diag = _find_table_header_band([
        _w("something", 20, 200),
        _w("else", 80, 200),
        _w("not", 20, 220),
        _w("header", 80, 220),
    ])
    assert hb is None
    assert diag["rows_scanned"] >= 1
    assert diag["selected_mode"] in {"not_found", "none", "no_words"}


def test_compare_parser_outputs_reports_overlap_metrics():
    base = [{"internal_code": "E1000"}, {"internal_code": "E2000"}]
    cand = [{"internal_code": "E1000"}, {"internal_code": "E3000"}]
    out = _compare_parser_outputs(base, cand)
    assert out["code_overlap"] == 1
    assert out["codes_only_base"] == 1
    assert out["codes_only_candidate"] == 1
    assert out["code_overlap_ratio"] == 0.5
