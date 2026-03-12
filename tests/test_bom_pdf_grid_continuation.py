import pytest

pdf = pytest.importorskip('pdfplumber')

from core.parsers import bom_pdf


class _FakePage:
    def __init__(self, lines, words, height=1000):
        self.lines = lines
        self._words = words
        self.height = height

    def extract_words(self, **kwargs):
        return self._words


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages


def _w(text, x0, x1, top, bottom):
    return {"text": text, "x0": x0, "x1": x1, "top": top, "bottom": bottom, "height": bottom - top}


def test_grid_continuation_inherits_columns_when_header_missing(tmp_path, monkeypatch):
    vxs = [10, 80, 220, 420, 560, 700, 780, 860, 940]
    page1_lines = [{"x0": x, "x1": x, "top": 160, "bottom": 760, "y0": 160, "y1": 760} for x in vxs]
    page1_words = [
        _w("Pos", 20, 50, 190, 200),
        _w("Codice", 100, 160, 190, 200),
        _w("Rev", 230, 260, 190, 200),
        _w("Descrizione", 440, 520, 190, 200),
        _w("Qty", 570, 600, 190, 200),
        _w("0001", 20, 50, 230, 240),
        _w("E11111", 95, 150, 230, 240),
        _w("DescA", 430, 500, 230, 240),
        _w("1", 570, 580, 230, 240),
    ]

    # continuation page: no visible vertical lines and no explicit header
    page2_lines = []
    page2_words = [
        _w("0002", 20, 50, 240, 250),
        _w("E22222", 95, 150, 240, 250),
        _w("DescB", 430, 500, 240, 250),
        _w("2", 570, 580, 240, 250),
        _w("Pagina 2", 800, 860, 960, 970),
    ]

    fake_pdf = _FakePdf([_FakePage(page1_lines, page1_words), _FakePage(page2_lines, page2_words)])

    lines, warns, diag = bom_pdf._extract_lines_from_grid_layout(fake_pdf)

    codes = {ln.get("internal_code") for ln in lines}
    assert "E11111" in codes
    assert "E22222" in codes

    page2 = diag["pages"][1]
    assert page2["continuation_page_detected"] is True
    assert page2["inherited_columns_from_previous_page"] is True
    assert page2["vertical_lines_strategy"] == "inherited_from_previous_page"


def test_write_pdf_diagnostics_creates_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bom_pdf._write_pdf_diagnostics(tmp_path / "ABC_BOM.pdf", {"summary": {"ok": True}})
    out = tmp_path / "diagnostics" / "pdf_debug" / "ABC_BOM.json"
    assert out.exists()
    assert '"ok": true' in out.read_text(encoding="utf-8").lower()
