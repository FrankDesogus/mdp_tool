import pytest

pdf = pytest.importorskip('pdfplumber')
from core.parsers.bom_pdf import _best_guess_key


def test_manufacturer_code_header_precedence_over_manufacturer():
    assert _best_guess_key('Manufacturer Code') == 'manufacturer_code'
    assert _best_guess_key('Codice costruttore') == 'manufacturer_code'
