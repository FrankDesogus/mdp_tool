import pytest

pdf = pytest.importorskip('pdfplumber')
from core.parsers.bom_pdf import _best_guess_key, extract_root_code_from_title, _align_by_pos


def test_manufacturer_code_header_precedence_over_manufacturer():
    assert _best_guess_key('Manufacturer Code') == 'manufacturer_code'
    assert _best_guess_key('Codice costruttore') == 'manufacturer_code'


def test_extract_root_code_from_title_ignores_revision_tokens():
    assert extract_root_code_from_title('E0029472 01-03') == 'E0029472'
    assert extract_root_code_from_title('E0029472 01') == 'E0029472'
    assert extract_root_code_from_title('Descrizione assembly E0029472 rev 01') == 'E0029472'
    assert extract_root_code_from_title('BOM E0029311 something') == 'E0029311'


def test_extract_root_code_from_title_discards_malformed_hyphenated_code():
    assert extract_root_code_from_title('E01813-31') is None


def test_align_by_pos_multiline_only_for_text_columns():
    pos, cols = _align_by_pos(
        ['0001', '0002'],
        [
            ['E100', 'E200', 'EXTRA_CODE_SHOULD_NOT_APPEND'],
            ['Desc 1', 'Desc 2', 'continued'],
        ],
        multiline_allowed_idx={1},
    )
    assert pos == ['0001', '0002']
    assert cols[0] == ['E100', 'E200']
    assert cols[1] == ['Desc 1', 'Desc 2\ncontinued']
