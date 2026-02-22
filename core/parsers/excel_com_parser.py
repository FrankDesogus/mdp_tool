import re
from typing import List, Tuple, Dict, Optional

from core.domain.models import MdpRow

try:
    import win32com.client  # pywin32
except ImportError:
    win32com = None


# -----------------------------
# Helpers
# -----------------------------

def _cell_text(cell) -> str:
    """Return trimmed string value from an Excel COM cell."""
    v = cell.Value
    if v is None:
        return ""
    return str(v).strip()


def _float_or_zero(v) -> float:
    """Parse numbers coming from Excel (including comma decimals)."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _merge_area_bounds(cell) -> Tuple[int, int, int, int]:
    """Return bounds (r1,c1,r2,c2) of merge area; if not merged return the cell itself."""
    try:
        ma = cell.MergeArea
        return ma.Row, ma.Column, ma.Row + ma.Rows.Count - 1, ma.Column + ma.Columns.Count - 1
    except Exception:
        return cell.Row, cell.Column, cell.Row, cell.Column


def _norm_header(s: str) -> str:
    """
    Normalize header labels:
    - uppercase
    - remove spaces and punctuation like '.', '-', '_', '/', '\'
    Examples: 'Q.TY' -> 'QTY', 'Q TY' -> 'QTY'
    """
    s = (s or "").strip().upper()
    s = re.sub(r"[.\-_/\\\s]+", "", s)
    return s


# Map normalized labels to canonical keys
_HEADER_ALIASES: Dict[str, set] = {
    "CODE": {"CODE", "CODICE", "PN", "PARTNUMBER", "PARTNO"},
    "DESCRIPTION": {"DESCRIPTION", "DESCRIZIONE", "DESC"},
    "REV": {"REV", "REV.", "REVISION", "REVIS", "REVISIONS"},
    "QTY": {"QTY", "QTA", "QUANTITY", "QUANTITA", "QUANTITÀ"},
}


def _map_header_label(raw: str) -> Optional[str]:
    """Return canonical header key for a given raw cell string, or None if unknown."""
    norm = _norm_header(raw)
    for canonical, variants in _HEADER_ALIASES.items():
        if norm in variants:
            return canonical
    return None


def _find_header_row_and_blocks(ws, scan_rows: int = 250, scan_cols: int = 120) -> Tuple[int, Dict[str, Tuple[int, int]]]:
    """
    Find header row and column blocks via MergeArea.
    Returns:
        header_row: int
        blocks: dict like {"CODE": (c1,c2), "DESCRIPTION": (c1,c2), "REV": (c1,c2), "QTY": (c1,c2)}
    """
    for r in range(1, scan_rows + 1):
        found: Dict[str, Tuple[int, int]] = {}

        for c in range(1, scan_cols + 1):
            raw = _cell_text(ws.Cells(r, c))
            if not raw:
                continue

            key = _map_header_label(raw)
            if not key:
                continue

            _, c1, _, c2 = _merge_area_bounds(ws.Cells(r, c))
            found.setdefault(key, (c1, c2))

        # minimal condition: CODE + DESCRIPTION must exist
        if "CODE" in found and "DESCRIPTION" in found:
            # Ensure REV and QTY are present on same row (second scan pass on this row)
            for key in ("REV", "QTY"):
                if key in found:
                    continue
                for c in range(1, scan_cols + 1):
                    raw = _cell_text(ws.Cells(r, c))
                    if not raw:
                        continue
                    k2 = _map_header_label(raw)
                    if k2 == key:
                        _, c1, _, c2 = _merge_area_bounds(ws.Cells(r, c))
                        found[key] = (c1, c2)
                        break

            missing = [k for k in ("REV", "QTY") if k not in found]
            if missing:
                raise RuntimeError(f"Header trovato alla riga {r} ma mancano: {missing}")

            return r, found

    raise RuntimeError("Impossibile trovare la riga header (CODE/DESCRIPTION/REV/QTY).")


def _first_non_empty_in_block(ws, row: int, c1: int, c2: int) -> Tuple[str, int]:
    """
    Scan columns c1..c2 on a given row and return:
      (text, col_index_of_first_non_empty)
    If none found, returns ("", -1).
    """
    for c in range(c1, c2 + 1):
        t = _cell_text(ws.Cells(row, c))
        if t:
            return t, c
    return "", -1


# -----------------------------
# Public API
# -----------------------------

def parse_mdp_excel_via_com(path: str, sheet_index: int = 1) -> List[MdpRow]:
    """
    Robust PBS/MDP parser for the real template:
    - detects header row dynamically
    - detects column blocks via merged headers
    - reads CODE and DESCRIPTION from the *first non-empty cell inside their blocks*
      (important: CODE can be indented too)
    - reads REV and QTY from their first column (usually not indented)
    - level computed from DESCRIPTION indentation (desc_col - desc_block_start)
    """
    if win32com is None:
        raise RuntimeError("pywin32 non installato (pip install pywin32) o non disponibile su questa piattaforma.")

    excel = win32com.client.DispatchEx("Excel.Application")
    # NON toccare Visible
    # NON toccare DisplayAlerts

    wb = None
    try:
        wb = excel.Workbooks.Open(path, ReadOnly=True)
        ws = wb.Worksheets(sheet_index)

        header_row, blocks = _find_header_row_and_blocks(ws)

        code_c1, code_c2 = blocks["CODE"]
        desc_c1, desc_c2 = blocks["DESCRIPTION"]
        rev_c1, _ = blocks["REV"]
        qty_c1, _ = blocks["QTY"]

        start_row = header_row + 1
        used_rows = ws.UsedRange.Rows.Count

        rows: List[MdpRow] = []
        empty_streak = 0

        # PBS reali possono avere righe vuote/annotazioni subito dopo l'header.
        EMPTY_STREAK_STOP = 25

        for r in range(start_row, used_rows + 1):
            code, _code_col = _first_non_empty_in_block(ws, r, code_c1, code_c2)
            desc_text, desc_col = _first_non_empty_in_block(ws, r, desc_c1, desc_c2)

            # end-of-table handling
            if not code and not desc_text:
                empty_streak += 1
                if empty_streak >= EMPTY_STREAK_STOP:
                    break
                continue
            empty_streak = 0

            rev = _cell_text(ws.Cells(r, rev_c1))
            qty = _float_or_zero(ws.Cells(r, qty_c1).Value)

            # level derived from DESCRIPTION indentation inside the DESCRIPTION block
            if desc_col == -1:
                desc_col = desc_c1
            level = max(0, desc_col - desc_c1)

            # normalize code spacing, but DO NOT split by spaces/underscore
            code = re.sub(r"\s+", " ", code).strip()

            rows.append(MdpRow(
                src_row=r,
                code=code,
                description=desc_text,
                rev=rev,
                qty=qty,
                desc_col=desc_col,
                level=level,
            ))

        return rows

    finally:
        try:
            if wb is not None:
                wb.Close(SaveChanges=False)
        except Exception:
            pass
        try:
            excel.Quit()
        except Exception:
            pass
