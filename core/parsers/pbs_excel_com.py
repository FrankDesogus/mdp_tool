import re
from typing import List, Tuple, Dict, Optional
import os

from core.domain.models import MdpRow

try:
    import win32com.client  # pywin32
except ImportError:
    win32com = None


# -----------------------------
# Shared helpers
# -----------------------------
def _apply_excel_safe_flags(excel) -> None:
    if excel is None:
        return
    try:
        excel.Visible = False
    except Exception:
        pass
    try:
        excel.DisplayAlerts = False
    except Exception:
        pass
    try:
        excel.ScreenUpdating = False
    except Exception:
        pass
    try:
        excel.EnableEvents = False
    except Exception:
        pass
    try:
        excel.Calculation = -4135  # xlCalculationManual
    except Exception:
        pass
    try:
        excel.AskToUpdateLinks = False
    except Exception:
        pass


def _open_workbook_safe(excel, path: str):
    try:
        return excel.Workbooks.Open(
            str(path),
            UpdateLinks=0,
            ReadOnly=True,
            IgnoreReadOnlyRecommended=True,
            AddToMru=False,
        )
    except Exception:
        return excel.Workbooks.Open(str(path), ReadOnly=True)


def _norm_header(s: str) -> str:
    s = (s or "").strip().upper()
    s = re.sub(r"[.\-_/\\\s]+", "", s)
    return s


_HEADER_ALIASES: Dict[str, set] = {
    "CODE": {"CODE", "CODICE", "PN", "PARTNUMBER", "PARTNO"},
    "DESCRIPTION": {"DESCRIPTION", "DESCRIZIONE", "DESC"},
    "REV": {"REV", "REV.", "REVISION", "REVIS", "REVISIONS"},
    "QTY": {"QTY", "QTA", "QUANTITY", "QUANTITA", "QUANTITÀ"},
}


def _map_header_label(raw: str) -> Optional[str]:
    norm = _norm_header(raw)
    for canonical, variants in _HEADER_ALIASES.items():
        if norm in variants:
            return canonical
    return None


def _float_or_zero(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0


# -----------------------------
# LEGACY parser (your logic, safer open/flags)
# -----------------------------
def _cell_text(cell) -> str:
    try:
        v = cell.Value
    except Exception:
        return ""
    if v is None:
        return ""
    return str(v).strip()


def _merge_area_bounds(cell) -> Tuple[int, int, int, int]:
    try:
        ma = cell.MergeArea
        return ma.Row, ma.Column, ma.Row + ma.Rows.Count - 1, ma.Column + ma.Columns.Count - 1
    except Exception:
        return cell.Row, cell.Column, cell.Row, cell.Column


def _find_header_row_and_blocks_legacy(ws, scan_rows: int = 250, scan_cols: int = 120) -> Tuple[int, Dict[str, Tuple[int, int]]]:
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

        if "CODE" in found and "DESCRIPTION" in found:
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


def _first_non_empty_in_block_legacy(ws, row: int, c1: int, c2: int) -> Tuple[str, int]:
    for c in range(c1, c2 + 1):
        t = _cell_text(ws.Cells(row, c))
        if t:
            return t, c
    return "", -1


def _parse_pbs_legacy(ws, sheet_index: int = 1) -> List[MdpRow]:
    header_row, blocks = _find_header_row_and_blocks_legacy(ws)

    code_c1, code_c2 = blocks["CODE"]
    desc_c1, desc_c2 = blocks["DESCRIPTION"]
    rev_c1, _ = blocks["REV"]
    qty_c1, _ = blocks["QTY"]

    start_row = header_row + 1
    used = ws.UsedRange
    last_used_row = used.Row + used.Rows.Count - 1

    rows: List[MdpRow] = []
    empty_streak = 0
    EMPTY_STREAK_STOP = 25

    for r in range(start_row, last_used_row + 1):
        code, _code_col = _first_non_empty_in_block_legacy(ws, r, code_c1, code_c2)
        desc_text, desc_col = _first_non_empty_in_block_legacy(ws, r, desc_c1, desc_c2)

        if not code and not desc_text:
            empty_streak += 1
            if empty_streak >= EMPTY_STREAK_STOP:
                break
            continue
        empty_streak = 0

        rev = _cell_text(ws.Cells(r, rev_c1))
        try:
            qty = _float_or_zero(ws.Cells(r, qty_c1).Value)
        except Exception:
            qty = 0.0

        if desc_col == -1:
            desc_col = desc_c1
        level = max(0, desc_col - desc_c1)

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


# -----------------------------
# BULK parser (stable: Range.Value in chunks)
# -----------------------------
def _range_values(ws, r1: int, c1: int, r2: int, c2: int):
    rng = ws.Range(ws.Cells(r1, c1), ws.Cells(r2, c2))
    vals = rng.Value
    if vals is None:
        return tuple()
    if not isinstance(vals, tuple):
        return ((vals,),)
    if len(vals) > 0 and not isinstance(vals[0], tuple):
        return (vals,)
    return vals


def _find_header_row_bulk(matrix, scan_rows: int, scan_cols: int) -> Tuple[int, Dict[str, int]]:
    for r in range(min(scan_rows, len(matrix))):
        row = matrix[r]
        found: Dict[str, int] = {}
        for c in range(min(scan_cols, len(row))):
            raw = row[c]
            if raw is None:
                continue
            key = _map_header_label(str(raw))
            if not key:
                continue
            found.setdefault(key, c + 1)

        if "CODE" in found and "DESCRIPTION" in found:
            # ensure REV/QTY
            for c in range(min(scan_cols, len(row))):
                raw = row[c]
                if raw is None:
                    continue
                k2 = _map_header_label(str(raw))
                if k2 in ("REV", "QTY") and k2 not in found:
                    found[k2] = c + 1

            missing = [k for k in ("REV", "QTY") if k not in found]
            if missing:
                raise RuntimeError(f"Header trovato alla riga {r+1} ma mancano: {missing}")
            return r + 1, found

    raise RuntimeError("Impossibile trovare la riga header (CODE/DESCRIPTION/REV/QTY).")


def _guess_blocks(col_map: Dict[str, int], scan_cols: int) -> Dict[str, Tuple[int, int]]:
    code_c = col_map["CODE"]
    desc_c = col_map["DESCRIPTION"]
    rev_c = col_map["REV"]
    qty_c = col_map["QTY"]

    def _block(start: int, end: int) -> Tuple[int, int]:
        if end < start:
            end = start
        return max(1, start), min(scan_cols, end)

    return {
        "CODE": _block(code_c, max(code_c, desc_c - 1)),
        "DESCRIPTION": _block(desc_c, max(desc_c, rev_c - 1)),
        "REV": (rev_c, rev_c),
        "QTY": (qty_c, qty_c),
    }


def _first_non_empty_in_block_row(row_vals, c1: int, c2: int) -> Tuple[str, int]:
    for c in range(c1, c2 + 1):
        idx = c - 1
        if idx < 0 or idx >= len(row_vals):
            continue
        v = row_vals[idx]
        if v is None:
            continue
        t = str(v).strip()
        if t:
            return t, c
    return "", -1


def _parse_pbs_bulk(ws) -> List[MdpRow]:
    SCAN_ROWS = 250
    SCAN_COLS = 120

    matrix = _range_values(ws, 1, 1, SCAN_ROWS, SCAN_COLS)
    header_row, col_map = _find_header_row_bulk(matrix, SCAN_ROWS, SCAN_COLS)
    blocks = _guess_blocks(col_map, SCAN_COLS)

    code_c1, code_c2 = blocks["CODE"]
    desc_c1, desc_c2 = blocks["DESCRIPTION"]
    rev_c1, _ = blocks["REV"]
    qty_c1, _ = blocks["QTY"]

    start_row = header_row + 1
    used = ws.UsedRange
    last_used_row = used.Row + used.Rows.Count - 1

    rows: List[MdpRow] = []
    empty_streak = 0
    EMPTY_STREAK_STOP = 25

    CHUNK = 400
    r = start_row
    while r <= last_used_row:
        r2 = min(last_used_row, r + CHUNK - 1)
        chunk_vals = _range_values(ws, r, 1, r2, SCAN_COLS)

        for i, row_vals in enumerate(chunk_vals):
            abs_r = r + i

            code, _ = _first_non_empty_in_block_row(row_vals, code_c1, code_c2)
            desc_text, desc_col = _first_non_empty_in_block_row(row_vals, desc_c1, desc_c2)

            if not code and not desc_text:
                empty_streak += 1
                if empty_streak >= EMPTY_STREAK_STOP:
                    return rows
                continue
            empty_streak = 0

            rev = ""
            if 0 <= (rev_c1 - 1) < len(row_vals) and row_vals[rev_c1 - 1] is not None:
                rev = str(row_vals[rev_c1 - 1]).strip()

            qty = 0.0
            if 0 <= (qty_c1 - 1) < len(row_vals):
                qty = _float_or_zero(row_vals[qty_c1 - 1])

            if desc_col == -1:
                desc_col = desc_c1
            level = max(0, desc_col - desc_c1)

            code = re.sub(r"\s+", " ", code).strip()

            rows.append(MdpRow(
                src_row=abs_r,
                code=code,
                description=desc_text,
                rev=rev,
                qty=qty,
                desc_col=desc_col,
                level=level,
            ))

        r = r2 + 1

    return rows


# -----------------------------
# Public API
# -----------------------------
def parse_pbs_excel_via_com(path: str, sheet_index: int = 1) -> List[MdpRow]:
    if win32com is None:
        raise RuntimeError("pywin32 non installato (pip install pywin32) o non disponibile su questa piattaforma.")
    print(f"[PBS] Using PBS file: {path}")

    excel = win32com.client.DispatchEx("Excel.Application")
    _apply_excel_safe_flags(excel)

    wb = None
    ws = None
    try:
        wb = _open_workbook_safe(excel, path)
        ws = wb.Worksheets(sheet_index)

        # 1) BULK first (stabile)
        try:
            rows = _parse_pbs_bulk(ws)
            if rows:
                return rows
        except Exception:
            # fall back to legacy
            pass

        # 2) LEGACY fallback (compatibilità)
        return _parse_pbs_legacy(ws)

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
        try:
            del ws
        except Exception:
            pass
        try:
            del wb
        except Exception:
            pass
        try:
            del excel
        except Exception:
            pass


# Backward-compat alias (if something imports parse_mdp_excel_via_com)
parse_mdp_excel_via_com = parse_pbs_excel_via_com
