# core/parsers/bom_excel.py
from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import win32com.client
import shutil
import tempfile


# =========================
# Config: colonne tabella
# =========================
TABLE_HEADER_ALIASES = {
    "pos": {"pos", "pos.", "position"},
    "qty": {"q.ty", "qty", "qta", "q.tà", "q.tà.", "qta.", "q'ty", "quantity"},
    "um": {"um", "u.m.", "u.m", "uom", "unit"},
    "internal_code": {"internal code", "internalcode", "code", "component code", "componentcode"},
    "description": {"description", "descrizione", "desc"},
    "refdes": {"ref. designator", "ref designator", "refdes", "reference designator", "ref.des"},
    "notes": {"notes", "note"},
    "manufacturer": {"manufacturer", "produttore", "mfr"},
    "manufacturer_code": {"manufacturer code", "mfr code", "mfrcode", "cod. produttore", "cod produttore"},
    "rev": {"rev", "revision"},
    "val": {"val", "value"},
    "rat": {"rat", "rating"},
    "tol": {"tol", "tolerance"},
    "tecn": {"tecn", "tech", "technology"},
}

DEFAULT_CONTINUATION_FIELDS = {
    "description",
    "refdes",
    "notes",
    "manufacturer",
    "manufacturer_code",
    "val",
    "rat",
    "tol",
    "tecn",
}

FIELD_JOIN_SEP = {
    "refdes": ", ",
}

MAX_SCAN_ROWS = 120
MAX_SCAN_COLS = 60
MAX_EMPTY_RUN = 30
MAX_TOTAL_ROWS = 20000

BOM_HEADER_SCAN_ROWS = 60
BOM_HEADER_SCAN_COLS = 30

# Bulk reading chunk size for table lines
TABLE_READ_COLS = 60   # enough for alias scanning + table columns
TABLE_CHUNK_ROWS = 400


# =========================
# Helpers
# =========================
def _norm(s) -> str:
    """Lowercase + strip + spazi multipli -> singolo (per confronto alias tabella)."""
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _norm_key(s) -> str:
    """
    Normalizzazione aggressiva per etichette:
    - uppercase
    - lascia solo A-Z e 0-9
    """
    if s is None:
        return ""
    s = str(s).strip().upper()
    s = re.sub(r"[^A-Z0-9]+", "", s)
    return s


def _raw(s) -> str:
    return "" if s is None else str(s).strip()


def _cell_ref(r: int, c: int) -> str:
    # mantengo la tua notazione R1C1 (utile per debug)
    return f"R{r}C{c}"


def _is_pos_value(v: str) -> bool:
    v = (v or "").strip()
    return v.isdigit()


def _parse_float(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None

    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")

    try:
        return float(s)
    except Exception:
        return None


def _extract_header_from_filename(path: Path) -> tuple[str, str, str]:
    name = path.stem
    m_rev = re.search(r"\bREV\s+([A-Z0-9]+)\b", name, re.IGNORECASE)
    rev = m_rev.group(1) if m_rev else ""
    m_code = re.search(r"\b\d{6,}\b", name)
    code = m_code.group(0) if m_code else ""
    title = name
    return code, rev, title


def _range_values_2d(ws, r1: int, c1: int, r2: int, c2: int):
    """
    Bulk read Excel range -> tuple of rows.

    🔒 Stabilizzazione coerente:
    - restituisce SEMPRE una matrice rettangolare (righe x colonne)
      anche se Excel ritorna None/scalare/tuple "degenere".
    - evita edge-case che portano a scan incompleti o mismatch.
    """
    rows = max(0, r2 - r1 + 1)
    cols = max(0, c2 - c1 + 1)
    if rows == 0 or cols == 0:
        return tuple()

    rng = ws.Range(ws.Cells(r1, c1), ws.Cells(r2, c2))
    vals = rng.Value

    # Caso: Excel ritorna None
    if vals is None:
        return tuple(tuple(None for _ in range(cols)) for _ in range(rows))

    # Caso: 1x1 -> scalare
    if not isinstance(vals, tuple):
        # riempi comunque la matrice richiesta:
        # se chiedi 1x1 ok, se chiedi NxM e Excel torna scalare (raro) metti scalare in [0][0]
        mat = [[None for _ in range(cols)] for _ in range(rows)]
        mat[0][0] = vals
        return tuple(tuple(r) for r in mat)

    # Caso: una riga -> tuple di scalari
    if len(vals) > 0 and not isinstance(vals[0], tuple):
        row = list(vals)
        # pad/truncate
        row = (row + [None] * cols)[:cols]
        mat = [row] + [[None] * cols for _ in range(rows - 1)]
        return tuple(tuple(r) for r in mat)

    # Caso: tuple di tuple (normale)
    mat = []
    for rr in range(rows):
        if rr < len(vals) and isinstance(vals[rr], tuple):
            row = list(vals[rr])
        else:
            row = []
        row = (row + [None] * cols)[:cols]
        mat.append(row)

    return tuple(tuple(r) for r in mat)


def _value_right(ws, r: int, c: int, max_offset: int = 8) -> tuple[str, str]:
    """Primo valore non vuoto a destra della cella (r,c)."""
    # (attualmente non più usato nel bulk header scan, ma lo tengo perché fa parte del tuo set helper)
    for off in range(1, max_offset + 1):
        v = _raw(ws.Cells(r, c + off).Value)
        if v:
            return v, _cell_ref(r, c + off)
    return "", ""


def _extract_pn_from_cell_text(cell_text: str) -> str:
    t = (cell_text or "").strip()
    if not t:
        return ""
    m = re.search(r"\bP\s*/\s*N\b\s*[:\-]?\s*([A-Z0-9\-_.]+)", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"\bPN\b\s*[:\-]?\s*([A-Z0-9\-_.]+)", t, re.IGNORECASE)
    return m2.group(1).strip() if m2 else ""


def _extract_rev_from_cell_text(cell_text: str) -> str:
    t = (cell_text or "").strip()
    if not t:
        return ""
    m = re.search(r"\bREV(?:ISION)?\b\s*[:\-]?\s*([A-Z0-9]+)\b", t, re.IGNORECASE)
    return (m.group(1).strip() if m else "")


def _extract_bom_header_from_sheet(ws) -> tuple[str, str, list[str], dict]:
    warnings: list[str] = []
    meta: dict = {
        "code_source": "",
        "rev_source": "",
        "code_cell": "",
        "rev_cell": "",
        "code_label_cell": "",
        "rev_label_cell": "",
        "code_raw_label": "",
        "rev_raw_label": "",
    }

    code = ""
    rev = ""

    # ✅ 1 sola chiamata COM
    matrix = _range_values_2d(ws, 1, 1, BOM_HEADER_SCAN_ROWS, BOM_HEADER_SCAN_COLS)

    for r0, row in enumerate(matrix):
        r = r0 + 1
        row_raw = [_raw(v) for v in row]
        row_normkey = [_norm_key(v) for v in row]

        for c0 in range(min(BOM_HEADER_SCAN_COLS, len(row_raw))):
            c = c0 + 1
            cell_text = row_raw[c0]
            label = row_normkey[c0]

            # ---- CODE (P/N) ----
            if not code:
                inline = _extract_pn_from_cell_text(cell_text) or ""
                if inline:
                    code = inline
                    meta["code_source"] = "sheet_inline"
                    meta["code_cell"] = _cell_ref(r, c)

            if not code and label in {"PN", "PARTNUMBER", "PARTNO"}:
                # ✅ value-right senza COM: guarda nella matrice
                for off in range(1, 9):
                    idx = c0 + off
                    if idx >= len(row_raw):
                        break
                    v = row_raw[idx]
                    if v:
                        code = v
                        meta["code_source"] = "sheet_label_right"
                        meta["code_label_cell"] = _cell_ref(r, c)
                        meta["code_raw_label"] = cell_text
                        meta["code_cell"] = _cell_ref(r, idx + 1)
                        break

            # ---- REV ----
            if not rev and (label.startswith("REV") or label == "REVISION"):
                inline_rev = _extract_rev_from_cell_text(cell_text)
                if inline_rev:
                    rev = inline_rev
                    meta["rev_source"] = "sheet_inline"
                    meta["rev_cell"] = _cell_ref(r, c)
                else:
                    for off in range(1, 9):
                        idx = c0 + off
                        if idx >= len(row_raw):
                            break
                        v = row_raw[idx]
                        if v:
                            rev = v
                            meta["rev_source"] = "sheet_label_right"
                            meta["rev_label_cell"] = _cell_ref(r, c)
                            meta["rev_raw_label"] = cell_text
                            meta["rev_cell"] = _cell_ref(r, idx + 1)
                            break

        if code and rev:
            break

    if not code:
        warnings.append(
            "Header BOM: codice assieme (P/N) non trovato nel foglio, uso fallback da filename se disponibile."
        )
        meta["code_source"] = meta["code_source"] or "missing_in_sheet"

    if not rev:
        warnings.append("Header BOM: REV non trovata nel foglio, uso fallback da filename se disponibile.")
        meta["rev_source"] = meta["rev_source"] or "missing_in_sheet"

    return code.strip(), rev.strip(), warnings, meta


def _find_table_header_row_and_map(ws) -> tuple[int | None, dict[str, int], list[str]]:
    """Trova la riga header della tabella componenti e crea una mappa canonical->col (BULK)."""
    warnings: list[str] = []
    best_row = None
    best_score = 0
    best_map: dict[str, int] = {}

    matrix = _range_values_2d(ws, 1, 1, MAX_SCAN_ROWS, MAX_SCAN_COLS)

    for r0 in range(min(MAX_SCAN_ROWS, len(matrix))):
        found: dict[str, int] = {}
        row = matrix[r0]

        for c0 in range(min(MAX_SCAN_COLS, len(row))):
            cell = _norm(row[c0])
            if not cell:
                continue

            for canonical, aliases in TABLE_HEADER_ALIASES.items():
                if cell in aliases:
                    found[canonical] = c0 + 1  # 1-based col

        score = len(found.keys() & {"pos", "qty", "um", "description"})
        if "internal_code" in found:
            score += 1
        if "refdes" in found:
            score += 1
        if "manufacturer" in found:
            score += 1
        if "manufacturer_code" in found:
            score += 1

        if score > best_score:
            best_score = score
            best_row = r0 + 1
            best_map = found

        if {"pos", "description", "qty"}.issubset(found.keys()):
            return r0 + 1, found, warnings

    if not best_row or best_score < 3:
        warnings.append("Header tabella BOM non riconosciuto con sufficiente confidenza (colonne chiave non trovate).")
        return None, {}, warnings

    if "pos" not in best_map:
        warnings.append("Header tabella trovato ma colonna POS non identificata (parser userà fallback).")

    return best_row, best_map, warnings


def _row_looks_like_repeated_header_vals(row_vals, col_map: dict[str, int]) -> bool:
    """Riconosce header ripetuti dentro la tabella (NO COM: usa row_vals)."""
    pos_col = col_map.get("pos")
    if pos_col:
        v = _norm(row_vals[pos_col - 1] if (pos_col - 1) < len(row_vals) else None)
        if v in TABLE_HEADER_ALIASES["pos"]:
            return True

    hits = 0
    for _, c in col_map.items():
        idx = c - 1
        if idx < 0 or idx >= len(row_vals):
            continue
        v = _norm(row_vals[idx])
        if not v:
            continue
        for aliases in TABLE_HEADER_ALIASES.values():
            if v in aliases:
                hits += 1
                break

    return hits >= 3


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

    # 🔒 due flag “safe” coerenti (riduce blocchi e macro)
    try:
        excel.AutomationSecurity = 3  # msoAutomationSecurityForceDisable
    except Exception:
        pass
    try:
        excel.Interactive = False
    except Exception:
        pass


def _open_workbook_safe(excel, path: Path):
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


# =========================
# Public API
# =========================
def parse_bom_excel_raw(path: Path, excel=None) -> dict:
    created_excel = False
    wb = None
    ws = None
    warnings: list[str] = []

    # gestione copia locale (solo se UNC)
    opened_path = path
    tmp_dir: str | None = None

    try:
        if excel is None:
            excel = win32com.client.DispatchEx("Excel.Application")
            created_excel = True

        _apply_excel_safe_flags(excel)

        # =========================
        # COPY_TO_LOCAL (minimo, ma molto utile su share)
        # =========================
        try:
            p_str = str(path)
            if p_str.startswith("\\\\"):  # UNC path
                tmp_dir = tempfile.mkdtemp(prefix="mdp_bom_")
                local_path = Path(tmp_dir) / path.name
                shutil.copy2(path, local_path)
                opened_path = local_path
        except Exception as e:
            warnings.append(f"COPY_TO_LOCAL fallita per {path.name}: {type(e).__name__}: {e}")
            opened_path = path  # fallback: apri da rete

        # open workbook
        wb = _open_workbook_safe(excel, opened_path)
        ws = wb.Sheets(1)

        # 1) BOM header (assiemi) da foglio
        sheet_code, sheet_rev, w_hdr, hdr_meta = _extract_bom_header_from_sheet(ws)
        warnings.extend(w_hdr)

        # fallback da filename (non deduzione) -> usa SEMPRE il path originale
        file_code, file_rev, title = _extract_header_from_filename(path)

        final_code = sheet_code or file_code
        final_rev = sheet_rev or file_rev

        code_source = hdr_meta.get("code_source", "")
        rev_source = hdr_meta.get("rev_source", "")

        if not sheet_code and final_code:
            code_source = "filename_fallback"
        if not sheet_rev and final_rev:
            rev_source = "filename_fallback"

        header = {
            "code": final_code,
            "rev": final_rev,
            "title": title,
            "date": "",
            "code_source": code_source,
            "rev_source": rev_source,
            "code_cell": hdr_meta.get("code_cell", ""),
            "rev_cell": hdr_meta.get("rev_cell", ""),
            "code_label_cell": hdr_meta.get("code_label_cell", ""),
            "rev_label_cell": hdr_meta.get("rev_label_cell", ""),
            "code_raw_label": hdr_meta.get("code_raw_label", ""),
            "rev_raw_label": hdr_meta.get("rev_raw_label", ""),
        }

        # 2) header tabella componenti (BULK)
        header_row, col_map, w_tbl = _find_table_header_row_and_map(ws)
        warnings.extend(w_tbl)

        if not header_row:
            return {"header": header, "lines": [], "warnings": warnings}

        cont_fields = set(col_map.keys()) - {"pos", "qty", "um", "internal_code"}
        cont_fields |= (DEFAULT_CONTINUATION_FIELDS & set(col_map.keys()))

        # UsedRange bounds
        used = ws.UsedRange
        last_row = used.Row + used.Rows.Count - 1
        end_row = min(last_row, header_row + MAX_TOTAL_ROWS)

        def _get_from_rowvals(row_vals, field: str) -> str:
            c = col_map.get(field)
            if not c:
                return ""
            idx = c - 1
            if idx < 0 or idx >= len(row_vals):
                return ""
            return _raw(row_vals[idx])

        def _row_has_any_data_vals(row_vals) -> bool:
            for f in col_map.keys():
                if _get_from_rowvals(row_vals, f).strip():
                    return True
            return False

        lines: list[dict] = []
        current: Optional[dict] = None
        empty_run = 0

        # Read table rows in chunks (bulk)
        r = header_row + 1
        while r <= end_row:
            r2 = min(end_row, r + TABLE_CHUNK_ROWS - 1)
            chunk = _range_values_2d(ws, r, 1, r2, TABLE_READ_COLS)

            for i, row_vals in enumerate(chunk):
                abs_r = r + i

                if not _row_has_any_data_vals(row_vals):
                    empty_run += 1
                    if empty_run >= MAX_EMPTY_RUN:
                        r = end_row + 1
                        break
                    continue
                empty_run = 0

                if _row_looks_like_repeated_header_vals(row_vals, col_map):
                    continue

                pos = _get_from_rowvals(row_vals, "pos").strip()

                if pos and _is_pos_value(pos):
                    rec = {k: "" for k in col_map.keys()}
                    rec["pos"] = pos.zfill(4)

                    for f in col_map.keys():
                        if f == "pos":
                            continue
                        rec[f] = _get_from_rowvals(row_vals, f).strip()

                    if "qty" in rec:
                        rec["qty"] = _parse_float(str(rec["qty"]))

                    lines.append(rec)
                    current = rec
                    continue

                # continuation row
                if current is not None:
                    for f in cont_fields:
                        v = _get_from_rowvals(row_vals, f).strip()
                        if not v:
                            continue
                        prev = (current.get(f) or "").strip()
                        sep = FIELD_JOIN_SEP.get(f, " ")
                        current[f] = (prev + sep + v).strip() if prev else v
                    continue

                warnings.append(f"Riga {abs_r}: contenuto senza POS e senza record precedente a cui collegare.")

            r = r2 + 1

        out_lines: list[dict] = []
        for rec in lines:
            out_lines.append(
                {
                    "pos": rec.get("pos", ""),
                    "internal_code": rec.get("internal_code", ""),
                    "description": rec.get("description", ""),
                    "qty": rec.get("qty", None),
                    "um": rec.get("um", ""),
                    "rev": rec.get("rev", ""),
                    "val": rec.get("val", ""),
                    "rat": rec.get("rat", ""),
                    "tol": rec.get("tol", ""),
                    "refdes": rec.get("refdes", ""),
                    "tecn": rec.get("tecn", ""),
                    "notes": rec.get("notes", ""),
                    "manufacturer": rec.get("manufacturer", ""),
                    "manufacturer_code": rec.get("manufacturer_code", ""),
                }
            )

        return {"header": header, "lines": out_lines, "warnings": warnings}

    finally:
        # chiusura workbook
        try:
            if wb is not None:
                wb.Close(False)
        except Exception:
            pass

        # chiusura excel se creato qui
        if created_excel:
            try:
                excel.Quit()
            except Exception:
                pass

        # cleanup copia locale
        try:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
