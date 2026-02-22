# core/parsers/totalization_excel.py
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import win32com.client  # type: ignore


# -------------------------
# Output model
# -------------------------
@dataclass(frozen=True)
class TotalizationParseResult:
    # PN -> qty totale per 1 prodotto finito (duplicati SUMMATI)
    qty_by_code: Dict[str, Decimal]
    warnings: List[str]

    # diagnostica duplicati
    row_count_by_code: Dict[str, int] = field(default_factory=dict)
    duplicate_codes: List[str] = field(default_factory=list)

    # EXTRA diagnostica: righe Excel (1-based) dove il PN compare (prime N)
    sample_rows_by_code: Dict[str, List[int]] = field(default_factory=dict)

    # EXTRA diagnostica: raw_code così come letto (repr) nelle prime N righe
    sample_raw_codes_by_code: Dict[str, List[str]] = field(default_factory=dict)

    # info tabella riconosciuta
    sheet_used: str = ""
    header_row_1based: int = 0
    code_col_1based: int = 0
    qty_col_1based: int = 0
    # info stop tabella
    stop_reason: str = ""


# -------------------------
# Helpers
# -------------------------
def _looks_like_formula_text(x: object) -> bool:
    """
    Riconosce il caso in cui Excel/COM ci sta dando una formula "come testo":
    es: "=SOMMA(A1:A3)" oppure "=A1*2"
    In questo caso NON è un numero e non va interpretato come tale.
    """
    if not isinstance(x, str):
        return False
    s = x.strip()
    return s.startswith("=") and len(s) > 1


def _to_decimal(x: object) -> Optional[Decimal]:
    if x is None:
        return None
    if isinstance(x, Decimal):
        return x
    try:
        # Excel COM spesso restituisce float/int/str/variant
        if isinstance(x, (int, float)):
            return Decimal(str(x))

        # Se arriva una "formula come testo", non possiamo convertirla in numero
        if _looks_like_formula_text(x):
            return None

        s = str(x).strip()
        if not s:
            return None
        s = s.replace(",", ".")
        return Decimal(s)
    except (InvalidOperation, ValueError, TypeError):
        return None


def _norm_text(x: object) -> str:
    return ("" if x is None else str(x)).strip()


def _norm_code(x: object) -> str:
    """
    Normalizzazione codice PN (coerente con l'obiettivo attuale):
    - NON rimuove spazi interni/invisibili (evita normalizzazioni aggressive)
    - uppercase
    - considera "vuoto" un codice che, dopo strip, è vuoto
    """
    if x is None:
        return ""
    s_raw = str(x)
    if not s_raw.strip():
        return ""
    return s_raw.upper()


def _is_plausible_pn(code: str) -> bool:
    """
    Filtra codici "spazzatura" che Excel COM può restituire come valori:
    - "", "0", "0.0", "0,0"
    - solo numeri/punti/virgole (es: "0.0", "123", "12.34") -> NON è un PN
    Regola: deve contenere almeno una lettera A-Z.
    """
    if not code:
        return False
    if code in {"0", "0.0", "0,0"}:
        return False
    return any("A" <= ch <= "Z" for ch in code)


def _norm_header(x: object) -> str:
    return " ".join(_norm_text(x).lower().split())


def _as_2d_rows(value) -> List[Tuple]:
    """
    Converte UsedRange.Value (scalare / tuple / tuple di tuple)
    in una lista di tuple (righe).
    """
    if value is None:
        return []

    # singola cella
    if not isinstance(value, tuple):
        return [(value,)]

    # una sola riga: tuple di valori
    if value and not isinstance(value[0], tuple):
        return [tuple(value)]

    # matrice: tuple di tuple
    return [tuple(r) for r in value]


def _excel_col_letter(col_1based: int) -> str:
    """1 -> A, 2 -> B, ..., 27 -> AA"""
    n = col_1based
    out = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        out = chr(65 + r) + out
    return out


# -------------------------
# Legacy loader (kept)
# -------------------------
def _load_excel_rows_win32(path: Path) -> Dict[str, List[Tuple]]:
    """
    Legge tutti i fogli tramite Excel COM e ritorna:
    {sheet_name: [ (cell1, cell2, ...), ... ]}

    Funziona per .xls, .xlsx, .xlsm, ecc.

    Gestione formule (coerente):
    - forza ricalcolo (se possibile)
    - legge Value2 (valori calcolati, più affidabile)
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    excel = None
    wb = None
    sheets_rows: Dict[str, List[Tuple]] = {}

    prev_calc = None
    xlCalculationAutomatic = -4105  # costante Excel

    try:
        excel = win32com.client.DispatchEx("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        try:
            excel.AskToUpdateLinks = False
        except Exception:
            pass

        try:
            prev_calc = excel.Calculation
            excel.Calculation = xlCalculationAutomatic
        except Exception:
            prev_calc = None

        wb = excel.Workbooks.Open(str(path), ReadOnly=True, UpdateLinks=0)

        try:
            wb.ForceFullCalculation = True
            wb.FullCalculationOnLoad = True
        except Exception:
            pass

        try:
            excel.CalculateFullRebuild()
        except Exception:
            try:
                wb.Calculate()
            except Exception:
                pass

        for ws in wb.Worksheets:
            name = str(ws.Name)
            used = ws.UsedRange
            rows = _as_2d_rows(used.Value2)

            def row_has_any(r: Tuple) -> bool:
                return any(_norm_text(c) for c in r)

            last_idx = -1
            for i, r in enumerate(rows):
                if row_has_any(r):
                    last_idx = i
            rows = rows[: last_idx + 1] if last_idx >= 0 else []

            sheets_rows[name] = rows

        return sheets_rows

    finally:
        try:
            if wb is not None:
                wb.Close(SaveChanges=False)
        except Exception:
            pass
        try:
            if excel is not None:
                if prev_calc is not None:
                    try:
                        excel.Calculation = prev_calc
                    except Exception:
                        pass
                excel.Quit()
        except Exception:
            pass


# -------------------------
# New loader: reads only target sheet + knows hidden rows when filter exists
# -------------------------
def _load_totalization_rows_with_hidden_win32(
    path: Path,
    target_sheet_upper: str,
) -> Tuple[str, int, List[Tuple], List[bool], bool]:
    """
    Legge SOLO il foglio target e ritorna:
    (sheet_name_original, used_first_row_1based, rows_matrix, hidden_by_row, filter_detected)

    hidden_by_row ha stessa lunghezza di rows_matrix:
      - se filter_detected=True: True se quella riga Excel è nascosta (filtrata)
      - se filter_detected=False: lista di False (non scartiamo righe hidden)
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    excel = None
    wb = None

    prev_calc = None
    xlCalculationAutomatic = -4105  # costante Excel

    try:
        excel = win32com.client.DispatchEx("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        try:
            excel.AskToUpdateLinks = False
        except Exception:
            pass

        # calcolo automatico
        try:
            prev_calc = excel.Calculation
            excel.Calculation = xlCalculationAutomatic
        except Exception:
            prev_calc = None

        wb = excel.Workbooks.Open(str(path), ReadOnly=True, UpdateLinks=0)

        # forzo recalcolo
        try:
            wb.ForceFullCalculation = True
            wb.FullCalculationOnLoad = True
        except Exception:
            pass
        try:
            excel.CalculateFullRebuild()
        except Exception:
            try:
                wb.Calculate()
            except Exception:
                pass

        ws = None
        sheet_name = ""
        for w in wb.Worksheets:
            if str(w.Name).strip().upper() == target_sheet_upper:
                ws = w
                sheet_name = str(w.Name)
                break

        if ws is None:
            return "", 0, [], [], False

        # rilevazione filtro: se c'è un AutoFilter attivo sul foglio
        filter_detected = False
        try:
            # AutoFilterMode True se il foglio ha frecce filtro attive
            # FilterMode True se c'è un filtro applicato che sta nascondendo righe
            filter_detected = bool(getattr(ws, "AutoFilterMode", False)) or bool(getattr(ws, "FilterMode", False))
        except Exception:
            filter_detected = False

        used = ws.UsedRange
        used_first_row_1based = int(used.Row)

        rows = _as_2d_rows(used.Value2)

        # taglia righe vuote finali
        def row_has_any(r: Tuple) -> bool:
            return any(_norm_text(c) for c in r)

        last_idx = -1
        for i, r in enumerate(rows):
            if row_has_any(r):
                last_idx = i
        rows = rows[: last_idx + 1] if last_idx >= 0 else []

        if not rows:
            return sheet_name, used_first_row_1based, [], [], filter_detected

        # hidden mask: SOLO se filtro rilevato, altrimenti tutto False
        hidden_by_row: List[bool] = []
        if filter_detected:
            for i in range(len(rows)):
                excel_row = used_first_row_1based + i
                try:
                    hidden = bool(ws.Rows(excel_row).Hidden)
                except Exception:
                    hidden = False
                hidden_by_row.append(hidden)
        else:
            hidden_by_row = [False] * len(rows)

        return sheet_name, used_first_row_1based, rows, hidden_by_row, filter_detected

    finally:
        try:
            if wb is not None:
                wb.Close(SaveChanges=False)
        except Exception:
            pass
        try:
            if excel is not None:
                if prev_calc is not None:
                    try:
                        excel.Calculation = prev_calc
                    except Exception:
                        pass
                excel.Quit()
        except Exception:
            pass


def _contains_any(h: str, tokens: set[str]) -> bool:
    return any(t in h for t in tokens)


def _pick_best_header_row(
    rows: List[Tuple],
    scan_rows: int,
    strong_code_tokens: set[str],
    strong_qty_tokens: set[str],
    code_tokens: set[str],
    qty_tokens: set[str],
) -> Optional[Tuple[int, int, int]]:
    """
    Ritorna (header_row_idx, code_col_idx, qty_col_idx) 0-based.
    Usa uno scoring per evitare falsi positivi.
    """
    best: Optional[Tuple[int, int, int, int]] = None
    # best = (score, header_row_idx, code_col_idx, qty_col_idx)

    for i, r in enumerate(rows[:scan_rows]):
        headers = [_norm_header(x) for x in r]

        c_idx: Optional[int] = None
        q_idx: Optional[int] = None
        score = 0

        # 1) match forti (prioritari)
        for j, h in enumerate(headers):
            if not h:
                continue
            if c_idx is None and _contains_any(h, strong_code_tokens):
                c_idx = j
                score += 100
            if q_idx is None and _contains_any(h, strong_qty_tokens):
                q_idx = j
                score += 100

        # 2) match normali (fallback)
        for j, h in enumerate(headers):
            if not h:
                continue
            if c_idx is None and _contains_any(h, code_tokens):
                c_idx = j
                score += 10
            if q_idx is None and _contains_any(h, qty_tokens):
                q_idx = j
                score += 10

        if c_idx is None or q_idx is None:
            continue

        cand = (score, i, c_idx, q_idx)
        if best is None or cand[0] > best[0]:
            best = cand

        if score >= 200:
            break

    if best is None:
        return None

    _, header_row_idx, code_col_idx, qty_col_idx = best
    return header_row_idx, code_col_idx, qty_col_idx


# -------------------------
# Public API
# -------------------------
def parse_totalization_xlsx(path: Path) -> TotalizationParseResult:
    """
    Legge una totalizzazione flat PN -> QTY totale.

    ✅ Usa win32com (Excel COM): legge anche .xls
    ✅ USA SOLO il foglio "TOTALIZZAZIONE"
    ✅ Legge i valori calcolati (Value2), quindi "come lo vedi in Excel"
    ✅ Trova header con priorità a:
       - "INTERNAL CODE"  (codice)
       - "Q.Ty"           (quantità)

    ✅ Duplicati: somma le quantità (non sovrascrive)
    ✅ Robustezza: legge SOLO la tabella e si ferma quando finisce
       (stop dopo N righe consecutive senza codice+qty)
    ✅ Filtra codici spazzatura numerici richiedendo almeno una lettera
    ✅ Diagnostica: sample_rows_by_code (righe Excel reali)
    ✅ Se ESISTE un filtro nel foglio (AutoFilter/FilterMode):
       ignora tutte le righe filtrate/nascoste (Hidden=True) -> comportamento coerente con Excel
    """
    path = Path(path)
    warnings: List[str] = []

    # Token "forti"
    strong_code_tokens = {"internal code"}
    strong_qty_tokens = {"q.ty", "q ty", "q.t y"}

    # Token generici (fallback)
    code_tokens = {
        "pn", "part", "part number", "partnumber",
        "internal code",
        "code", "codice", "codice interno", "item", "articolo",
    }
    qty_tokens = {
        "q.ty", "qty", "qta", "quantity",
        "qty_total", "qty total", "quantità", "quantita",
        "totale",
    }

    # ----------------------------------------------------------
    # Regola: usa SOLO il foglio "TOTALIZZAZIONE" + informazioni su righe filtrate
    # ----------------------------------------------------------
    TARGET_SHEET_NAME = "TOTALIZZAZIONE"
    sheet_name, used_first_row_1based, rows, hidden_by_row, filter_detected = _load_totalization_rows_with_hidden_win32(
        path=path,
        target_sheet_upper=TARGET_SHEET_NAME,
    )

    if not sheet_name:
        return TotalizationParseResult(
            qty_by_code={},
            warnings=[f"Foglio '{TARGET_SHEET_NAME}' non trovato nel file Excel."],
        )

    if not rows:
        return TotalizationParseResult(
            qty_by_code={},
            warnings=[f"Foglio '{sheet_name}' vuoto o non leggibile (UsedRange vuoto)."],
            sheet_used=sheet_name,
        )

    # ----------------------------------------------------------
    # Header detection
    # ----------------------------------------------------------
    SCAN_HEADER_ROWS = 300
    header_pick = _pick_best_header_row(
        rows=rows,
        scan_rows=min(SCAN_HEADER_ROWS, len(rows)),
        strong_code_tokens=strong_code_tokens,
        strong_qty_tokens=strong_qty_tokens,
        code_tokens=code_tokens,
        qty_tokens=qty_tokens,
    )
    if header_pick is None:
        return TotalizationParseResult(
            qty_by_code={},
            warnings=[
                f"Header non trovato nel foglio '{sheet_name}' "
                "(cercavo 'INTERNAL CODE' e 'Q.Ty' con fallback su token generici)."
            ],
            sheet_used=sheet_name,
        )

    header_row_idx, code_col, qty_col = header_pick

    # ----------------------------------------------------------
    # Lettura tabella (stop a fine tabella)
    # ----------------------------------------------------------
    qty_by_code: Dict[str, Decimal] = {}
    row_count_by_code: Dict[str, int] = {}
    sample_rows_by_code: Dict[str, List[int]] = {}
    sample_raw_codes_by_code: Dict[str, List[str]] = {}

    valid_rows_read = 0
    skipped_empty_code = 0
    skipped_bad_qty = 0
    skipped_implausible_code = 0
    skipped_qty_formula_as_text = 0
    skipped_hidden_rows = 0

    # Stop quando troviamo molte righe consecutive vuote (fine tabella)
    EMPTY_STREAK_BREAK = 30
    empty_streak = 0
    stop_reason = ""

    # Valutazione righe valide (solo log)
    MAX_EVAL_ROWS = 8000
    valid_eval = 0

    # scan di valutazione: rispetta righe filtrate SOLO se filtro rilevato
    for abs_i, r in enumerate(rows[header_row_idx + 1: header_row_idx + 1 + MAX_EVAL_ROWS], start=header_row_idx + 1):
        if filter_detected and 0 <= abs_i < len(hidden_by_row) and hidden_by_row[abs_i]:
            continue

        if code_col >= len(r) or qty_col >= len(r):
            continue

        code = _norm_code(r[code_col])
        qty_cell = r[qty_col]
        qty = _to_decimal(qty_cell)

        if _looks_like_formula_text(qty_cell):
            continue

        if not code or not _is_plausible_pn(code):
            continue
        if qty is None or qty <= 0:
            continue
        valid_eval += 1

    # Enumerate sulle righe dati: così calcoliamo la riga Excel reale
    for rel_idx, r in enumerate(rows[header_row_idx + 1:], start=1):
        # indice assoluto nella matrice rows (0-based)
        abs_i = header_row_idx + rel_idx  # es: prima riga dati => header_row_idx+1

        # riga Excel reale (1-based) considerando l'inizio del UsedRange
        excel_row_1based = used_first_row_1based + abs_i

        # Se c'è filtro: ignora righe filtrate/nascoste
        if filter_detected and 0 <= abs_i < len(hidden_by_row) and hidden_by_row[abs_i]:
            skipped_hidden_rows += 1
            continue

        # Protezioni su lunghezza riga
        if code_col >= len(r) or qty_col >= len(r):
            empty_streak += 1
            if empty_streak >= EMPTY_STREAK_BREAK:
                stop_reason = f"STOP: {EMPTY_STREAK_BREAK} righe consecutive senza colonne code/qty (fine UsedRange tabella)"
                break
            continue

        raw_code = r[code_col]
        raw_qty = r[qty_col]

        code = _norm_code(raw_code)

        # Se qty è formula come testo, contala e scarta
        if _looks_like_formula_text(raw_qty):
            skipped_qty_formula_as_text += 1
            continue

        qty = _to_decimal(raw_qty)

        # Riga "vuota" (niente codice e niente qty): avanza streak e magari stoppa
        if (not code) and (qty is None):
            empty_streak += 1
            if empty_streak >= EMPTY_STREAK_BREAK:
                stop_reason = f"STOP: {EMPTY_STREAK_BREAK} righe consecutive vuote (fine tabella)"
                break
            continue

        # Riga non vuota -> reset streak
        empty_streak = 0

        if not code:
            skipped_empty_code += 1
            continue

        # filtra codici spazzatura numerici
        if not _is_plausible_pn(code):
            skipped_implausible_code += 1
            continue

        if qty is None or qty <= 0:
            skipped_bad_qty += 1
            continue

        valid_rows_read += 1

        # Somma duplicati
        prev = qty_by_code.get(code)
        if prev is None:
            qty_by_code[code] = qty
            row_count_by_code[code] = 1
        else:
            qty_by_code[code] = prev + qty
            row_count_by_code[code] = row_count_by_code.get(code, 1) + 1

        # Salva sample righe (prime 5) per debug in Excel
        lst = sample_rows_by_code.get(code)
        if lst is None:
            sample_rows_by_code[code] = [excel_row_1based]
        else:
            if len(lst) < 5:
                lst.append(excel_row_1based)

        # Salva sample raw_code repr (prime 5)
        raw_lst = sample_raw_codes_by_code.get(code)
        raw_repr = repr(raw_code)
        if raw_lst is None:
            sample_raw_codes_by_code[code] = [f"riga {excel_row_1based}: {raw_repr}"]
        else:
            if len(raw_lst) < 5:
                raw_lst.append(f"riga {excel_row_1based}: {raw_repr}")

    if not stop_reason:
        stop_reason = "STOP: fine UsedRange (nessuna condizione di break raggiunta)"

    duplicate_codes = sorted([c for c, n in row_count_by_code.items() if n > 1])

    warnings.append(
        f"Tabella riconosciuta: sheet='{sheet_name}', used_first_row={used_first_row_1based}, "
        f"header_row={used_first_row_1based + header_row_idx}, "
        f"code_col={code_col + 1}({_excel_col_letter(code_col+1)}), "
        f"qty_col={qty_col + 1}({_excel_col_letter(qty_col+1)}), valid_rows_eval={valid_eval}, "
        f"filter_detected={filter_detected}"
    )
    warnings.append(
        "Righe dati lette: "
        f"valid={valid_rows_read}, pn_unici={len(qty_by_code)}, codici_duplicati={len(duplicate_codes)}, "
        f"skipped_empty_code={skipped_empty_code}, skipped_bad_qty={skipped_bad_qty}, "
        f"skipped_implausible_code={skipped_implausible_code}, skipped_qty_formula_as_text={skipped_qty_formula_as_text}, "
        f"skipped_hidden_rows={skipped_hidden_rows}"
    )
    warnings.append(f"{stop_reason} | empty_streak_break={EMPTY_STREAK_BREAK}")

    if skipped_qty_formula_as_text > 0:
        warnings.append(
            "Nota: trovate celle QTY che sembrano formule ma sono testo (iniziano con '='); "
            "Excel non le calcola perché non sono formule vere. Verifica formattazione/cella nel file."
        )

    # Diagnostica duplicati (prime 20)
    if duplicate_codes:
        warnings.append("Dettaglio duplicati (prime 20):")
        for c in duplicate_codes[:20]:
            rows_ = sample_rows_by_code.get(c, [])
            raws_ = sample_raw_codes_by_code.get(c, [])
            warnings.append(f"  DUP '{c}' rows={rows_} raw={raws_}")

    return TotalizationParseResult(
        qty_by_code=qty_by_code,
        warnings=warnings,
        row_count_by_code=row_count_by_code,
        duplicate_codes=duplicate_codes,
        sample_rows_by_code=sample_rows_by_code,
        sample_raw_codes_by_code=sample_raw_codes_by_code,
        sheet_used=sheet_name,
        header_row_1based=used_first_row_1based + header_row_idx,
        code_col_1based=code_col + 1,
        qty_col_1based=qty_col + 1,
        stop_reason=stop_reason,
    )
