# core/parsers/bom_pdf.py
from __future__ import annotations

from pathlib import Path
import logging
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber


# ============================================================
# Header parsing (IT/EN labels)
# ============================================================
# Esempi reali:
#   "Codice / Code   E0255369 01     Revisione / Revision   01"
#   "Codice E0224105 01  Revisione 02"
_RX_CODE_REV = re.compile(
    r"(?:Codice\s*/\s*Code|Codice|Code)\s+([A-Z0-9 _.\-]+?)\s+"
    r"(?:Revisione\s*/\s*Revision|Revisione|Revision)\s+([A-Z0-9]+)",
    re.IGNORECASE,
)
_RX_TITLE = re.compile(r"(?:Descrizione\s*/\s*Description|Descrizione|Description)\s+(.+)", re.IGNORECASE)
_RX_PRINT_DATE = re.compile(
    r"(?:Data di Stampa|Printing Date)\s*/?\s*[\r\n ]*(\d{2}/\d{2}/\d{4})",
    re.IGNORECASE,
)

# Feature flag (safe rollout)
#  - default ON
#  - set BOM_PDF_LAYOUT_FALLBACK=0 to disable
_ENABLE_LAYOUT_FALLBACK = os.getenv("BOM_PDF_LAYOUT_FALLBACK", "1").strip() not in {"0", "false", "False"}
_DEBUG_PDF = os.getenv("BOM_PDF_DEBUG", "0").strip() in {"1", "true", "True"}
_LOG = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================
def _norm(s: Any) -> str:
    return ("" if s is None else str(s)).strip()


def _split_cell(cell: Any) -> List[str]:
    """
    pdfplumber spesso restituisce dati multi-riga in una sola cella con '\n'.
    """
    if cell is None:
        return []
    s = str(cell)
    parts = [p.strip() for p in s.split("\n")]
    return [p for p in parts if p and p.lower() != "null"]


def _normalize_pos(pos: str) -> str:
    p = (pos or "").strip()
    if p.isdigit():
        return p.zfill(4)
    return p

def _looks_like_type_token(t: str) -> bool:
    tt = (t or "").strip().lower()
    if not tt:
        return False
    # token molto corto (es. DD) o parole generiche documentali
    if len(tt) <= 4:
        return True
    return tt in {"disegno", "drawing", "documento", "document", "doc", "dwg"}


def _looks_like_document_type(s: str) -> bool:
    """
    Righe "documentali" (es. Disegno / Drawing / Documento) NON hanno POS numerica.
    In questi casi vogliamo comunque estrarre code/descrizione (qty spesso assente).
    """
    t = (s or "").strip().lower()
    return t in {
        "disegno",
        "drawing",
        "documento",
        "document",
        "doc",
        "dwg",
    }


def _norm_header_token(s: str) -> str:
    """
    Normalizza token di header tabella, per robustezza su encoding strani:
      - lowercase
      - unicode normalize + rimozione diacritici
      - ripara mojibake comuni (es. "Q.tï¿½" ~ "Q.tà")
      - rimuove punctuation più comune e spazi
      - tiene solo [a-z0-9]
    """
    t = (s or "").strip().lower()
    if not t:
        return ""

    # unicode normalize + remove diacritics (q.tà -> q.ta)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))

    # ripara mojibake comuni: "ï¿½" / "�" spesso rappresentano "à"
    t = t.replace("ï¿½", "a").replace("�", "a")

    # togli punteggiatura "u.m." "q.tà" "qt?!" ecc + spazi
    t = re.sub(r"[\s\.\,\;\:\(\)\[\]\{\}\-_/\\\?\!]+", "", t)

    # tieni solo alfanumerico per rendere robusto il matching
    t = re.sub(r"[^a-z0-9]+", "", t)

    return t


def _tokenize_header_row(row: List[Any]) -> List[str]:
    toks: List[str] = []
    for cell in row or []:
        for part in _split_cell(cell):
            part = part.strip()
            if part:
                toks.append(part)
    return toks


# ============================================================
# Header detection (table body header)
# ============================================================
def _looks_like_body_header_row(row: List[Any]) -> bool:
    """
    Riconosce header tabella BOM.
    Deve contenere almeno:
      - pos/item/riga
      - code/codice/materiale
      - qty/quant/qta/qt?
    (um/unit e desc sono un plus)
    """
    tokens = _tokenize_header_row(row)
    joined = " ".join(tokens)
    j = _norm(joined).lower()

    has_pos = any(k in j for k in ("pos", "riga", "item"))
    has_code = any(k in j for k in ("codice", "code", "materiale", "material"))
    # qty permissivo (ma non iper-generico)
    has_qty = ("qty" in j) or ("quant" in j) or ("qta" in j) or (re.search(r"\bq\s*\.?\s*t", j) is not None)
    return bool(has_pos and has_code and has_qty)


# ============================================================
# Column mapping (INTELLIGENT + safe QTY)
# ============================================================
_HEADER_SYNONYMS: Dict[str, str] = {
    # POS
    "pos": "pos",
    "riga": "pos",
    "item": "pos",
    "nr": "pos",
    "n": "pos",
    # TYPE
    "tipo": "type",
    "type": "type",
    # CODE
    "codice": "code",
    "code": "code",
    "materiale": "code",
    "material": "code",
    "pn": "code",
    "partnumber": "code",
    # REV
    "rev": "rev",
    "revisione": "rev",
    "revision": "rev",
    # DESC
    "descrizione": "desc",
    "description": "desc",
    "desc": "desc",
    # UM / UNIT
    "um": "um",
    "um.": "um",
    "u.m": "um",
    "u.m.": "um",
    "unit": "um",
    "unita": "um",
    "unita'": "um",
    "uom": "um",
    # QTY
    "qty": "qty",
    "qta": "qty",
    "qt": "qty",
    "quant": "qty",
    "quantita": "qty",
    "quantita'": "qty",
    # NOTES
    "note": "notes",
    "notes": "notes",
    # MANUFACTURER
    "produttore": "manufacturer",
    "manufacturer": "manufacturer",
    "ditta": "manufacturer",
    "company": "manufacturer",
    "ragsoc": "manufacturer",
    "compname": "manufacturer",
    "ragsoccompname": "manufacturer",
    "ragionesociale": "manufacturer",
    "companyname": "manufacturer",
    # MANUFACTURER CODE
    "codicecostruttore": "manufacturer_code",
    "codiceproduttore": "manufacturer_code",
    "manufacturercode": "manufacturer_code",
    "trade": "manufacturer_code",
}


def _is_qty_header_token(token: str) -> bool:
    """
    Conservative qty detector.
    Accetta varianti rotte tipo:
      - "qt?", "qt?!"
      - "q.tà", "qta", "qty"
      - "quant", "quant."
    Evita falsi positivi su parole lunghe/non pertinenti.
    """
    if not token:
        return False
    t = _norm_header_token(token)
    if not t:
        return False

    if t in {"qty", "qta", "qt", "quant", "quantita"}:
        return True

    if "qt" in t and len(t) <= 7 and (t.startswith("qt") or t.startswith("qta")):
        return True

    if "quant" in t and len(t) <= 10:
        return True

    return False


def _best_guess_key(token: str) -> Optional[str]:
    """
    Converte un token header in chiave canonica, con match sia esatto che fuzzy.
    """
    if not token:
        return None

    t = _norm_header_token(token)
    if not t:
        return None

    if t in _HEADER_SYNONYMS:
        return _HEADER_SYNONYMS[t]

    if _is_qty_header_token(token):
        return "qty"

    if "revis" in t or t == "rev":
        return "rev"
    if "descr" in t or "descrip" in t:
        return "desc"
    if "cod" in t or "material" in t or t == "pn":
        return "code"
    if "pos" in t or "riga" in t or "item" in t:
        return "pos"
    if "um" in t or "unit" in t or "uom" in t:
        return "um"
    if "tipo" in t or "type" in t:
        return "type"
    if "note" in t:
        return "notes"
    if "costrutt" in t or ("manufactur" in t and "code" in t) or "trade" in t:
        return "manufacturer_code"
    if "manufactur" in t or "produtt" in t or "ditta" in t or "company" in t:
        return "manufacturer"

    return None


def _build_col_map(header_row: List[Any]) -> Dict[str, int]:
    """
    Costruisce mapping colonna->indice in base al testo dell'header tabella.
    Non dipende da indici fissi.
    """
    col_map: Dict[str, int] = {}

    for i, cell in enumerate(header_row or []):
        parts = _split_cell(cell)
        if not parts:
            continue

        key: Optional[str] = None
        for p in parts:
            key = _best_guess_key(p)
            if key:
                break

        if not key:
            key = _best_guess_key(" ".join(parts))

        if not key:
            continue

        if key not in col_map:
            col_map[key] = i

    return col_map


def _validate_minimum_colmap(col_map: Dict[str, int]) -> bool:
    """
    Colonne minime indispensabili:
      - code
    """
    return "code" in col_map


# ============================================================
# Text fallback parsing (POS-based, PN-like aware)
# ============================================================
_POS_RE = re.compile(r"^\s*(\d{4})\s+(.*)$")
_PN_LIKE_RE = re.compile(r"^(?:[A-Z]{1,3}\d{4,}|[A-Z]\d{6,}|\d{6,})$", re.IGNORECASE)
_REV_LIKE_RE = re.compile(r"^[A-Z0-9]{1,3}$", re.IGNORECASE)
_QTY_TOKEN_RE = re.compile(r"^\d+(?:[.,]\d+)?$")

_KNOWN_UM = {
    "NR", "N", "PZ", "PZA", "PZI", "PCS", "PC", "EA", "EACH",
    "MM", "CM", "M", "MT", "GR", "G", "KG", "ML", "L", "LT",
}


def _try_extract_um_qty_from_tail(tokens: List[str]) -> Tuple[str, str, List[str]]:
    if not tokens:
        return "", "", tokens

    qty_i = None
    qty = ""
    for i in range(len(tokens) - 1, -1, -1):
        t = tokens[i].strip()
        t2 = t.replace(",", ".")
        if _QTY_TOKEN_RE.match(t2):
            qty_i = i
            qty = t
            break

    if qty_i is None:
        return "", "", tokens

    um = ""
    if qty_i - 1 >= 0:
        cand_um = tokens[qty_i - 1].strip().upper()
        if cand_um in _KNOWN_UM or len(cand_um) <= 4:
            um = tokens[qty_i - 1].strip()
            desc_tokens = tokens[: qty_i - 1]
            return um, qty, desc_tokens

    desc_tokens = tokens[:qty_i]
    return "", qty, desc_tokens


def _parse_lines_from_text(pages_text: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    buf: Optional[str] = None

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        s = buf.strip()
        buf = None

        m = _POS_RE.match(s)
        if not m:
            return

        pos = _normalize_pos(m.group(1))
        tail = (m.group(2) or "").strip()
        if not tail:
            return

        tokens = [t for t in tail.split() if t.strip()]
        if len(tokens) < 1:
            return

        pn_idx: Optional[int] = None
        for i, t in enumerate(tokens):
            if _PN_LIKE_RE.match(t):
                pn_idx = i
                break
        if pn_idx is None:
            return

        code = tokens[pn_idx].strip()
        if not code:
            return

        # ✅ NEW: molti codici sono "E1234567 01" (suffix a 2 cifre parte del codice)
        # Se dopo il PN-like c'è un token a 2 cifre, e dopo ancora una rev-like,
        # allora uniamo nel codice e prendiamo la rev dal terzo token.
        rev = ""
        rest_tokens: List[str]

        if pn_idx + 2 < len(tokens) and re.fullmatch(r"\d{2}", tokens[pn_idx + 1]) and _REV_LIKE_RE.match(tokens[pn_idx + 2]):
            code = f"{code} {tokens[pn_idx + 1].strip()}"
            rev = tokens[pn_idx + 2].strip()
            rest_tokens = tokens[pn_idx + 3 :]
        elif pn_idx + 1 < len(tokens) and _REV_LIKE_RE.match(tokens[pn_idx + 1]):
            rev = tokens[pn_idx + 1].strip()
            rest_tokens = tokens[pn_idx + 2 :]
        else:
            rest_tokens = tokens[pn_idx + 1 :]


        um, qty, desc_tokens = _try_extract_um_qty_from_tail(rest_tokens)
        desc = " ".join(desc_tokens).strip()

        out.append(
            {
                "pos": pos,
                "internal_code": code,
                "rev": rev,
                "description": desc,
                "um": (um or "").strip(),
                "qty": (qty or "").strip() or None,
            }
        )

    for text in pages_text:
        for line in (text or "").splitlines():
            if not line.strip():
                continue
            if _POS_RE.match(line):
                flush()
                buf = line.strip()
            else:
                if buf:
                    buf += " " + line.strip()
                else:
                    # ✅ NEW: prova a catturare righe senza POS (documentali) nel text fallback
                    s = line.strip()
                    tokens = [t for t in s.split() if t.strip()]
                    if len(tokens) < 3:
                        continue

                    pn_idx = None
                    for i, t in enumerate(tokens):
                        if _PN_LIKE_RE.match(t):
                            pn_idx = i
                            break
                    if pn_idx is None:
                        continue

                    # type = token prima del PN-like, se plausibile
                    type_s = tokens[pn_idx - 1].strip() if pn_idx - 1 >= 0 else ""
                    if not _looks_like_type_token(type_s):
                        continue

                    code = tokens[pn_idx].strip()
                    if not code:
                        continue

                    # stesso fix "suffix 01 parte del codice"
                    rev = ""
                    rest_tokens: List[str] = []
                    if pn_idx + 2 < len(tokens) and re.fullmatch(r"\d{2}", tokens[pn_idx + 1]) and _REV_LIKE_RE.match(
                            tokens[pn_idx + 2]):
                        code = f"{code} {tokens[pn_idx + 1].strip()}"
                        rev = tokens[pn_idx + 2].strip()
                        rest_tokens = tokens[pn_idx + 3:]
                    elif pn_idx + 1 < len(tokens) and _REV_LIKE_RE.match(tokens[pn_idx + 1]):
                        rev = tokens[pn_idx + 1].strip()
                        rest_tokens = tokens[pn_idx + 2:]
                    else:
                        rest_tokens = tokens[pn_idx + 1:]

                    # qty/um: per documentali teniamo qty vuota (fedele al PDF)
                    desc = " ".join(rest_tokens).strip()

                    out.append(
                        {
                            "pos": "",
                            "type": type_s,
                            "internal_code": code,
                            "rev": rev,
                            "description": desc,
                            "um": "",
                            "qty": None,
                        }
                    )

    flush()
    return out


# ============================================================
# Table extraction (standard + aggressive)
# ============================================================
TABLE_SETTINGS_AGGRESSIVE = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 20,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
    "intersection_tolerance": 3,
    "text_tolerance": 3,
}


def _get_cell_lines(row: List[Any], idx: int) -> List[str]:
    if idx < 0:
        return []
    cell = row[idx] if idx < len(row) else None
    return _split_cell(cell)


def _align_by_pos(pos_list: List[str], cols: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    n = len(pos_list)

    def norm_len(a: List[str]) -> List[str]:
        if len(a) < n:
            return a + [""] * (n - len(a))
        if len(a) > n:
            return a[:n]
        return a

    pos_list = norm_len(pos_list)
    cols = [norm_len(c) for c in cols]
    return pos_list, cols


def _is_numeric_pos_list(pos_list: List[str]) -> bool:
    if not pos_list:
        return False
    return all((p or "").strip().isdigit() for p in pos_list if (p or "").strip())


def _extract_lines_from_tables(pdf: pdfplumber.PDF, aggressive: bool) -> Tuple[List[Dict[str, Any]], bool, List[str]]:
    """
    Ritorna (lines, found_any_body_table).
    """
    lines: List[Dict[str, Any]] = []
    found_body = False
    debug_notes: List[str] = []


    for page in pdf.pages:
        tables = page.extract_tables(table_settings=TABLE_SETTINGS_AGGRESSIVE) if aggressive else page.extract_tables()
        tables = tables or []

        for table in tables:
            if not table:
                continue

            header_row_idx: Optional[int] = None
            header_row: Optional[List[Any]] = None
            for i, r in enumerate(table):
                if r and _looks_like_body_header_row(r):
                    header_row_idx = i
                    header_row = r
                    break
            if header_row_idx is None or header_row is None:
                continue

            found_body = True
            col_map = _build_col_map(header_row)
            if _DEBUG_PDF:
                debug_notes.append(f"[tables] page={page.page_number} aggressive={int(aggressive)} col_map={col_map}")

            if not _validate_minimum_colmap(col_map):
                continue

            max_needed = max(col_map.values()) if col_map else 0

            for row in table[header_row_idx + 1 :]:
                if not row:
                    continue

                cols_split: List[List[str]] = []
                for ci in range(max_needed + 1):
                    cell = row[ci] if ci < len(row) else None
                    cols_split.append(_split_cell(cell))

                pos_idx = col_map.get("pos", -1)
                pos_list = _get_cell_lines(row, pos_idx) if pos_idx >= 0 else []
                is_numeric_pos = _is_numeric_pos_list(pos_list)

                # righe senza POS numerica: gestisci "documentali" se TYPE è presente e coerente
                if not is_numeric_pos:
                    type_idx = col_map.get("type", -1)
                    type_s = " ".join(_get_cell_lines(row, type_idx)).strip() if type_idx >= 0 else ""

                    code_idx = col_map.get("code", -1)
                    code_s = " ".join(_get_cell_lines(row, code_idx)).strip() if code_idx >= 0 else ""

                    if _looks_like_document_type(type_s) and code_s:
                        rev = " ".join(_get_cell_lines(row, col_map.get("rev", -1))).strip() if "rev" in col_map else ""
                        desc = " ".join(_get_cell_lines(row, col_map.get("desc", -1))).strip() if "desc" in col_map else ""
                        um = " ".join(_get_cell_lines(row, col_map.get("um", -1))).strip() if "um" in col_map else ""
                        qty = None  # IMPORTANT: documentali -> qty deve restare vuoto

                        item: Dict[str, Any] = {
                            "pos": "",
                            "type": type_s,
                            "internal_code": code_s,
                            "rev": rev,
                            "description": desc,
                            "um": um,
                            "qty": qty,
                        }

                        if "notes" in col_map:
                            item["notes"] = " ".join(_get_cell_lines(row, col_map["notes"])).strip()
                        if "manufacturer_code" in col_map:
                            item["manufacturer_code"] = " ".join(_get_cell_lines(row, col_map["manufacturer_code"])).strip()
                        if "manufacturer" in col_map:
                            item["manufacturer"] = " ".join(_get_cell_lines(row, col_map["manufacturer"])).strip()

                        lines.append(item)
                    continue

                # multi-line expansion (POS è la base)
                n = len(pos_list)
                pos_list = [p.strip() for p in pos_list]
                pos_list, cols_split = _align_by_pos(pos_list, cols_split)

                for i in range(n):
                    pos = _normalize_pos((pos_list[i] or "").strip())
                    if not pos:
                        continue

                    code = (cols_split[col_map["code"]][i] or "").strip() if "code" in col_map else ""
                    if not code:
                        continue

                    rev = (cols_split[col_map["rev"]][i] or "").strip() if "rev" in col_map else ""
                    desc = (cols_split[col_map["desc"]][i] or "").strip() if "desc" in col_map else ""
                    um = (cols_split[col_map["um"]][i] or "").strip() if "um" in col_map else ""
                    qraw = (cols_split[col_map["qty"]][i] or "").strip() if "qty" in col_map else ""
                    qty = qraw or None

                    item: Dict[str, Any] = {
                        "pos": pos,
                        "internal_code": code,
                        "rev": rev,
                        "description": desc,
                        "um": um,
                        "qty": qty,
                    }

                    if "notes" in col_map:
                        item["notes"] = (cols_split[col_map["notes"]][i] or "").strip()
                    if "manufacturer_code" in col_map:
                        item["manufacturer_code"] = (cols_split[col_map["manufacturer_code"]][i] or "").strip()
                    if "manufacturer" in col_map:
                        item["manufacturer"] = (cols_split[col_map["manufacturer"]][i] or "").strip()

                    lines.append(item)

    return lines, found_body, debug_notes

# ============================================================
# NEW: Misalignment detection (Rev -> Qty, shift, etc.)
# ============================================================
def _looks_like_misaligned_qty(lines: List[Dict[str, Any]]) -> bool:
    """
    Detector "hard" ma generico:
      - molte righe hanno qty == rev (numerico corto)
      - molte righe hanno qty numerico piccolo ma um vuota
    Se scatta, proviamo layout-based parser.
    """
    if not lines:
        return False

    checked = 0
    suspicious = 0

    for ln in lines[:500]:
        qty = _norm(ln.get("qty"))
        rev = _norm(ln.get("rev"))
        um = _norm(ln.get("um"))

        if not qty:
            continue
        if not re.fullmatch(r"\d{1,3}", qty):
            continue

        checked += 1

        if qty == rev and qty:
            suspicious += 1
            continue

        if not um:
            suspicious += 1

    if checked == 0:
        return False

    return (suspicious / checked) >= 0.35  # abbastanza severo per non rompere i casi buoni


# ============================================================
# NEW: Layout-based parsing (words + x/y)
# ============================================================
def _is_pos_token(s: str) -> bool:
    s = (s or "").strip()
    return bool(re.fullmatch(r"\d{4}", s))


def _header_key_from_word(text: str) -> Optional[str]:
    """
    Mappa una parola di header (singola) in chiave canonica.
    Supporta mojibake e varianti IT/EN.
    """
    t = _norm_header_token(text)
    if not t:
        return None

    # match diretti più comuni (incl. mojibake già normalizzato)
    if t in {"pos", "riga", "item", "nr", "n"}:
        return "pos"
    if t in {"tipo", "type"}:
        return "type"
    if t in {"codice", "code", "materiale", "material", "pn", "partnumber"}:
        return "code"
    if t in {"rev", "revisione", "revision"} or "revis" in t:
        return "rev"
    if t in {"descrizione", "description", "desc"} or "descr" in t or "descrip" in t:
        return "desc"
    if t in {"um", "um.", "um", "unit", "unita", "uom"} or "unit" in t or t.startswith("um"):
        return "um"
    if _is_qty_header_token(text) or t in {"qty", "qta", "qt", "quant", "quantita"}:
        return "qty"
    if t in {"note", "notes"}:
        return "notes"
    if "manufactur" in t and "code" in t:
        return "manufacturer_code"
    if "manufactur" in t or "produtt" in t or "company" in t or "ditta" in t:
        return "manufacturer"
    if t == "trade":
        return "manufacturer_code"

    return None


def _cluster_words_by_y(words: List[Dict[str, Any]], y_tol: float) -> List[List[Dict[str, Any]]]:
    if not words:
        return []
    words = sorted(words, key=lambda w: (float(w["top"]), float(w["x0"])))
    rows: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = [words[0]]
    cur_y = float(words[0]["top"])

    for w in words[1:]:
        y = float(w["top"])
        if abs(y - cur_y) <= y_tol:
            cur.append(w)
        else:
            rows.append(cur)
            cur = [w]
            cur_y = y
    rows.append(cur)
    return rows


def _find_table_header_band(page_words: List[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    """
    Trova la banda Y dell'header tabella BOM.
    Strategia: raggruppa per righe, cerca riga con almeno 3 chiavi tra:
      pos, code, rev, desc, qty
    """
    if not page_words:
        return None

    # stima tolleranza y in modo semplice
    heights = [float(w.get("height", 0.0)) for w in page_words if float(w.get("height", 0.0)) > 0]
    y_tol = 3.0 if not heights else max(2.0, min(4.5, (sum(heights) / len(heights)) * 0.6))

    rows = _cluster_words_by_y(page_words, y_tol=y_tol)
    best_row = None

    for rw in rows:
        keys = set()
        for w in rw:
            k = _header_key_from_word(w.get("text", ""))
            if k:
                keys.add(k)

        score = sum(k in keys for k in ("pos", "code", "rev", "desc", "qty"))
        if ("pos" in keys and "code" in keys and score >= 3):
            best_row = rw
            break

    if not best_row:
        return None

    y0 = min(float(w["top"]) for w in best_row) - 3.0
    y1 = max(float(w["bottom"]) for w in best_row) + 18.0  # include eventuale header multi-line
    return (y0, y1)


def _build_x_columns_from_header(header_words: List[Dict[str, Any]], page_width: float) -> List[Tuple[str, float, float]]:
    """
    Costruisce colonne dinamiche da x0 delle label header.
    Ritorna lista di (key, x_left, x_right) ordinate da sinistra a destra.
    """
    # per ogni key prendiamo il min x0 osservato
    key_x: Dict[str, float] = {}
    for w in header_words:
        k = _header_key_from_word(w.get("text", ""))
        if not k:
            continue
        x0 = float(w["x0"])
        if k not in key_x or x0 < key_x[k]:
            key_x[k] = x0

    if not key_x:
        return []

    ordered = sorted(key_x.items(), key=lambda kv: kv[1])  # (key, x)
    xs = [x for _, x in ordered]
    keys = [k for k, _ in ordered]

    # bounds = midpoints
    bounds: List[float] = [0.0]
    for a, b in zip(xs, xs[1:]):
        bounds.append((a + b) / 2.0)
    bounds.append(float(page_width))

    cols: List[Tuple[str, float, float]] = []
    for i, k in enumerate(keys):
        cols.append((k, bounds[i], bounds[i + 1]))

    return cols


def _assign_row_words_to_cols(row_words: List[Dict[str, Any]], cols: List[Tuple[str, float, float]]) -> Dict[str, str]:
    buckets: Dict[str, List[str]] = {k: [] for (k, _, _) in cols}
    for w in row_words:
        txt = _norm(w.get("text"))
        if not txt:
            continue
        xc = (float(w["x0"]) + float(w["x1"])) / 2.0
        for k, x0, x1 in cols:
            if x0 <= xc < x1:
                buckets[k].append(txt)
                break

    out: Dict[str, str] = {}
    for k, parts in buckets.items():
        s = " ".join(p for p in parts if p)
        s = re.sub(r"\s+", " ", s).strip()
        out[k] = s
    return out


def _extract_lines_from_layout(pdf: pdfplumber.PDF) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Layout-based parser:
      - page.extract_words() -> words with x0/x1/top/bottom
      - detect header band
      - build columns by x-range from header tokens
      - cluster y into rows
      - map words into columns by x center
    """
    warnings: List[str] = []
    lines: List[Dict[str, Any]] = []

    for page_idx, page in enumerate(pdf.pages):
        words = page.extract_words() or []
        if not words:
            continue

        hb = _find_table_header_band(words)
        if not hb:
            continue

        y0, y1 = hb
        header_words = [w for w in words if float(w["top"]) >= y0 and float(w["top"]) <= y1]
        cols = _build_x_columns_from_header(header_words, page.width)

        if not cols:
            warnings.append(f"[layout] page {page_idx+1}: header found but columns not built")
            continue

        # y tolerance for body
        heights = [float(w.get("height", 0.0)) for w in words if float(w.get("height", 0.0)) > 0]
        y_tol = 3.0 if not heights else max(2.0, min(4.5, (sum(heights) / len(heights)) * 0.6))

        # body words: under header; avoid footer area
        body_words = [w for w in words if float(w["top"]) > y1 + 1.0 and float(w["top"]) < float(page.height) - 35.0]
        row_clusters = _cluster_words_by_y(body_words, y_tol=y_tol)

        for rw in row_clusters:
            rw = sorted(rw, key=lambda w: float(w["x0"]))
            if not rw:
                continue

            # filter: must look like a BOM row (pos 4 digits somewhere left)
            first = _norm(rw[0].get("text"))
            has_pos = _is_pos_token(first) or any(_is_pos_token(_norm(w.get("text"))) and float(w.get("x0", 9999)) < 90 for w in rw)
            if not has_pos:
                continue

            mapped = _assign_row_words_to_cols(rw, cols)

            # Build item in SAME schema as existing parser
            pos = _normalize_pos(mapped.get("pos", "").strip())
            code = mapped.get("code", "").strip()
            if not code:
                continue

            item: Dict[str, Any] = {
                "pos": pos,
                "internal_code": code,
                "rev": mapped.get("rev", "").strip(),
                "description": mapped.get("desc", "").strip(),
                "um": mapped.get("um", "").strip(),
                # IMPORTANT: se QTY non c'è nel PDF, resta vuoto
                "qty": (mapped.get("qty", "").strip() or None),
            }

            # Optional fields if present in header
            if "type" in mapped and mapped.get("type", "").strip():
                item["type"] = mapped.get("type", "").strip()
            if "notes" in mapped and mapped.get("notes", "").strip():
                item["notes"] = mapped.get("notes", "").strip()
            if "manufacturer" in mapped and mapped.get("manufacturer", "").strip():
                item["manufacturer"] = mapped.get("manufacturer", "").strip()
            if "manufacturer_code" in mapped and mapped.get("manufacturer_code", "").strip():
                item["manufacturer_code"] = mapped.get("manufacturer_code", "").strip()

            lines.append(item)

    if not lines:
        warnings.append("[layout] Nessuna riga BOM estratta con layout parser.")
    return lines, warnings


# ============================================================
# Public API
# ============================================================
def parse_bom_pdf_raw(path: Path) -> dict:
    header: Dict[str, str] = {"code": "", "rev": "", "title": "", "date": ""}
    warnings: List[str] = []

    with pdfplumber.open(path) as pdf:
        # header: prova su 1-2 pagine (a volte l'header è spezzato)
        header_text = []
        for p in pdf.pages[:2]:
            header_text.append(p.extract_text() or "")
        text = "\n".join(header_text)

        m = _RX_CODE_REV.search(text)
        if m:
            header["code"] = m.group(1).strip()
            header["rev"] = m.group(2).strip()

        mt = _RX_TITLE.search(text)
        if mt:
            header["title"] = mt.group(1).strip()

        md = _RX_PRINT_DATE.search(text)
        if md:
            d = md.group(1).strip()
            try:
                dd, mm, yy = d.split("/")
                header["date"] = f"{yy}-{mm}-{dd}"
            except Exception:
                header["date"] = d

        if not header["code"]:
            raise ValueError(f"BOM PDF senza header riconoscibile (code/rev): {path.name}")

        # 1) Tables standard
        lines, found_body, table_debug = _extract_lines_from_tables(pdf, aggressive=False)
        # 2) Tables aggressive fallback
        if not lines:
            lines, found_body_aggr, table_debug_aggr = _extract_lines_from_tables(pdf, aggressive=True)
            found_body = found_body or found_body_aggr
            table_debug.extend(table_debug_aggr)


        # 3) Text fallback (POS-based)
        #    Nota: questo NON prenderà righe "Disegno" (perché non hanno POS).
        if not lines:
            pages_text = [(p.extract_text() or "") for p in pdf.pages]
            lines = _parse_lines_from_text(pages_text)

        # 4) NEW: Layout fallback (words/x-y) when:
        #    - no lines
        #    - or strong misalignment detected
        if _ENABLE_LAYOUT_FALLBACK and (not lines or _looks_like_misaligned_qty(lines)):
            layout_lines, layout_warn = _extract_lines_from_layout(pdf)
            if layout_lines:
                if lines:
                    warnings.append("Fallback layout-based attivato (incoerenza tabelle: possibile misalignment Rev->Qty).")
                else:
                    warnings.append("Fallback layout-based attivato (nessuna riga da tables/text).")
                warnings.extend(layout_warn)
                lines = layout_lines
            else:
                warnings.extend(layout_warn)

        if _DEBUG_PDF and table_debug:
            warnings.extend(table_debug)
            for n in table_debug:
                _LOG.info(n)


        if not lines:
            if found_body:
                warnings.append("Nessuna riga BOM estratta (tabelle BOM trovate ma parsing fallito).")
            else:
                warnings.append("Nessuna riga BOM estratta (nessuna tabella BOM riconosciuta e text fallback vuoto).")

    return {"header": header, "lines": lines, "warnings": warnings}
