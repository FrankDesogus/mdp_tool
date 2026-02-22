# core/parsers/bom_pdf.py
from __future__ import annotations

from pathlib import Path
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
    # (scelta conservativa: meglio 'a' che un carattere rotto che impedisce il match)
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
    "quantita": "qty",
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

    # match diretti
    if t in {"qty", "qta", "qt", "quant", "quantita"}:
        return True

    # "qt?" / "qt?!" / "qta?" dopo norm -> "qt" o "qta"
    # regola: contiene "qt" ed è corto
    if "qt" in t and len(t) <= 7 and (t.startswith("qt") or t.startswith("qta")):
        return True

    # "quant" ecc (ma non troppo lungo)
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

    # match diretto
    if t in _HEADER_SYNONYMS:
        return _HEADER_SYNONYMS[t]

    # qty: detector conservativo dedicato
    if _is_qty_header_token(token):
        return "qty"

    # match parziale per casi comuni
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
    if "manufactur" in t or "produtt" in t or "ditta" in t or "company" in t:
        return "manufacturer"
    if "costrutt" in t or ("manufactur" in t and "code" in t) or "trade" in t:
        return "manufacturer_code"

    return None


def _build_col_map(header_row: List[Any]) -> Dict[str, int]:
    """
    Costruisce mapping colonna->indice in base al testo dell'header tabella.
    Non dipende da indici fissi.

    Strategia:
      - per ogni colonna i, collassa testo cella (anche multiline)
      - prova a riconoscere 1 key canonica
      - prima occorrenza vince
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
    qty/pos possono mancare in alcuni template: qty resta None, pos può essere gestito diversamente.
    """
    return "code" in col_map


# ============================================================
# Text fallback parsing (POS-based, PN-like aware)
# ============================================================
# Prima avevate:
#   POS CODE REV RESTO...
# Ma in BOM con colonna TYPE, il testo è spesso:
#   0030 Minuteria C0020628 03 DESC...
# Quindi qui cerchiamo il primo PN-like token dopo la POS.

_POS_RE = re.compile(r"^\s*(\d{4})\s+(.*)$")

# abbastanza conservativo:
# - E... / C... / Q... ecc con molte cifre
# - oppure numerico lungo
_PN_LIKE_RE = re.compile(
    r"^(?:[A-Z]{1,3}\d{4,}|[A-Z]\d{6,}|\d{6,})$",
    re.IGNORECASE,
)

# rev tipica: 01, 02, A, B, A1, 10 ...
_REV_LIKE_RE = re.compile(r"^[A-Z0-9]{1,3}$", re.IGNORECASE)

_QTY_TOKEN_RE = re.compile(r"^\d+(?:[.,]\d+)?$")

_KNOWN_UM = {
    "NR",
    "N",
    "PZ",
    "PZA",
    "PZI",
    "PCS",
    "PC",
    "EA",
    "EACH",
    "MM",
    "CM",
    "M",
    "MT",
    "GR",
    "G",
    "KG",
    "ML",
    "L",
    "LT",
}


def _try_extract_um_qty_from_tail(tokens: List[str]) -> Tuple[str, str, List[str]]:
    """
    Cerca (UM, QTY) dalla coda della riga.
    Ritorna: (um, qty_string, desc_tokens)
    Se non li trova, um/qty vuoti e desc_tokens = tokens
    """
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
    """
    Fallback: estrae righe in base a POS=4 cifre.
    Gestisce continuazioni di description su righe successive.
    Robust: cerca il primo token PN-like dopo la POS (evita "MINUTERIA" come code).
    """
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

        rev = ""
        rest_tokens: List[str]
        if pn_idx + 1 < len(tokens) and _REV_LIKE_RE.match(tokens[pn_idx + 1]):
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


def _extract_lines_from_tables(pdf: pdfplumber.PDF, aggressive: bool) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Ritorna (lines, found_any_body_table).
    """
    lines: List[Dict[str, Any]] = []
    found_body = False

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
                        # IMPORTANT: per righe documentali NON trattiamo mai qraw come qty (spesso è Rif.*)
                        # qraw = " ".join(_get_cell_lines(row, col_map.get("qty", -1))).strip() if "qty" in col_map else ""
                        qty = None

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

    return lines, found_body


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
        lines, found_body = _extract_lines_from_tables(pdf, aggressive=False)

        # 2) Tables aggressive fallback
        if not lines:
            lines, found_body_aggr = _extract_lines_from_tables(pdf, aggressive=True)
            found_body = found_body or found_body_aggr

        # 3) Text fallback (POS-based)
        #    Nota: questo NON prenderà righe "Disegno" (perché non hanno POS).
        if not lines:
            pages_text = [(p.extract_text() or "") for p in pdf.pages]
            lines = _parse_lines_from_text(pages_text)

        if not lines:
            if found_body:
                warnings.append("Nessuna riga BOM estratta (tabelle BOM trovate ma parsing fallito).")
            else:
                warnings.append("Nessuna riga BOM estratta (nessuna tabella BOM riconosciuta e text fallback vuoto).")

    return {"header": header, "lines": lines, "warnings": warnings}
