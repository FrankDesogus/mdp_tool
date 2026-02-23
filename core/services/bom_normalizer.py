# mdp_tool/core/services/bom_normalizer.py
from __future__ import annotations

import math
import os
import re
import logging
from pathlib import Path
from typing import Optional, Iterable, List

from core.domain.models import (
    BomLineKind,
    NormalizedBomLine,
    BomHeader,
    BomDocument,
)

# ✅ NEW: alias/canonicalizzazione codici (external -> internal)
from core.services.code_aliases import canonicalize_code

_LOG = logging.getLogger(__name__)
_DEBUG_DIAG = os.getenv("MDP_DEBUG_DIAGNOSTICS", "0").strip() in {"1", "true", "True"}


def _clean_key_text(value: object) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# --- regole base, estendibili ---
_REF_UNITS = {"REF", "REFERENCE", "DOC", "DOCUMENT", "DWG"}


def parse_qty(value) -> Optional[float]:
    """
    Parsing robusto quantità:
    - None / "" / "-" => None
    - numeri excel -> float
    - stringhe "1,00" "1.00" " 2 " => float
    - stringhe non numeriche => None
    """
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s or s == "-":
        return None

    # normalizza "1,00" -> "1.00"
    s = s.replace(",", ".")
    # rimuovi spazi e caratteri non numerici "soft" (es. "1.00 ")
    s = re.sub(r"\s+", "", s)

    try:
        return float(s)
    except ValueError:
        return None


def normalize_unit(u: Optional[str]) -> str:
    return (u or "").strip().upper()


def classify_line(qty: Optional[float], unit: str, line_type: str = "", description: str = "") -> BomLineKind:
    """
    Regole:
    - TYPE documentale (es. Disegno/Drawing/Documento/DWG/DD) => DOCUMENT_REF
    - UM in REF_UNITS => DOCUMENT_REF (anche se qty manca)
    - qty è None => UNKNOWN
    - qty presente => MATERIAL
    """
    t = (line_type or "").strip().upper()
    d = (description or "").strip().upper()

    # tipo documentale (robusto anche se UM/qty mancano o le colonne sono "sporche")
    if t in {"DD", "DOC", "DOCUMENTO", "DISEGNO", "DRAWING", "DWG"}:
        return BomLineKind.DOCUMENT_REF
    if "DISEGNO" in d or "DRAWING" in d or "DOCUMENT" in d:
        return BomLineKind.DOCUMENT_REF

    if unit in _REF_UNITS:
        return BomLineKind.DOCUMENT_REF
    if qty is None:
        return BomLineKind.UNKNOWN
    return BomLineKind.MATERIAL


def build_bom_document(
    *,
    path: Path,
    header_code: str,
    header_rev: str,
    header_title: str = "",
    doc_date_iso: str = "",
    raw_lines: Iterable[dict],
) -> BomDocument:
    # ✅ NEW: canonicalizza header_code (codice esterno -> codice interno)
    header_code = canonicalize_code((header_code or "").strip())

    header = BomHeader(
        code=(header_code or "").strip(),
        revision=(header_rev or "").strip(),
        title=(header_title or "").strip(),
        doc_date_iso=(doc_date_iso or "").strip(),
    )

    lines: List[NormalizedBomLine] = []
    raw_count = 0
    raw_with_mfr = 0

    for raw_line in raw_lines:
        if not isinstance(raw_line, dict):
            raise TypeError(f"BOM normalizer: riga non-dict: {type(raw_line)!r}")

        pos = _clean_key_text(raw_line.get("pos") or "")

        # NEW: tipo riga (per distinguere documenti vs materiali)
        line_type = _clean_key_text(raw_line.get("type") or "")

        # component code nella tabella = INTERNAL CODE
        internal_code = _clean_key_text(
            raw_line.get("internal_code")
            or raw_line.get("code")          # fallback eventuale
            or ""
        )

        # ✅ NEW: canonicalizza internal_code (codice esterno -> codice interno)
        if internal_code:
            internal_code = canonicalize_code(internal_code)

        description = str(raw_line.get("description") or "").strip()
        rev = _clean_key_text(raw_line.get("rev") or "")

        unit = normalize_unit(_clean_key_text(raw_line.get("um") or raw_line.get("unit")))
        qty = parse_qty(raw_line.get("qty"))

        # NEW: classificazione anche su base "type" / descrizione
        kind = classify_line(qty, unit, line_type=line_type, description=description)

        # extra colonne reali del tuo Excel
        val = str(raw_line.get("val") or "").strip()
        rat = str(raw_line.get("rat") or "").strip()
        tol = str(raw_line.get("tol") or "").strip()
        refdes = str(raw_line.get("refdes") or "").strip()
        tecn = str(raw_line.get("tecn") or "").strip()
        notes = str(raw_line.get("notes") or "").strip()

        manufacturer = str(raw_line.get("manufacturer") or "").strip()
        manufacturer_code = str(raw_line.get("manufacturer_code") or "").strip()
        raw_count += 1
        if manufacturer:
            raw_with_mfr += 1
        lines.append(
            NormalizedBomLine(
                pos=pos,
                internal_code=internal_code,
                description=description,
                qty=qty,
                unit=unit,
                kind=kind,
                type=line_type,
                val=val,
                rat=rat,
                tol=tol,
                refdes=refdes,
                tecn=tecn,
                notes=notes,
                manufacturer=manufacturer,
                manufacturer_code=manufacturer_code,
                rev=rev,
            )
        )

    if _DEBUG_DIAG:
        norm_with_mfr = sum(1 for ln in lines if (ln.manufacturer or "").strip())
        pct_raw = (100.0 * raw_with_mfr / raw_count) if raw_count else 0.0
        pct_norm = (100.0 * norm_with_mfr / len(lines)) if lines else 0.0
        _LOG.info("[diag] BOM manufacturer coverage: raw=%s/%s (%.1f%%) normalized=%s/%s (%.1f%%) file=%s", raw_with_mfr, raw_count, pct_raw, norm_with_mfr, len(lines), pct_norm, path.name)


    return BomDocument(
        path=path,
        header=header,
        lines=lines,
    )
