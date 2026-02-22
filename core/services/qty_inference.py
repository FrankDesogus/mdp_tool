from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -------------------------
# Output model
# -------------------------
@dataclass(frozen=True)
class InferredQtyRow:
    parent_code: str
    parent_rev: str
    child_code: str
    pos: str

    # NEW (non rompe nulla: default vuoti)
    unit: str = ""
    kind: str = ""
    parent_bom_path: str = ""

    original_qty: str = ""  # sempre "" qui (mancante) ma utile per CSV uniformi
    inferred_qty: str = ""  # già formattata
    status: str = ""        # "INFERRED" | "AMBIGUOUS" | "NOT_CALCULABLE" | "CHECK"
    method: str = ""        # breve etichetta del metodo usato
    confidence: float = 0.0 # 0..1
    note: str = ""          # note per diagnostica


# -------------------------
# Parsing warning "Qty mancante ..." (fallback)
# -------------------------
_QTY_MISSING_RE = re.compile(
    r"""
    Qty\ mancante:\s*
    parent=(?P<pcode>.+?)\s+rev\s+(?P<prev>.+?)\s*
    ->\s*child=(?P<ccode>.+?)\s*
    \(pos=(?P<pos>.*?)\)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _parse_missing_qty_warnings(warnings: Iterable[str]) -> List[Tuple[str, str, str, str]]:
    """
    Estrae (parent_code, parent_rev, child_code, pos) dai warning dell'exploder.
    Ritorna solo quelli matchati.
    """
    out: List[Tuple[str, str, str, str]] = []
    for w in warnings or []:
        s = (w or "").strip()
        m = _QTY_MISSING_RE.search(s)
        if not m:
            continue
        out.append(
            (
                (m.group("pcode") or "").strip(),
                (m.group("prev") or "").strip(),
                (m.group("ccode") or "").strip(),
                (m.group("pos") or "").strip(),
            )
        )
    return out


# -------------------------
# Decimal helpers
# -------------------------
def _to_decimal(x: object) -> Optional[Decimal]:
    if x is None:
        return None
    if isinstance(x, Decimal):
        return x
    try:
        s = str(x).strip()
        if not s:
            return None
        s = s.replace(",", ".")
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None


def _is_close(a: Decimal, b: Decimal, tol: Decimal = Decimal("0.02")) -> bool:
    return abs(a - b) <= tol


def _fmt_decimal(d: Decimal) -> str:
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


# -------------------------
# Quantizzazione per UoM
# -------------------------
_DEFAULT_STEPS = (Decimal("1"), Decimal("0.5"), Decimal("0.25"), Decimal("0.1"), Decimal("0.01"))

# policy semplice: puoi estenderla quando vuoi (MT, MM, CM, KG, ecc.)
_STEPS_BY_UOM: Dict[str, Tuple[Decimal, ...]] = {
    "NR": (Decimal("1"),),  # solo interi
    # esempio: per metri spesso ha senso 0.01 o 0.001, ma dipende dal tuo dominio
    "MT": (Decimal("0.1"), Decimal("0.01"), Decimal("0.001")),
}


def _quantize_by_steps(x: Decimal, steps: Tuple[Decimal, ...], tol: Decimal = Decimal("0.02")) -> Tuple[Decimal, str, str]:
    """
    Prova a "snappare" x su una lista di step ammessi.
    Ritorna: (snapped_value, flag, step_label)
      flag: OK_STEP | CHECK
      step_label: es. "1", "0.25", "0.01", "RAW"
    """
    if x is None:
        return Decimal("0"), "CHECK", "RAW"

    # prova ogni step in ordine (dal più grande al più piccolo o come passata)
    for step in steps:
        if step <= 0:
            continue
        q = (x / step).to_integral_value(rounding="ROUND_HALF_UP") * step
        if _is_close(x, q, tol=tol):
            return q, "OK_STEP", _fmt_decimal(step)

    # fallback: 3 decimali e CHECK
    return x.quantize(Decimal("0.001")), "CHECK", "RAW"


def _steps_for_uom(uom: str) -> Tuple[Decimal, ...]:
    u = (uom or "").strip().upper()
    if not u:
        return _DEFAULT_STEPS
    return _STEPS_BY_UOM.get(u, _DEFAULT_STEPS)


# -------------------------
# Code normalization helpers
# -------------------------
_SUFFIXES_FOR_LOOKUP = ("ASSY", "PRTL", "ASSD", "ELTD", "BLKD", "DS")


def _normalize_code_for_lookup(code: str) -> str:
    c = (code or "").strip()
    for suf in _SUFFIXES_FOR_LOOKUP:
        if c.endswith(suf) and len(c) > len(suf):
            return c[: -len(suf)]
    return c


# -------------------------
# Occurrences(parent) from PBS (override)
# -------------------------
def compute_occurrences_from_pbs(pbs: object) -> Dict[str, Decimal]:
    occ: Dict[str, Decimal] = {}

    if pbs is None or not getattr(pbs, "rows", None):
        return occ

    rows = list(getattr(pbs, "rows", []) or [])
    if not rows:
        return occ

    stack: List[Tuple[int, Decimal]] = []

    for r in rows:
        code = str(getattr(r, "code", "") or "").strip()
        if not code:
            continue

        level = int(getattr(r, "level", 0) or 0)

        qty = _to_decimal(getattr(r, "qty", None))
        if qty is None or qty <= 0:
            qty = Decimal("1")

        while stack and stack[-1][0] >= level:
            stack.pop()

        if not stack:
            occ_here = Decimal("1") * qty
        else:
            parent_occ = stack[-1][1]
            occ_here = parent_occ * qty

        occ[code] = occ.get(code, Decimal("0")) + occ_here
        norm = _normalize_code_for_lookup(code)
        if norm != code:
            occ[norm] = occ.get(norm, Decimal("0")) + occ_here

        stack.append((level, occ_here))

    return occ


# -------------------------
# Occurrences(parent) "safe" from explosion
# -------------------------
def _compute_occurrences_safe(explosion: object) -> Dict[str, Decimal]:
    occ: Dict[str, Decimal] = {}

    root_code = str(getattr(explosion, "root_code", "") or "").strip()
    if root_code:
        occ[root_code] = Decimal("1")

    for e in getattr(explosion, "edges", []) or []:
        child_code = str(getattr(e, "child_code", "") or "").strip()
        qty = _to_decimal(getattr(e, "qty", None))
        if not child_code or qty is None:
            continue
        occ[child_code] = occ.get(child_code, Decimal("0")) + qty

        norm = _normalize_code_for_lookup(child_code)
        if norm != child_code:
            occ[norm] = occ.get(norm, Decimal("0")) + qty

    return occ


def _child_parent_index(
    explosion: object,
    missing_edges: Optional[Iterable[object]] = None,
) -> Tuple[Dict[str, set], Dict[Tuple[str, str], int]]:
    """
    parents_by_child e count_by_parent_child calcolati:
      - dagli edge "safe" (qty nota)
      - + dai missing edges (qty mancante) se forniti

    Questo evita un bug importante: un child con qty sempre mancante
    non apparirebbe mai negli edge => sembrerebbe "unique parent" anche se non lo è.
    """
    parents_by_child: Dict[str, set] = {}
    count_by_parent_child: Dict[Tuple[str, str], int] = {}

    # safe edges
    for e in getattr(explosion, "edges", []) or []:
        p = str(getattr(e, "parent_code", "") or "").strip()
        c = str(getattr(e, "child_code", "") or "").strip()
        if not p or not c:
            continue
        parents_by_child.setdefault(c, set()).add(p)
        count_by_parent_child[(p, c)] = count_by_parent_child.get((p, c), 0) + 1

    # missing edges (structured)
    for me in (missing_edges or []):
        p = str(getattr(me, "parent_code", "") or "").strip()
        c = str(getattr(me, "child_code", "") or "").strip()
        if not p or not c:
            continue
        parents_by_child.setdefault(c, set()).add(p)
        count_by_parent_child[(p, c)] = count_by_parent_child.get((p, c), 0) + 1

    return parents_by_child, count_by_parent_child


# -------------------------
# Public API
# -------------------------
def infer_missing_edge_quantities(
    explosion: object,
    totals_override: Optional[Dict[str, object]] = None,
    occ_override: Optional[Dict[str, Decimal]] = None,
) -> List[InferredQtyRow]:
    """
    Inferisce qty mancanti basandosi su:
      inferred ≈ total(child) / occurrences(parent)

    total(child) viene preso da:
      - totals_override (se fornita)  [prioritario]
      - explosion.qty_by_code         [fallback]

    occurrences(parent) viene preso da:
      - occ_override (es. calcolata dal PBS) [prioritario]
      - safe occ dall'explosion (solo edge con qty nota)
      - fallback qty_by_code[parent]
      - root fallback

    Sicurezza:
    - se child ha più parent -> AMBIGUOUS
    - se (parent, child) compare più volte -> AMBIGUOUS
    - se total(child) o occurrences(parent) non disponibili -> NOT_CALCULABLE

    Quantizzazione:
    - usa steps dipendenti da UoM (NR => interi; altri => griglia)
    """
    if explosion is None:
        return []

    # ✅ Preferisci dataset strutturato dall'exploder (nuovo)
    structured_missing = list(getattr(explosion, "missing_edges", []) or [])

    # fallback legacy: parse warnings
    legacy_missing = _parse_missing_qty_warnings(getattr(explosion, "warnings", []) or [])

    # costruisci lista uniforme
    # item = (parent_code, parent_rev, child_code, pos, unit, kind, bom_path, source)
    missing: List[Tuple[str, str, str, str, str, str, str, str]] = []

    if structured_missing:
        for me in structured_missing:
            missing.append(
                (
                    str(getattr(me, "parent_code", "") or "").strip(),
                    str(getattr(me, "parent_rev", "") or "").strip(),
                    str(getattr(me, "child_code", "") or "").strip(),
                    str(getattr(me, "pos", "") or "").strip(),
                    str(getattr(me, "unit", "") or "").strip(),
                    str(getattr(me, "kind", "") or "").strip(),
                    str(getattr(me, "parent_bom_path", "") or "").strip(),
                    "structured",
                )
            )
    else:
        for pcode, prev, ccode, pos in legacy_missing:
            missing.append((pcode, prev, ccode, pos, "", "", "", "legacy_warning"))

    if not missing:
        return []

    # base totals da explode + override esterno
    base_qty_by_code: Dict[str, object] = getattr(explosion, "qty_by_code", {}) or {}
    qty_by_code: Dict[str, object] = dict(base_qty_by_code)

    if totals_override:
        for k, v in totals_override.items():
            kk = str(k).strip()
            if kk:
                qty_by_code[kk] = v

    safe_occ = _compute_occurrences_safe(explosion)

    # ⚠️ importante: multi-parent detection deve considerare anche missing_edges
    parents_by_child, count_by_parent_child = _child_parent_index(explosion, missing_edges=structured_missing)

    rows: List[InferredQtyRow] = []

    for parent_code, parent_rev, child_code, pos, unit, kind, bom_path, src in missing:
        # --- ambiguity checks (su codici as-is)
        pset = parents_by_child.get(child_code, set())
        if len(pset) >= 2:
            rows.append(
                InferredQtyRow(
                    parent_code=parent_code,
                    parent_rev=parent_rev,
                    child_code=child_code,
                    pos=pos,
                    unit=unit,
                    kind=kind,
                    parent_bom_path=bom_path,
                    original_qty="",
                    inferred_qty="",
                    status="AMBIGUOUS",
                    method="total(child)/occ(parent)",
                    confidence=0.0,
                    note=f"Child sotto più parent: {sorted(pset)} | src={src}",
                )
            )
            continue

        if count_by_parent_child.get((parent_code, child_code), 0) >= 2:
            rows.append(
                InferredQtyRow(
                    parent_code=parent_code,
                    parent_rev=parent_rev,
                    child_code=child_code,
                    pos=pos,
                    unit=unit,
                    kind=kind,
                    parent_bom_path=bom_path,
                    original_qty="",
                    inferred_qty="",
                    status="AMBIGUOUS",
                    method="total(child)/occ(parent)",
                    confidence=0.0,
                    note="Coppia (parent,child) presente in più righe -> divisione non univoca",
                )
            )
            continue

        # --- total(child)
        total_child = _to_decimal(qty_by_code.get(child_code))
        total_child_src = "qty_by_code[child]"

        if total_child is None or total_child <= 0:
            c_norm = _normalize_code_for_lookup(child_code)
            if c_norm != child_code:
                total_child = _to_decimal(qty_by_code.get(c_norm))
                if total_child is not None and total_child > 0:
                    total_child_src = f"qty_by_code[normalize(child)={c_norm}]"

        if total_child is None or total_child <= 0:
            rows.append(
                InferredQtyRow(
                    parent_code=parent_code,
                    parent_rev=parent_rev,
                    child_code=child_code,
                    pos=pos,
                    unit=unit,
                    kind=kind,
                    parent_bom_path=bom_path,
                    original_qty="",
                    inferred_qty="",
                    status="NOT_CALCULABLE",
                    method="total(child)/occ(parent)",
                    confidence=0.0,
                    note="total(child) mancante/non valido (né totals_override né explosion.qty_by_code)",
                )
            )
            continue

        # --- occurrences(parent)
        occ_parent: Optional[Decimal] = None
        occ_src = ""

        # 1) PBS override
        if occ_override:
            occ_parent = occ_override.get(parent_code)
            occ_src = "pbs_occ[parent]"
            if occ_parent is None or occ_parent <= 0:
                p_norm = _normalize_code_for_lookup(parent_code)
                if p_norm != parent_code:
                    occ_parent = occ_override.get(p_norm)
                    if occ_parent is not None and occ_parent > 0:
                        occ_src = f"pbs_occ[normalize(parent)={p_norm}]"

        # 2) safe occ da explosion (edge con qty nota)
        if occ_parent is None or occ_parent <= 0:
            occ_parent = safe_occ.get(parent_code)
            occ_src = "safe_occ[parent]"
            if occ_parent is None or occ_parent <= 0:
                p_norm = _normalize_code_for_lookup(parent_code)
                if p_norm != parent_code:
                    occ_parent = safe_occ.get(p_norm)
                    if occ_parent is not None and occ_parent > 0:
                        occ_src = f"safe_occ[normalize(parent)={p_norm}]"

        # 3) fallback: se parent compare nella totalizzazione come componente
        if occ_parent is None or occ_parent <= 0:
            occ_parent = _to_decimal(qty_by_code.get(parent_code))
            occ_src = "qty_by_code[parent]"
            if occ_parent is None or occ_parent <= 0:
                p_norm = _normalize_code_for_lookup(parent_code)
                if p_norm != parent_code:
                    occ_parent = _to_decimal(qty_by_code.get(p_norm))
                    if occ_parent is not None and occ_parent > 0:
                        occ_src = f"qty_by_code[normalize(parent)={p_norm}]"

        # 4) root fallback
        if occ_parent is None or occ_parent <= 0:
            root_code = str(getattr(explosion, "root_code", "") or "").strip()
            if parent_code == root_code:
                occ_parent = Decimal("1")
                occ_src = "root=1"

        if occ_parent is None or occ_parent <= 0:
            rows.append(
                InferredQtyRow(
                    parent_code=parent_code,
                    parent_rev=parent_rev,
                    child_code=child_code,
                    pos=pos,
                    unit=unit,
                    kind=kind,
                    parent_bom_path=bom_path,
                    original_qty="",
                    inferred_qty="",
                    status="NOT_CALCULABLE",
                    method="total(child)/occ(parent)",
                    confidence=0.0,
                    note=f"occurrences(parent) non calcolabile (parent='{parent_code}', norm='{_normalize_code_for_lookup(parent_code)}')",
                )
            )
            continue

        # --- inference
        inferred_raw = total_child / occ_parent

        steps = _steps_for_uom(unit)
        quantized, qflag, step_used = _quantize_by_steps(inferred_raw, steps=steps, tol=Decimal("0.02"))
        inferred_str = _fmt_decimal(quantized)

        if qflag == "CHECK":
            status = "CHECK"
            confidence = 0.4
            note = (
                f"Snapping non stabile su steps={','.join(_fmt_decimal(s) for s in steps)} "
                f"(raw={_fmt_decimal(inferred_raw)} -> {inferred_str}) | "
                f"UoM={unit or '-'} step=RAW | total_src={total_child_src} | occ_src={occ_src} | src={src}"
            )
        else:
            # confidenza base alta; se UoM non nota, abbassa un filo
            if (unit or "").strip().upper() == "NR":
                confidence = 0.90
            elif unit:
                confidence = 0.80
            else:
                confidence = 0.70

            status = "INFERRED"
            note = (
                f"Snapped (raw={_fmt_decimal(inferred_raw)} -> {inferred_str}) "
                f"UoM={unit or '-'} step={step_used} | total_src={total_child_src} | occ_src={occ_src} | src={src}"
            )

        rows.append(
            InferredQtyRow(
                parent_code=parent_code,
                parent_rev=parent_rev,
                child_code=child_code,
                pos=pos,
                unit=unit,
                kind=kind,
                parent_bom_path=bom_path,
                original_qty="",
                inferred_qty=inferred_str,
                status=status,
                method="total(child)/occ(parent)",
                confidence=confidence,
                note=note,
            )
        )

    return rows


def write_inferred_qty_csv(rows: Sequence[InferredQtyRow], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "parent_code",
                "parent_rev",
                "child_code",
                "pos",
                "unit",
                "kind",
                "parent_bom_path",
                "original_qty",
                "inferred_qty",
                "status",
                "method",
                "confidence",
                "note",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.parent_code,
                    r.parent_rev,
                    r.child_code,
                    r.pos,
                    r.unit,
                    r.kind,
                    r.parent_bom_path,
                    r.original_qty,
                    r.inferred_qty,
                    r.status,
                    r.method,
                    f"{r.confidence:.2f}",
                    r.note,
                ]
            )
