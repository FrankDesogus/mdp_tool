# core/services/parent_child_compare.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from core.domain.models import (
    BomDocument,
    BomLineKind,
    LinkResult,
    LinkStatus,
    PbsDocument,
)
from core.services.bom_prefilter import normalize_alnum


# -----------------------------------------------------------------------------
# Heuristics / Domain rules
# -----------------------------------------------------------------------------
# IMPORTANT: questi suffissi sono "documentali" nel tuo PBS: esistono in PBS ma
# non devono essere richiesti in BOM come child "materiale".
#
# Li ho ricavati dai tuoi tree PBS (ASSD/PRTL/ELTD/MCHD/OUTL/3D_S/USRM/TCHS/GRBF/PCBD/POST/POSB/MFGS...)
# e li ho resi estendibili senza toccare l'algoritmo.
DOCUMENT_SUFFIXES: Tuple[str, ...] = (
    "ASSD",
    "MCHD",
    "PRTL",
    "ELTD",
    "OUTL",
    "3D_S",
    "USRM",
    "TCHS",
    "GRBF",
    "PCBD",
    "POST",
    "POSB",
    "MFGS",
)

# Se in futuro vuoi davvero "optional", qui puoi aggiungere suffissi specifici.
# Per ora non lo usiamo per evitare false classificazioni.
OPTIONAL_SUFFIXES: Tuple[str, ...] = ()


def _norm_code(code: str) -> str:
    return normalize_alnum(code or "")


def _is_zero_qty(qty: object) -> bool:
    """
    PBS qty può essere str/int/float/Decimal.
    Trattiamo 0 / "0" / "0.0" come zero.
    """
    if qty is None:
        return False
    try:
        if isinstance(qty, Decimal):
            return qty == Decimal(0)
    except Exception:
        pass
    try:
        # numeri
        if isinstance(qty, (int, float)):
            return float(qty) == 0.0
    except Exception:
        pass
    try:
        s = str(qty).strip().replace(",", ".")
        if not s:
            return False
        return float(s) == 0.0
    except Exception:
        return False


def _looks_like_document(code: str) -> bool:
    c = _norm_code(code)
    if not c:
        return False
    return any(c.endswith(suf) for suf in DOCUMENT_SUFFIXES)


def _looks_optional(code: str) -> bool:
    c = _norm_code(code)
    if not c:
        return False
    return any(c.endswith(suf) for suf in OPTIONAL_SUFFIXES)


def _levenshtein_leq(a: str, b: str, max_dist: int = 1) -> Tuple[bool, int]:
    """
    (ok, dist) con ok=True se dist <= max_dist.
    Ottimizzato per max_dist piccolo (qui usiamo 1).
    """
    a = a or ""
    b = b or ""
    if a == b:
        return True, 0
    if abs(len(a) - len(b)) > max_dist:
        return False, max_dist + 1

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        min_row = cur[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(
                prev[j] + 1,        # delete
                cur[j - 1] + 1,     # insert
                prev[j - 1] + cost  # subst
            ))
            if cur[j] < min_row:
                min_row = cur[j]
        if min_row > max_dist:
            return False, max_dist + 1
        prev = cur

    dist = prev[-1]
    return (dist <= max_dist), dist


# -----------------------------------------------------------------------------
# PBS relationship extraction (parent → children) by level-stack
# -----------------------------------------------------------------------------
def build_pbs_parent_children(
    pbs: PbsDocument
) -> Tuple[
    Dict[str, Set[str]],        # children_by_parent
    Dict[str, int],             # src_row_by_code
    Dict[str, str],             # rev_by_code
    Dict[str, Optional[str]],   # parent_by_child (immediate)
    Dict[str, object],          # qty_by_code (raw qty)
]:
    children_by_parent: Dict[str, Set[str]] = {}
    src_row_by_code: Dict[str, int] = {}
    rev_by_code: Dict[str, str] = {}
    parent_by_child: Dict[str, Optional[str]] = {}
    qty_by_code: Dict[str, object] = {}

    stack: List[Tuple[int, str]] = []
    rows = list(getattr(pbs, "rows", []) or [])
    for r in rows:
        code = _norm_code(getattr(r, "code", "") or "")
        if not code:
            continue

        level = int(getattr(r, "level", 0) or 0)
        src_row_by_code[code] = int(getattr(r, "src_row", -1) or -1)
        rev_by_code[code] = (getattr(r, "rev", "") or "").strip()
        qty_by_code[code] = getattr(r, "qty", None)

        while stack and stack[-1][0] >= level:
            stack.pop()

        parent_code: Optional[str] = stack[-1][1] if stack else None
        parent_by_child[code] = parent_code

        if parent_code:
            children_by_parent.setdefault(parent_code, set()).add(code)

        stack.append((level, code))

    return children_by_parent, src_row_by_code, rev_by_code, parent_by_child, qty_by_code


def has_unexploded_ancestor(code: str, parent_by_child: Dict[str, Optional[str]], unexploded: Set[str]) -> bool:
    cur = code
    seen = set()
    while True:
        if cur in seen:
            return False
        seen.add(cur)
        parent = parent_by_child.get(cur)
        if not parent:
            return False
        if parent in unexploded:
            return True
        cur = parent


# -----------------------------------------------------------------------------
# BOM children extraction (per parent)
# -----------------------------------------------------------------------------
def bom_children_set(bom: BomDocument) -> Tuple[Set[str], Dict[str, List[str]]]:
    """
    Set di child codes (normalizzati) dalle righe BOM.
    Esclude DOCUMENT_REF e codici vuoti.
    Nota: qui non filtro document-like: lo facciamo DOPO, in modo simmetrico con PBS.
    """
    out: Set[str] = set()
    pos_by_child: Dict[str, List[str]] = {}

    for line in (getattr(bom, "lines", []) or []):
        kind = getattr(line, "kind", BomLineKind.UNKNOWN)
        if kind == BomLineKind.DOCUMENT_REF:
            continue

        child = _norm_code(getattr(line, "internal_code", "") or "")
        if not child:
            continue

        out.add(child)
        pos = str(getattr(line, "pos", "") or "").strip()
        if pos:
            pos_by_child.setdefault(child, []).append(pos)

    return out, pos_by_child


# -----------------------------------------------------------------------------
# Output model
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ParentChildMismatch:
    severity: str              # INFO/WARN/ERROR
    category: str              # CHILD_MISSING_IN_BOM, CHILD_MISSING_IN_BOM_OPTIONAL, TYPO_SUSPECT_CHILD_MISMATCH, ...
    root_vs_cascade: str       # ROOT or CASCADE

    parent_code: str
    parent_rev: str
    parent_pbs_src_row: int
    bom_path: str

    pbs_child: str = ""
    bom_child: str = ""
    bom_pos: str = ""
    suggestion: str = ""
    note: str = ""


# -----------------------------------------------------------------------------
# Suggestions
# -----------------------------------------------------------------------------
def _suggest_best_match(bom_child: str, missing_pbs_children: Sequence[str]) -> str:
    """
    Cerca match forte: edit distance <= 1 oppure prefisso numerico lungo.
    Ritorna suggestion string o "".
    """
    b = _norm_code(bom_child)
    if not b or not missing_pbs_children:
        return ""

    best = ""
    best_dist = 999

    for cand in missing_pbs_children:
        c = _norm_code(cand)
        ok, dist = _levenshtein_leq(b, c, max_dist=1)
        if ok and dist < best_dist:
            best = c
            best_dist = dist

    if best:
        return f"Sostituire '{bom_child}' con '{best}' (edit_distance={best_dist})"

    # fallback: match prefisso numerico >= 6
    b_num = "".join(ch for ch in b if ch.isdigit())
    best2 = ""
    best_pref = 0
    for cand in missing_pbs_children:
        c = _norm_code(cand)
        c_num = "".join(ch for ch in c if ch.isdigit())
        pref = 0
        n = min(len(b_num), len(c_num))
        while pref < n and b_num[pref] == c_num[pref]:
            pref += 1
        if pref > best_pref:
            best_pref = pref
            best2 = c

    if best2 and best_pref >= 6:
        return f"Possibile refuso: '{bom_child}' vs '{best2}' (numeric_prefix={best_pref})"

    return ""


def _suggest_from_candidates(pbs_child: str, extra_bom_children: Sequence[str]) -> str:
    """
    Quando un PBS child manca in BOM, prova a trovare un extra BOM child simile.
    """
    p = _norm_code(pbs_child)
    if not p or not extra_bom_children:
        return ""

    best = ""
    best_dist = 999
    for cand in extra_bom_children:
        c = _norm_code(cand)
        ok, dist = _levenshtein_leq(p, c, max_dist=1)
        if ok and dist < best_dist:
            best = c
            best_dist = dist

    if best:
        return f"In BOM esiste '{best}' simile a '{pbs_child}' (edit_distance={best_dist})"

    return ""


# -----------------------------------------------------------------------------
# Main diagnosis
# -----------------------------------------------------------------------------
def diagnose_parent_children_mismatches(
    *,
    pbs: PbsDocument,
    boms: Sequence[BomDocument],
    links: Sequence[LinkResult],
    explosion: object,
    unexploded_codes: Set[str],
) -> List[ParentChildMismatch]:
    """
    Confronta, per ogni parent P con link OK, i figli PBS vs figli BOM (stesso parent).

    Scelte di dominio (in base ai tuoi feedback):
    - CHILD_EXTRA_IN_BOM: NON emesso (normale) tranne se serve a trovare un typo (caso A).
    - Documenti (ASSD/PRTL/ELTD/...): filtrati da entrambe le parti → niente falsi positivi.
    - Missing in BOM:
        - se qty PBS == 0 (ma non document-like): OPTIONAL (WARN)
        - altrimenti: ERROR se ROOT, WARN se CASCADE
    """
    (
        children_by_parent,
        src_row_by_code,
        _rev_by_code,
        parent_by_child,
        qty_by_code,
    ) = build_pbs_parent_children(pbs)

    # Indicizza BOM per path
    bom_by_path: Dict[Path, BomDocument] = {getattr(b, "path"): b for b in boms}

    # Reached set (per distinguere ROOT vs CASCADE)
    reached: Set[str] = set()
    try:
        reached.add(_norm_code(getattr(explosion, "root_code", "") or ""))
    except Exception:
        pass
    for e in (getattr(explosion, "edges", []) or []):
        pc = _norm_code(getattr(e, "parent_code", "") or "")
        if pc:
            reached.add(pc)

    out: List[ParentChildMismatch] = []

    for lr in links:
        if getattr(lr, "status", None) != LinkStatus.OK:
            continue

        bom_ref = getattr(lr, "bom", None)
        if not bom_ref:
            continue

        bom_path = getattr(bom_ref, "path", None)
        if not bom_path:
            continue

        bom_doc = bom_by_path.get(Path(bom_path))
        if not bom_doc:
            continue

        parent = _norm_code(getattr(lr, "assembly_code", "") or "")
        if not parent:
            continue

        parent_rev = (getattr(lr, "assembly_rev", "") or "").strip()
        parent_src_row = src_row_by_code.get(parent, -1)

        # ROOT vs CASCADE: se parent non raggiunto o ha antenato unexploded => cascata
        is_cascade = (parent not in reached) or has_unexploded_ancestor(parent, parent_by_child, unexploded_codes)
        root_vs_cascade = "CASCADE" if is_cascade else "ROOT"

        # PBS children set (filtra document-like)
        raw_pbs_children = children_by_parent.get(parent, set())
        pbs_children = {c for c in raw_pbs_children if not _looks_like_document(c)}

        # BOM children set (filtra document-like)
        raw_bom_children, pos_by_child = bom_children_set(bom_doc)
        bom_children = {c for c in raw_bom_children if not _looks_like_document(c)}

        # Differenze
        missing_in_bom = sorted(list(pbs_children - bom_children))
        extra_in_bom = sorted(list(bom_children - pbs_children))

        # 1) Missing: (PBS child non presente nella BOM di parent)
        for c in missing_in_bom:
            # OPTIONAL: qty == 0 (ma non document-like, perché già filtrato)
            raw_qty = qty_by_code.get(c)
            optional_by_qty = _is_zero_qty(raw_qty) or _looks_optional(c)

            if optional_by_qty:
                sev = "WARN"  # anche se ROOT, qty==0 => non lo vogliamo come ERROR
                cat = "CHILD_MISSING_IN_BOM_OPTIONAL"
                note = "PBS child con qty=0 (non document). Trattato come optional/gestito altrove."
            else:
                sev = "WARN" if is_cascade else "ERROR"
                cat = "CHILD_MISSING_IN_BOM"
                note = ""

            suggestion = _suggest_from_candidates(c, extra_in_bom)
            out.append(
                ParentChildMismatch(
                    severity=sev,
                    category=cat,
                    root_vs_cascade=root_vs_cascade,
                    parent_code=parent,
                    parent_rev=parent_rev,
                    parent_pbs_src_row=parent_src_row,
                    bom_path=str(getattr(bom_doc, "path", "")),
                    pbs_child=c,
                    suggestion=suggestion,
                    note=note,
                )
            )

        # 2) Extra: NON emettiamo CHILD_EXTRA_IN_BOM (normale)
        #    Ma se un extra BOM child è simile a un missing PBS child, allora è tipicamente il caso A:
        #       PBS: P→Y
        #       BOM: P→Z (simile a Y)
        #
        #    Quindi emettiamo SOLO i TYPO_SUSPECT.
        if extra_in_bom and missing_in_bom:
            for c in extra_in_bom:
                sug = _suggest_best_match(c, missing_in_bom)
                if not sug:
                    continue

                # Se il codice suggerito è document-like, non lo segnaliamo (falso tipo ASSY↔ASSD)
                # (in pratica non dovrebbe accadere perché document-like è filtrato)
                if " con '" in sug:
                    try:
                        proposed = sug.split(" con '", 1)[1].split("'", 1)[0]
                        if _looks_like_document(proposed):
                            continue
                    except Exception:
                        pass

                sev = "WARN" if is_cascade else "ERROR"
                out.append(
                    ParentChildMismatch(
                        severity=sev,
                        category="TYPO_SUSPECT_CHILD_MISMATCH",
                        root_vs_cascade=root_vs_cascade,
                        parent_code=parent,
                        parent_rev=parent_rev,
                        parent_pbs_src_row=parent_src_row,
                        bom_path=str(getattr(bom_doc, "path", "")),
                        bom_child=c,
                        bom_pos=";".join(pos_by_child.get(c, [])),
                        suggestion=sug,
                    )
                )

    return out


# -----------------------------------------------------------------------------
# CSV writer + summary
# -----------------------------------------------------------------------------
def write_parent_child_mismatch_csv(rows: Sequence[ParentChildMismatch], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "severity",
        "category",
        "root_vs_cascade",
        "parent_code",
        "parent_rev",
        "parent_pbs_src_row",
        "bom_path",
        "pbs_child",
        "bom_child",
        "bom_pos",
        "suggestion",
        "note",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({
                "severity": r.severity,
                "category": r.category,
                "root_vs_cascade": r.root_vs_cascade,
                "parent_code": r.parent_code,
                "parent_rev": r.parent_rev,
                "parent_pbs_src_row": r.parent_pbs_src_row,
                "bom_path": r.bom_path,
                "pbs_child": r.pbs_child,
                "bom_child": r.bom_child,
                "bom_pos": r.bom_pos,
                "suggestion": r.suggestion,
                "note": r.note,
            })
    return len(rows)


def summarize_parent_child_mismatches(rows: Sequence[ParentChildMismatch]) -> str:
    if not rows:
        return "Nessun mismatch parent→children rilevato."

    n_err = sum(1 for r in rows if r.severity == "ERROR")
    n_warn = sum(1 for r in rows if r.severity == "WARN")
    n_info = sum(1 for r in rows if r.severity == "INFO")

    root_err = sum(1 for r in rows if r.severity == "ERROR" and r.root_vs_cascade == "ROOT")
    cas_err = sum(1 for r in rows if r.severity == "ERROR" and r.root_vs_cascade == "CASCADE")

    return (
        f"Parent→Children compare: ERROR={n_err} (ROOT={root_err}, CASCADE={cas_err}) | "
        f"WARN={n_warn} | INFO={n_info} | rows={len(rows)}"
    )
