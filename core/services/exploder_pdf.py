# core/services/exploder_pdf.py
# NOTE:
#   This module is intentionally a "twin" of core/services/exploder.py.
#   We DO NOT modify the original exploder.
#
#   Key difference for PDF-only workflow:
#     - child REV is taken from PBS if available,
#     - otherwise from the BOM line itself (line.rev), which is typical in BOM PDF tables.
#
#   Noise control:
#     - missing qty and non-positive qty warnings can be enabled/disabled via policy flags
#       (warn_missing_qty, warn_non_positive_qty). By default they are OFF to reduce noise.
#
#   IMPORTANT (coherence fix):
#     - Header PN indexing MUST be canonicalized using header REV context, otherwise
#       PN like "E0216161 01" risk being indexed as "E021616101" and never matched by children.

from __future__ import annotations

import re
import os
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Set, Tuple

from core.domain.models import BomDocument, BomLineKind, PbsDocument
from core.services.pn_canonical import canonicalize_pn, canonicalize_rev


_LOG = logging.getLogger(__name__)
_WU_DEBUG_ENV = "BOM_PDF_WU_DEBUG"
_WU_DEBUG_TARGET = "166104001"


def _wu_debug_enabled() -> bool:
    return (os.getenv(_WU_DEBUG_ENV, "") or "").strip() == "1"


def _wu_has_target(value: str) -> bool:
    return _WU_DEBUG_TARGET in ((value or "").strip())


# -------------------------
# Normalizers & helpers
# -------------------------
def _norm_rev(rev: str) -> str:
    return canonicalize_rev(rev or "")


def _canon_code(code: str, *, rev: str = "") -> str:
    """
    Canonical PN with optional REV context.

    Why:
      - headers often include " ... 01" suffix inside the code string
      - if we canonicalize without REV context, that suffix might be kept
        and indexing becomes inconsistent with line canonicalization.
    """
    r = _norm_rev(rev)
    return canonicalize_pn(code or "", rev=(r if r else None))


def _to_decimal(qty: Optional[float]) -> Optional[Decimal]:
    if qty is None:
        return None
    try:
        # float -> string to reduce binary artifacts
        return Decimal(str(qty))
    except Exception:
        return None


def _norm_uom(u: str) -> str:
    """
    Normalizza unità di misura in modo conservativo.
    - Tutti i sinonimi "pezzi" -> NR
    - Mantiene le unità note (MM, CM, M, GR, KG, ML, L)
    - Altrimenti ritorna la stringa upper pulita.
    """
    u = (u or "").strip().upper()
    if not u:
        return ""
    if u in ("NR", "N", "N°", "PZ", "PZA", "PZI", "PC", "PCS", "EA", "EACH", "PCE", "PZ."):
        return "NR"
    if u in ("MM",):
        return "MM"
    if u in ("CM",):
        return "CM"
    if u in ("M", "MT"):
        return "M"
    if u in ("GR", "G"):
        return "GR"
    # ✅ NEW: KGM è un caso reale nei PDF ("KGM 0.025"), normalizziamo a KG
    if u in ("KG", "KGM"):
        return "KG"
    if u in ("ML",):
        return "ML"
    if u in ("L", "LT"):
        return "L"
    return u


# -------------------------
# REV ordering (pragmatic, robust)
# -------------------------
_rev_re = re.compile(r"^([A-Z]+)?(\d+)?$")


def _rev_sort_key(rev: str) -> tuple:
    """
    Ordering key for REV across messy real data.
    Supports:
      - 'A', 'B', 'C'
      - '01', '02'
      - 'A01' (letters + digits)
    Fallback: lexicographic.
    """
    r = _norm_rev(rev)
    if not r:
        return (0, "", -1, "")
    m = _rev_re.match(r)
    if not m:
        return (1, r, -1, r)
    letters = m.group(1) or ""
    num = m.group(2)
    num_i = int(num) if num is not None else -1
    return (2, letters, num_i, r)


# -------------------------
# Numeric root helper (for suspicious header mismatch detection)
# -------------------------
_num_root_re = re.compile(r"(\d{6,})")  # pragmatic: at least 6 digits


def _numeric_root(code: str) -> str:
    """
    Extract a numeric root from a PN (e.g. '231018117ASSY' -> '231018117').
    If none found, returns ''.
    """
    s = _canon_code(code, rev="")
    if not s:
        return ""
    m = _num_root_re.search(s)
    return m.group(1) if m else ""


# -------------------------
# Domain policy & result models
# -------------------------
@dataclass(frozen=True)
class ExplodePolicy:
    """
    Matching policy tra righe BOM e sotto-BOM.

    - strict_rev: legacy flag (manteniamo compatibilità; non usarlo come unica leva)
    - explode_documents: se True esplode anche linee DOCUMENT_REF (di default False)

    - root_strict_rev: se True, la ROOT deve matchare esattamente (PN, REV)
    - recursive_fallback: se True, in ricorsione applica fallback se (PN, REV) non trovato
    - recursive_pick_highest_rev: se True e ci sono più BOM per PN, sceglie la rev “maggiore”

    Noise control:
    - warn_missing_qty: emette warning testuali per qty mancanti (default False)
    - warn_non_positive_qty: emette warning testuali per qty <= 0 (default False)
    """
    strict_rev: bool = True
    explode_documents: bool = False

    root_strict_rev: bool = True
    recursive_fallback: bool = True
    recursive_pick_highest_rev: bool = True

    warn_missing_qty: bool = False
    warn_non_positive_qty: bool = False


@dataclass(frozen=True)
class ExplosionEdge:
    parent_code: str
    parent_rev: str
    child_code: str
    child_rev: str  # può essere "" se ignota in riga/PBS
    qty: Decimal
    depth: int
    path: Tuple[str, ...]  # sequenza di code attraversate (pn)
    parent_bom_path: str = ""
    description: str = ""
    manufacturer: str = ""
    manufacturer_code: str = ""


@dataclass(frozen=True)
class MissingEdge:
    parent_code: str
    parent_rev: str
    child_code: str
    pos: str
    unit: str
    kind: str
    parent_bom_path: str = ""


@dataclass(frozen=True)
class BomCandidateInfo:
    bom_path: str
    header_code: str
    header_rev: str


@dataclass(frozen=True)
class BomResolutionTrace:
    """
    A single resolution attempt: "I was looking for PN X (rev Y) in context K"
    and what happened.
    """
    context: str  # "ROOT" | "EXPLODE_CHILD"
    expected_code: str
    expected_rev: str

    source_parent_code: str = ""  # only for EXPLODE_CHILD
    source_parent_rev: str = ""

    direct_candidates_count: int = 0
    suspicious_same_root: List[BomCandidateInfo] = field(default_factory=list)

    selected_bom_path: str = ""
    selected_header_code: str = ""
    selected_header_rev: str = ""

    outcome: str = ""  # "OK" | "MISSING_FILE" | "AMBIGUOUS" | "FALLBACK" | "SUSPICIOUS_HEADER_MISMATCH"
    note: str = ""
    suggestion: str = ""


def _trace(base: BomResolutionTrace, **overrides) -> BomResolutionTrace:
    return BomResolutionTrace(**{**base.__dict__, **overrides})


@dataclass
class ExplosionResult:
    root_code: str
    root_rev: str

    qty_by_code: Dict[str, Decimal] = field(default_factory=dict)
    qty_by_code_rev: Dict[Tuple[str, str], Decimal] = field(default_factory=dict)

    edges: List[ExplosionEdge] = field(default_factory=list)
    available_bom_codes: Set[str] = field(default_factory=set)

    exploded_assemblies: Set[Tuple[str, str]] = field(default_factory=set)
    missing_sub_boms: Set[str] = field(default_factory=set)
    rev_mismatch_sub_boms: List[Tuple[str, str, str]] = field(default_factory=list)
    cycles: List[Tuple[str, ...]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    resolution_traces: List[BomResolutionTrace] = field(default_factory=list)
    missing_edges: List[MissingEdge] = field(default_factory=list)

    uom_by_code: Dict[str, str] = field(default_factory=dict)
    non_nr_codes: Set[str] = field(default_factory=set)


# -------------------------
# Index builders (COHERENT)
# -------------------------
def index_boms_by_code_rev(boms: Iterable[BomDocument]) -> Dict[Tuple[str, str], BomDocument]:
    """
    IMPORTANT: header PN must be canonicalized using header REV context.
    """
    idx: Dict[Tuple[str, str], BomDocument] = {}
    for b in boms:
        hr = _norm_rev(getattr(b.header, "revision", "") or "")
        hc = _canon_code(getattr(b.header, "code", "") or "", rev=hr)
        k = (hc, hr)
        if hc and k not in idx:
            idx[k] = b
    return idx


def index_boms_by_code(boms: Iterable[BomDocument]) -> Dict[str, List[BomDocument]]:
    """
    IMPORTANT: index by canonical header code using header REV context.
    """
    idx: Dict[str, List[BomDocument]] = {}
    for b in boms:
        hr = _norm_rev(getattr(b.header, "revision", "") or "")
        hc = _canon_code(getattr(b.header, "code", "") or "", rev=hr)
        if not hc:
            continue
        idx.setdefault(hc, []).append(b)
    return idx


def index_boms_by_numeric_root(boms: Iterable[BomDocument]) -> Dict[str, List[BomDocument]]:
    idx: Dict[str, List[BomDocument]] = {}
    for b in boms:
        hr = _norm_rev(getattr(b.header, "revision", "") or "")
        hc = _canon_code(getattr(b.header, "code", "") or "", rev=hr)
        root = _numeric_root(hc)
        if not root:
            continue
        idx.setdefault(root, []).append(b)
    return idx


# -------------------------
# PBS helpers
# -------------------------
def find_root_from_pbs(pbs: PbsDocument) -> Tuple[str, str]:
    if not pbs.rows:
        raise ValueError("PBS vuoto: impossibile determinare root.")
    min_level = min(r.level for r in pbs.rows)
    for r in pbs.rows:
        if r.level == min_level and (r.code or "").strip():
            return _canon_code(r.code, rev=_norm_rev(r.rev)), _norm_rev(r.rev)
    raise ValueError("PBS senza code valorizzati: impossibile determinare root.")


# -------------------------
# BOM resolver (same logic, but now coherent codes)
# -------------------------
def choose_child_bom(
    *,
    child_code: str,
    child_rev: str,
    by_code_rev: Dict[Tuple[str, str], BomDocument],
    by_code: Dict[str, List[BomDocument]],
    by_numeric_root: Dict[str, List[BomDocument]],
    policy: ExplodePolicy,
    result: ExplosionResult,
    context_parent_code: str = "",
    context_parent_rev: str = "",
) -> Optional[BomDocument]:
    c = _canon_code(child_code, rev=child_rev)
    r = _norm_rev(child_rev)

    if not c:
        return None

    cands = by_code.get(c, [])

    trace = BomResolutionTrace(
        context="EXPLODE_CHILD",
        expected_code=c,
        expected_rev=r,
        source_parent_code=_canon_code(context_parent_code, rev=context_parent_rev),
        source_parent_rev=_norm_rev(context_parent_rev),
        direct_candidates_count=len(cands),
    )

    def _suspicious_same_root_candidates() -> List[BomCandidateInfo]:
        root = _numeric_root(c)
        if not root:
            return []
        alts = by_numeric_root.get(root, [])
        out: List[BomCandidateInfo] = []
        for b in alts:
            bhr = _norm_rev(getattr(b.header, "revision", "") or "")
            bhc = _canon_code(getattr(b.header, "code", "") or "", rev=bhr)
            if bhc == c:
                continue
            out.append(
                BomCandidateInfo(
                    bom_path=str(getattr(b, "path", "")),
                    header_code=bhc,
                    header_rev=bhr,
                )
            )
        return out

    def _pick_best(cands_: List[BomDocument]) -> Optional[BomDocument]:
        if not cands_:
            return None
        if len(cands_) == 1:
            return cands_[0]
        if not policy.recursive_pick_highest_rev:
            result.warnings.append(
                f"EXPLODE_AMBIGUOUS: PN={c} ha {len(cands_)} BOM candidate e fallback max-rev disabilitato."
            )
            result.resolution_traces.append(
                _trace(
                    trace,
                    outcome="AMBIGUOUS",
                    note=f"Multiple BOM candidates for PN={c} and fallback max-rev disabled.",
                    suggestion="Abilita recursive_pick_highest_rev oppure risolvi ambiguità nel dataset (una sola BOM per PN).",
                )
            )
            return None
        return max(cands_, key=lambda b: _rev_sort_key(getattr(b.header, "revision", "") or ""))

    if r:
        k = (c, r)
        if k in by_code_rev:
            b = by_code_rev[k]
            bhr = _norm_rev(getattr(b.header, "revision", "") or "")
            bhc = _canon_code(getattr(b.header, "code", "") or "", rev=bhr)
            result.resolution_traces.append(
                _trace(
                    trace,
                    selected_bom_path=str(getattr(b, "path", "")),
                    selected_header_code=bhc,
                    selected_header_rev=bhr,
                    outcome="OK",
                    note="Strict match (PN,REV) trovato.",
                )
            )
            return b

        if not policy.recursive_fallback:
            suspicious = _suspicious_same_root_candidates()
            if suspicious:
                result.resolution_traces.append(
                    _trace(
                        trace,
                        suspicious_same_root=suspicious,
                        outcome="SUSPICIOUS_HEADER_MISMATCH",
                        note=f"Nessuna BOM con header (PN,REV)=({c},{r}). Trovate BOM con stesso numeric-root ma header diverso.",
                        suggestion="Verifica header PN nelle BOM candidate (probabile template/copia-incolla).",
                    )
                )
            else:
                result.resolution_traces.append(
                    _trace(
                        trace,
                        outcome="MISSING_FILE",
                        note=f"Nessuna BOM con header (PN,REV)=({c},{r}) e fallback disabilitato.",
                        suggestion="Verifica presenza BOM o abilita recursive_fallback per proseguire.",
                    )
                )
            return None

        best = _pick_best(cands)
        if best is None:
            if len(cands) > 1:
                result.warnings.append(
                    f"EXPLODE_AMBIGUOUS: PN={c} richiesto REV={r}, ma ci sono {len(cands)} BOM (nessuna match strict)."
                )
                result.resolution_traces.append(
                    _trace(
                        trace,
                        outcome="AMBIGUOUS",
                        note=f"Strict rev mismatch and multiple candidates ({len(cands)}) for PN={c}.",
                        suggestion="Riduci le candidate o abilita criterio di scelta univoco (max rev).",
                    )
                )
            else:
                suspicious = _suspicious_same_root_candidates()
                if suspicious:
                    result.resolution_traces.append(
                        _trace(
                            trace,
                            suspicious_same_root=suspicious,
                            outcome="SUSPICIOUS_HEADER_MISMATCH",
                            note=f"Strict (PN,REV) non trovato e nessuna BOM indicizzata per PN={c}. Trovate BOM con stesso numeric-root ma header diverso.",
                            suggestion="Controlla gli header PN delle BOM indicate: probabile PN header errato.",
                        )
                    )
                else:
                    result.resolution_traces.append(
                        _trace(
                            trace,
                            outcome="MISSING_FILE",
                            note=f"Nessuna BOM indicizzata per PN={c} (richiesta REV={r}).",
                            suggestion="Verifica se la BOM esiste ma non è stata scoperta/parseata, oppure se il PN è un refuso/variante.",
                        )
                    )
            return None

        chosen_rev = _norm_rev(getattr(best.header, "revision", "") or "")
        result.rev_mismatch_sub_boms.append((c, r, chosen_rev))
        result.resolution_traces.append(
            _trace(
                trace,
                selected_bom_path=str(getattr(best, "path", "")),
                selected_header_code=_canon_code(getattr(best.header, "code", "") or "", rev=chosen_rev),
                selected_header_rev=chosen_rev,
                outcome="FALLBACK",
                note=f"Strict (PN,REV)=({c},{r}) non trovato; scelto fallback REV={chosen_rev}.",
                suggestion="Se il fallback non è desiderato, allinea le REV o aggiungi BOM per la REV richiesta.",
            )
        )
        return best

    # rev ignota
    if not cands:
        suspicious = _suspicious_same_root_candidates()
        if suspicious:
            result.resolution_traces.append(
                _trace(
                    trace,
                    suspicious_same_root=suspicious,
                    outcome="SUSPICIOUS_HEADER_MISMATCH",
                    note=f"Nessuna BOM indicizzata per PN={c} (rev ignota). Esistono BOM con stesso numeric-root ma header diverso.",
                    suggestion="Verifica header PN nelle BOM candidate: probabile PN header errato o variante non normalizzata.",
                )
            )
        else:
            result.resolution_traces.append(
                _trace(
                    trace,
                    outcome="MISSING_FILE",
                    note=f"Nessuna BOM indicizzata per PN={c} (rev ignota).",
                    suggestion="Verifica presenza BOM o refuso/variante PN.",
                )
            )
        return None

    if len(cands) == 1:
        b = cands[0]
        bhr = _norm_rev(getattr(b.header, "revision", "") or "")
        result.resolution_traces.append(
            _trace(
                trace,
                selected_bom_path=str(getattr(b, "path", "")),
                selected_header_code=_canon_code(getattr(b.header, "code", "") or "", rev=bhr),
                selected_header_rev=bhr,
                outcome="OK",
                note="Unica BOM candidata per PN (rev ignota).",
            )
        )
        return b

    if not policy.recursive_fallback:
        result.warnings.append(
            f"EXPLODE_AMBIGUOUS: PN={c} rev ignota, {len(cands)} BOM candidate (fallback disabilitato)."
        )
        result.resolution_traces.append(
            _trace(
                trace,
                outcome="AMBIGUOUS",
                note=f"{len(cands)} BOM candidate per PN={c} e rev ignota; fallback disabilitato.",
                suggestion="Abilita recursive_fallback/max-rev oppure rendi univoco il dataset (una sola BOM per PN).",
            )
        )
        return None

    best = _pick_best(cands)
    if best is None:
        return None

    chosen_rev = _norm_rev(getattr(best.header, "revision", "") or "")
    result.warnings.append(
        f"EXPLODE_FALLBACK_MAX_REV: PN={c} rev ignota, uso REV={chosen_rev} tra {len(cands)} BOM"
    )
    result.resolution_traces.append(
        _trace(
            trace,
            selected_bom_path=str(getattr(best, "path", "")),
            selected_header_code=_canon_code(getattr(best.header, "code", "") or "", rev=chosen_rev),
            selected_header_rev=chosen_rev,
            outcome="FALLBACK",
            note=f"Rev ignota; scelto fallback max-rev REV={chosen_rev} tra {len(cands)} candidate.",
            suggestion="Se il fallback non è desiderato, fornisci la REV in input (riga BOM/PBS) o riduci le BOM candidate.",
        )
    )
    return best


# -------------------------
# Main entrypoint
# -------------------------
def explode_boms_pdf(
    *,
    root_code: str,
    root_rev: str,
    boms: Iterable[BomDocument],
    pbs: Optional[PbsDocument] = None,
    policy: Optional[ExplodePolicy] = None,
) -> ExplosionResult:
    """
    PDF-friendly exploder:
    - Keeps the same behavior of explode_boms (PBS-first selection where applicable)
    - BUT: for child lines, if PBS doesn't provide a REV, uses line.rev from BOM table.
    """
    policy = policy or ExplodePolicy()

    by_code_rev = index_boms_by_code_rev(boms)
    by_code = index_boms_by_code(boms)
    by_numeric_root = index_boms_by_numeric_root(boms)

    root_rev_n = _norm_rev(root_rev)
    root_code_n = _canon_code(root_code, rev=root_rev_n)

    res = ExplosionResult(root_code=root_code_n, root_rev=root_rev_n)
    res.available_bom_codes = set(by_code.keys())

    # PBS index (optional)
    pbs_rev_by_code: Dict[str, str] = {}
    if pbs is not None:
        for r in pbs.rows:
            c = _canon_code(getattr(r, "code", "") or "", rev=_norm_rev(getattr(r, "rev", "") or ""))
            if not c:
                continue
            rv = _norm_rev(getattr(r, "rev", "") or "")
            if not rv:
                continue
            if c in pbs_rev_by_code and pbs_rev_by_code[c] != rv:
                res.warnings.append(f"PBS: PN {c} compare con REV diverse: {pbs_rev_by_code[c]} vs {rv}")
            pbs_rev_by_code[c] = rv

    # -------------------------
    # ROOT selection
    # -------------------------
    root = by_code_rev.get((root_code_n, root_rev_n))
    if root is None:
        root_trace = BomResolutionTrace(
            context="ROOT",
            expected_code=root_code_n,
            expected_rev=root_rev_n,
            direct_candidates_count=len(by_code.get(root_code_n, [])),
        )

        if not policy.root_strict_rev:
            cands = by_code.get(root_code_n, [])
            if len(cands) == 1:
                root = cands[0]
                chosen_rev = _norm_rev(getattr(root.header, "revision", "") or "")
                res.warnings.append(
                    f"ROOT_FALLBACK_SINGLE: PN={root_code_n} REV={root_rev_n}, uso unica BOM REV={chosen_rev}"
                )
                res.rev_mismatch_sub_boms.append((root_code_n, root_rev_n, chosen_rev))
                res.resolution_traces.append(
                    _trace(
                        root_trace,
                        selected_bom_path=str(getattr(root, "path", "")),
                        selected_header_code=_canon_code(getattr(root.header, "code", "") or "", rev=chosen_rev),
                        selected_header_rev=chosen_rev,
                        outcome="FALLBACK",
                        note="ROOT strict non trovata; uso unica candidata per PN.",
                        suggestion="Allinea la REV input o aggiungi la BOM della REV richiesta se vuoi strict root.",
                    )
                )
            elif len(cands) > 1:
                if policy.recursive_pick_highest_rev:
                    root = max(cands, key=lambda b: _rev_sort_key(getattr(b.header, "revision", "") or ""))
                    chosen_rev = _norm_rev(getattr(root.header, "revision", "") or "")
                    res.warnings.append(
                        f"ROOT_FALLBACK_MAX_REV: PN={root_code_n} REV={root_rev_n}, uso REV={chosen_rev} tra {len(cands)} BOM"
                    )
                    res.rev_mismatch_sub_boms.append((root_code_n, root_rev_n, chosen_rev))
                    res.resolution_traces.append(
                        _trace(
                            root_trace,
                            selected_bom_path=str(getattr(root, "path", "")),
                            selected_header_code=_canon_code(getattr(root.header, "code", "") or "", rev=chosen_rev),
                            selected_header_rev=chosen_rev,
                            outcome="FALLBACK",
                            note=f"ROOT strict non trovata; scelto max-rev tra {len(cands)} candidate.",
                            suggestion="Se vuoi strict root, rendi disponibile la BOM della REV o gestisci fallback consapevolmente.",
                        )
                    )
                else:
                    res.errors.append(
                        f"Nessuna BOM root trovata per (PN,REV)=({root_code_n},{root_rev_n}) e {len(cands)} candidate per PN."
                    )
                    res.resolution_traces.append(
                        _trace(
                            root_trace,
                            outcome="AMBIGUOUS",
                            note=f"ROOT non trovata strict e {len(cands)} candidate per PN; max-rev disabilitato.",
                            suggestion="Abilita recursive_pick_highest_rev oppure rendi univoco il dataset.",
                        )
                    )
                    return res
            else:
                res.errors.append(f"Nessuna BOM root trovata per (PN,REV)=({root_code_n},{root_rev_n}).")
                res.resolution_traces.append(
                    _trace(
                        root_trace,
                        outcome="MISSING_FILE",
                        note="ROOT strict non trovata e nessuna candidata per PN.",
                        suggestion="Verifica che la BOM root esista e sia stata scoperta/parseata.",
                    )
                )
                return res
        else:
            res.errors.append(f"Nessuna BOM root trovata per (PN,REV)=({root_code_n},{root_rev_n}).")
            res.resolution_traces.append(
                _trace(
                    root_trace,
                    outcome="MISSING_FILE",
                    note="ROOT strict non trovata.",
                    suggestion="Verifica che la BOM root (PN,REV) esista nel dataset.",
                )
            )
            return res

    # root ok
    chosen_rev = _norm_rev(getattr(root.header, "revision", "") or "")
    res.resolution_traces.append(
        BomResolutionTrace(
            context="ROOT",
            expected_code=root_code_n,
            expected_rev=root_rev_n,
            direct_candidates_count=len(by_code.get(root_code_n, [])),
            selected_bom_path=str(getattr(root, "path", "")),
            selected_header_code=_canon_code(getattr(root.header, "code", "") or "", rev=chosen_rev),
            selected_header_rev=chosen_rev,
            outcome="OK",
            note="ROOT strict match (PN,REV) trovata.",
        )
    )

    # -------------------------
    # Accumulators
    # -------------------------
    def _accumulate(code: str, rev: str, qty: Decimal) -> None:
        code_n = _canon_code(code, rev=rev)
        rev_n = _norm_rev(rev)
        if not code_n:
            return
        res.qty_by_code[code_n] = res.qty_by_code.get(code_n, Decimal("0")) + qty
        res.qty_by_code_rev[(code_n, rev_n)] = res.qty_by_code_rev.get((code_n, rev_n), Decimal("0")) + qty
        if _wu_debug_enabled() and (_wu_has_target(code) or _wu_has_target(code_n)):
            _LOG.info(
                "[WU_DEBUG][flat-accumulate] pn_display=%s rev_display=%s key_usata_per_grouping=%s qty_accumulated=%s qty_accumulated_by_code_rev=%s",
                code,
                rev_n,
                code_n,
                res.qty_by_code.get(code_n),
                res.qty_by_code_rev.get((code_n, rev_n)),
            )

    def _track_uom(child_code: str, raw_uom: object) -> str:
        uom = _norm_uom("" if raw_uom is None else str(raw_uom))
        if uom and child_code not in res.uom_by_code:
            res.uom_by_code[child_code] = uom
        if uom and uom != "NR":
            res.non_nr_codes.add(child_code)
        return uom

    # -------------------------
    # Recursive walk
    # -------------------------
    def _walk(
        *,
        parent_bom: BomDocument,
        parent_multiplier: Decimal,
        depth: int,
        stack: Tuple[str, ...],
    ) -> None:
        prev = _norm_rev(getattr(parent_bom.header, "revision", "") or "")
        pcode = _canon_code(getattr(parent_bom.header, "code", "") or "", rev=prev)

        res.exploded_assemblies.add((pcode, prev))

        for line in parent_bom.lines:
            internal = getattr(line, "internal_code", "") or ""
            if not internal.strip():
                continue

            if (not policy.explode_documents) and line.kind in (BomLineKind.DOCUMENT_REF,):
                continue

            # child_rev resolution (PBS-first, then line.rev)
            child_rev = ""
            # PBS provides rev by canonical PN; but we need a stable key → try with line.rev first if present
            raw_lr = (getattr(line, "rev", "") or "") if hasattr(line, "rev") else ""
            lr_n = _norm_rev(raw_lr)

            # canonical child code using the best rev we know at this moment (line rev if present)
            child_code_guess = _canon_code(internal, rev=lr_n)

            # PBS-first override
            child_rev = pbs_rev_by_code.get(child_code_guess, "")
            if not child_rev:
                child_rev = lr_n  # PDF-only: line.rev

            child_code = _canon_code(internal, rev=child_rev)
            if not child_code:
                continue

            if _wu_debug_enabled() and (
                _wu_has_target(internal)
                or _wu_has_target(child_code)
                or _wu_has_target(pcode)
            ):
                _LOG.info(
                    "[WU_DEBUG][explode-edge] parent_raw=%s child_raw=%s parent_key=%s child_key=%s child_rev=%s canonical_child_key=%s",
                    getattr(parent_bom.header, "code", "") or "",
                    internal,
                    pcode,
                    child_code,
                    _norm_rev(child_rev),
                    _canon_code(internal, rev=child_rev),
                )

            _track_uom(child_code, getattr(line, "unit", "") or "")

            q = _to_decimal(getattr(line, "qty", None))

            if q is None:
                # ✅ Qty mancante = leaf nel tree (robusto, non dipende da kind)
                eff_qty = Decimal("0")
                new_path = stack + (child_code,)

                if child_code in stack:
                    res.cycles.append(new_path)
                    res.errors.append(f"Ciclo rilevato: {' -> '.join(new_path)}")
                    continue

                res.edges.append(
                    ExplosionEdge(
                        parent_code=pcode,
                        parent_rev=prev,
                        child_code=child_code,
                        child_rev=_norm_rev(child_rev),
                        qty=eff_qty,  # 0 solo per compatibilità (leaf)
                        depth=depth,
                        path=new_path,
                        parent_bom_path=str(getattr(parent_bom, "path", "")),
                        description=(getattr(line, "description", "") or "").strip(),
                        manufacturer=(getattr(line, "manufacturer", "") or "").strip(),
                        manufacturer_code=(getattr(line, "manufacturer_code", "") or "").strip(),
                    )
                )

                # opzionale: mantieni anche missing_edges per auditing
                res.missing_edges.append(
                    MissingEdge(
                        parent_code=pcode,
                        parent_rev=prev,
                        child_code=child_code,
                        pos=str(getattr(line, "pos", "") or ""),
                        unit=str(getattr(line, "unit", "") or ""),
                        kind=str(getattr(line, "kind", "") or ""),
                        parent_bom_path=str(getattr(parent_bom, "path", "")),
                    )
                )
                if policy.warn_missing_qty:
                    res.warnings.append(
                        f"Qty mancante (leaf): parent={pcode} rev {prev} -> child={child_code} (pos={getattr(line, 'pos', '')})"
                    )

                # niente accumulate, niente ricorsione
                continue


            if q <= 0:
                if policy.warn_non_positive_qty:
                    res.warnings.append(
                        f"Qty non positiva ({q}): parent={pcode} rev {prev} -> child={child_code} (pos={getattr(line, 'pos', '')})"
                    )
                continue

            eff_qty = parent_multiplier * q

            _accumulate(child_code, child_rev, eff_qty)

            new_path = stack + (child_code,)

            if child_code in stack:
                res.cycles.append(new_path)
                res.errors.append(f"Ciclo rilevato: {' -> '.join(new_path)}")
                continue

            res.edges.append(
                ExplosionEdge(
                    parent_code=pcode,
                    parent_rev=prev,
                    child_code=child_code,
                    child_rev=_norm_rev(child_rev),
                    qty=eff_qty,
                    depth=depth,
                    path=new_path,
                    parent_bom_path=str(getattr(parent_bom, "path", "")),
                    description=(getattr(line, "description", "") or "").strip(),
                    manufacturer=(getattr(line, "manufacturer", "") or "").strip(),
                    manufacturer_code=(getattr(line, "manufacturer_code", "") or "").strip(),

                )
            )

            child_bom = choose_child_bom(
                child_code=child_code,
                child_rev=child_rev,
                by_code_rev=by_code_rev,
                by_code=by_code,
                by_numeric_root=by_numeric_root,
                policy=policy,
                result=res,
                context_parent_code=pcode,
                context_parent_rev=prev,
            )

            if child_bom is None:
                # missing as "sub BOM" means: the PN exists as a BOM header somewhere, but couldn't resolve
                if child_code in by_code:
                    res.missing_sub_boms.add(child_code)
                continue

            _walk(
                parent_bom=child_bom,
                parent_multiplier=eff_qty,
                depth=depth + 1,
                stack=new_path,
            )

    _walk(parent_bom=root, parent_multiplier=Decimal("1"), depth=1, stack=(root_code_n,))

    return res
