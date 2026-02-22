# core/services/exploder.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Set, Tuple

from core.domain.models import BomDocument, BomLineKind, PbsDocument, key_code_rev


def _norm_code(code: str) -> str:
    return (code or "").strip().upper()


def _norm_rev(rev: str) -> str:
    return (rev or "").strip().strip().upper()


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
    if u in ("KG",):
        return "KG"
    if u in ("ML",):
        return "ML"
    if u in ("L", "LT"):
        return "L"
    return u


# --- REV ordering (pragmatic, robust) ---
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


# --- Numeric root helper (for suspicious header mismatch detection) ---
_num_root_re = re.compile(r"(\d{6,})")  # pragmatic: at least 6 digits


def _numeric_root(code: str) -> str:
    """
    Extract a numeric root from a PN (e.g. '231018117ASSY' -> '231018117').
    If none found, returns ''.
    """
    s = _norm_code(code)
    if not s:
        return ""
    m = _num_root_re.search(s)
    return m.group(1) if m else ""


@dataclass(frozen=True)
class ExplodePolicy:
    """
    Matching policy tra righe BOM e sotto-BOM.

    - strict_rev: legacy flag (manteniamo compatibilità; non usarlo come unica leva)
    - explode_documents: se True esplode anche linee DOCUMENT_REF (di default False)

    - root_strict_rev: se True, la ROOT deve matchare esattamente (PN, REV) da PBS
    - recursive_fallback: se True, in ricorsione applica fallback se (PN, REV PBS) non trovato
    - recursive_pick_highest_rev: se True e ci sono più BOM per PN, sceglie la rev “maggiore”
    """
    strict_rev: bool = True
    explode_documents: bool = False

    root_strict_rev: bool = True
    recursive_fallback: bool = True
    recursive_pick_highest_rev: bool = True


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


# ✅ NEW: structured missing edge (qty missing on a BOM line)
# (Compatibile: non cambia nulla di esistente; aggiunge solo capacità)
@dataclass(frozen=True)
class MissingEdge:
    parent_code: str
    parent_rev: str
    child_code: str
    pos: str
    unit: str
    kind: str
    parent_bom_path: str = ""


# --- NEW: structured resolution trace (for explainable diagnostics) ---
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

    # direct candidates (matched by PN) - from by_code
    direct_candidates_count: int = 0

    # suspicious alternatives: same numeric root but different header code
    suspicious_same_root: List[BomCandidateInfo] = field(default_factory=list)

    selected_bom_path: str = ""
    selected_header_code: str = ""
    selected_header_rev: str = ""

    outcome: str = ""  # "OK" | "MISSING_FILE" | "AMBIGUOUS" | "FALLBACK" | "REV_MISMATCH" | "SUSPICIOUS_HEADER_MISMATCH"
    note: str = ""     # short human-readable explanation
    suggestion: str = ""  # short actionable suggestion


def _trace(base: BomResolutionTrace, **overrides) -> BomResolutionTrace:
    """
    Costruisce un trace facendo override in modo sicuro.
    Evita l'errore: got multiple values for keyword argument 'selected_bom_path'
    """
    return BomResolutionTrace(**{**base.__dict__, **overrides})


@dataclass
class ExplosionResult:
    root_code: str
    root_rev: str

    # flat aggregata
    qty_by_code: Dict[str, Decimal] = field(default_factory=dict)
    qty_by_code_rev: Dict[Tuple[str, str], Decimal] = field(default_factory=dict)

    # trace
    edges: List[ExplosionEdge] = field(default_factory=list)

    # dataset info (per report): PN per cui esiste almeno una BOM nel folder
    available_bom_codes: Set[str] = field(default_factory=set)

    # diagnostica
    exploded_assemblies: Set[Tuple[str, str]] = field(default_factory=set)
    missing_sub_boms: Set[str] = field(default_factory=set)  # PN con BOM presenti ma non selezionabili / non esplosi
    rev_mismatch_sub_boms: List[Tuple[str, str, str]] = field(default_factory=list)  # (pn, expected_rev, found_rev)
    cycles: List[Tuple[str, ...]] = field(default_factory=list)  # path dove si è rilevato un ciclo
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # --- NEW: structured audit traces for explainability / CSV ---
    resolution_traces: List[BomResolutionTrace] = field(default_factory=list)

    # ✅ NEW: structured list of missing qty edges (for qty inference with UoM)
    missing_edges: List[MissingEdge] = field(default_factory=list)

    # ✅ NEW: UoM tracking per PN (serve per regole NR / non-NR)
    uom_by_code: Dict[str, str] = field(default_factory=dict)
    non_nr_codes: Set[str] = field(default_factory=set)


def index_boms_by_code_rev(boms: Iterable[BomDocument]) -> Dict[Tuple[str, str], BomDocument]:
    idx: Dict[Tuple[str, str], BomDocument] = {}
    for b in boms:
        k = key_code_rev(b.header.code, b.header.revision)
        # se ce ne fossero più di una per stessa chiave, tieni la prima e segnala a livello superiore (linker)
        if k not in idx:
            idx[k] = b
    return idx


def index_boms_by_code(boms: Iterable[BomDocument]) -> Dict[str, List[BomDocument]]:
    idx: Dict[str, List[BomDocument]] = {}
    for b in boms:
        c = _norm_code(b.header.code)
        idx.setdefault(c, []).append(b)
    return idx


def index_boms_by_numeric_root(boms: Iterable[BomDocument]) -> Dict[str, List[BomDocument]]:
    """
    Lightweight index to detect cases like:
      expected PN = 2310181171A01
      but BOM exists with header PN = 231018117ASSY (same numeric root).
    This helps explain "missing" caused by wrong header PN.
    """
    idx: Dict[str, List[BomDocument]] = {}
    for b in boms:
        root = _numeric_root(b.header.code)
        if not root:
            continue
        idx.setdefault(root, []).append(b)
    return idx


def find_root_from_pbs(pbs: PbsDocument) -> Tuple[str, str]:
    if not pbs.rows:
        raise ValueError("PBS vuoto: impossibile determinare root.")
    min_level = min(r.level for r in pbs.rows)
    # prima riga al livello minimo
    for r in pbs.rows:
        if r.level == min_level and (r.code or "").strip():
            return _norm_code(r.code), _norm_rev(r.rev)
    raise ValueError("PBS senza code valorizzati: impossibile determinare root.")


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
    """
    (INVARIATO: stesso tuo codice)
    """
    c = _norm_code(child_code)
    r = _norm_rev(child_rev)

    if not c:
        return None

    cands = by_code.get(c, [])

    trace = BomResolutionTrace(
        context="EXPLODE_CHILD",
        expected_code=c,
        expected_rev=r,
        source_parent_code=_norm_code(context_parent_code),
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
            hcode = _norm_code(b.header.code)
            if hcode == c:
                continue
            out.append(
                BomCandidateInfo(
                    bom_path=str(getattr(b, "path", "")),
                    header_code=hcode,
                    header_rev=_norm_rev(b.header.revision),
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
        return max(cands_, key=lambda b: _rev_sort_key(b.header.revision))

    if r:
        k = (c, r)
        if k in by_code_rev:
            b = by_code_rev[k]
            result.resolution_traces.append(
                _trace(
                    trace,
                    selected_bom_path=str(getattr(b, "path", "")),
                    selected_header_code=_norm_code(b.header.code),
                    selected_header_rev=_norm_rev(b.header.revision),
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

        chosen_rev = _norm_rev(best.header.revision)
        if len(cands) == 1:
            result.warnings.append(
                f"EXPLODE_FALLBACK_SINGLE: PN={c} richiesto REV={r}, uso unica BOM disponibile REV={chosen_rev}"
            )
        else:
            result.warnings.append(
                f"EXPLODE_FALLBACK_MAX_REV: PN={c} richiesto REV={r}, uso REV={chosen_rev} tra {len(cands)} BOM"
            )

        result.rev_mismatch_sub_boms.append((c, r, chosen_rev))

        result.resolution_traces.append(
            _trace(
                trace,
                selected_bom_path=str(getattr(best, "path", "")),
                selected_header_code=_norm_code(best.header.code),
                selected_header_rev=chosen_rev,
                outcome="FALLBACK",
                note=f"Strict (PN,REV)=({c},{r}) non trovato; scelto fallback REV={chosen_rev}.",
                suggestion="Se il fallback non è desiderato, allinea le REV o aggiungi BOM per la REV richiesta.",
            )
        )
        return best

    if not cands:
        suspicious = _suspicious_same_root_candidates()
        if suspicious:
            result.resolution_traces.append(
                _trace(
                    trace,
                    suspicious_same_root=suspicious,
                    outcome="SUSPICIOUS_HEADER_MISMATCH",
                    note=f"Nessuna BOM indicizzata per PN={c} (rev PBS ignota). Esistono BOM con stesso numeric-root ma header diverso.",
                    suggestion="Verifica header PN nelle BOM candidate: probabile PN header errato o variante non normalizzata.",
                )
            )
        else:
            result.resolution_traces.append(
                _trace(
                    trace,
                    outcome="MISSING_FILE",
                    note=f"Nessuna BOM indicizzata per PN={c} (rev PBS ignota).",
                    suggestion="Verifica presenza BOM o refuso/variante PN.",
                )
            )
        return None

    if len(cands) == 1:
        b = cands[0]
        result.resolution_traces.append(
            _trace(
                trace,
                selected_bom_path=str(getattr(b, "path", "")),
                selected_header_code=_norm_code(b.header.code),
                selected_header_rev=_norm_rev(b.header.revision),
                outcome="OK",
                note="Unica BOM candidata per PN (rev ignota).",
            )
        )
        return b

    if not policy.recursive_fallback:
        result.warnings.append(
            f"EXPLODE_AMBIGUOUS: PN={c} rev PBS ignota, {len(cands)} BOM candidate (fallback disabilitato)."
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

    chosen_rev = _norm_rev(best.header.revision)
    result.warnings.append(
        f"EXPLODE_FALLBACK_MAX_REV: PN={c} rev PBS ignota, uso REV={chosen_rev} tra {len(cands)} BOM"
    )
    result.resolution_traces.append(
        _trace(
            trace,
            selected_bom_path=str(getattr(best, "path", "")),
            selected_header_code=_norm_code(best.header.code),
            selected_header_rev=chosen_rev,
            outcome="FALLBACK",
            note=f"Rev ignota; scelto fallback max-rev REV={chosen_rev} tra {len(cands)} candidate.",
            suggestion="Se il fallback non è desiderato, fornisci la REV in PBS o riduci le BOM candidate.",
        )
    )
    return best


def explode_boms(
    *,
    root_code: str,
    root_rev: str,
    boms: Iterable[BomDocument],
    pbs: Optional[PbsDocument] = None,
    policy: Optional[ExplodePolicy] = None,
) -> ExplosionResult:
    policy = policy or ExplodePolicy()

    by_code_rev = index_boms_by_code_rev(boms)
    by_code = index_boms_by_code(boms)
    by_numeric_root = index_boms_by_numeric_root(boms)

    root_code_n = _norm_code(root_code)
    root_rev_n = _norm_rev(root_rev)

    res = ExplosionResult(root_code=root_code_n, root_rev=root_rev_n)
    res.available_bom_codes = set(by_code.keys())

    # PBS index: REV "vera" per PN (quando disponibile). In BOM non esiste rev di riga.
    pbs_rev_by_code: Dict[str, str] = {}
    if pbs is not None:
        for r in pbs.rows:
            c = _norm_code(getattr(r, "code", ""))
            if not c:
                continue
            rv = _norm_rev(getattr(r, "rev", ""))
            if not rv:
                continue
            if c in pbs_rev_by_code and pbs_rev_by_code[c] != rv:
                res.warnings.append(f"PBS: PN {c} compare con REV diverse: {pbs_rev_by_code[c]} vs {rv}")
            pbs_rev_by_code[c] = rv

    # --- ROOT selection: strict by PBS by default ---
    root = by_code_rev.get((root_code_n, root_rev_n))
    if root is None:
        # (INVARIATO: tuo codice root selection)
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
                res.warnings.append(
                    f"ROOT_FALLBACK_SINGLE: PN={root_code_n} PBS REV={root_rev_n}, uso unica BOM REV={_norm_rev(root.header.revision)}"
                )
                res.rev_mismatch_sub_boms.append((root_code_n, root_rev_n, _norm_rev(root.header.revision)))

                res.resolution_traces.append(
                    _trace(
                        root_trace,
                        selected_bom_path=str(getattr(root, "path", "")),
                        selected_header_code=_norm_code(root.header.code),
                        selected_header_rev=_norm_rev(root.header.revision),
                        outcome="FALLBACK",
                        note="ROOT strict non trovata; uso unica candidata per PN.",
                        suggestion="Allinea la REV PBS o aggiungi la BOM della REV richiesta se vuoi strict root.",
                    )
                )
            elif len(cands) > 1:
                if policy.recursive_pick_highest_rev:
                    root = max(cands, key=lambda b: _rev_sort_key(b.header.revision))
                    res.warnings.append(
                        f"ROOT_FALLBACK_MAX_REV: PN={root_code_n} PBS REV={root_rev_n}, uso REV={_norm_rev(root.header.revision)} tra {len(cands)} BOM"
                    )
                    res.rev_mismatch_sub_boms.append((root_code_n, root_rev_n, _norm_rev(root.header.revision)))

                    res.resolution_traces.append(
                        _trace(
                            root_trace,
                            selected_bom_path=str(getattr(root, "path", "")),
                            selected_header_code=_norm_code(root.header.code),
                            selected_header_rev=_norm_rev(root.header.revision),
                            outcome="FALLBACK",
                            note=f"ROOT strict non trovata; scelto max-rev tra {len(cands)} candidate.",
                            suggestion="Se vuoi strict root, rendi disponibile la BOM della REV PBS o abilita/disabilita fallback consapevolmente.",
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
                            suggestion="Abilita recursive_pick_highest_rev oppure risolvi ambiguità nel dataset (una sola BOM per PN).",
                        )
                    )
                    return res
            else:
                root_suspicious: List[BomCandidateInfo] = []
                root_root = _numeric_root(root_code_n)
                if root_root:
                    for b in by_numeric_root.get(root_root, []):
                        hcode = _norm_code(b.header.code)
                        if hcode == root_code_n:
                            continue
                        root_suspicious.append(
                            BomCandidateInfo(
                                bom_path=str(getattr(b, "path", "")),
                                header_code=hcode,
                                header_rev=_norm_rev(b.header.revision),
                            )
                        )

                res.errors.append(f"Nessuna BOM root trovata per (PN,REV)=({root_code_n},{root_rev_n}).")
                if root_suspicious:
                    res.resolution_traces.append(
                        _trace(
                            root_trace,
                            suspicious_same_root=root_suspicious,
                            outcome="SUSPICIOUS_HEADER_MISMATCH",
                            note="ROOT strict non trovata; esistono BOM con stesso numeric-root ma header diverso.",
                            suggestion="Controlla header PN nelle BOM candidate: probabile PN header errato.",
                        )
                    )
                else:
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
            root_suspicious: List[BomCandidateInfo] = []
            root_root = _numeric_root(root_code_n)
            if root_root:
                for b in by_numeric_root.get(root_root, []):
                    hcode = _norm_code(b.header.code)
                    if hcode == root_code_n:
                        continue
                    root_suspicious.append(
                        BomCandidateInfo(
                            bom_path=str(getattr(b, "path", "")),
                            header_code=hcode,
                            header_rev=_norm_rev(b.header.revision),
                        )
                    )

            res.errors.append(f"Nessuna BOM root trovata per (PN,REV)=({root_code_n},{root_rev_n}).")
            if root_suspicious:
                res.resolution_traces.append(
                    _trace(
                        root_trace,
                        suspicious_same_root=root_suspicious,
                        outcome="SUSPICIOUS_HEADER_MISMATCH",
                        note="ROOT strict non trovata; esistono BOM con stesso numeric-root ma header diverso.",
                        suggestion="Controlla header PN nelle BOM candidate (probabile errore umano/template).",
                    )
                )
            else:
                res.resolution_traces.append(
                    _trace(
                        root_trace,
                        outcome="MISSING_FILE",
                        note="ROOT strict non trovata.",
                        suggestion="Verifica che la BOM root (PN,REV) esista nel dataset.",
                    )
                )
            return res

    else:
        res.resolution_traces.append(
            BomResolutionTrace(
                context="ROOT",
                expected_code=root_code_n,
                expected_rev=root_rev_n,
                direct_candidates_count=len(by_code.get(root_code_n, [])),
                selected_bom_path=str(getattr(root, "path", "")),
                selected_header_code=_norm_code(root.header.code),
                selected_header_rev=_norm_rev(root.header.revision),
                outcome="OK",
                note="ROOT strict match (PN,REV) trovata.",
            )
        )

    def _accumulate(code: str, rev: str, qty: Decimal) -> None:
        code = _norm_code(code)
        rev = _norm_rev(rev)
        if not code:
            return
        res.qty_by_code[code] = res.qty_by_code.get(code, Decimal("0")) + qty
        res.qty_by_code_rev[(code, rev)] = res.qty_by_code_rev.get((code, rev), Decimal("0")) + qty

    def _track_uom(child_code: str, raw_uom: object) -> str:
        """
        Track UoM in modo stabile:
        - salva la prima UoM non-vuota vista per quel PN
        - se la UoM è != NR, marca il PN come non-NR
        """
        uom = _norm_uom("" if raw_uom is None else str(raw_uom))
        if uom and child_code not in res.uom_by_code:
            res.uom_by_code[child_code] = uom
        if uom and uom != "NR":
            res.non_nr_codes.add(child_code)
        return uom

    def _walk(
        *,
        parent_bom: BomDocument,
        parent_multiplier: Decimal,
        depth: int,
        stack: Tuple[str, ...],
    ) -> None:
        pcode = _norm_code(parent_bom.header.code)
        prev = _norm_rev(parent_bom.header.revision)

        key = (pcode, prev)
        res.exploded_assemblies.add(key)

        for line in parent_bom.lines:
            if not line.internal_code:
                continue

            # filtra documentazione se richiesto
            if (not policy.explode_documents) and line.kind in (BomLineKind.DOCUMENT_REF,):
                continue

            child_code = _norm_code(line.internal_code)
            _track_uom(child_code, getattr(line, "unit", ""))

            q = _to_decimal(line.qty)
            if q is None:
                # ✅ keep legacy warning for compatibility
                res.warnings.append(
                    f"Qty mancante: parent={pcode} rev {prev} -> child={child_code} (pos={line.pos})"
                )

                # ✅ NEW: structured missing edge capture (with UoM)
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
                continue

            if q <= 0:
                res.warnings.append(
                    f"Qty non positiva ({q}): parent={pcode} rev {prev} -> child={child_code} (pos={line.pos})"
                )
                continue

            # Nel tuo contesto reale: la REV dei componenti non è nella BOM ma nel PBS.
            child_rev = pbs_rev_by_code.get(child_code, "")

            eff_qty = parent_multiplier * q

            # accumula SEMPRE la riga (flat include anche assembly)
            _accumulate(child_code, child_rev, eff_qty)

            new_path = stack + (child_code,)

            # cycle detection (code-only)
            if child_code in stack:
                res.cycles.append(new_path)
                res.errors.append(f"Ciclo rilevato: {' -> '.join(new_path)}")
                continue

            edge = ExplosionEdge(
                parent_code=pcode,
                parent_rev=prev,
                child_code=child_code,
                child_rev=child_rev,
                qty=eff_qty,
                depth=depth,
                path=new_path,
                parent_bom_path=str(parent_bom.path),
            )
            res.edges.append(edge)

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
                # non esplodibile -> foglia (oppure BOM presente ma non selezionabile)
                if child_code in by_code:
                    res.missing_sub_boms.add(child_code)
                continue

            _walk(parent_bom=child_bom, parent_multiplier=eff_qty, depth=depth + 1, stack=new_path)

    _walk(parent_bom=root, parent_multiplier=Decimal("1"), depth=1, stack=(root_code_n,))

    return res
