# core/services/unexploded_diagnosis.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import csv
import difflib

from core.domain.models import BomDocument, PbsDocument
from core.services.exploder import ExplosionResult, _norm_code, _norm_rev


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class Occurrence:
    parent_bom: str          # es: "2310182171A01 REV A"
    parent_bom_path: str
    pos: str
    qty: Optional[float]
    description: str


@dataclass(frozen=True)
class DiagnosisRow:
    pbs_code: str
    pbs_rev: str
    diagnosis: str

    candidate_code: str = ""
    parent_bom: str = ""
    pos: str = ""
    note: str = ""

    # --- NEW: chain-break aggregation ---
    root_chain_break: str = ""       # "1" for root, "0" for cascade, "" unknown
    cascade_depth: str = ""          # "0" root, "1..n" cascade
    classification: str = ""         # "ROOT" | "CASCADE" | ""
    not_reached_by_root: str = ""    # root_code for cascades


# -----------------------------
# Helpers
# -----------------------------
def _num_root(code: str, min_digits: int = 6) -> Optional[str]:
    """
    Estrae il prefisso numerico contiguo iniziale.
    Ritorna None se < min_digits.
    """
    s = _norm_code(code)
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    if i >= min_digits:
        return s[:i]
    return None


def _suffix(code: str) -> str:
    s = _norm_code(code)
    nr = _num_root(s)
    if not nr:
        return ""
    return s[len(nr):]


def _levenshtein(a: str, b: str, max_dist: int = 3) -> int:
    """
    Levenshtein con early-exit (cutoff) per performance.
    Ritorna un valore > max_dist se supera la soglia.
    """
    a = _norm_code(a)
    b = _norm_code(b)
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # DP su riga
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        # best-in-row for cutoff
        row_min = cur[0]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            v = min(ins, dele, sub)
            cur.append(v)
            if v < row_min:
                row_min = v
        if row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


def _bom_header_str(b: BomDocument) -> str:
    return f"{_norm_code(b.header.code)} REV {_norm_rev(b.header.revision)}"


def _pbs_rev_map(pbs: PbsDocument) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for r in pbs.rows:
        c = _norm_code(r.code)
        if not c:
            continue
        rv = _norm_rev(r.rev)
        if rv:
            m[c] = rv
    return m


def _parse_parent_header(parent_bom: str) -> Tuple[str, str]:
    """
    Prova a parsare stringhe tipo: "2310182171A01 REV A"
    Ritorna (code, rev) normalizzati. Se non parseable: ("","")
    """
    s = (parent_bom or "").strip()
    if not s:
        return "", ""
    up = s.upper()
    if " REV " not in up:
        return _norm_code(s), ""
    try:
        left, right = up.split(" REV ", 1)
        return _norm_code(left.strip()), _norm_rev(right.strip())
    except Exception:
        return "", ""


def _pn_from_warning(w: str) -> str:
    """
    Estrae PN da warning che contengono 'PN=XXXX'.
    Ritorna '' se non parseable.
    """
    if not w:
        return ""
    if "PN=" not in w:
        return ""
    try:
        frag = w.split("PN=", 1)[1]
        pn = frag.split(" ", 1)[0].strip()
        return _norm_code(pn)
    except Exception:
        return ""


def _pbs_effective_children_count_map(pbs: PbsDocument) -> Dict[str, int]:
    """
    Mappa: parent_code -> count(direct_children with qty > 0).

    Serve per ridurre falsi allarmi su nodi PBS non-manufacturing:
    se un nodo ha solo figli con qty==0 (ELTD/ASSD/BLKD/USRM...), lo
    classifichiamo come informativo (non "errore").
    """
    # Build parent pointers / adjacency via row.level
    rows = list(getattr(pbs, "rows", []) or [])
    if not rows:
        return {}

    # Normalize minimal node view
    nodes: List[Tuple[str, int, float]] = []
    for r in rows:
        code = _norm_code(getattr(r, "code", "") or "")
        if not code:
            nodes.append(("", 0, 0.0))
            continue
        level = int(getattr(r, "level", 0) or 0)
        try:
            qty = float(getattr(r, "qty", 0) or 0)
        except Exception:
            qty = 0.0
        nodes.append((code, level, qty))

    # stack of indices
    stack: List[int] = []
    parent_idx: List[Optional[int]] = [None] * len(nodes)

    for i, (code, level, qty) in enumerate(nodes):
        while stack and nodes[stack[-1]][1] >= level:
            stack.pop()
        parent_idx[i] = stack[-1] if stack else None
        stack.append(i)

    cnt: Dict[str, int] = {}
    for i, (code, level, qty) in enumerate(nodes):
        pi = parent_idx[i]
        if pi is None:
            continue
        pcode = nodes[pi][0]
        if not pcode:
            continue
        if qty and qty > 0:
            cnt[pcode] = cnt.get(pcode, 0) + 1

    return cnt


# -----------------------------
# Index builder
# -----------------------------
@dataclass
class ChildIndex:
    occ_by_child: Dict[str, List[Occurrence]]
    child_by_numroot: Dict[str, Set[str]]
    child_by_prefix: Dict[str, Set[str]]


def build_child_index(boms: Iterable[BomDocument]) -> ChildIndex:
    occ_by_child: Dict[str, List[Occurrence]] = {}
    child_by_numroot: Dict[str, Set[str]] = {}
    child_by_prefix: Dict[str, Set[str]] = {}

    for bom in boms:
        parent = _bom_header_str(bom)
        parent_path = str(bom.path)

        for line in bom.lines:
            cc = _norm_code(line.internal_code)
            if not cc:
                continue

            occ = Occurrence(
                parent_bom=parent,
                parent_bom_path=parent_path,
                pos=str(line.pos or ""),
                qty=line.qty,
                description=(line.description or "").strip(),
            )
            occ_by_child.setdefault(cc, []).append(occ)

            nr = _num_root(cc)
            if nr:
                child_by_numroot.setdefault(nr, set()).add(cc)

            # prefisso "stabile" per restringere typo: prime 8 cifre se disponibili
            pref = cc[:10] if len(cc) >= 10 else cc[:8]
            if pref:
                child_by_prefix.setdefault(pref, set()).add(cc)

    return ChildIndex(
        occ_by_child=occ_by_child,
        child_by_numroot=child_by_numroot,
        child_by_prefix=child_by_prefix,
    )


# -----------------------------
# Core diagnosis
# -----------------------------
def diagnose_unexploded_assemblies(
    *,
    pbs: PbsDocument,
    boms: Sequence[BomDocument],
    explosion: ExplosionResult,
    unexploded_codes: Iterable[str],
    max_candidates: int = 3,
) -> List[DiagnosisRow]:
    pbs_rev_by_code = _pbs_rev_map(pbs)
    idx = build_child_index(boms)

    # NEW: to reduce false positives on non-manufacturing nodes
    pbs_effective_children_cnt = _pbs_effective_children_count_map(pbs)

    # -----------------------------------
    # NEW: reachability from actual explosion
    # -----------------------------------
    reached_as_child: Set[str] = set()
    for e in getattr(explosion, "edges", []) or []:
        reached_as_child.add(_norm_code(getattr(e, "child_code", "")))

    exploded_codes: Set[str] = set()
    for (c, _r) in getattr(explosion, "exploded_assemblies", set()) or set():
        exploded_codes.add(_norm_code(c))

    # quick helpers for "selection fail" signals
    rev_mismatch_map: Dict[str, Tuple[str, str]] = {
        _norm_code(pn): (_norm_rev(exp), _norm_rev(found))
        for (pn, exp, found) in getattr(explosion, "rev_mismatch_sub_boms", []) or []
    }

    # NEW: detect fallback/ambiguity warnings from exploder
    fallback_set: Set[str] = set()
    ambig_set: Set[str] = set()

    for w in getattr(explosion, "warnings", []) or []:
        # legacy warning string tipo: "Ambiguità BOM per PN=XXXX ..."
        if "Ambiguità BOM per PN=" in w:
            try:
                frag = w.split("Ambiguità BOM per PN=", 1)[1]
                pn = frag.split(" ", 1)[0].strip()
                if pn:
                    ambig_set.add(_norm_code(pn))
            except Exception:
                pass

        # new warnings from exploder policy
        if "EXPLODE_FALLBACK_" in w or "ROOT_FALLBACK_" in w:
            pn = _pn_from_warning(w)
            if pn:
                fallback_set.add(pn)
        if "EXPLODE_AMBIGUOUS" in w:
            pn = _pn_from_warning(w)
            if pn:
                ambig_set.add(pn)

    rows: List[DiagnosisRow] = []

    for raw_code in sorted({_norm_code(x) for x in unexploded_codes if _norm_code(x)}):
        pbs_rev = pbs_rev_by_code.get(raw_code, "")

        # -----------------------------------
        # NEW: PBS-only "non manufacturing" node => informational
        # If a PBS node has NO direct children with qty>0, do not flag it as error.
        # We keep it visible but classify as INFO.
        # -----------------------------------
        if pbs_effective_children_cnt.get(raw_code, 0) == 0:
            rows.append(
                DiagnosisRow(
                    pbs_code=raw_code,
                    pbs_rev=pbs_rev,
                    diagnosis="INFO_PBS_ONLY_ZERO_QTY_CHILDREN",
                    candidate_code=raw_code,
                    note="PBS node has no direct manufacturing children (all direct children qty=0) -> informational",
                )
            )
            continue

        occs_exact = idx.occ_by_child.get(raw_code, [])

        # -----------------------------------
        # CASE A (NEW): reached in real explosion but not exploded as parent => selection fail / stop in recursion
        # -----------------------------------
        if raw_code in reached_as_child and raw_code not in exploded_codes:
            note_bits: List[str] = []
            diag = "REACHED_SELECTION_FAIL"

            if raw_code in getattr(explosion, "missing_sub_boms", set()) or set():
                note_bits.append(
                    "BOM exists for PN but not selectable during recursion (rev mismatch / ambiguity / policy stop)"
                )

            if raw_code in rev_mismatch_map:
                exp, found = rev_mismatch_map[raw_code]
                note_bits.append(f"rev mismatch: PBS={exp} chosen={found}")

            if raw_code in fallback_set:
                note_bits.append("fallback used to select a BOM revision (see warnings)")

            if raw_code in ambig_set:
                note_bits.append("multiple BOM candidates for same PN (ambiguous)")

            # include occurrence context if available (useful for human action)
            if occs_exact:
                o = occs_exact[0]
                rows.append(
                    DiagnosisRow(
                        pbs_code=raw_code,
                        pbs_rev=pbs_rev,
                        diagnosis=diag,
                        candidate_code=raw_code,
                        parent_bom=o.parent_bom,
                        pos=o.pos,
                        note="; ".join(note_bits) if note_bits else "reached as child but never exploded as parent",
                    )
                )
            else:
                # reached but no occurrence in global index (rare): still produce row
                rows.append(
                    DiagnosisRow(
                        pbs_code=raw_code,
                        pbs_rev=pbs_rev,
                        diagnosis=diag,
                        candidate_code=raw_code,
                        note="; ".join(note_bits) if note_bits else "reached as child but never exploded as parent",
                    )
                )
            continue

        # -----------------------------------
        # CASE B (NEW): not reached in real explosion => chain break above (or PBS contains unreachable nodes)
        # -----------------------------------
        if raw_code not in reached_as_child:
            # If it appears as a child in some BOM(s), try to infer chain-break parent that was not exploded.
            if occs_exact:
                # pick an occurrence where parent is NOT exploded (best hint)
                chain_hint: Optional[Occurrence] = None
                for o in occs_exact:
                    pc, _pr = _parse_parent_header(o.parent_bom)
                    if pc and pc not in exploded_codes:
                        chain_hint = o
                        break
                if chain_hint is None:
                    chain_hint = occs_exact[0]

                note_bits: List[str] = []
                note_bits.append("not reached in exploded tree (chain breaks above)")

                pc, pr = _parse_parent_header(chain_hint.parent_bom)
                if pc:
                    if pc in exploded_codes:
                        # parent exploded but child not reached => inconsistent indexes or filtering (documents)
                        note_bits.append(
                            f"parent {pc} appears exploded; verify filters (explode_documents) or code normalization"
                        )
                    else:
                        note_bits.append(
                            f"appears under parent {pc} (not exploded) -> fix/select parent's BOM chain first"
                        )

                rows.append(
                    DiagnosisRow(
                        pbs_code=raw_code,
                        pbs_rev=pbs_rev,
                        diagnosis="NOT_REACHED_CHAIN_BREAK",
                        candidate_code=raw_code,
                        parent_bom=chain_hint.parent_bom,
                        pos=chain_hint.pos,
                        note="; ".join(note_bits),
                    )
                )
                continue

            # else: never appears as exact child in any BOM -> fall through to variants/typos/missing link
            # (existing logic below)

        # -----------------------------------
        # Existing logic (kept): if cited as exact child -> previously REV_LOOKUP_FAIL
        # We keep it for cases where explosion edges are not available or for robustness.
        # -----------------------------------
        if occs_exact:
            note_bits: List[str] = []
            diag = "REV_LOOKUP_FAIL"

            if raw_code in getattr(explosion, "missing_sub_boms", set()) or set():
                note_bits.append("present as child, BOM exists but not selectable (strict rev or ambiguity)")

            if raw_code in rev_mismatch_map:
                exp, found = rev_mismatch_map[raw_code]
                note_bits.append(f"rev mismatch: requested={exp} found={found}")

            if raw_code in fallback_set:
                note_bits.append("fallback used to select a BOM revision (see warnings)")

            if raw_code in ambig_set:
                note_bits.append("multiple BOM candidates for same PN (ambiguous)")

            # scegli 1 occorrenza “più utile” (prima)
            o = occs_exact[0]
            rows.append(
                DiagnosisRow(
                    pbs_code=raw_code,
                    pbs_rev=pbs_rev,
                    diagnosis=diag,
                    candidate_code=raw_code,
                    parent_bom=o.parent_bom,
                    pos=o.pos,
                    note="; ".join(note_bits) if note_bits else "present as child but never exploded",
                )
            )
            continue

        # CASE 1: mai citato come child esatto => prova CODE_VARIANT, poi LIKELY_TYPO, altrimenti MISSING_LINK
        nr = _num_root(raw_code)
        if nr and nr in idx.child_by_numroot:
            # candidati con stesso root numerico (variante suffisso)
            cands = sorted(idx.child_by_numroot[nr])
            # filtra quelli con suffisso diverso (se suffisso esiste)
            sfx = _suffix(raw_code)
            variant = [c for c in cands if c != raw_code and (_suffix(c) != sfx)]
            if variant:
                best = variant[0]
                o = idx.occ_by_child.get(best, [None])[0]
                rows.append(
                    DiagnosisRow(
                        pbs_code=raw_code,
                        pbs_rev=pbs_rev,
                        diagnosis="CODE_VARIANT",
                        candidate_code=best,
                        parent_bom=(o.parent_bom if o else ""),
                        pos=(o.pos if o else ""),
                        note=f"same numeric root={nr}, suffix mismatch ({_suffix(raw_code)} vs {_suffix(best)})",
                    )
                )
                continue

        # LIKELY_TYPO: restringi candidati per performance
        search_pool: Set[str] = set()
        if nr and nr in idx.child_by_numroot:
            search_pool = set(idx.child_by_numroot[nr])
        else:
            pref = raw_code[:10] if len(raw_code) >= 10 else raw_code[:8]
            search_pool = set(idx.child_by_prefix.get(pref, set()))

        # fallback prudente (se pool vuoto): usa solo “simili” via difflib su tutto, ma limitato
        if not search_pool:
            # evita O(N^2) su dataset enormi: usa difflib.get_close_matches
            all_codes = list(idx.occ_by_child.keys())
            close = difflib.get_close_matches(raw_code, all_codes, n=max_candidates, cutoff=0.90)
            search_pool = set(close)

        scored: List[Tuple[int, float, str]] = []
        for c in search_pool:
            if c == raw_code:
                continue
            dist = _levenshtein(raw_code, c, max_dist=2)
            if dist <= 2:
                ratio = difflib.SequenceMatcher(None, raw_code, c).ratio()
                scored.append((dist, ratio, c))
            else:
                # prova ratio alta anche se dist>2 (swap/insert particolari)
                ratio = difflib.SequenceMatcher(None, raw_code, c).ratio()
                if ratio >= 0.92:
                    scored.append((dist, ratio, c))

        if scored:
            scored.sort(key=lambda t: (t[0], -t[1]))
            best_dist, best_ratio, best = scored[0]
            o = idx.occ_by_child.get(best, [None])[0]
            rows.append(
                DiagnosisRow(
                    pbs_code=raw_code,
                    pbs_rev=pbs_rev,
                    diagnosis="LIKELY_TYPO",
                    candidate_code=best,
                    parent_bom=(o.parent_bom if o else ""),
                    pos=(o.pos if o else ""),
                    note=f"edit_distance={best_dist}, ratio={best_ratio:.2f}",
                )
            )
            continue

        # niente trovato
        rows.append(
            DiagnosisRow(
                pbs_code=raw_code,
                pbs_rev=pbs_rev,
                diagnosis="MISSING_LINK",
                note="never appears as child in any BOM",
            )
        )

    return rows


def write_unexploded_report_csv(rows: Sequence[DiagnosisRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pbs_code", "pbs_rev", "diagnosis",
            "candidate_code", "parent_bom", "pos", "note",
            "classification", "root_chain_break", "cascade_depth", "not_reached_by_root",
        ])
        for r in rows:
            w.writerow([
                r.pbs_code, r.pbs_rev, r.diagnosis,
                r.candidate_code, r.parent_bom, r.pos, r.note,
                r.classification, r.root_chain_break, r.cascade_depth, r.not_reached_by_root,
            ])
