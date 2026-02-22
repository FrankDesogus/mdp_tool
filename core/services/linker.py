# mdp_tool/core/services/linker.py
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from core.domain.models import (
    PbsDocument,
    BomDocument,
    BomRef,
    LinkResult,
    LinkStatus,
    key_code_rev,
)


# --- NEW: numeric root helper (same idea used in exploder) ---
_num_root_re = re.compile(r"(\d{6,})")  # at least 6 digits is a good industrial heuristic


def _norm_code(code: str) -> str:
    return (code or "").strip().upper()


def _numeric_root(code: str) -> str:
    """
    Extract numeric root from a PN, e.g. '231018117ASSY' -> '231018117'.
    Returns '' if not found.
    """
    s = _norm_code(code)
    if not s:
        return ""
    m = _num_root_re.search(s)
    return m.group(1) if m else ""


def index_boms(boms: Iterable[BomDocument]) -> Dict[Tuple[str, str], List[BomDocument]]:
    idx: Dict[Tuple[str, str], List[BomDocument]] = defaultdict(list)
    for b in boms:
        k = key_code_rev(b.header.code, b.header.revision)
        idx[k].append(b)
    return idx


# --- NEW: secondary index by numeric root (to detect header PN wrong / variants) ---
def index_boms_by_numeric_root(boms: Iterable[BomDocument]) -> Dict[str, List[BomDocument]]:
    idx: Dict[str, List[BomDocument]] = defaultdict(list)
    for b in boms:
        root = _numeric_root(b.header.code)
        if root:
            idx[root].append(b)
    return idx


def link_pbs_to_boms(
    pbs: PbsDocument,
    boms: Iterable[BomDocument],
) -> List[LinkResult]:
    """
    Per ogni riga PBS (assembly) crea un LinkResult:
    - OK se esiste una BOM unica con stesso (code, rev)
    - MISSING se 0
    - AMBIGUOUS se >1

    Nota: qui non decidiamo *quali* righe PBS siano assembly vs leaf: per ora
    linkiamo tutte le righe con code non vuoto (poi affiniamo con regole MDP).

    NEW:
    - se status=MISSING, cerchiamo BOM con stesso numeric-root ma header PN diverso:
      tipico caso "file esiste, ma header PN sbagliato / variante (ASSY/ASSD/...)"
    """
    idx = index_boms(boms)
    by_root = index_boms_by_numeric_root(boms)

    results: List[LinkResult] = []

    for r in pbs.rows:
        code = (r.code or "").strip()
        rev = (r.rev or "").strip()
        if not code:
            continue

        k = key_code_rev(code, rev)
        matches = idx.get(k, [])

        if len(matches) == 1:
            m = matches[0]
            results.append(
                LinkResult(
                    assembly_code=code,
                    assembly_rev=rev,
                    status=LinkStatus.OK,
                    bom=BomRef(path=m.path, code=m.header.code, revision=m.header.revision),
                    message="Match strict OK (code+rev).",
                )
            )
            continue

        if len(matches) == 0:
            # NEW: suspicious detection based on numeric root
            expected_norm = _norm_code(code)
            root = _numeric_root(expected_norm)

            suspicious: List[BomRef] = []
            if root:
                for b in by_root.get(root, []):
                    hcode = _norm_code(b.header.code)
                    if hcode == expected_norm:
                        continue
                    suspicious.append(BomRef(path=b.path, code=b.header.code, revision=b.header.revision))

            if suspicious:
                # Keep status=MISSING (strict missing), but attach candidates for explainability/action
                results.append(
                    LinkResult(
                        assembly_code=code,
                        assembly_rev=rev,
                        status=LinkStatus.MISSING,
                        candidates=tuple(suspicious),
                        message=(
                            "Nessuna BOM trovata per (code+rev) in modo strict. "
                            "Però esistono BOM con stesso numeric-root ma header PN diverso: "
                            "probabile PN header errato / variante (ASSY/ASSD/...) / refuso."
                        ),
                    )
                )
            else:
                results.append(
                    LinkResult(
                        assembly_code=code,
                        assembly_rev=rev,
                        status=LinkStatus.MISSING,
                        message="Nessuna BOM trovata per (code+rev).",
                    )
                )
            continue

        # len(matches) > 1
        cands = tuple(BomRef(path=m.path, code=m.header.code, revision=m.header.revision) for m in matches)
        results.append(
            LinkResult(
                assembly_code=code,
                assembly_rev=rev,
                status=LinkStatus.AMBIGUOUS,
                candidates=cands,
                message="Più BOM trovate per (code+rev): ambiguità bloccante.",
            )
        )

    return results
