# core/services/chain_break_aggregation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from core.domain.models import PbsDocument
from core.services.exploder import ExplosionResult, _norm_code
from core.services.unexploded_diagnosis import DiagnosisRow, _parse_parent_header


@dataclass(frozen=True)
class ChainBreakRootRow:
    root_code: str
    root_rev: str

    bom_parent_code: str
    bom_parent_rev: str
    bom_parent_pos: str

    pbs_children_qty_count: int
    impacted_descendants_count: int

    suspected_reason: str
    suggested_action: str


def _build_pbs_parent_children(pbs: PbsDocument) -> Tuple[List[dict], Dict[int, List[int]]]:
    """
    Build a row-tree using row.level (stack).
    nodes[i] = {code, rev, qty, level}
    children[parent_i] = [child_i, ...]
    """
    nodes: List[dict] = []
    children: Dict[int, List[int]] = {}

    stack: List[int] = []  # indices of nodes, monotone by level
    for r in pbs.rows:
        code = _norm_code(r.code)
        rev = (r.rev or "").strip().upper()
        level = int(getattr(r, "level", 0) or 0)
        qty = float(getattr(r, "qty", 0) or 0)

        node_i = len(nodes)
        nodes.append({"code": code, "rev": rev, "qty": qty, "level": level})

        while stack and nodes[stack[-1]]["level"] >= level:
            stack.pop()

        if stack:
            parent_i = stack[-1]
            children.setdefault(parent_i, []).append(node_i)

        stack.append(node_i)

    return nodes, children


def _effective_direct_children_count(nodes: List[dict], children: Dict[int, List[int]], i: int) -> int:
    # “manufacturing children” = direct children with qty > 0
    return sum(1 for ci in children.get(i, []) if float(nodes[ci]["qty"]) > 0)


def _nearest_root_ancestor(
    nodes: List[dict],
    children: Dict[int, List[int]],
    node_index: int,
    root_codes: set,
) -> Tuple[str, int]:
    """
    Return (root_code, depth_from_root) for this node occurrence, or ("", -1).
    Depth is measured in PBS edges (not BOM).
    """
    # reconstruct parent pointers cheaply by scanning stack-like:
    # we can build parent array once:
    raise RuntimeError("internal helper must be wrapped by aggregate_chain_breaks (parent array built there)")


def aggregate_chain_breaks(
    *,
    pbs: PbsDocument,
    explosion: ExplosionResult,
    diag_rows: Sequence[DiagnosisRow],
) -> Tuple[List[DiagnosisRow], List[ChainBreakRootRow]]:
    """
    1) Identify root chain-break nodes:
       - in PBS has direct children with qty>0
       - appears as item in BOM explosion (reached)
       - but is a leaf in BOM explosion (not in exploded_assemblies)
    2) Classify diag rows: ROOT vs CASCADE; assign NOT_REACHED_BY_ROOT and cascade_depth
    3) Emit chain_break_roots rows with impact counts and suspected reason from resolution_traces
    """
    nodes, child_map = _build_pbs_parent_children(pbs)

    # build parent pointers
    parent: List[Optional[int]] = [None] * len(nodes)
    stack: List[int] = []
    for i, n in enumerate(nodes):
        lvl = n["level"]
        while stack and nodes[stack[-1]]["level"] >= lvl:
            stack.pop()
        parent[i] = stack[-1] if stack else None
        stack.append(i)

    # build “reached as item in BOM”
    reached_codes = set(getattr(explosion, "qty_by_code", {}).keys()) | {getattr(explosion, "root_code", "")}
    # “exploded as assembly”
    exploded_codes = {c for (c, _rv) in getattr(explosion, "exploded_assemblies", set())}

    # for each PBS node occurrence, decide if it qualifies as root
    root_codes: set = set()
    direct_children_count_by_code: Dict[str, int] = {}

    for i, n in enumerate(nodes):
        c = n["code"]
        if not c:
            continue
        eff_children = _effective_direct_children_count(nodes, child_map, i)
        direct_children_count_by_code[c] = max(direct_children_count_by_code.get(c, 0), eff_children)

        if eff_children <= 0:
            continue
        if c not in reached_codes:
            continue
        if c in exploded_codes:
            continue
        root_codes.add(c)

    # Map code -> best (root_code, depth) by scanning occurrences
    best_root_for_code: Dict[str, Tuple[str, int]] = {}

    for i, n in enumerate(nodes):
        code = n["code"]
        if not code:
            continue

        # climb ancestors to find nearest root
        cur = i
        depth = 0
        found_root = ""
        found_depth = -1

        if code in root_codes:
            found_root = code
            found_depth = 0
        else:
            depth = 0
            while parent[cur] is not None:
                cur = parent[cur]  # type: ignore[assignment]
                depth += 1
                ac = nodes[cur]["code"]
                if ac in root_codes:
                    found_root = ac
                    found_depth = depth
                    break

        if found_root:
            prev = best_root_for_code.get(code)
            if prev is None or found_depth < prev[1]:
                best_root_for_code[code] = (found_root, found_depth)

    # Helper: pick a trace for a root_code
    def _trace_reason(root_code: str) -> Tuple[str, str]:
        traces = [t for t in getattr(explosion, "resolution_traces", []) if _norm_code(t.expected_code) == root_code]
        # prefer child-resolution failures (most useful)
        traces.sort(key=lambda t: (t.context != "EXPLODE_CHILD", t.outcome == "OK"))
        if not traces:
            return ("unknown", "check explode_resolution_traces.csv")
        t = traces[0]
        outcome = (t.outcome or "").strip().upper()
        note = (t.note or "").strip()
        sugg = (t.suggestion or "").strip() or "check BOM header / alias / linking rule"

        if outcome == "SUSPICIOUS_HEADER_MISMATCH":
            return ("alias mismatch", sugg)
        if outcome == "AMBIGUOUS":
            return ("no sub-bom resolved (ambiguous)", sugg)
        if outcome == "REV_MISMATCH":
            return ("rev mismatch / selection blocked", sugg)
        if outcome == "MISSING_FILE":
            return ("no sub-bom resolved (missing file)", sugg)
        # fallback
        if note:
            return (note[:80], sugg)
        return ("no sub-bom resolved", sugg)

    # Update diag rows with ROOT/CASCADE classification
    updated: List[DiagnosisRow] = []
    for r in diag_rows:
        c = _norm_code(r.pbs_code)
        root, depth = best_root_for_code.get(c, ("", -1))

        # default fields
        root_flag = ""
        cascade_depth = ""
        classification = ""
        by_root = ""

        if root:
            if root == c:
                classification = "ROOT"
                root_flag = "1"
                cascade_depth = "0"
            else:
                classification = "CASCADE"
                root_flag = "0"
                cascade_depth = str(depth if depth >= 0 else "")
                by_root = root

        updated.append(
            DiagnosisRow(
                pbs_code=r.pbs_code,
                pbs_rev=r.pbs_rev,
                diagnosis=r.diagnosis,
                candidate_code=r.candidate_code,
                parent_bom=r.parent_bom,
                pos=r.pos,
                note=r.note,

                root_chain_break=root_flag,
                cascade_depth=cascade_depth,
                classification=classification,
                not_reached_by_root=by_root,
            )
        )

    # Root report rows
    # Impact count: number of distinct codes classified CASCADE under that root
    impacted: Dict[str, set] = {rc: set() for rc in root_codes}
    for r in updated:
        if r.classification == "CASCADE" and r.not_reached_by_root:
            impacted.setdefault(r.not_reached_by_root, set()).add(_norm_code(r.pbs_code))

    # Find BOM parent info for each root (from diag row of that root)
    diag_by_code = { _norm_code(r.pbs_code): r for r in updated }

    root_rows: List[ChainBreakRootRow] = []
    for rc in sorted(root_codes):
        dr = diag_by_code.get(rc)
        bom_parent_code, bom_parent_rev = ("", "")
        bom_parent_pos = ""
        root_rev = (dr.pbs_rev if dr else "")

        if dr:
            bom_parent_code, bom_parent_rev = _parse_parent_header(dr.parent_bom)
            bom_parent_pos = dr.pos or ""

        pbs_children_qty_count = int(direct_children_count_by_code.get(rc, 0))
        impacted_descendants_count = len(impacted.get(rc, set()))

        suspected_reason, suggested_action = _trace_reason(rc)

        root_rows.append(
            ChainBreakRootRow(
                root_code=rc,
                root_rev=root_rev,
                bom_parent_code=bom_parent_code,
                bom_parent_rev=bom_parent_rev,
                bom_parent_pos=bom_parent_pos,
                pbs_children_qty_count=pbs_children_qty_count,
                impacted_descendants_count=impacted_descendants_count,
                suspected_reason=suspected_reason,
                suggested_action=suggested_action,
            )
        )

    return updated, root_rows


def write_chain_break_roots_csv(rows: Sequence[ChainBreakRootRow], out_path) -> None:
    import csv
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "root_code",
            "root_rev",
            "bom_parent_code",
            "bom_parent_rev",
            "bom_parent_pos",
            "pbs_children_qty_count",
            "impacted_descendants_count",
            "suspected_reason",
            "suggested_action",
        ])
        for r in rows:
            w.writerow([
                r.root_code,
                r.root_rev,
                r.bom_parent_code,
                r.bom_parent_rev,
                r.bom_parent_pos,
                r.pbs_children_qty_count,
                r.impacted_descendants_count,
                r.suspected_reason,
                r.suggested_action,
            ])
