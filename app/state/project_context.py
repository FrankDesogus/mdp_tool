# app/state/project_context.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import os
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from core.services.part_master import PartInfo, build_part_master, lookup_part_info, missing_lookup_samples
# Import "best effort" per typing: non devono rompere se i moduli non sono disponibili.
try:
    from core.domain.models import PbsDocument, MdpRow  # type: ignore
except Exception:  # pragma: no cover
    PbsDocument = Any  # type: ignore
    MdpRow = Any  # type: ignore

try:
    from core.use_case.analyze_folder_pdf import AnalyzeFolderPdfResult  # type: ignore
except Exception:  # pragma: no cover
    AnalyzeFolderPdfResult = Any  # type: ignore

try:
    from core.use_case.analyze_folder import AnalyzeFolderResult  # type: ignore
except Exception:  # pragma: no cover
    AnalyzeFolderResult = Any  # type: ignore


_LOG = logging.getLogger(__name__)
_DEBUG_DIAG = os.getenv("MDP_DEBUG_DIAGNOSTICS", "0").strip() in {"1", "true", "True"}
_WU_DEBUG_ENV = "BOM_PDF_WU_DEBUG"
_WU_DEBUG_TARGET = "166104001"


def _find_pbs_root_row(pbs: Any) -> Tuple[int, Any]:
    """
    Ritorna (row_index, row_obj) della root PBS (best effort):
    - livello minimo
    - prima riga (ordine file)
    """
    rows = list(getattr(pbs, "rows", []) or [])
    if not rows:
        return -1, None
    min_level = min(int(getattr(r, "level", 0) or 0) for r in rows)
    for i, r in enumerate(rows):
        if int(getattr(r, "level", 0) or 0) == min_level:
            return i, r
    return 0, rows[0]


def _find_pbs_root_code_rev(pbs: Any) -> Tuple[str, str]:
    idx, r = _find_pbs_root_row(pbs)
    if r is None:
        return "", ""
    return (getattr(r, "code", "") or "").strip(), (getattr(r, "rev", "") or "").strip()


def _build_pbs_tree_indices_by_id(
    pbs: Any,
) -> Tuple[
    str,  # pbs_root_id
    Dict[str, List[str]],  # children_by_parent_id
    Dict[str, Any],  # row_by_id
    Dict[str, str],  # code_by_id
    Dict[str, str],  # parent_id_by_child_id
    Dict[str, List[Any]],  # occurrences_by_code (lista righe per code)
]:
    """
    PBS "occurrence-based" (robusto ai duplicati):
      - Ogni riga PBS diventa un nodo distinto (pbs_node_id)
      - La tree usa node_id come chiave (non code)

    node_id: "r{index}" (stabile rispetto all'ordine file).
    """
    rows = list(getattr(pbs, "rows", []) or [])
    if not rows:
        return "", {}, {}, {}, {}, {}

    # baseline livelli
    min_level = min(int(getattr(r, "level", 0) or 0) for r in rows)

    # root row
    root_idx, _root_row = _find_pbs_root_row(pbs)
    pbs_root_id = f"r{root_idx}" if root_idx >= 0 else "r0"

    children_dd: DefaultDict[str, List[str]] = defaultdict(list)
    row_by_id: Dict[str, Any] = {}
    code_by_id: Dict[str, str] = {}
    parent_by_child: Dict[str, str] = {}
    occ_dd: DefaultDict[str, List[Any]] = defaultdict(list)

    # stack: [(norm_level, node_id)]
    stack: List[Tuple[int, str]] = []

    for i, r in enumerate(rows):
        node_id = f"r{i}"
        code = (getattr(r, "code", "") or "").strip()
        rev = (getattr(r, "rev", "") or "").strip()  # non usato per key, ma utile al display
        lvl = int(getattr(r, "level", 0) or 0) - min_level

        row_by_id[node_id] = r
        code_by_id[node_id] = code
        if code:
            occ_dd[code].append(r)

        # pop finché lo stack non è il parent diretto
        while stack and stack[-1][0] >= lvl:
            stack.pop()

        if stack:
            parent_id = stack[-1][1]
            children_dd[parent_id].append(node_id)
            parent_by_child[node_id] = parent_id

        stack.append((lvl, node_id))

    return pbs_root_id, dict(children_dd), row_by_id, code_by_id, parent_by_child, dict(occ_dd)


def _build_pbs_code_based_indices_from_id_tree(
    pbs_root_id: str,
    children_by_parent_id: Dict[str, List[str]],
    code_by_id: Dict[str, str],
) -> Dict[str, List[str]]:
    """
    Compat / lookup: costruisce una mappa parent_code -> [child_code,...] *lossy*.
    NON usarla per renderizzare la tree se hai duplicati.
    """
    out: DefaultDict[str, List[str]] = defaultdict(list)
    stack = [pbs_root_id]
    seen = set()

    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)

        pcode = code_by_id.get(pid, "")
        for cid in children_by_parent_id.get(pid, []):
            ccode = code_by_id.get(cid, "")
            if pcode and ccode:
                out[pcode].append(ccode)
            stack.append(cid)

    return dict(out)


def _detect_bom_self_child_edges(edges: List[Any]) -> List[Any]:
    """
    Edge patologici: parent_code == child_code
    """
    bad: List[Any] = []
    for e in edges:
        p = (getattr(e, "parent_code", "") or "").strip()
        c = (getattr(e, "child_code", "") or "").strip()
        if p and c and p == c:
            bad.append(e)
    return bad


def _detect_bom_cycles_best_effort(
    children_by_parent: Dict[str, List[Any]],
    max_cycles: int = 50,
) -> List[List[str]]:
    """
    Cycle detection sul grafo manufacturing (codici).
    Best-effort, iterativo (no recursion depth).
    Ritorna una lista di cicli come path di codici (es: ["A","B","C","A"]).

    Nota: in grafi grandi può essere costoso; limitiamo max_cycles.
    """
    # adjacency: parent_code -> [child_code,...]
    adj: Dict[str, List[str]] = {}
    for p, edges in children_by_parent.items():
        ch: List[str] = []
        for e in edges:
            c = (getattr(e, "child_code", "") or "").strip()
            if c:
                ch.append(c)
        if ch:
            adj[p] = ch

    cycles: List[List[str]] = []
    visited_global: Set[str] = set()

    # DFS iterativa con stack di (node, iterator_index)
    for start in list(adj.keys()):
        if start in visited_global:
            continue

        stack: List[Tuple[str, int]] = [(start, 0)]
        parent: Dict[str, str] = {}
        in_stack: Set[str] = {start}

        while stack and len(cycles) < max_cycles:
            node, idx = stack[-1]
            visited_global.add(node)

            neigh = adj.get(node, [])
            if idx >= len(neigh):
                # finished node
                stack.pop()
                in_stack.discard(node)
                continue

            # advance iterator
            nxt = neigh[idx]
            stack[-1] = (node, idx + 1)

            if nxt not in adj:
                # leaf / not expanded
                continue

            if nxt in in_stack:
                # found back-edge => cycle
                # reconstruct path from node back to nxt
                path = [nxt]
                cur = node
                while True:
                    path.append(cur)
                    if cur == nxt:
                        break
                    if cur not in parent:
                        break
                    cur = parent[cur]
                path.reverse()
                # ensure cycle ends at start node again
                if path and path[0] == path[-1]:
                    cycles.append(path)
                else:
                    # normalize to explicit closure
                    if path:
                        cycles.append(path + [path[0]])
                continue

            if nxt not in parent:
                parent[nxt] = node
            stack.append((nxt, 0))
            in_stack.add(nxt)

    return cycles


@dataclass
class ProjectContext:
    """
    Contesto unico in memoria per la GUI.

    ⚠️ Distinzione fondamentale:
      - PBS Tree (documentale/logica) -> usare pbs_*_id (occurrence-based) quando disponibile
      - BOM Explosion Tree (manufacturing) -> usare bom_* + explosion (derivato da result.explosion)

    Supporta sia:
      - PDF_ONLY (AnalyzeFolderPdfResult: roots + explosions)
      - PBS_EXCEL (AnalyzeFolderResult: pbs + boms + explosion)
    """

    folder: Path
    mode: str = ""  # "PDF_ONLY" | "PBS_EXCEL" | ...

    # teniamo SEMPRE il result grezzo, così non perdi niente
    result: Any | None = None

    # compat: se ti serve distinguere
    pdf_result: Optional[Any] = None
    excel_result: Optional[Any] = None

    # ------------------------
    # Roots
    # ------------------------
    # Root manufacturing (root dell'explosion)
    root_code: str = ""
    root_rev: str = ""

    # Root PBS (root logica/documentale) [codice/rev]
    pbs_root_code: str = ""
    pbs_root_rev: str = ""

    # Root PBS (occurrence id)
    pbs_root_id: str = ""

    # ------------------------
    # Explosion (manufacturing)
    # ------------------------
    explosion: Any | None = None  # ExplosionResult (Any per evitare import pesanti)

    # Indici manufacturing (BOM explosion)
    bom_children_by_parent: Dict[str, List[Any]] = None  # edges by parent
    bom_parents_by_child: Dict[str, Set[str]] = None
    bom_occurrences_by_code: Dict[str, List[Any]] = None

    # Diagnostics manufacturing (B3)
    bom_self_child_edges: List[Any] = None
    bom_cycles: List[List[str]] = None  # lista cicli (best-effort)

    # ------------------------
    # PBS (documentale)
    # ------------------------
    pbs: Any | None = None  # PbsDocument (Any per non importare sempre)

    # NEW (robusto): tree per occorrenze
    pbs_children_by_parent_id: Dict[str, List[str]] = None
    pbs_row_by_id: Dict[str, Any] = None
    pbs_code_by_id: Dict[str, str] = None
    pbs_parent_id_by_child_id: Dict[str, str] = None

    # Compat / lookup (lossy) + supporto
    pbs_children_by_parent: Dict[str, List[str]] = None  # ⚠️ lossy, non usare per renderizzare se duplicati
    pbs_node_by_code: Dict[str, Any] = None              # first occurrence (best effort)
    pbs_occurrences_by_code: Dict[str, List[Any]] = None # occurrences vere per code (lista righe)

    # ------------------------
    # Legacy aliases (backward compatibility)
    # ------------------------
    # NOTE: mantenuti per non rompere viste esistenti che assumono “tree=manufacturing”.
    children_by_parent: Dict[str, List[Any]] = None  # alias di bom_children_by_parent
    parents_by_child: Dict[str, Set[str]] = None
    occurrences_by_code: Dict[str, List[Any]] = None

    # ------------------------
    # Master data e totali
    # ------------------------
    part_master: Dict[str, PartInfo] = None
    uom_by_code: Dict[str, str] = None
    qty_by_code: Dict[str, Any] = None

    def __post_init__(self) -> None:
        # manufacturing indices
        if self.bom_children_by_parent is None:
            self.bom_children_by_parent = {}
        if self.bom_parents_by_child is None:
            self.bom_parents_by_child = {}
        if self.bom_occurrences_by_code is None:
            self.bom_occurrences_by_code = {}

        # manufacturing diagnostics
        if self.bom_self_child_edges is None:
            self.bom_self_child_edges = []
        if self.bom_cycles is None:
            self.bom_cycles = []

        # PBS indices (ID-based)
        if self.pbs_children_by_parent_id is None:
            self.pbs_children_by_parent_id = {}
        if self.pbs_row_by_id is None:
            self.pbs_row_by_id = {}
        if self.pbs_code_by_id is None:
            self.pbs_code_by_id = {}
        if self.pbs_parent_id_by_child_id is None:
            self.pbs_parent_id_by_child_id = {}

        # PBS compat/lookup
        if self.pbs_children_by_parent is None:
            self.pbs_children_by_parent = {}
        if self.pbs_node_by_code is None:
            self.pbs_node_by_code = {}
        if self.pbs_occurrences_by_code is None:
            self.pbs_occurrences_by_code = {}

        # legacy aliases (default to manufacturing)
        if self.children_by_parent is None:
            self.children_by_parent = self.bom_children_by_parent
        if self.parents_by_child is None:
            self.parents_by_child = self.bom_parents_by_child
        if self.occurrences_by_code is None:
            self.occurrences_by_code = self.bom_occurrences_by_code

        if self.part_master is None:
            self.part_master = {}
        if self.uom_by_code is None:
            self.uom_by_code = {}
        if self.qty_by_code is None:
            self.qty_by_code = {}

    # ------------------------------------------------------------
    # ✅ ENTRYPOINT PRINCIPALE usato dalla GUI
    # ------------------------------------------------------------
    @classmethod
    def from_any_result(cls, folder: Path, result: Any, mode: str = "") -> "ProjectContext":
        """
        Factory unica usata dalla GUI.

        Regola:
          - se result ha (roots, explosions) => PDF_ONLY
          - se result ha (explosion) => PBS_EXCEL
        """
        folder = Path(folder).expanduser().resolve()
        mode = mode or ""

        if hasattr(result, "explosions") and hasattr(result, "roots"):
            return cls._from_pdf_only(folder=folder, result=result, mode=mode)

        if hasattr(result, "explosion"):
            return cls._from_pbs_excel(folder=folder, result=result, mode=mode)

        name = result.__class__.__name__ if result is not None else ""
        if name == "AnalyzeFolderPdfResult":
            return cls._from_pdf_only(folder=folder, result=result, mode=mode)
        if name == "AnalyzeFolderResult":
            return cls._from_pbs_excel(folder=folder, result=result, mode=mode)

        raise TypeError(f"Unknown analysis result type: {name}")

    @classmethod
    def from_any_result_legacy(cls, result: Any) -> "ProjectContext":
        base_dir = Path(getattr(result, "base_dir", "."))
        return cls.from_any_result(folder=base_dir, result=result, mode="")

    @classmethod
    def from_analyze_result(cls, r: AnalyzeFolderPdfResult) -> "ProjectContext":
        base_dir = Path(getattr(r, "base_dir", "."))
        return cls._from_pdf_only(folder=base_dir, result=r, mode="PDF_ONLY")

    # ------------------------
    # Internal builders
    # ------------------------
    @classmethod
    def _from_pdf_only(cls, folder: Path, result: Any, mode: str) -> "ProjectContext":
        roots = list(getattr(result, "roots", []) or [])
        if not roots:
            raise ValueError("AnalyzeFolderPdfResult has no roots.")

        root_code, root_rev = roots[0]
        explosions = getattr(result, "explosions", {}) or {}
        if _DEBUG_DIAG:
            available = sorted(f"{k[0]}:{k[1]}" for k in explosions.keys())
            _LOG.debug(
                "[ROOT_DEBUG] ui_selected_root=%s:%s explosion_lookup_key=%s:%s available=%s",
                root_code,
                root_rev,
                root_code,
                root_rev,
                available,
            )
        exp = explosions.get((root_code, root_rev))
        if exp is None:
            raise ValueError("AnalyzeFolderPdfResult has no explosion for the selected root.")

        edges = list(getattr(exp, "edges", []) or [])
        uom_by_code = dict(getattr(exp, "uom_by_code", {}) or {})
        qty_by_code = dict(getattr(exp, "qty_by_code", {}) or {})

        bom_children, bom_parents, bom_occ = cls._build_indices_from_edges(edges)

        # B3 manufacturing diagnostics
        self_child = _detect_bom_self_child_edges(edges)
        cycles = _detect_bom_cycles_best_effort(bom_children)

        boms = list(getattr(result, "boms", []) or [])
        part_master = build_part_master(boms)

        # In PDF_ONLY non abbiamo PBS: per uniformità, impostiamo pbs_root = root manufacturing.
        ctx = cls(
            folder=folder,
            mode=mode or "PDF_ONLY",
            result=result,
            pdf_result=result,
            excel_result=None,
            root_code=str(root_code or ""),
            root_rev=str(root_rev or ""),
            pbs_root_code=str(root_code or ""),
            pbs_root_rev=str(root_rev or ""),
            pbs_root_id="",
            explosion=exp,
            bom_children_by_parent=bom_children,
            bom_parents_by_child=bom_parents,
            bom_occurrences_by_code=bom_occ,
            bom_self_child_edges=self_child,
            bom_cycles=cycles,
            # legacy aliases
            children_by_parent=bom_children,
            parents_by_child=bom_parents,
            occurrences_by_code=bom_occ,
            part_master=part_master,
            uom_by_code=uom_by_code,
            qty_by_code=qty_by_code,
            # PBS fields remain empty/None
            pbs=None,
            pbs_children_by_parent_id={},
            pbs_row_by_id={},
            pbs_code_by_id={},
            pbs_parent_id_by_child_id={},
            pbs_children_by_parent={},
            pbs_node_by_code={},
            pbs_occurrences_by_code={},
        )
        cls._emit_lookup_diagnostics(part_master, edges)
        return ctx

    @classmethod
    def _from_pbs_excel(cls, folder: Path, result: Any, mode: str) -> "ProjectContext":
        exp = getattr(result, "explosion", None)
        pbs = getattr(result, "pbs", None)

        # Master data dalle BOM (best effort)
        boms = list(getattr(result, "boms", []) or [])
        part_master = build_part_master(boms)

        # PBS indices (robusto, occurrence-based)
        pbs_root_code, pbs_root_rev = "", ""
        pbs_root_id = ""
        pbs_children_by_parent_id: Dict[str, List[str]] = {}
        pbs_row_by_id: Dict[str, Any] = {}
        pbs_code_by_id: Dict[str, str] = {}
        pbs_parent_id_by_child: Dict[str, str] = {}
        pbs_occ_by_code: Dict[str, List[Any]] = {}
        pbs_children_code_based: Dict[str, List[str]] = {}
        pbs_first_row_by_code: Dict[str, Any] = {}

        if pbs is not None:
            pbs_root_code, pbs_root_rev = _find_pbs_root_code_rev(pbs)
            (
                pbs_root_id,
                pbs_children_by_parent_id,
                pbs_row_by_id,
                pbs_code_by_id,
                pbs_parent_id_by_child,
                pbs_occ_by_code,
            ) = _build_pbs_tree_indices_by_id(pbs)

            # compat/lookup (lossy) + first occurrence
            pbs_children_code_based = _build_pbs_code_based_indices_from_id_tree(
                pbs_root_id=pbs_root_id,
                children_by_parent_id=pbs_children_by_parent_id,
                code_by_id=pbs_code_by_id,
            )
            for code, occs in pbs_occ_by_code.items():
                if occs:
                    pbs_first_row_by_code[code] = occs[0]

        # Costruisci comunque un contesto valido anche senza explosion (GUI può mostrare PBS + diagnostics)
        if exp is None:
            return cls(
                folder=folder,
                mode=mode or "PBS_EXCEL",
                result=result,
                pdf_result=None,
                excel_result=result,
                root_code="",
                root_rev="",
                pbs_root_code=str(pbs_root_code),
                pbs_root_rev=str(pbs_root_rev),
                pbs_root_id=str(pbs_root_id),
                explosion=None,
                bom_children_by_parent={},
                bom_parents_by_child={},
                bom_occurrences_by_code={},
                bom_self_child_edges=[],
                bom_cycles=[],
                # legacy aliases (vuoti)
                children_by_parent={},
                parents_by_child={},
                occurrences_by_code={},
                part_master=part_master,
                uom_by_code={},
                qty_by_code={},
                # PBS
                pbs=pbs,
                pbs_children_by_parent_id=pbs_children_by_parent_id,
                pbs_row_by_id=pbs_row_by_id,
                pbs_code_by_id=pbs_code_by_id,
                pbs_parent_id_by_child_id=pbs_parent_id_by_child,
                pbs_children_by_parent=pbs_children_code_based,
                pbs_node_by_code=pbs_first_row_by_code,
                pbs_occurrences_by_code=pbs_occ_by_code,
            )

        root_code = getattr(exp, "root_code", "") or ""
        root_rev = getattr(exp, "root_rev", "") or ""

        edges = list(getattr(exp, "edges", []) or [])
        uom_by_code = dict(getattr(exp, "uom_by_code", {}) or {})
        qty_by_code = dict(getattr(exp, "qty_by_code", {}) or {})

        bom_children, bom_parents, bom_occ = cls._build_indices_from_edges(edges)

        # B3 manufacturing diagnostics
        self_child = _detect_bom_self_child_edges(edges)
        cycles = _detect_bom_cycles_best_effort(bom_children)

        ctx = cls(
            folder=folder,
            mode=mode or "PBS_EXCEL",
            result=result,
            pdf_result=None,
            excel_result=result,
            root_code=str(root_code),
            root_rev=str(root_rev),
            pbs_root_code=str(pbs_root_code),
            pbs_root_rev=str(pbs_root_rev),
            pbs_root_id=str(pbs_root_id),
            explosion=exp,
            bom_children_by_parent=bom_children,
            bom_parents_by_child=bom_parents,
            bom_occurrences_by_code=bom_occ,
            bom_self_child_edges=self_child,
            bom_cycles=cycles,
            # legacy aliases
            children_by_parent=bom_children,
            parents_by_child=bom_parents,
            occurrences_by_code=bom_occ,
            part_master=part_master,
            uom_by_code=uom_by_code,
            qty_by_code=qty_by_code,
            # PBS
            pbs=pbs,
            pbs_children_by_parent_id=pbs_children_by_parent_id,
            pbs_row_by_id=pbs_row_by_id,
            pbs_code_by_id=pbs_code_by_id,
            pbs_parent_id_by_child_id=pbs_parent_id_by_child,
            pbs_children_by_parent=pbs_children_code_based,
            pbs_node_by_code=pbs_first_row_by_code,
            pbs_occurrences_by_code=pbs_occ_by_code,
        )
        cls._emit_lookup_diagnostics(part_master, edges)
        return ctx

    def part_info_for(self, code: str, rev: str = "") -> Optional[PartInfo]:
        return lookup_part_info(self.part_master, code, rev)

    @staticmethod
    def _emit_lookup_diagnostics(part_master: Dict[str, PartInfo], edges: List[Any]) -> None:
        if not _DEBUG_DIAG:
            return
        total = 0
        miss = 0
        pairs: List[Tuple[str, str]] = []
        for e in edges:
            code = (getattr(e, "child_code", "") or "").strip()
            rev = (getattr(e, "child_rev", "") or "").strip()
            if not code:
                continue
            total += 1
            if lookup_part_info(part_master, code, rev) is None:
                miss += 1
                pairs.append((code, rev))
        pct = (100.0 * miss / total) if total else 0.0
        sample = missing_lookup_samples(part_master, pairs, sample_size=20)
        _LOG.info("[diag] part_master lookup misses: %s/%s (%.1f%%)%s", miss, total, pct, f" sample={sample}" if sample else "")

    @staticmethod
    def _build_indices_from_edges(
        edges: List[Any],
    ) -> tuple[Dict[str, List[Any]], Dict[str, Set[str]], Dict[str, List[Any]]]:
        children_by_parent_dd: DefaultDict[str, List[Any]] = defaultdict(list)
        parents_by_child_dd: DefaultDict[str, Set[str]] = defaultdict(set)
        occurrences_by_code_dd: DefaultDict[str, List[Any]] = defaultdict(list)

        for e in edges:
            parent = (getattr(e, "parent_code", "") or "").strip()
            child = (getattr(e, "child_code", "") or "").strip()
            if parent and child:
                children_by_parent_dd[parent].append(e)
                parents_by_child_dd[child].add(parent)
                occurrences_by_code_dd[child].append(e)

        parents_by_child = {k: set(v) for k, v in parents_by_child_dd.items()}
        occurrences_by_code = dict(occurrences_by_code_dd)
        if (os.getenv(_WU_DEBUG_ENV, "").strip() == "1"):
            for child_key in sorted(k for k in parents_by_child.keys() if _WU_DEBUG_TARGET in (k or "")):
                parents_list = sorted(parents_by_child.get(child_key, set()))
                _LOG.info(
                    "[WU_DEBUG][where-used-index] child_key=%s parents_list=%s count=%s",
                    child_key,
                    parents_list,
                    len(parents_list),
                )

        return (
            dict(children_by_parent_dd),
            parents_by_child,
            occurrences_by_code,
        )
