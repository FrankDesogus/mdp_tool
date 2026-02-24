# core/use_case/analyze_folder_pdf.py
from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ✅ PDF-only discovery
from core.parsers.discovery_pdf import DiscoveryPdfResult, discover_folder_pdf
from core.parsers.bom_pdf import parse_bom_pdf_raw, extract_root_code_from_title

from core.domain.models import BomDocument
from core.services.bom_normalizer import build_bom_document
from core.services.exploder_pdf import ExplodePolicy, explode_boms_pdf
from core.services.part_master import build_part_master, lookup_part_info

# ✅ PN canonicalization
from core.services.pn_canonical import canonicalize_part_number, canonicalize_rev

_DEBUG_PDF = os.getenv("BOM_PDF_DEBUG", "0").strip() in {"1", "true", "True"}

_LOG = logging.getLogger(__name__)
_WU_DEBUG_ENV = "BOM_PDF_WU_DEBUG"
_WU_DEBUG_TARGET = "166104001"


# -------------------------
# Report strutturati
# -------------------------
@dataclass(frozen=True)
class Issue:
    level: str  # "INFO" | "WARN" | "ERROR"
    message: str
    path: Optional[Path] = None
    code: str = ""


@dataclass
class AnalyzeFolderPdfResult:
    base_dir: Path
    discovery: DiscoveryPdfResult

    boms: List[BomDocument] = field(default_factory=list)

    # atteso: 1 sola root (code,rev). rev può essere "" (auto)
    roots: List[Tuple[str, str]] = field(default_factory=list)

    # una explosion per root (code,rev)
    explosions: Dict[Tuple[str, str], object] = field(default_factory=dict)

    issues: List[Issue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.level == "ERROR" for i in self.issues)


# -------------------------
# Helpers
# -------------------------
def _norm_rev(rev: str) -> str:
    return canonicalize_rev(rev or "")


def _canon_code(code: str, suffix: str) -> str:
    return canonicalize_part_number(code or "", suffix=_norm_rev(suffix))


def _canon_header_code(code: str, rev: str) -> str:
    """
    Canonicalizza PN header usando REV se disponibile.
    Esempi (dipende dall'implementazione di canonicalize_pn):
      - code='E0224103 01', rev='01'
      - code='E022410301',  rev='01'
      - code='E0224103',    rev=''
    """
    return _canon_code(code, rev)


def _canon_line_code(code: str, line_rev: str) -> str:
    """
    Canonicalizza PN di riga usando line.rev (se presente).
    """
    return _canon_code(code, line_rev)


def _norm_key(code: str) -> str:
    return (code or "").strip().upper()

def _fmt_qty(q: object) -> str:
    if q is None:
        return ""
    try:
        if isinstance(q, Decimal):
            return f"{q.normalize()}"
    except Exception:
        pass
    return str(q)


def _select_canonical_root_code(header: dict) -> str:
    """
    Source of truth per la root della BOM:
      1) header.root_code (se già valorizzato)
      2) estrazione da header.title
      3) fallback conservativo da header.code (es. "E0029472 01")
    """
    root = str(header.get("root_code") or "").strip().upper()
    if root:
        return root

    title = str(header.get("title") or "")
    root = extract_root_code_from_title(title) or ""
    if root:
        return root

    code = str(header.get("code") or "")
    return extract_root_code_from_title(code) or ""


def _select_header_code_effective(header: dict) -> str:
    """
    Header code da usare per il nodo BOM: usa la stessa canonicalizzazione
    impiegata nella costruzione del grafo.

    Nota:
    - root_code serve per inferenza root, NON per sostituire sempre header.code.
    - se header.code manca, fallback a root_code per non perdere completamente il nodo.
    """
    header_code_raw = str(header.get("code") or "").strip()
    header_rev = str(header.get("rev") or header.get("revision") or "")
    if header_code_raw:
        return _canon_header_code(header_code_raw, header_rev)
    return _canon_header_code(_select_canonical_root_code(header), "")


def build_bom_graph(
    boms: Iterable[BomDocument],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]], Set[str], Dict[str, str]]:
    """
    Come prima, ma ritorna anche alias_base_to_full.
    """
    # PASS 1: raccogli header_nodes + header_revs
    header_revs: Dict[str, Set[str]] = {}
    header_nodes: Set[str] = set()

    boms_list = list(boms)

    for b in boms_list:
        raw_hc = getattr(b.header, "code", "") or ""
        raw_hr = getattr(b.header, "revision", "") or ""
        parent = _canon_header_code(raw_hc, raw_hr)
        prev = _norm_rev(raw_hr)
        if not parent:
            continue
        header_nodes.add(parent)
        header_revs.setdefault(parent, set()).add(prev)

    # alias map (safe/univoca) costruita dagli header
    alias_base_to_full = build_base_to_full_alias_from_headers(header_nodes)

    # PASS 2: costruisci archi con child risolto via alias
    children_of: Dict[str, Set[str]] = {}
    parents_of: Dict[str, Set[str]] = {}

    for b in boms_list:
        raw_hc = getattr(b.header, "code", "") or ""
        raw_hr = getattr(b.header, "revision", "") or ""

        parent = _canon_header_code(raw_hc, raw_hr)
        if not parent:
            continue

        for ln in getattr(b, "lines", []) or []:
            raw_cc = getattr(ln, "internal_code", "") or ""
            if not raw_cc.strip():
                continue

            raw_lr = (getattr(ln, "rev", "") or "") if hasattr(ln, "rev") else ""
            child = _canon_line_code(raw_cc, raw_lr)
            if not child:
                continue

            # ✅ resolve base->full ONLY if safe & child isn't already header
            child_resolved = resolve_child_with_alias(
                child,
                header_nodes=header_nodes,
                alias_base_to_full=alias_base_to_full,
            )

            children_of.setdefault(parent, set()).add(child_resolved)
            parents_of.setdefault(child_resolved, set()).add(parent)

    return children_of, parents_of, header_revs, header_nodes, alias_base_to_full


def build_base_to_full_alias_from_headers(header_nodes: Set[str]) -> Dict[str, str]:
    """
    Costruisce una mappa base->full usando SOLO gli header.
    Regola conservativa:
      - alias solo per forme esplicitamente suffissate con "-NN" (es: E0254438-01 -> E0254438)
      - mapping valido SOLO se quel base mappa a UN SOLO header full

    Evita falsi alias su codici che terminano naturalmente con 2 cifre (es: E0181296).
    """
    buckets: Dict[str, Set[str]] = {}
    for full in header_nodes:
        if len(full) < 4:
            continue
        if full[-3] != "-" or not full[-2:].isdigit():
            continue
        base = full[:-3]
        if base:
            buckets.setdefault(base, set()).add(full)

    alias: Dict[str, str] = {}
    for base, fulls in buckets.items():
        if len(fulls) == 1:
            alias[base] = next(iter(fulls))
    return alias


def resolve_child_with_alias(
    child: str,
    *,
    header_nodes: Set[str],
    alias_base_to_full: Dict[str, str],
) -> str:
    """
    Se child NON è un header node ma esiste alias univoco base->full, risolve al full.
    """
    if not child:
        return child
    if child in header_nodes:
        return child
    return alias_base_to_full.get(child, child)


def _write_alias_suspects_csv(
    *,
    alias_base_to_full: Dict[str, str],
    parents_of: Dict[str, Set[str]],
    header_nodes: Set[str],
    out_path: Path,
) -> int:
    """
    Diagnostica: evidenzia base citati come child (indegree>0) ma non header,
    e il full corrispondente che invece è header.
    """
    rows = []
    for base, full in sorted(alias_base_to_full.items()):
        base_is_header = base in header_nodes
        full_is_header = full in header_nodes
        base_indeg = len(parents_of.get(base, set()) or [])
        full_indeg = len(parents_of.get(full, set()) or [])
        rows.append((base, full, base_is_header, full_is_header, base_indeg, full_indeg))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base", "full", "base_is_header", "full_is_header", "base_indegree", "full_indegree"])
        w.writerows(rows)
    return len(rows)


def infer_single_root_from_graph(
    *,
    parents_of: Dict[str, Set[str]],
    header_revs: Dict[str, Set[str]],
    header_nodes: Set[str],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Root = header node che non compare mai come child (indegree=0).

    Returns:
      roots: list of (code, rev)
        - atteso 1 elemento
        - rev = unica rev se presente, altrimenti "" (auto)
        - se multiple root: ritorna la lista (rev="") e lascia al chiamante decidere (no euristiche)
      debug_msgs: log diagnostici
    """
    debug: List[str] = []
    root_codes = sorted([h for h in header_nodes if not parents_of.get(h)])

    debug.append(f"header_nodes={len(header_nodes)} indegree0_roots={len(root_codes)}")

    if not root_codes:
        return [], debug

    if len(root_codes) > 1:
        debug.append("multiple roots detected (no heuristics applied).")
        for rc in root_codes[:30]:
            revs = sorted(header_revs.get(rc, set()))
            debug.append(f"root_candidate={rc} revs={revs}")
        return [(rc, "") for rc in root_codes], debug

    root_code = root_codes[0]
    revs = sorted(header_revs.get(root_code, set()))

    if len(revs) == 1:
        return [(root_code, revs[0])], debug

    debug.append(f"root has ambiguous revs: {revs}")
    return [(root_code, "")], debug


def _extract_folder_hint(folder: Path) -> Optional[Tuple[str, str]]:
    """
    Estrae hint (base, rev) dal nome cartella.
    Esempio: "E0181296 01-06" -> ("E0181296", "06")
    """
    text = (folder.name or "").upper()
    m = re.search(r"(E\d{7,8})\s*0?1\s*[-_ ]\s*(\d{1,2})", text)
    if not m:
        return None
    base = canonicalize_part_number(m.group(1))
    rev = _norm_rev(m.group(2))
    return (base, rev)


def _reachable_header_count(
    root_code: str,
    *,
    children_of: Dict[str, Set[str]],
    header_nodes: Set[str],
) -> int:
    seen: Set[str] = set()
    stack = [root_code]

    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for child in children_of.get(node, set()) or set():
            if child not in seen:
                stack.append(child)

    return len(seen & header_nodes)


def rank_roots(
    *,
    roots: List[Tuple[str, str]],
    folder_hint: Optional[Tuple[str, str]],
    children_of: Dict[str, Set[str]],
    header_nodes: Set[str],
) -> Tuple[Tuple[str, str], List[Dict[str, object]]]:
    """
    Ranking roots:
      1) folder match (base+rev) con priorità altissima
      2) reachable_count su header_nodes
      3) outdegree
      4) code lessicografico (stabilità)
    """
    rows: List[Dict[str, object]] = []

    for code, rev in roots:
        root_key = _norm_key(code)
        effective_rev = _norm_rev(rev)
        folder_match = 0
        if folder_hint:
            hint_code, hint_rev = folder_hint
            if _norm_key(hint_code) == root_key and _norm_rev(hint_rev) == effective_rev:
                folder_match = 1

        reachable_count = _reachable_header_count(code, children_of=children_of, header_nodes=header_nodes)
        outdegree = len(children_of.get(code, set()) or set())
        score = (1_000_000 if folder_match else 0) + (reachable_count * 1_000) + outdegree

        reason = []
        if folder_match:
            reason.append("folder_hint_match")
        reason.append(f"reachable={reachable_count}")
        reason.append(f"outdegree={outdegree}")

        rows.append(
            {
                "root_candidate": code,
                "root_rev": rev,
                "score": score,
                "outdegree": outdegree,
                "reachable_count": reachable_count,
                "folder_match": folder_match,
                "reason": ",".join(reason),
            }
        )

    rows.sort(
        key=lambda r: (
            -int(r["folder_match"]),
            -int(r["reachable_count"]),
            -int(r["outdegree"]),
            str(r["root_candidate"]),
        )
    )
    best = rows[0]
    return (str(best["root_candidate"]), str(best["root_rev"])), rows


# -------------------------
# Diagnostics writers (NEW/RESTORED)
# -------------------------
def _write_root_candidates_csv(
    *,
    roots: List[Tuple[str, str]],
    parents_of: Dict[str, Set[str]],
    header_revs: Dict[str, Set[str]],
    out_path: Path,
) -> int:
    """
    Diagnostica root candidates anche quando non esplodiamo.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["root_code", "root_rev", "indegree", "parents", "header_revs_seen"])
        for code, rev in roots:
            parents = sorted(parents_of.get(code, set()) or [])
            indeg = len(parents)
            seen_revs = sorted(header_revs.get(code, set()) or [])
            w.writerow(
                [
                    code,
                    rev,
                    indeg,
                    ";".join(parents),
                    ";".join(seen_revs),
                ]
            )
    return len(roots)


def _write_graph_stats_csv(
    *,
    children_of: Dict[str, Set[str]],
    parents_of: Dict[str, Set[str]],
    header_nodes: Set[str],
    out_path: Path,
    limit: int = 5000,
) -> int:
    """
    Dump base del grafo per capire dove si spezza:
      - node
      - is_header
      - indegree
      - outdegree

    limit: evita file enormi, ma 216 BOM non è un problema.
    """
    nodes: Set[str] = set()
    nodes.update(header_nodes)
    nodes.update(parents_of.keys())
    nodes.update(children_of.keys())

    # includi anche figli citati
    for p, chs in children_of.items():
        nodes.add(p)
        for c in chs:
            nodes.add(c)

    rows = []
    for n in nodes:
        indeg = len(parents_of.get(n, set()) or [])
        outdeg = len(children_of.get(n, set()) or [])
        rows.append((n, "Y" if n in header_nodes else "N", indeg, outdeg))

    rows.sort(key=lambda x: (-x[2], -x[3], x[0]))  # indegree desc, outdegree desc, pn asc
    rows = rows[: max(1, limit)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pn", "is_header", "indegree", "outdegree"])
        w.writerows(rows)
    return len(rows)


def _write_root_ranking_csv(*, ranking_rows: List[Dict[str, object]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["root_candidate", "root_rev", "score", "outdegree", "reachable_count", "folder_match", "reason"])
        for row in ranking_rows:
            w.writerow(
                [
                    row.get("root_candidate", ""),
                    row.get("root_rev", ""),
                    row.get("score", 0),
                    row.get("outdegree", 0),
                    row.get("reachable_count", 0),
                    row.get("folder_match", 0),
                    row.get("reason", ""),
                ]
            )
    return len(ranking_rows)


def _write_bom_explosion_tree_txt(explosion: object, out_path: Path, boms: Optional[list[object]] = None) -> int:
    if not explosion or not getattr(explosion, "edges", None):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("(Explode vuoto)\n", encoding="utf-8")
        return 0

    part_master = build_part_master(boms or [])

    lines: List[str] = []
    root = f"{getattr(explosion, 'root_code', '')} REV {getattr(explosion, 'root_rev', '')}"
    lines.append(f"ROOT: {root}")
    lines.append("")


    for e in getattr(explosion, "edges", []) or []:
        depth = int(getattr(e, "depth", 1) or 1)
        indent = "  " * max(0, depth - 1)
        description = (getattr(e, "description", "") or "").strip()


        parent = f"{getattr(e, 'parent_code', '')} REV {getattr(e, 'parent_rev', '')}"
        child_code = (getattr(e, "child_code", "") or "").strip()
        child_rev = (getattr(e, "child_rev", "") or "").strip()
        child = f"{child_code} REV {child_rev or '-'}"
        qty = _fmt_qty(getattr(e, "qty", ""))

        info = lookup_part_info(part_master, child_code, child_rev)
        desc = (("" if info is None else info.description) or (getattr(e, "description", "") or "")).strip()
        mfr = (("" if info is None else info.manufacturer) or (getattr(e, "manufacturer", "") or "")).strip()
        mfr_code = (("" if info is None else info.manufacturer_code) or (getattr(e, "manufacturer_code", "") or "")).strip()


        meta_parts = []
        if desc:
            meta_parts.append(f"desc={desc}")
        if mfr:
            meta_parts.append(f"mfr={mfr}")
        if mfr_code:
            meta_parts.append(f"mfr_code={mfr_code}")

        meta = f" [{' | '.join(meta_parts)}]" if meta_parts else ""

        lines.append(f"{indent}- {child} x{qty}{meta}   (parent: {parent})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _write_bom_flat_qty_csv(explosion: object, out_path: Path) -> int:
    qty_map = getattr(explosion, "qty_by_code", None)
    if not explosion or not qty_map:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        return 0

    rows = [(pn, _fmt_qty(qty)) for pn, qty in qty_map.items()]
    rows.sort(key=lambda x: x[0])

    if (os.getenv(_WU_DEBUG_ENV, "").strip() == "1"):
        qty_by_code_rev = getattr(explosion, "qty_by_code_rev", {}) or {}
        for pn, qty in rows:
            if _WU_DEBUG_TARGET not in pn:
                continue
            revs = [rev for (code, rev) in qty_by_code_rev.keys() if code == pn]
            rev_display = ",".join(sorted({(r or "").strip() for r in revs}))
            _LOG.info(
                "[WU_DEBUG][flat-csv-row] pn_display=%s rev_display=%s key_usata_per_grouping=%s qty_accumulated=%s",
                pn,
                rev_display,
                pn,
                qty,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PN", "QTY_TOTAL"])
        w.writerows(rows)

    return len(rows)


def _write_explode_resolution_traces_csv(explosion: object, out_path: Path) -> int:
    traces = getattr(explosion, "resolution_traces", None)
    if not explosion or not traces:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "context",
                "expected_code",
                "expected_rev",
                "source_parent_code",
                "source_parent_rev",
                "direct_candidates_count",
                "selected_header_code",
                "selected_header_rev",
                "selected_bom_path",
                "outcome",
                "note",
                "suggestion",
            ]
        )
        for t in traces:
            w.writerow(
                [
                    getattr(t, "context", ""),
                    getattr(t, "expected_code", ""),
                    getattr(t, "expected_rev", ""),
                    getattr(t, "source_parent_code", ""),
                    getattr(t, "source_parent_rev", ""),
                    getattr(t, "direct_candidates_count", 0),
                    getattr(t, "selected_header_code", ""),
                    getattr(t, "selected_header_rev", ""),
                    getattr(t, "selected_bom_path", ""),
                    getattr(t, "outcome", ""),
                    getattr(t, "note", ""),
                    getattr(t, "suggestion", ""),
                ]
            )
    return len(traces)


def _write_missing_edges_csv(explosion: object, out_path: Path) -> int:
    missing = getattr(explosion, "missing_edges", None)
    if not explosion or not missing:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parent_code", "parent_rev", "child_code", "pos", "unit", "kind", "parent_bom_path"])
        for m in missing:
            w.writerow(
                [
                    getattr(m, "parent_code", ""),
                    getattr(m, "parent_rev", ""),
                    getattr(m, "child_code", ""),
                    getattr(m, "pos", ""),
                    getattr(m, "unit", ""),
                    getattr(m, "kind", ""),
                    getattr(m, "parent_bom_path", ""),
                ]
            )
    return len(missing)


# -------------------------
# Use case
# -------------------------
class AnalyzeFolderPdfUseCase:
    """
    Entry point BOM-only (PDF).
    - Non richiede PBS.
    - Scansiona SOLO BOM PDF (discovery_pdf).
    - Parsea + normalizza le BOM.
    - Costruisce grafo globale (canonicalizzato) e inferisce ROOT (indegree=0).
    - Se root unica: explode UNA SOLA VOLTA dalla ROOT e produce i file classici.
    - Se root multiple: NON explode, ma produce diagnostica grafo/root in _diagnostics.
    """

    def run(self, folder: str | Path, progress_cb=None, log_cb=None, **kwargs) -> AnalyzeFolderPdfResult:
        base_dir = Path(folder).expanduser().resolve()

        def _log(level: str, msg: str) -> None:
            if log_cb:
                try:
                    log_cb(level, msg)
                except Exception:
                    pass

        def _prog(done: int, total: int, msg: str) -> None:
            if progress_cb:
                try:
                    progress_cb(done, total, msg)
                except Exception:
                    pass

        def _add_issue(level: str, message: str, path: Optional[Path], code: str) -> None:
            result.issues.append(Issue(level, message, path, code=code))
            _log(level, message)

        _log("INFO", f"[FOLDER] {base_dir}")

        # ✅ PDF-only discovery
        discovery = discover_folder_pdf(base_dir)
        result = AnalyzeFolderPdfResult(base_dir=base_dir, discovery=discovery)

        # ✅ ripristiniamo _diagnostics SEMPRE (anche se root multiple/early return)
        diag_dir = base_dir / "_diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        # 1) BOM PDF discovery
        bom_pdf_paths = list(discovery.bom_pdf)
        _add_issue(
            "INFO",
            f"[DISCOVERY] BOM PDF trovate: {len(bom_pdf_paths)}",
            base_dir,
            "DISCOVERY_BOM_PDF",
        )

        if not bom_pdf_paths:
            _add_issue(
                "ERROR",
                "Nessun file BOM PDF trovato (serve un .pdf con 'BOM' nel nome).",
                base_dir,
                "NO_BOM_PDF",
            )
            _prog(0, 1, "Nessuna BOM PDF trovata")
            return result

        # progress: parsing stage
        total_pdfs = len(bom_pdf_paths)
        _prog(0, total_pdfs, "Inizio parsing BOM PDF...")

        # 2) Parse + normalize
        bom_docs: List[BomDocument] = []
        for idx, p in enumerate(bom_pdf_paths, start=1):
            _prog(idx - 1, total_pdfs, f"Parsing {p.name} ({idx}/{total_pdfs})")
            _log("INFO", f"[PARSE] {p.name} ({idx}/{total_pdfs})")

            try:
                raw = parse_bom_pdf_raw(p)
                header = raw.get("header", {}) or {}
                raw_lines = raw.get("lines", []) or []

                # ⚠️ per ora NON spammiamo warnings: le salviamo ma non intasiamo
                for w in (raw.get("warnings") or []):
                    _add_issue("WARN", f"[BOM_PDF_PARSE_WARN] {p.name}: {w}", p, "BOM_PDF_PARSE")

                header_title = str(header.get("title") or "")
                header_code_raw = str(header.get("code") or "")
                header_rev = str(header.get("rev") or header.get("revision") or "")
                root_code = _select_canonical_root_code(header)
                header_code_effective = _select_header_code_effective(header)

                if _DEBUG_PDF:
                    _log(
                        "DEBUG",
                        f"[PDF_DEBUG] file={p.name} title_raw={header_title!r} header_code={header_code_raw!r} "
                        f"root_code={root_code or '(none)'} rev_header={header_rev or '(none)'}",
                    )
                if root_code and ("-" in root_code or " " in root_code):
                    _add_issue(
                        "WARN",
                        f"[ROOT_CODE] root_code non canonico: {root_code!r} (title={header_title!r})",
                        p,
                        "ROOT_CODE_NON_CANONICAL",
                    )

                bom = build_bom_document(
                    path=p,
                    header_code=header_code_effective,
                    header_rev=header_rev,
                    header_title=header_title,
                    doc_date_iso=str(header.get("date") or header.get("doc_date_iso") or ""),
                    raw_lines=raw_lines,
                )
                bom_docs.append(bom)
            except Exception as e:
                _add_issue(
                    "ERROR",
                    f"[BOM_PDF_PARSE] Errore parsing/normalizzazione: {p.name}: {type(e).__name__}: {e}",
                    p,
                    "BOM_PDF_PARSE",
                )

        _prog(total_pdfs, total_pdfs, "Parsing PDF completato")
        result.boms = bom_docs

        if not bom_docs:
            _add_issue("ERROR", "Nessuna BOM PDF parseata correttamente.", base_dir, "NO_BOMS")
            return result

        _add_issue(
            "INFO",
            f"[BOM_PDF_PARSE] BOM parseate OK: {len(bom_docs)}",
            base_dir,
            "BOM_PDF_PARSE_OK",
        )

        # 3) Build grafo globale + root discovery (indegree=0)
        _log("INFO", "[GRAPH] Build grafo globale...")
        _prog(total_pdfs, total_pdfs, "Costruzione grafo globale...")

        children_of, parents_of, header_revs, header_nodes, alias_base_to_full = build_bom_graph(bom_docs)

        # ✅ diagnostica grafo sempre (anche quando root multiple)
        _write_graph_stats_csv(
            children_of=children_of,
            parents_of=parents_of,
            header_nodes=header_nodes,
            out_path=diag_dir / "graph_stats.csv",
            limit=5000,
        )
        # ✅ diagnostica alias sempre (serve a provare i mismatch base->full)
        _write_alias_suspects_csv(
            alias_base_to_full=alias_base_to_full,
            parents_of=parents_of,
            header_nodes=header_nodes,
            out_path=diag_dir / "alias_suspects.csv",
        )

        roots, dbg = infer_single_root_from_graph(
            parents_of=parents_of,
            header_revs=header_revs,
            header_nodes=header_nodes,
        )

        if roots and _DEBUG_PDF:
            candidates = ", ".join(f"{c}:{r or '-'}" for c, r in roots)
            _log("DEBUG", f"[ROOT_DEBUG] root_candidates=[{candidates}]")
        elif _DEBUG_PDF:
            _log("DEBUG", "[ROOT_DEBUG] root_candidates=[]")

        if _DEBUG_PDF:
            graph_nodes = set(header_nodes) | set(parents_of.keys()) | set(children_of.keys())
            dash_nodes = sorted(n for n in graph_nodes if "-01" in (n or ""))
            _log(
                "DEBUG",
                f"[GRAPH_DEBUG] nodes={len(graph_nodes)} nodes_with_-01={len(dash_nodes)} sample={dash_nodes[:20]}",
            )

        if not roots:
            _add_issue(
                "ERROR",
                "[ROOT_INFERENCE] Nessuna root trovata (indegree=0). Dataset incoerente o canonicalizzazione insufficiente.",
                base_dir,
                "ROOT_INFERENCE_NONE",
            )
            _add_issue("INFO", f"[ROOT_DEBUG] {' | '.join(dbg)}", base_dir, "ROOT_DEBUG")
            return result
        _write_root_candidates_csv(
            roots=roots,
            parents_of=parents_of,
            header_revs=header_revs,
            out_path=diag_dir / "root_candidates.csv",
        )

        folder_hint = _extract_folder_hint(base_dir)
        selected_root, ranking_rows = rank_roots(
            roots=roots,
            folder_hint=folder_hint,
            children_of=children_of,
            header_nodes=header_nodes,
        )
        _write_root_ranking_csv(ranking_rows=ranking_rows, out_path=diag_dir / "root_ranking.csv")

        root_code, root_rev = selected_root
        result.roots = [(root_code, root_rev)]

        if len(roots) > 1:
            _add_issue(
                "WARN",
                f"[ROOT_INFERENCE] Trovate {len(roots)} roots. Selezionata migliore via ranking: {root_code} REV {root_rev or '(auto)'}.",
                base_dir,
                "ROOT_INFERENCE_MULTI",
            )
        else:
            _add_issue(
                "INFO",
                f"[ROOT_INFERENCE] Root unica: {root_code} REV {root_rev or '(auto)'}",
                base_dir,
                "ROOT_INFERENCE_OK",
            )

        top_reason = ranking_rows[0].get("reason", "") if ranking_rows else ""
        _add_issue("INFO", f"[ROOT_DEBUG] {' | '.join(dbg)} | selected_reason={top_reason}", base_dir, "ROOT_DEBUG")
        if _DEBUG_PDF:
            _log("DEBUG", f"[ROOT_DEBUG] root_selected={root_code} rev={root_rev or '(auto)'}")
        rows_for_root = sum(
            len(getattr(b, "lines", []) or [])
            for b in bom_docs
            if _norm_key(getattr(getattr(b, "header", None), "code", "")) == _norm_key(root_code)
        )
        if _DEBUG_PDF:
            _log("DEBUG", f"[ROOT_DEBUG] bom_rows_for_selected_root={rows_for_root}")
        _add_issue(
            "INFO",
            f"[DIAGNOSTICS] Creati: {(diag_dir / 'graph_stats.csv').name}, {(diag_dir / 'root_candidates.csv').name}, {(diag_dir / 'root_ranking.csv').name}",
            diag_dir,
            "DIAGNOSTICS",
        )

        # 4) Explode UNA sola volta (da root)
        _log("INFO", f"[EXPLODE] Avvio explode da root={root_code} rev={root_rev or '(auto)'}")
        _prog(0, 1, "Explode in corso...")

        policy = ExplodePolicy(
            strict_rev=True,
            explode_documents=True,
            root_strict_rev=False,
            recursive_fallback=True,
            recursive_pick_highest_rev=True,
            warn_missing_qty=False,
            warn_non_positive_qty=False,
        )

        exp = explode_boms_pdf(
            root_code=root_code,
            root_rev=root_rev,
            boms=bom_docs,
            pbs=None,
            policy=policy,
        )
        result.explosions[(root_code, root_rev)] = exp
        if (root_code, root_rev) not in result.explosions:
            available = ", ".join(f"{k[0]}:{k[1]}" for k in sorted(result.explosions.keys()))
            _log("DEBUG", f"[ROOT_DEBUG] selected_root_missing_from_explosions selected={root_code}:{root_rev} available=[{available}]")

        safe_root = f"{root_code}_REV_{(root_rev or 'AUTO')}".replace(" ", "_").replace("/", "_")
        tree_txt = diag_dir / f"bom_explosion_tree__{safe_root}.txt"
        flat_csv = diag_dir / f"bom_flat_qty__{safe_root}.csv"
        traces_csv = diag_dir / f"explode_resolution_traces__{safe_root}.csv"
        missing_csv = diag_dir / f"explode_missing_edges__{safe_root}.csv"

        _write_bom_explosion_tree_txt(exp, tree_txt, boms=list(getattr(result, "boms", []) or []))
        _write_bom_flat_qty_csv(exp, flat_csv)
        _write_explode_resolution_traces_csv(exp, traces_csv)
        _write_missing_edges_csv(exp, missing_csv)

        n_edges = len(getattr(exp, "edges", []) or [])
        n_items = len(getattr(exp, "qty_by_code", {}) or {})
        n_missing = len(getattr(exp, "missing_sub_boms", set()) or set())
        n_err = len(getattr(exp, "errors", []) or [])

        level = "INFO" if n_err == 0 else "WARN"
        _add_issue(
            level,
            f"[EXPLODE] ROOT {root_code} REV {root_rev or '(auto)'}: edges={n_edges} items={n_items} "
            f"missing_sub_boms={n_missing} errors={n_err} "
            f"| outputs: {tree_txt.name}, {flat_csv.name}",
            diag_dir,
            "EXPLODE",
        )

        for e in (getattr(exp, "errors", []) or [])[:40]:
            _add_issue("ERROR", f"[EXPLODE_ERR] {e}", diag_dir, "EXPLODE_ERR")

        _prog(1, 1, "Explode completato")
        return result


if __name__ == "__main__":
    import sys

    folder_arg = sys.argv[1] if len(sys.argv) > 1 else "."
    res = AnalyzeFolderPdfUseCase().run(folder_arg)

    print("\n=== AnalyzeFolderPdfUseCase ===")
    for i in res.issues:
        print(f"{i.level:<5} {i.code:<22} {i.message}")
