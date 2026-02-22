# core/use_case/analyze_folder_pdf.py
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ✅ PDF-only discovery
from core.parsers.discovery_pdf import DiscoveryPdfResult, discover_folder_pdf
from core.parsers.bom_pdf import parse_bom_pdf_raw

from core.domain.models import BomDocument
from core.services.bom_normalizer import build_bom_document
from core.services.exploder_pdf import ExplodePolicy, explode_boms_pdf
from core.services.part_master import build_part_master, lookup_part_info

# ✅ PN canonicalization
from core.services.pn_canonical import canonicalize_pn, canonicalize_rev


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


def _canon_header_code(code: str, rev: str) -> str:
    """
    Canonicalizza PN header usando REV se disponibile.
    Esempi (dipende dall'implementazione di canonicalize_pn):
      - code='E0224103 01', rev='01'
      - code='E022410301',  rev='01'
      - code='E0224103',    rev=''
    """
    return canonicalize_pn(code or "", rev=_norm_rev(rev))


def _canon_line_code(code: str, line_rev: str) -> str:
    """
    Canonicalizza PN di riga usando line.rev (se presente).
    """
    return canonicalize_pn(code or "", rev=_norm_rev(line_rev))


def _fmt_qty(q: object) -> str:
    if q is None:
        return ""
    try:
        if isinstance(q, Decimal):
            return f"{q.normalize()}"
    except Exception:
        pass
    return str(q)


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
      - se header termina con 2 cifre (es: ...01), base = header[:-2]
      - mapping valido SOLO se quel base mappa a UN SOLO header full
    """
    buckets: Dict[str, Set[str]] = {}
    for full in header_nodes:
        if len(full) >= 2 and full[-2:].isdigit():
            base = full[:-2]
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

                bom = build_bom_document(
                    path=p,
                    header_code=str(header.get("code") or ""),
                    header_rev=str(header.get("rev") or header.get("revision") or ""),
                    header_title=str(header.get("title") or ""),
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

        if not roots:
            _add_issue(
                "ERROR",
                "[ROOT_INFERENCE] Nessuna root trovata (indegree=0). Dataset incoerente o canonicalizzazione insufficiente.",
                base_dir,
                "ROOT_INFERENCE_NONE",
            )
            _add_issue("INFO", f"[ROOT_DEBUG] {' | '.join(dbg)}", base_dir, "ROOT_DEBUG")
            return result

        if len(roots) > 1:
            # niente euristiche: segnaliamo e stop, MA scriviamo diagnostica
            result.roots = roots
            _add_issue(
                "WARN",
                f"[ROOT_INFERENCE] Trovate {len(roots)} roots (atteso 1). Nessuna euristica applicata.",
                base_dir,
                "ROOT_INFERENCE_MULTI",
            )
            _add_issue("INFO", f"[ROOT_DEBUG] {' | '.join(dbg)}", base_dir, "ROOT_DEBUG")

            _write_root_candidates_csv(
                roots=roots,
                parents_of=parents_of,
                header_revs=header_revs,
                out_path=diag_dir / "root_candidates.csv",
            )

            _add_issue(
                "INFO",
                f"[DIAGNOSTICS] Creati: {(diag_dir / 'graph_stats.csv').name}, {(diag_dir / 'root_candidates.csv').name}",
                diag_dir,
                "DIAGNOSTICS",
            )
            _prog(1, 1, "Root multiple: analisi fermata (vedi diagnostica)")
            return result

        root_code, root_rev = roots[0]
        result.roots = [(root_code, root_rev)]
        _add_issue(
            "INFO",
            f"[ROOT_INFERENCE] Root unica: {root_code} REV {root_rev or '(auto)'}",
            base_dir,
            "ROOT_INFERENCE_OK",
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
