# core/use_case/analyze_folder.py
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Callable, Any

from core.parsers.discovery import DiscoveryResult, choose_single_pbs, discover_folder
from core.parsers.bom_excel import parse_bom_excel_raw
from core.parsers.bom_pdf import parse_bom_pdf_raw

from core.domain.models import BomDocument, LinkResult, PbsDocument
from core.services.bom_normalizer import build_bom_document
from core.services.linker import link_pbs_to_boms

from core.services.exploder import ExplodePolicy, explode_boms, find_root_from_pbs
from core.services.config_compare import compare_explosion_to_pbs, ConfigCompletenessReport

from core.services.bom_prefilter import (
    BomTarget,
    prefilter_boms_by_filename,
    build_targets_from_pbs,
    verify_bom_header_against_targets,  # kept for compatibility (not used after NEW swap)
    verify_bom_header_with_diagnosis,   # ✅ NEW
    BomHeaderVerification,              # ✅ NEW
)

from core.services.odoo_pbs_export import export_pbs_to_odoo_csv
from core.services.odoo_pbs_csv_check import check_pbs_odoo_csv

from core.services.unexploded_diagnosis import (
    diagnose_unexploded_assemblies,
    write_unexploded_report_csv,
)

# ✅ NEW: parent → children set compare (local mismatch diagnosis)
from core.services.parent_child_compare import (
    diagnose_parent_children_mismatches,
    write_parent_child_mismatch_csv,
    summarize_parent_child_mismatches,
)
from core.services.chain_break_aggregation import (
    aggregate_chain_breaks,
    write_chain_break_roots_csv,
)

# ✅ NEW: inferenza qty mancanti + occurrences PBS
from core.services.qty_inference import (
    infer_missing_edge_quantities,
    write_inferred_qty_csv,
    compute_occurrences_from_pbs,
)

# ✅ NEW: TOTALIZATION vs BOM EXPLOSION validation (definitive rules)
from core.services.totalization_compare import (
    compare_totalization_to_bom_explosion,
    write_totalization_vs_bom_validation_csv,
    summarize_totalization_validation,
)

# -------------------------
# Report strutturati
# -------------------------
@dataclass(frozen=True)
class Issue:
    level: str              # "INFO" | "WARN" | "ERROR"
    message: str
    path: Optional[Path] = None
    code: str = ""


@dataclass
class AnalyzeFolderResult:
    base_dir: Path
    discovery: DiscoveryResult

    pbs: Optional[PbsDocument] = None
    boms: List[BomDocument] = field(default_factory=list)

    links: List[LinkResult] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)

    explosion: Optional[object] = None
    config_report: Optional[ConfigCompletenessReport] = None

    @property
    def has_errors(self) -> bool:
        return any(i.level == "ERROR" for i in self.issues)

    @property
    def link_ok(self) -> bool:
        return not any(i.level == "ERROR" and (i.code or "").startswith("LINK") for i in self.issues)


# -------------------------
# ✅ NEW: Issue list che fa "streaming" su log_cb
# -------------------------
class _IssueList(list):
    def __init__(self, log_cb: Optional[Callable[[str, str], None]] = None):
        super().__init__()
        self._log_cb = log_cb

    def append(self, item: Issue) -> None:  # type: ignore[override]
        super().append(item)
        if self._log_cb:
            try:
                self._log_cb(item.level, item.message)
            except Exception:
                pass


# -------------------------
# Helper stampa/log ordinati (MODULE-LEVEL)
# -------------------------
def _fmt_qty(q: object) -> str:
    if q is None:
        return ""
    try:
        # ExplosionEdge.qty spesso è Decimal
        if isinstance(q, Decimal):
            return f"{q.normalize()}"
    except Exception:
        pass
    return str(q)


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
    except Exception:
        return None


def _qty_close(a: Optional[Decimal], b: Optional[Decimal], tol: Decimal = Decimal("0.0001")) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def _write_pbs_tree_txt(pbs: Optional[PbsDocument], out_path: Path) -> int:
    """
    Stampa l'albero PBS/MDP usando row.level come indent.
    """
    if not pbs or not getattr(pbs, "rows", None):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("(PBS vuoto)\n", encoding="utf-8")
        return 0

    rows = list(pbs.rows)
    min_level = min(int(getattr(x, "level", 0) or 0) for x in rows) if rows else 0

    lines: List[str] = []
    for r in rows:
        level = int(getattr(r, "level", 0) or 0)
        indent = "  " * max(0, level - min_level)
        code = (getattr(r, "code", "") or "").strip()
        rev = (getattr(r, "rev", "") or "").strip()
        qty = getattr(r, "qty", "")
        desc = (getattr(r, "description", "") or "").strip()
        lines.append(f"{indent}- {code}  REV {rev or '-'}  QTY {qty}  | {desc}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _write_bom_explosion_tree_txt(explosion: object, out_path: Path) -> int:
    """
    Stampa l'albero di esplosione BOM usando ExplosionEdge.depth come indent.
    Non dipende dal PBS.
    """
    if not explosion or not getattr(explosion, "edges", None):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("(Explode vuoto)\n", encoding="utf-8")
        return 0

    lines: List[str] = []
    root = f"{getattr(explosion, 'root_code', '')} REV {getattr(explosion, 'root_rev', '')}"
    lines.append(f"ROOT: {root}")
    lines.append("")

    for e in getattr(explosion, "edges", []) or []:
        depth = int(getattr(e, "depth", 1) or 1)
        indent = "  " * max(0, depth - 1)

        parent = f"{getattr(e, 'parent_code', '')} REV {getattr(e, 'parent_rev', '')}"
        child = f"{getattr(e, 'child_code', '')} REV {getattr(e, 'child_rev', '') or '-'}"
        qty = _fmt_qty(getattr(e, "qty", ""))

        lines.append(f"{indent}- {child}  x{qty}   (parent: {parent})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _write_bom_flat_qty_csv(explosion: object, out_path: Path) -> int:
    """
    CSV con quantità aggregate per PN (flat).
    NB: nel tuo contesto questa qty rappresenta la Q.Tà TOTALE LOTTO (per NR).
    """
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


def _chunk(items: List[str], size: int) -> List[List[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _format_missing_targets(missing: List[BomTarget], per_line: int = 8) -> str:
    entries = [f"{t.original_code} REV {t.rev or '?'}" for t in missing]
    lines: List[str] = []
    for group in _chunk(entries, per_line):
        lines.append("  - " + "; ".join(group))
    return "\n".join(lines)


def _truncate(s: object, max_len: int) -> str:
    text = "" if s is None else str(s)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _print_bom_lines_table(bom: BomDocument, max_rows: int = 15) -> None:
    lines = getattr(bom, "lines", []) or []
    if not lines:
        print("(BOM senza righe)")
        return

    header = (
        f"{'POS':>4}  {'CODE':<18}  {'QTY':>8}  {'UM':<4}  {'DESC':<38}  "
        f"{'VAL':<10}  {'REFDES':<18}  {'MFR':<14}  {'MFR_CODE':<16}"
    )
    print(header)
    print("-" * len(header))

    for l in lines[:max_rows]:
        pos = getattr(l, "pos", "")
        code = getattr(l, "internal_code", "")
        qty = getattr(l, "qty", "")
        um = getattr(l, "unit", "")
        desc = getattr(l, "description", "")
        val = getattr(l, "val", "")
        refdes = getattr(l, "refdes", "")
        mfr = getattr(l, "manufacturer", "")
        mfr_code = getattr(l, "manufacturer_code", "")

        row = (
            f"{str(pos):>4}  "
            f"{_truncate(code, 18):<18}  "
            f"{_truncate(qty, 8):>8}  "
            f"{_truncate(um, 4):<4}  "
            f"{_truncate(desc, 38):<38}  "
            f"{_truncate(val, 10):<10}  "
            f"{_truncate(refdes, 18):<18}  "
            f"{_truncate(mfr, 14):<14}  "
            f"{_truncate(mfr_code, 16):<16}"
        )
        print(row)

    if len(lines) > max_rows:
        print(f"... ({len(lines) - max_rows} righe non mostrate)")


# -------------------------
# COM Excel helpers - SOLO per ridurre crash/lentezza
# -------------------------
def _try_create_excel_app():
    """
    Crea UNA istanza Excel COM in modalità 'safe' per parsing massivo.
    Ritorna None se win32com non è disponibile.
    """
    try:
        import win32com.client  # type: ignore
    except Exception:
        return None

    excel = win32com.client.DispatchEx("Excel.Application")

    # Safe flags: riducono prompt/aggiornamenti e ricalcoli
    for setter in (
        ("Visible", False),
        ("DisplayAlerts", False),
        ("ScreenUpdating", False),
        ("EnableEvents", False),
        ("AskToUpdateLinks", False),
    ):
        try:
            setattr(excel, setter[0], setter[1])
        except Exception:
            pass

    try:
        # xlCalculationManual = -4135
        excel.Calculation = -4135
    except Exception:
        pass

    return excel


def _safe_quit_excel(excel) -> None:
    if excel is None:
        return
    try:
        excel.Quit()
    except Exception:
        pass


def _parse_bom_excel_with_optional_excel(path: Path, excel) -> dict:
    """
    Chiama parse_bom_excel_raw:
      - prima prova a passare excel=excel (nuova firma)
      - se non supportato, ripiega su firma vecchia (solo path)
    """
    try:
        return parse_bom_excel_raw(path, excel=excel)  # type: ignore[arg-type]
    except TypeError:
        return parse_bom_excel_raw(path)  # type: ignore[arg-type]


# -------------------------
# Diagnostics CSV writers
# -------------------------
def _write_rev_fallbacks_csv(explosion: object, out_path: Path) -> int:
    rows = list(getattr(explosion, "rev_mismatch_sub_boms", []) or [])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pn", "pbs_rev_requested", "bom_rev_selected"])
        for pn, exp, found in rows:
            w.writerow([pn, exp, found])
    return len(rows)


def _write_missing_sub_boms_csv(explosion: object, out_path: Path) -> int:
    missing = sorted(list(getattr(explosion, "missing_sub_boms", set()) or set()))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pn"])
        for pn in missing:
            w.writerow([pn])
    return len(missing)


def _write_cycles_csv(explosion: object, out_path: Path) -> int:
    cycles = list(getattr(explosion, "cycles", []) or [])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cycle_path"])
        for c in cycles:
            try:
                w.writerow([" -> ".join(list(c))])
            except Exception:
                w.writerow([str(c)])
    return len(cycles)


def _write_bom_header_audit_csv(rows: List[dict], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "file",
        "prefilter_strength",
        "prefilter_match_len",
        "prefilter_best_target",
        "pbs_src_row",
        "target_prtl_code",
        "target_prtl_rev",
        "expected_parent_code",
        "expected_parent_rev",
        "header_code",
        "header_rev",
        "header_code_source",
        "header_code_cell",
        "header_rev_source",
        "header_rev_cell",
        "header_code_label_cell",
        "header_rev_label_cell",
        "verification_reason",
        "suggestion",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    return len(rows)


def _write_explode_resolution_traces_csv(explosion: object, out_path: Path) -> int:
    traces = list(getattr(explosion, "resolution_traces", []) or [])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "context",
        "expected_code",
        "expected_rev",
        "source_parent_code",
        "source_parent_rev",
        "direct_candidates_count",
        "selected_bom_path",
        "selected_header_code",
        "selected_header_rev",
        "outcome",
        "note",
        "suggestion",
        "suspicious_same_root",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in traces:
            suspicious = ""
            try:
                lst = getattr(t, "suspicious_same_root", []) or []
                suspicious = " | ".join(
                    [
                        f"{getattr(x, 'header_code', '')} rev {getattr(x, 'header_rev', '')} @ {getattr(x, 'bom_path', '')}"
                        for x in lst
                    ]
                )
            except Exception:
                suspicious = ""

            w.writerow([
                getattr(t, "context", ""),
                getattr(t, "expected_code", ""),
                getattr(t, "expected_rev", ""),
                getattr(t, "source_parent_code", ""),
                getattr(t, "source_parent_rev", ""),
                getattr(t, "direct_candidates_count", 0),
                getattr(t, "selected_bom_path", ""),
                getattr(t, "selected_header_code", ""),
                getattr(t, "selected_header_rev", ""),
                getattr(t, "outcome", ""),
                getattr(t, "note", ""),
                getattr(t, "suggestion", ""),
                suspicious,
            ])

    return len(traces)


def _write_totalization_duplicates_csv(tot_res: object, out_path: Path) -> int:
    """
    CSV: codici duplicati nella totalizzazione (conteggio righe + qty somma).
    Usa i campi nuovi del parser: duplicate_codes, row_count_by_code, qty_by_code.
    """
    dup = list(getattr(tot_res, "duplicate_codes", []) or [])
    row_count = getattr(tot_res, "row_count_by_code", {}) or {}
    qty_by_code = getattr(tot_res, "qty_by_code", {}) or {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PN", "ROWS_COUNT", "QTY_SUM"])
        for pn in dup:
            w.writerow([pn, row_count.get(pn, 0), _fmt_qty(qty_by_code.get(pn))])

    return len(dup)


def _write_totalization_missing_vs_bom_csv(
    bom_flat_qty: Dict[str, Decimal],
    totals_override: Dict[str, Decimal],
    out_path: Path,
) -> int:
    """
    CSV: codici presenti in BOM flat (explosion.qty_by_code) ma non nella totalizzazione.
    """
    missing = sorted(set(bom_flat_qty) - set(totals_override))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PN", "QTY_BOM_FLAT"])
        for pn in missing:
            w.writerow([pn, _fmt_qty(bom_flat_qty.get(pn))])

    return len(missing)


def _write_totalization_extra_vs_bom_csv(
    bom_flat_qty: Dict[str, Decimal],
    totals_override: Dict[str, Decimal],
    out_path: Path,
) -> int:
    """
    CSV: codici presenti in Totalizzazione ma NON in BOM flat.
    Regola tua: se un internal code è sulla totalizzazione ma non nell'esplosione -> ERROR.
    """
    extra = sorted(set(totals_override) - set(bom_flat_qty))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PN", "QTY_TOTALIZATION"])
        for pn in extra:
            w.writerow([pn, _fmt_qty(totals_override.get(pn))])

    return len(extra)


def _write_totalization_qty_mismatches_csv(
    bom_flat_qty: Dict[str, Decimal],
    totals_override: Dict[str, Decimal],
    out_path: Path,
    tol: Decimal = Decimal("0.0001"),
) -> int:
    """
    CSV: mismatch quantità tra BOM flat (calcolata da esplosione) e Totalizzazione.
    Regola tua: mismatch quantità -> WARN.
    """
    rows: List[Tuple[str, str, str, str]] = []
    common = set(bom_flat_qty).intersection(set(totals_override))

    for pn in sorted(common):
        qb = _to_decimal(bom_flat_qty.get(pn))
        qt = _to_decimal(totals_override.get(pn))
        if qb is None or qt is None:
            continue
        if abs(qb - qt) > tol:
            rows.append((pn, _fmt_qty(qb), _fmt_qty(qt), _fmt_qty(qt - qb)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PN", "QTY_BOM_FLAT", "QTY_TOTALIZATION", "DELTA(TOT-BOM)"])
        w.writerows(rows)

    return len(rows)


def _write_totalization_missing_nr_vs_bom_csv(
    bom_flat_qty: Dict[str, Decimal],
    totals_override: Dict[str, Decimal],
    uom_by_code: Dict[str, str],
    out_path: Path,
) -> int:
    """
    CSV: codici presenti in BOM flat (qty>0) con UM=NR ma NON in totalizzazione => ERROR.
    Nota: uom_by_code viene dall'exploder (se non esiste, dict vuoto -> non segnala nulla).
    """
    rows: List[Tuple[str, str, str]] = []
    for pn in sorted(set(bom_flat_qty) - set(totals_override)):
        u = (uom_by_code.get(pn, "") or "").strip().upper()
        if u != "NR":
            continue
        q = _to_decimal(bom_flat_qty.get(pn))
        if q is None or q == 0:
            continue
        rows.append((pn, u, _fmt_qty(q)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PN", "UOM", "QTY_BOM_FLAT"])
        w.writerows(rows)

    return len(rows)


def _write_non_nr_estimates_from_totalization_csv(
    non_nr_codes: Set[str],
    totals_override: Dict[str, Decimal],
    uom_by_code: Dict[str, str],
    out_path: Path,
) -> int:
    """
    CSV: per codici non-NR (UM != NR) che l'esplosione non riesce a totalizzare,
    scrive una stima prendendo direttamente la totalizzazione.
    NB: richiede che l'exploder esponga `non_nr_codes` e `uom_by_code`.
    """
    if not non_nr_codes:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        return 0

    rows: List[Tuple[str, str, str]] = []
    for pn in sorted(non_nr_codes):
        u = (uom_by_code.get(pn, "") or "").strip().upper()
        q = totals_override.get(pn)
        if q is None:
            continue
        rows.append((pn, u or "?", _fmt_qty(q)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PN", "UOM", "QTY_ESTIMATED_FROM_TOTALIZATION"])
        w.writerows(rows)

    return len(rows)


class AnalyzeFolderUseCase:
    """
    Entry point per la GUI.
    Niente PyQt qui dentro.
    """

    def run(
        self,
        folder: str | Path,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        log_cb: Optional[Callable[[str, str], None]] = None,
    ) -> AnalyzeFolderResult:
        """
        progress_cb(done, total, message)
        log_cb(level, message)
        Entrambi opzionali per retrocompatibilità.
        """

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

        base_dir = Path(folder).expanduser().resolve()
        _log("INFO", f"[AnalyzeFolder] Folder selezionata: {base_dir}")

        _prog(0, 1, "Discovery cartella…")
        discovery = discover_folder(base_dir)

        result = AnalyzeFolderResult(base_dir=base_dir, discovery=discovery)
        # ✅ stream log su tutte le issues che vengono appese (senza cambiare le append esistenti)
        result.issues = _IssueList(log_cb)

        from core.parsers.totalization_excel import parse_totalization_xlsx

        # ==========================================================
        # 🔎 AUTO-LOAD TOTALIZZAZIONE ESTERNA (per inferenza qty BOM)
        # ==========================================================
        totals_override: Optional[Dict[str, Decimal]] = None

        _prog(0, 1, "Ricerca TOTALIZZ*.xls* (opzionale)…")
        try:
            candidates = sorted(base_dir.glob("TOTALIZZ*.xls*"))
            if candidates:
                tot_path = candidates[0]  # usa il primo trovato

                _log("INFO", f"[TOTALIZATION] Trovato: {tot_path.name} — parsing…")
                tot_res = parse_totalization_xlsx(tot_path)
                totals_override = tot_res.qty_by_code

                diag_dir = base_dir / "_diagnostics"
                diag_dir.mkdir(parents=True, exist_ok=True)

                dup_csv = diag_dir / "totalization_duplicates.csv"
                n_dup = _write_totalization_duplicates_csv(tot_res, dup_csv)

                if n_dup:
                    result.issues.append(
                        Issue(
                            "WARN",
                            f"[TOTALIZATION] Duplicati trovati: {n_dup} | CSV: {dup_csv.name}",
                            dup_csv,
                            code="TOTALIZATION_DUPLICATES",
                        )
                    )
                else:
                    result.issues.append(
                        Issue(
                            "INFO",
                            f"[TOTALIZATION] Nessun duplicato rilevato | CSV: {dup_csv.name}",
                            dup_csv,
                            code="TOTALIZATION_DUPLICATES",
                        )
                    )

                for w in tot_res.warnings:
                    result.issues.append(Issue("WARN", f"[TOTALIZATION] {w}", tot_path, code="TOTALIZATION"))

                result.issues.append(
                    Issue(
                        "INFO",
                        f"[TOTALIZATION] File trovato e caricato: {tot_path.name} | righe={len(totals_override)}",
                        tot_path,
                        code="TOTALIZATION",
                    )
                )
            else:
                result.issues.append(
                    Issue(
                        "INFO",
                        "[TOTALIZATION] Nessun file TOTALIZZ*.xls* trovato nella cartella progetto.",
                        base_dir,
                        code="TOTALIZATION",
                    )
                )

        except Exception as e:
            result.issues.append(
                Issue(
                    "WARN",
                    f"[TOTALIZATION] Errore lettura file totalizzazione: {e}",
                    base_dir,
                    code="TOTALIZATION",
                )
            )
            totals_override = None

        # 1) PBS: deve essere unico
        _prog(0, 1, "Ricerca PBS (unico)…")
        try:
            pbs_path = choose_single_pbs(discovery)
        except Exception as e:
            result.issues.append(Issue("ERROR", str(e), base_dir, code="DISCOVERY_PBS"))
            _prog(1, 1, "Errore: PBS non valido")
            return result

        # 2) parsing PBS
        _prog(0, 1, f"Parsing PBS: {pbs_path.name} …")
        try:
            from core.services.pbs_service import load_pbs
            result.pbs = load_pbs(pbs_path)
        except Exception as e:
            result.issues.append(Issue("ERROR", f"Errore parsing PBS: {e}", pbs_path, code="PBS_PARSE"))
            _prog(1, 1, "Errore parsing PBS")
            return result

        # 2b) export Odoo CSV
        _prog(0, 1, "Export PBS → Odoo CSV …")
        out_dir = base_dir / "_odoo_export"
        try:
            release_csv, lines_csv = export_pbs_to_odoo_csv(
                pbs=result.pbs,
                out_dir=out_dir,
                status="draft",
            )
            result.issues.append(
                Issue(
                    "INFO",
                    f"Export Odoo CSV creati in '{out_dir.name}': {release_csv.name}, {lines_csv.name}",
                    out_dir,
                    code="EXPORT_PBS_ODOO",
                )
            )
        except Exception as e:
            result.issues.append(Issue("ERROR", f"Errore export Odoo PBS->CSV: {e}", out_dir, code="EXPORT_PBS_ODOO"))
            _prog(1, 1, "Errore export PBS")
            return result

        # 2c) pre-flight check CSV (senza Odoo)
        _prog(0, 1, "Validazione CSV export (pre-flight)…")
        try:
            check_issues = check_pbs_odoo_csv(release_csv, lines_csv)
            if not check_issues:
                result.issues.append(
                    Issue(
                        "WARN",
                        "[CSV_CHECK] Nessuna issue ritornata dal check (lista vuota).",
                        out_dir,
                        code="CSV_CHECK_PBS_ODOO",
                    )
                )
            else:
                for ci in check_issues:
                    result.issues.append(
                        Issue(
                            level=ci.level,
                            message=f"[CSV_CHECK] {ci.message}",
                            path=out_dir,
                            code="CSV_CHECK_PBS_ODOO",
                        )
                    )

            if any(ci.level == "ERROR" for ci in check_issues):
                _prog(1, 1, "Errore CSV check")
                return result
        except Exception as e:
            result.issues.append(Issue("ERROR", f"Errore CSV check: {e}", out_dir, code="CSV_CHECK_PBS_ODOO"))
            _prog(1, 1, "Errore CSV check")
            return result

        # 3) BOM prefilter
        _prog(0, 1, "Costruzione target BOM da PBS + prefilter filename…")
        bom_targets: Set[BomTarget] = build_targets_from_pbs(result.pbs)
        expected_targets = len(bom_targets)

        all_bom_paths = list(discovery.bom_excel) + list(discovery.bom_pdf)
        filename_matches, bom_paths_to_open = prefilter_boms_by_filename(
            all_bom_paths,
            bom_targets,
            include_medium=True,
        )

        prefilter_by_path: Dict[Path, object] = {m.path: m for m in filename_matches}

        result.issues.append(
            Issue(
                "INFO",
                f"BOM prefilter: target attese dal PBS={expected_targets} | file BOM in cartella={len(all_bom_paths)} | candidate da aprire={len(bom_paths_to_open)}",
                base_dir,
                code="BOM_PREFILTER",
            )
        )

        if filename_matches:
            top = filename_matches[:8]
            pretty: List[str] = []
            for m in top:
                bt = f"{m.best_target.original_code} REV {m.best_target.rev}" if m.best_target else "None"
                pretty.append(
                    f"{m.strength:>6} len={getattr(m, 'match_len', 0):>2} "
                    f"rev={getattr(m, 'filename_rev', '') or '?':<2} "
                    f"score={m.score:.2f} | {m.path.name} -> {bt}"
                )
            result.issues.append(
                Issue(
                    "INFO",
                    "BOM prefilter top match:\n  - " + "\n  - ".join(pretty),
                    base_dir,
                    code="BOM_PREFILTER_TOP",
                )
            )

        bom_excel_to_open = [p for p in bom_paths_to_open if p.suffix.lower() in (".xls", ".xlsx", ".xlsm")]
        bom_pdf_to_open = [p for p in bom_paths_to_open if p.suffix.lower() == ".pdf"]

        # 4) parsing BOM raw + normalizzazione
        bom_docs: List[BomDocument] = []
        verified_targets: Set[BomTarget] = set()
        parsed_ok_count = 0
        hdr_mismatch_count = 0
        bom_header_audit_rows: List[dict] = []

        total_files_to_parse = len(bom_excel_to_open) + len(bom_pdf_to_open)
        parsed_so_far = 0
        _prog(0, max(1, total_files_to_parse), f"Parsing BOM… (0/{total_files_to_parse})")

        def _log_hdr_mismatch(file_path: Path, header_code: str, header_rev: str) -> None:
            nonlocal hdr_mismatch_count
            hdr_mismatch_count += 1

            m = prefilter_by_path.get(file_path)
            expected = ""
            if m and getattr(m, "best_target", None):
                bt = getattr(m, "best_target")
                expected = f" | expected={bt.original_code} REV {bt.rev}"
            ml = getattr(m, "match_len", 0) if m else 0

            result.issues.append(
                Issue(
                    "WARN",
                    f"BOM header mismatch vs PBS target: file={file_path.name} "
                    f"header=({header_code} rev {header_rev})"
                    f" | prefilter_len={ml}{expected}",
                    file_path,
                    code="BOM_HDR_MISMATCH",
                )
            )

        def _add_bom_header_audit_row(
            file_path: Path,
            header_code: str,
            header_rev: str,
            verification: BomHeaderVerification,
            header_meta: dict,
        ) -> None:
            m = prefilter_by_path.get(file_path)
            strength = getattr(m, "strength", "") if m else ""
            match_len = getattr(m, "match_len", 0) if m else 0

            bt = getattr(m, "best_target", None) if m else None
            prefilter_best_target = ""
            target_prtl_code = ""
            target_prtl_rev = ""
            expected_parent_code = ""
            expected_parent_rev = ""
            pbs_src_row = ""

            if bt is not None:
                prefilter_best_target = f"{getattr(bt, 'original_code', '')} REV {getattr(bt, 'rev', '')}"
                target_prtl_code = getattr(bt, "original_code", "") or ""
                target_prtl_rev = getattr(bt, "rev", "") or ""
                expected_parent_code = getattr(bt, "expected_parent_code", "") or ""
                expected_parent_rev = getattr(bt, "expected_parent_rev", "") or ""
                pbs_src_row_val = getattr(bt, "pbs_src_row", -1)
                pbs_src_row = "" if pbs_src_row_val in (-1, None) else str(pbs_src_row_val)

            bom_header_audit_rows.append(
                {
                    "file": file_path.name,
                    "prefilter_strength": strength,
                    "prefilter_match_len": match_len,
                    "prefilter_best_target": prefilter_best_target,
                    "pbs_src_row": pbs_src_row,
                    "target_prtl_code": target_prtl_code,
                    "target_prtl_rev": target_prtl_rev,
                    "expected_parent_code": expected_parent_code,
                    "expected_parent_rev": expected_parent_rev,
                    "header_code": header_code,
                    "header_rev": header_rev,
                    "header_code_source": header_meta.get("code_source", ""),
                    "header_code_cell": header_meta.get("code_cell", ""),
                    "header_rev_source": header_meta.get("rev_source", ""),
                    "header_rev_cell": header_meta.get("rev_cell", ""),
                    "header_code_label_cell": header_meta.get("code_label_cell", ""),
                    "header_rev_label_cell": header_meta.get("rev_label_cell", ""),
                    "verification_reason": getattr(verification, "reason", ""),
                    "suggestion": getattr(verification, "suggestion", ""),
                }
            )

        excel_app = None
        if bom_excel_to_open:
            _prog(parsed_so_far, max(1, total_files_to_parse), "Init Excel COM (se disponibile)…")
            try:
                excel_app = _try_create_excel_app()
                if excel_app is None:
                    result.issues.append(
                        Issue(
                            "WARN",
                            "Excel COM non disponibile: parsing BOM Excel userà la modalità legacy (più lenta/instabile).",
                            base_dir,
                            code="EXCEL_COM",
                        )
                    )
                else:
                    result.issues.append(
                        Issue(
                            "INFO",
                            "Excel COM singleton inizializzato per parsing BOM (stabilità/performance).",
                            base_dir,
                            code="EXCEL_COM",
                        )
                    )
            except Exception as e:
                excel_app = None
                result.issues.append(
                    Issue(
                        "WARN",
                        f"Impossibile inizializzare Excel COM singleton: {e}. Uso parsing legacy per BOM Excel.",
                        base_dir,
                        code="EXCEL_COM",
                    )
                )

        try:
            # BOM Excel
            for idx, xls_path in enumerate(bom_excel_to_open, start=1):
                _prog(parsed_so_far, max(1, total_files_to_parse), f"Parsing BOM Excel ({idx}/{len(bom_excel_to_open)}): {xls_path.name}")
                try:
                    raw = _parse_bom_excel_with_optional_excel(xls_path, excel_app)

                    for w in raw.get("warnings", []):
                        result.issues.append(Issue("WARN", f"BOM Excel: {w}", xls_path, code="BOM_XLS_WARN"))

                    bom_docs.append(
                        build_bom_document(
                            path=xls_path,
                            header_code=raw.get("header", {}).get("code", ""),
                            header_rev=raw.get("header", {}).get("rev", ""),
                            header_title=raw.get("header", {}).get("title", ""),
                            doc_date_iso=raw.get("header", {}).get("date", ""),
                            raw_lines=raw.get("lines", []),
                        )
                    )
                    parsed_ok_count += 1

                    _last = bom_docs[-1]
                    _hdr = getattr(_last, "header", None)
                    _hc = getattr(_hdr, "code", "") if _hdr is not None else getattr(_last, "header_code", "")
                    _hr = getattr(_hdr, "revision", "") if _hdr is not None else getattr(_last, "header_rev", "")

                    v = verify_bom_header_with_diagnosis(_hc, _hr, bom_targets)
                    _add_bom_header_audit_row(xls_path, _hc, _hr, v, raw.get("header", {}) or {})

                    if v.matched and v.best_target:
                        verified_targets.add(v.best_target)
                    else:
                        _log_hdr_mismatch(xls_path, _hc, _hr)

                except Exception as e:
                    msg = str(e)
                    if "0x80010108" in msg or "RPC_E_DISCONNECTED" in msg:
                        result.issues.append(
                            Issue(
                                "WARN",
                                f"Excel COM disconnesso durante parsing BOM '{xls_path.name}'. Dettaglio: {e}",
                                xls_path,
                                code="BOM_XLS_COM_DISCONNECTED",
                            )
                        )
                    else:
                        result.issues.append(Issue("WARN", f"Errore BOM Excel '{xls_path.name}': {e}", xls_path, code="BOM_XLS"))

                parsed_so_far += 1
                _prog(parsed_so_far, max(1, total_files_to_parse), f"Parsing BOM… ({parsed_so_far}/{total_files_to_parse})")

            # BOM PDF
            for idx, pdf_path in enumerate(bom_pdf_to_open, start=1):
                _prog(parsed_so_far, max(1, total_files_to_parse), f"Parsing BOM PDF ({idx}/{len(bom_pdf_to_open)}): {pdf_path.name}")
                try:
                    raw = parse_bom_pdf_raw(pdf_path)

                    for w in raw.get("warnings", []):
                        result.issues.append(Issue("WARN", f"BOM PDF: {w}", pdf_path, code="BOM_PDF_WARN"))

                    bom_docs.append(
                        build_bom_document(
                            path=pdf_path,
                            header_code=raw.get("header", {}).get("code", ""),
                            header_rev=raw.get("header", {}).get("rev", ""),
                            header_title=raw.get("header", {}).get("title", ""),
                            doc_date_iso=raw.get("header", {}).get("date", ""),
                            raw_lines=raw.get("lines", []),
                        )
                    )
                    parsed_ok_count += 1

                    _last = bom_docs[-1]
                    _hdr = getattr(_last, "header", None)
                    _hc = getattr(_hdr, "code", "") if _hdr is not None else getattr(_last, "header_code", "")
                    _hr = getattr(_hdr, "revision", "") if _hdr is not None else getattr(_last, "header_rev", "")

                    v = verify_bom_header_with_diagnosis(_hc, _hr, bom_targets)
                    _add_bom_header_audit_row(pdf_path, _hc, _hr, v, raw.get("header", {}) or {})

                    if v.matched and v.best_target:
                        verified_targets.add(v.best_target)
                    else:
                        _log_hdr_mismatch(pdf_path, _hc, _hr)

                except Exception as e:
                    result.issues.append(Issue("WARN", f"Errore BOM PDF '{pdf_path.name}': {e}", pdf_path, code="BOM_PDF"))

                parsed_so_far += 1
                _prog(parsed_so_far, max(1, total_files_to_parse), f"Parsing BOM… ({parsed_so_far}/{total_files_to_parse})")

        finally:
            _safe_quit_excel(excel_app)

        result.boms = bom_docs

        _prog(0, 1, "Export diagnostiche BOM header audit…")
        # write BOM header audit CSV
        try:
            diag_dir = base_dir / "_diagnostics"
            audit_csv = diag_dir / "bom_header_audit.csv"
            n_audit = _write_bom_header_audit_csv(bom_header_audit_rows, audit_csv)
            result.issues.append(
                Issue(
                    "INFO" if n_audit == 0 else "WARN",
                    f"[BOM_AUDIT] CSV creato: {audit_csv.name} (righe={n_audit})",
                    audit_csv,
                    code="BOM_AUDIT_CSV",
                )
            )
        except Exception as e:
            result.issues.append(Issue("WARN", f"[BOM_AUDIT] Errore export CSV audit header BOM: {e}", base_dir, code="BOM_AUDIT_CSV"))

        # 4b) Report BOM numerico + mancanti
        expected = len(bom_targets)
        opened = len(bom_paths_to_open)
        parsed = parsed_ok_count
        validated = len(verified_targets)
        mismatch = hdr_mismatch_count
        missing_count = max(0, expected - validated)

        result.issues.append(
            Issue(
                "INFO",
                "BOM summary: "
                f"expected={expected} | opened={opened} | parsed_ok={parsed} | "
                f"validated_header={validated} | header_mismatch={mismatch} | missing={missing_count}",
                base_dir,
                code="BOM_SUMMARY",
            )
        )

        if bom_targets:
            missing = sorted(bom_targets - verified_targets)
            if missing:
                msg = (
                    f"BOM mancanti (attese ma NON validate su header): {len(missing)} su {len(bom_targets)}\n"
                    f"{_format_missing_targets(missing, per_line=8)}"
                )
                result.issues.append(Issue("WARN", msg, base_dir, code="BOM_MISSING"))
            else:
                result.issues.append(
                    Issue(
                        "INFO",
                        f"Tutte le BOM target dal PBS risultano trovate e validate su header: {len(bom_targets)}/{len(bom_targets)}",
                        base_dir,
                        code="BOM_MISSING",
                    )
                )

        if (discovery.bom_excel or discovery.bom_pdf) and not bom_docs:
            result.issues.append(
                Issue(
                    "ERROR",
                    "Sono stati trovati file BOM ma tutte le BOM sono fallite in parsing/normalizzazione. Vedi warning BOM_XLS/BOM_PDF.",
                    base_dir,
                    code="BOM_ALL_FAILED",
                )
            )
        elif not bom_docs:
            result.issues.append(Issue("WARN", "Nessuna BOM parsabile trovata (Excel/PDF).", base_dir, code="BOM_NONE"))

        # 5) linking strict PBS -> BOM
        _prog(0, 1, "Linking PBS → BOM…")
        try:
            result.links = link_pbs_to_boms(result.pbs, result.boms)  # type: ignore[arg-type]
        except Exception as e:
            result.issues.append(Issue("ERROR", f"Errore linking PBS→BOM: {e}", base_dir, code="LINK_FATAL"))
            _prog(1, 1, "Errore linking PBS→BOM")
            return result

        link_ok = sum(1 for x in result.links if getattr(x.status, "value", "") == "OK")
        result.issues.append(Issue("INFO", f"LINK summary: OK={link_ok} | total_links={len(result.links)}", base_dir, code="LINK_SUMMARY"))

        # 6) Esplosione BOM ricorsiva + confronto con PBS
        _prog(0, 1, "Explosion BOM ricorsiva…")
        try:
            root_code, root_rev = find_root_from_pbs(result.pbs)  # type: ignore[arg-type]
            _log("INFO", f"[EXPLODE] Root da PBS: {root_code} rev {root_rev}")

            policy = ExplodePolicy(
                strict_rev=True,
                explode_documents=False,
                root_strict_rev=True,
                recursive_fallback=True,
                recursive_pick_highest_rev=True,
            )

            result.explosion = explode_boms(
                root_code=root_code,
                root_rev=root_rev,
                boms=result.boms,
                pbs=result.pbs,
                policy=policy,
            )

            # ✅ Export TXT/CSV
            _prog(0, 1, "Export tree TXT/CSV + totalization validation…")
            try:
                diag_dir = base_dir / "_diagnostics"
                diag_dir.mkdir(parents=True, exist_ok=True)

                pbs_txt = diag_dir / "pbs_tree.txt"
                bom_txt = diag_dir / "bom_explosion_tree.txt"
                flat_csv = diag_dir / "bom_flat_qty.csv"

                n_pbs = _write_pbs_tree_txt(result.pbs, pbs_txt)
                n_bom = _write_bom_explosion_tree_txt(result.explosion, bom_txt)
                n_flat = _write_bom_flat_qty_csv(result.explosion, flat_csv)

                result.issues.append(Issue("INFO", f"[EXPORT_TREE] PBS tree TXT creato: {pbs_txt.name} (righe={n_pbs})", pbs_txt, code="EXPORT_PBS_TREE"))
                result.issues.append(Issue("INFO", f"[EXPORT_TREE] BOM explosion TXT creato: {bom_txt.name} (righe={n_bom})", bom_txt, code="EXPORT_BOM_TREE"))
                result.issues.append(Issue("INFO", f"[EXPORT_TREE] BOM flat CSV creato: {flat_csv.name} (righe={n_flat})", flat_csv, code="EXPORT_BOM_FLAT"))

                # ==========================================================
                # ✅ TOTALIZATION vs BOM EXPLOSION (definitive business rules)
                # ==========================================================
                if totals_override:
                    validation_rows = compare_totalization_to_bom_explosion(
                        explosion=result.explosion,
                        totals_override=totals_override,
                    )

                    out_csv = diag_dir / "totalization_vs_bom_validation.csv"
                    write_totalization_vs_bom_validation_csv(validation_rows, out_csv)

                    summary = summarize_totalization_validation(validation_rows)

                    has_err = any(getattr(r, "status", "").startswith("ERROR") for r in validation_rows)
                    has_warn = any(getattr(r, "status", "").startswith("WARN") for r in validation_rows)
                    level = "ERROR" if has_err else ("WARN" if has_warn else "INFO")

                    result.issues.append(
                        Issue(
                            level,
                            f"[TOTALIZATION_VALIDATION] {summary} | CSV: {out_csv.name}",
                            out_csv,
                            code="TOTALIZATION_VALIDATION",
                        )
                    )
                else:
                    result.issues.append(
                        Issue(
                            "INFO",
                            "[TOTALIZATION_VALIDATION] Skip: totalizzazione non caricata.",
                            diag_dir,
                            code="TOTALIZATION_VALIDATION",
                        )
                    )

            except Exception as e:
                result.issues.append(Issue("WARN", f"[EXPORT_TREE] Errore export/compare: {e}", base_dir, code="EXPORT_TREE_ERR"))

            for w in getattr(result.explosion, "warnings", []) or []:
                result.issues.append(Issue("WARN", f"[EXPLODE] {w}", base_dir, code="EXPLODE_WARN"))
            for e in getattr(result.explosion, "errors", []) or []:
                result.issues.append(Issue("ERROR", f"[EXPLODE] {e}", base_dir, code="EXPLODE_ERROR"))

            # export resolution traces CSV
            _prog(0, 1, "Export explode resolution traces…")
            try:
                diag_dir = base_dir / "_diagnostics"
                tr_csv = diag_dir / "explode_resolution_traces.csv"
                n_tr = _write_explode_resolution_traces_csv(result.explosion, tr_csv)
                result.issues.append(
                    Issue(
                        "INFO" if n_tr == 0 else "WARN",
                        f"[EXPLODE_DIAG] Resolution traces CSV creato: {tr_csv.name} (righe={n_tr})",
                        tr_csv,
                        code="EXPLODE_RESOLUTION_TRACES_CSV",
                    )
                )
            except Exception as e:
                result.issues.append(Issue("WARN", f"[EXPLODE_DIAG] Errore export resolution traces CSV: {e}", base_dir, code="EXPLODE_DIAG"))

            diag_dir = base_dir / "_diagnostics"

            try:
                rev_csv = diag_dir / "explode_rev_fallbacks.csv"
                n_rev = _write_rev_fallbacks_csv(result.explosion, rev_csv)
                result.issues.append(
                    Issue(
                        "WARN" if n_rev else "INFO",
                        f"[EXPLODE_DIAG] {'Revision fallback/mismatch' if n_rev else 'Nessun revision fallback/mismatch (CSV vuoto)'}: {rev_csv.name} (righe={n_rev})",
                        rev_csv,
                        code="EXPLODE_REV_FALLBACK_CSV",
                    )
                )
            except Exception as e:
                result.issues.append(Issue("WARN", f"[EXPLODE_DIAG] Errore export rev fallbacks CSV: {e}", diag_dir, code="EXPLODE_DIAG"))

            try:
                miss_csv = diag_dir / "explode_missing_sub_boms.csv"
                n_miss = _write_missing_sub_boms_csv(result.explosion, miss_csv)
                if n_miss:
                    top = sorted(list(getattr(result.explosion, "missing_sub_boms", set()) or set()))[:30]
                    extra = f" | top30: {', '.join(top)}" if top else ""
                    result.issues.append(Issue("WARN", f"[EXPLODE_DIAG] Missing sub-BOM CSV creato: {miss_csv.name} (count={n_miss}){extra}", miss_csv, code="EXPLODE_MISSING_SUB_BOMS_CSV"))
                else:
                    result.issues.append(Issue("INFO", f"[EXPLODE_DIAG] Nessun missing_sub_boms rilevato (CSV vuoto): {miss_csv.name}", miss_csv, code="EXPLODE_MISSING_SUB_BOMS_CSV"))
            except Exception as e:
                result.issues.append(Issue("WARN", f"[EXPLODE_DIAG] Errore export missing sub-boms CSV: {e}", diag_dir, code="EXPLODE_DIAG"))

            try:
                cyc_csv = diag_dir / "explode_cycles.csv"
                n_cyc = _write_cycles_csv(result.explosion, cyc_csv)
                if n_cyc:
                    result.issues.append(Issue("WARN", f"[EXPLODE_DIAG] Cycles CSV creato: {cyc_csv.name} (count={n_cyc})", cyc_csv, code="EXPLODE_CYCLES_CSV"))
            except Exception:
                pass

            # ✅ QTY inference: usa occurrences dal PBS (molto più robusto)
            _prog(0, 1, "Inferenza qty mancanti…")
            try:
                diag_dir = base_dir / "_diagnostics"
                diag_dir.mkdir(parents=True, exist_ok=True)

                pbs_occ = compute_occurrences_from_pbs(result.pbs) if result.pbs else None

                inferred_rows = infer_missing_edge_quantities(
                    result.explosion,
                    totals_override=totals_override,
                    occ_override=pbs_occ,
                )
                out_csv = diag_dir / "qty_inference.csv"
                if inferred_rows:
                    write_inferred_qty_csv(inferred_rows, out_csv)

                n_total = len(inferred_rows)
                n_inferred = sum(1 for r in inferred_rows if getattr(r, "status", "") == "INFERRED")
                n_check = sum(1 for r in inferred_rows if getattr(r, "status", "") == "CHECK")
                n_amb = sum(1 for r in inferred_rows if getattr(r, "status", "") == "AMBIGUOUS")
                n_nc = sum(1 for r in inferred_rows if getattr(r, "status", "") == "NOT_CALCULABLE")

                if n_total == 0:
                    result.issues.append(Issue("INFO", "[QTY_INFERENCE] Nessuna riga qty mancante rilevata nei warning explode (CSV non creato).", diag_dir, code="QTY_INFERENCE"))
                else:
                    level = "INFO"
                    if (n_amb + n_nc) > 0:
                        level = "WARN"

                    result.issues.append(
                        Issue(
                            level,
                            f"[QTY_INFERENCE] CSV creato: {out_csv.name} | "
                            f"rows={n_total} | inferred={n_inferred} | check={n_check} | "
                            f"ambiguous={n_amb} | not_calculable={n_nc}",
                            out_csv,
                            code="QTY_INFERENCE",
                        )
                    )
            except Exception as e:
                result.issues.append(Issue("WARN", f"[QTY_INFERENCE] Errore durante inferenza qty mancanti: {e}", base_dir, code="QTY_INFERENCE"))

            # config compare
            _prog(0, 1, "Config compare: Explosion vs PBS…")
            result.config_report = compare_explosion_to_pbs(pbs=result.pbs, explosion=result.explosion)  # type: ignore[arg-type]
            result.issues.append(Issue("INFO", f"[CONFIG_COMPLETENESS] {result.config_report.summary()}", base_dir, code="CONFIG_COMPLETENESS"))

            if result.config_report.bom_introduced_not_in_pbs:
                top = list(result.config_report.bom_introduced_not_in_pbs.items())[:20]
                pretty = ", ".join([f"{pn} x{qty}" for pn, qty in top])
                result.issues.append(Issue("WARN", f"[CONFIG] PN introdotti da BOM ma assenti PBS (top 20): {pretty}", base_dir, code="CONFIG_INTRODUCED_NOT_IN_PBS"))

            if result.config_report.pbs_unexploded_assemblies:
                top = sorted(list(result.config_report.pbs_unexploded_assemblies))[:30]
                result.issues.append(Issue("WARN", f"[CONFIG] Assiemi PBS mai esplosi (top 30): " + ", ".join(top), base_dir, code="CONFIG_PBS_UNEXPLODED"))

                diag_rows = diagnose_unexploded_assemblies(
                    pbs=result.pbs,
                    boms=result.boms,
                    explosion=result.explosion,
                    unexploded_codes=result.config_report.pbs_unexploded_assemblies,
                )

                diag_dir = base_dir / "_diagnostics"
                diag_dir.mkdir(parents=True, exist_ok=True)

                diag_rows2, root_rows = aggregate_chain_breaks(
                    pbs=result.pbs,
                    explosion=result.explosion,
                    diag_rows=diag_rows,
                )

                csv_path = diag_dir / "pbs_unexploded_diagnosis.csv"
                write_unexploded_report_csv(diag_rows2, csv_path)

                roots_csv = diag_dir / "chain_break_roots.csv"
                write_chain_break_roots_csv(root_rows, roots_csv)

                preview = diag_rows2[:30]
                pretty_diag = "\n  - " + "\n  - ".join(
                    f"{r.pbs_code} REV {r.pbs_rev}: {r.diagnosis}"
                    f" | class={r.classification or '-'}"
                    f" | root={r.not_reached_by_root or ('SELF' if r.classification == 'ROOT' else '-')}"
                    f" | cand={r.candidate_code or '-'}"
                    f" | parent={r.parent_bom or '-'} pos={r.pos or '-'}"
                    f" | {r.note or ''}".rstrip()
                    for r in preview
                )

                root_summary = ""
                if root_rows:
                    top_roots = sorted(root_rows, key=lambda x: x.impacted_descendants_count, reverse=True)[:10]
                    root_summary = "\nROOT CAUSES (top):\n  - " + "\n  - ".join(
                        f"{r.root_code} REV {r.root_rev or '-'} | impacted={r.impacted_descendants_count}"
                        f" | parent={r.bom_parent_code or '-'} pos={r.bom_parent_pos or '-'}"
                        f" | reason={r.suspected_reason}"
                        for r in top_roots
                    )

                result.issues.append(
                    Issue(
                        "WARN",
                        f"[UNEXPLODED_DIAG] CSV creato: {csv_path.name} (righe={len(diag_rows2)})"
                        f"\n[CHAIN_BREAK_ROOTS] CSV creato: {roots_csv.name} (righe={len(root_rows)})"
                        f"{root_summary}"
                        f"{pretty_diag}",
                        csv_path,
                        code="UNEXPLODED_DIAG",
                    )
                )

            # parent -> children compare
            _prog(0, 1, "Diagnosi parent→children mismatches…")
            try:
                diag_dir = base_dir / "_diagnostics"
                diag_dir.mkdir(parents=True, exist_ok=True)

                unexploded_set = set(getattr(result.config_report, "pbs_unexploded_assemblies", set()) or set())

                pc_rows = diagnose_parent_children_mismatches(
                    pbs=result.pbs,
                    boms=result.boms,
                    links=result.links,
                    explosion=result.explosion,
                    unexploded_codes=unexploded_set,
                )

                pc_csv = diag_dir / "parent_child_mismatches.csv"
                write_parent_child_mismatch_csv(pc_rows, pc_csv)

                summary = summarize_parent_child_mismatches(pc_rows)

                if pc_rows:
                    preview = pc_rows[:30]
                    pretty = "\n  - " + "\n  - ".join(
                        f"{r.severity} {r.category} [{r.root_vs_cascade}] "
                        f"parent={r.parent_code} rev={r.parent_rev or '-'} "
                        f"| pbs_child={r.pbs_child or '-'} bom_child={r.bom_child or '-'} pos={r.bom_pos or '-'} "
                        f"| file={Path(r.bom_path).name if r.bom_path else '-'} "
                        f"| sug={r.suggestion or '-'}"
                        for r in preview
                    )
                    result.issues.append(
                        Issue(
                            "WARN" if any(x.severity == "ERROR" for x in pc_rows) else "INFO",
                            f"[PARENT_CHILD_COMPARE] {summary} | CSV: {pc_csv.name} (righe={len(pc_rows)})"
                            f"{pretty}",
                            pc_csv,
                            code="PARENT_CHILD_COMPARE",
                        )
                    )
                else:
                    result.issues.append(Issue("INFO", f"[PARENT_CHILD_COMPARE] {summary} | CSV: {pc_csv.name}", pc_csv, code="PARENT_CHILD_COMPARE"))

            except Exception as e:
                result.issues.append(Issue("WARN", f"[PARENT_CHILD_COMPARE] Errore diagnosi parent→children: {e}", base_dir, code="PARENT_CHILD_COMPARE"))

            if result.config_report.revision_mismatches:
                top = result.config_report.revision_mismatches[:30]
                result.issues.append(Issue("WARN", "[CONFIG] Mismatch revisioni (top 30):\n  - " + "\n  - ".join(top), base_dir, code="CONFIG_REV_MISMATCH"))

        except Exception as e:
            result.issues.append(Issue("ERROR", f"Errore esplosione/config compare: {e}", base_dir, code="CONFIG_FATAL"))

        _prog(1, 1, "Analisi completata")
        _log("INFO", "[AnalyzeFolder] Done.")
        return result


if __name__ == "__main__":
    import faulthandler
    faulthandler.dump_traceback_later(30, repeat=True)

    uc = AnalyzeFolderUseCase()
    res = uc.run(r"Z:\JOB\E-00211 - ELT_4 EWE Control Panel\MDP EWE CONTROL PANEL")

    print("PBS file:", res.pbs.path if res.pbs else None)
    print("PBS rows:", len(res.pbs.rows) if res.pbs else 0)
    print("BOM:", len(res.boms))

    if res.boms:
        print("\n--- DEBUG BOM (prima BOM, righe ordinate) ---")
        bom0 = res.boms[0]
        hdr = getattr(bom0, "header", None)
        hc = getattr(hdr, "code", "") if hdr is not None else getattr(bom0, "header_code", "")
        hr = getattr(hdr, "revision", "") if hdr is not None else getattr(bom0, "header_rev", "")
        ht = getattr(hdr, "title", "") if hdr is not None else ""
        print(f"HEADER: code={hc} | rev={hr} | title={ht}")
        _print_bom_lines_table(bom0, max_rows=15)

    print("LINK OK:", sum(1 for x in res.links if getattr(x.status, "value", "") == "OK"), "OK")

    for i in res.issues:
        p = f" ({i.path})" if i.path else ""
        c = f" [{i.code}]" if i.code else ""
        print(f"{i.level}{c}: {i.message}{p}")

    if res.pbs:
        for r in res.pbs.rows:
            indent = "  " * int(getattr(r, "level", 0) or 0)
            print(f"{getattr(r, 'src_row', ''):>5} | {indent}{r.code} | {r.rev} | {r.qty} | {r.description}")
