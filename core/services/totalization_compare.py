# core/services/totalization_compare.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


# =========================
# Domain output model
# =========================
@dataclass(frozen=True)
class TotalizationValidationRow:
    pn: str
    uom: str
    qty_bom: Optional[Decimal]
    qty_totalization: Optional[Decimal]
    status: str
    note: str = ""


# =========================
# Internal helpers
# =========================
def _norm_pn(x: object) -> str:
    return ("" if x is None else str(x)).strip()


def _norm_uom(x: object) -> str:
    return ("" if x is None else str(x)).strip().upper()


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


def _fmt_decimal(x: Optional[Decimal]) -> str:
    if x is None:
        return ""
    try:
        return f"{x.normalize()}"
    except Exception:
        return str(x)


def _qty_equal(a: Optional[Decimal], b: Optional[Decimal], tol: Decimal = Decimal("0.0001")) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def _get_explosion_maps(explosion: object) -> tuple[Dict[str, Decimal], Dict[str, str], Set[str]]:
    """
    Estrae in modo safe i campi introdotti dall'exploder:
      - qty_by_code : Dict[PN, Decimal]
      - uom_by_code : Dict[PN, str]
      - non_nr_codes: Set[PN]
    """
    qty_by_code = getattr(explosion, "qty_by_code", {}) or {}
    uom_by_code = getattr(explosion, "uom_by_code", {}) or {}
    non_nr_codes = set(getattr(explosion, "non_nr_codes", set()) or set())

    # normalizza chiavi/valori
    qty_norm: Dict[str, Decimal] = {}
    for k, v in qty_by_code.items():
        pn = _norm_pn(k)
        dv = _to_decimal(v)
        if pn and dv is not None:
            qty_norm[pn] = dv

    uom_norm: Dict[str, str] = {}
    for k, v in uom_by_code.items():
        pn = _norm_pn(k)
        if pn:
            uom_norm[pn] = _norm_uom(v)

    non_nr_norm = {_norm_pn(x) for x in non_nr_codes if _norm_pn(x)}

    # fallback: se non_nr_codes non c'è/è vuoto, deriviamo dai UOM != NR
    if not non_nr_norm and uom_norm:
        non_nr_norm = {pn for pn, u in uom_norm.items() if u and u != "NR"}

    return qty_norm, uom_norm, non_nr_norm


# =========================
# Public API
# =========================
def compare_totalization_to_bom_explosion(
    explosion: object,
    totals_override: Dict[str, Decimal],
    tol: Decimal = Decimal("0.0001"),
) -> List[TotalizationValidationRow]:
    """
    Confronta TOTALIZZAZIONE vs BOM EXPLOSION applicando le regole definitive:

    ERRORI BLOCCANTI
    1) PN in TOTAL ma NON in BOM EXPLOSION -> ERROR_EXTRA_IN_TOTAL
    2) PN in BOM EXPLOSION con qty>0 e UOM=NR ma NON in TOTAL -> ERROR_MISSING_IN_TOTAL

    WARNING
    3) PN presente in entrambi con UOM=NR e qty diverse -> WARN_QTY_MISMATCH

    CASO SPECIALE NON-NR
    4) Se UOM != NR: NON confrontare qty, NON generare mismatch/error -> SKIPPED_NON_NR
    """
    if not explosion:
        return []

    bom_qty_by_code, uom_by_code, non_nr_codes = _get_explosion_maps(explosion)

    # normalizza totalizzazione
    tot_qty_by_code: Dict[str, Decimal] = {}
    for k, v in (totals_override or {}).items():
        pn = _norm_pn(k)
        dv = _to_decimal(v)
        if pn and dv is not None:
            tot_qty_by_code[pn] = dv

    bom_set = set(bom_qty_by_code.keys())
    tot_set = set(tot_qty_by_code.keys())
    all_pn = sorted(bom_set.union(tot_set))

    rows: List[TotalizationValidationRow] = []

    for pn in all_pn:
        qb = bom_qty_by_code.get(pn)
        qt = tot_qty_by_code.get(pn)

        uom = _norm_uom(uom_by_code.get(pn, ""))

        # Se non sappiamo la UOM ma abbiamo pn in non_nr_codes, forziamo NON-NR
        is_non_nr = (pn in non_nr_codes) or (uom and uom != "NR")

        # --- Regola 4: NON-NR => skip qty compare, ma comunque output riga
        if is_non_nr:
            # se UOM manca, mettiamo "?" (così sai che non era disponibile)
            out_uom = uom or "?"
            rows.append(
                TotalizationValidationRow(
                    pn=pn,
                    uom=out_uom,
                    qty_bom=qb,
                    qty_totalization=qt,
                    status="SKIPPED_NON_NR",
                    note="UOM != NR: qty non confrontate; usare totalizzazione solo per inferenza qty mancanti BOM.",
                )
            )
            continue

        # qui siamo in NR (o sconosciuto ma non marcato non-NR)
        out_uom = uom or "NR"

        in_bom = pn in bom_set
        in_tot = pn in tot_set

        # --- Regola 1: extra in totalizzazione
        if in_tot and not in_bom:
            rows.append(
                TotalizationValidationRow(
                    pn=pn,
                    uom=out_uom,
                    qty_bom=None,
                    qty_totalization=qt,
                    status="ERROR_EXTRA_IN_TOTAL",
                    note="PN presente in Totalizzazione ma assente in BOM EXPLOSION.",
                )
            )
            continue

        # --- Regola 2: missing in totalizzazione (solo se qty_bom > 0 e UOM=NR)
        if in_bom and not in_tot:
            qb_dec = _to_decimal(qb)
            if qb_dec is not None and qb_dec > 0 and out_uom == "NR":
                rows.append(
                    TotalizationValidationRow(
                        pn=pn,
                        uom=out_uom,
                        qty_bom=qb_dec,
                        qty_totalization=None,
                        status="ERROR_MISSING_IN_TOTAL",
                        note="PN presente in BOM EXPLOSION (qty>0, UOM=NR) ma assente in Totalizzazione.",
                    )
                )
            else:
                # caso raro: qty 0 o uom non NR (ma qui non_nr l'abbiamo già gestito)
                rows.append(
                    TotalizationValidationRow(
                        pn=pn,
                        uom=out_uom,
                        qty_bom=qb_dec,
                        qty_totalization=None,
                        status="OK",
                        note="Assente in totalizzazione ma qty BOM non >0 (o condizione non applicabile).",
                    )
                )
            continue

        # --- Regola 3: mismatch qty (solo NR, presenti entrambi)
        if in_bom and in_tot:
            qb_dec = _to_decimal(qb)
            qt_dec = _to_decimal(qt)

            if qb_dec is None or qt_dec is None:
                rows.append(
                    TotalizationValidationRow(
                        pn=pn,
                        uom=out_uom,
                        qty_bom=qb_dec,
                        qty_totalization=qt_dec,
                        status="OK",
                        note="Quantità non confrontabile (valore non numerico o mancante).",
                    )
                )
                continue

            if not _qty_equal(qb_dec, qt_dec, tol=tol):
                rows.append(
                    TotalizationValidationRow(
                        pn=pn,
                        uom=out_uom,
                        qty_bom=qb_dec,
                        qty_totalization=qt_dec,
                        status="WARN_QTY_MISMATCH",
                        note="QTY_TOTALIZZAZIONE != QTY_BOM_EXPLOSION (UOM=NR).",
                    )
                )
            else:
                rows.append(
                    TotalizationValidationRow(
                        pn=pn,
                        uom=out_uom,
                        qty_bom=qb_dec,
                        qty_totalization=qt_dec,
                        status="OK",
                        note="",
                    )
                )
            continue

        # fallback (non dovrebbe capitare)
        rows.append(
            TotalizationValidationRow(
                pn=pn,
                uom=out_uom,
                qty_bom=qb,
                qty_totalization=qt,
                status="OK",
                note="",
            )
        )

    return rows


def write_totalization_vs_bom_validation_csv(rows: Iterable[TotalizationValidationRow], out_path: Path) -> int:
    """
    Scrive:
    PN | UOM | QTY_BOM | QTY_TOTALIZZAZIONE | STATUS | NOTE
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_list = list(rows)
    rows_list.sort(key=lambda r: r.pn)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PN", "UOM", "QTY_BOM", "QTY_TOTALIZZAZIONE", "STATUS", "NOTE"])
        for r in rows_list:
            w.writerow(
                [
                    r.pn,
                    r.uom,
                    _fmt_decimal(_to_decimal(r.qty_bom)),
                    _fmt_decimal(_to_decimal(r.qty_totalization)),
                    r.status,
                    r.note or "",
                ]
            )

    return len(rows_list)


def summarize_totalization_validation(rows: Iterable[TotalizationValidationRow]) -> str:
    """
    Ritorna una stringa pronta per Issue/report:
    - counts per STATUS
    - totale righe
    """
    rows_list = list(rows)
    total = len(rows_list)

    by_status: Dict[str, int] = {}
    for r in rows_list:
        by_status[r.status] = by_status.get(r.status, 0) + 1

    # ordine utile
    order = [
        "ERROR_EXTRA_IN_TOTAL",
        "ERROR_MISSING_IN_TOTAL",
        "WARN_QTY_MISMATCH",
        "SKIPPED_NON_NR",
        "OK",
    ]
    parts = []
    for k in order:
        if k in by_status:
            parts.append(f"{k}={by_status[k]}")
    # eventuali status futuri
    for k in sorted(set(by_status) - set(order)):
        parts.append(f"{k}={by_status[k]}")

    return f"rows={total} | " + " | ".join(parts)
