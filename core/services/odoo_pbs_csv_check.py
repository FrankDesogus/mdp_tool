from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Dict, Optional
from datetime import datetime


@dataclass(frozen=True)
class CsvCheckIssue:
    level: str   # INFO/WARN/ERROR
    message: str


def _must_have(headers: List[str], required: List[str]) -> List[str]:
    missing = [c for c in required if c not in headers]
    return missing


def check_pbs_odoo_csv(release_csv: Path, lines_csv: Path) -> List[CsvCheckIssue]:
    issues: List[CsvCheckIssue] = []

    # --- release ---
    with release_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        headers = r.fieldnames or []
        missing = _must_have(headers, ["x_name", "source_filename", "source_hash", "parsed_at", "status"])
        if missing:
            issues.append(CsvCheckIssue("ERROR", f"pbs_release.csv: colonne mancanti: {missing}"))
            return issues

        rows = list(r)
        if len(rows) != 1:
            issues.append(CsvCheckIssue("ERROR", f"pbs_release.csv: attesa 1 riga, trovate {len(rows)}"))
            return issues

        release_name = (rows[0].get("x_name") or "").strip()
        if not release_name:
            issues.append(CsvCheckIssue("ERROR", "pbs_release.csv: x_name (Descrizione) vuoto"))

        # parsed_at format
        try:
            datetime.strptime(rows[0].get("parsed_at", ""), "%Y-%m-%d %H:%M:%S")
        except Exception:
            issues.append(CsvCheckIssue("WARN", "pbs_release.csv: parsed_at non nel formato YYYY-MM-DD HH:MM:SS"))

    # --- lines ---
    with lines_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        headers = r.fieldnames or []
        required = ["id", "release_id", "sequence", "src_row", "code", "x_name", "rev", "qty", "level", "expected_bom_key", "parent_line_id/id"]
        missing = _must_have(headers, required)
        if missing:
            issues.append(CsvCheckIssue("ERROR", f"pbs_lines.csv: colonne mancanti: {missing}"))
            return issues

        rows = list(r)
        if not rows:
            issues.append(CsvCheckIssue("ERROR", "pbs_lines.csv: nessuna riga"))
            return issues

        ids: List[str] = [(x.get("id") or "").strip() for x in rows]
        if any(not x for x in ids):
            issues.append(CsvCheckIssue("ERROR", "pbs_lines.csv: trovato id vuoto"))

        # unique id
        seen: Set[str] = set()
        dup: Set[str] = set()
        for i in ids:
            if i in seen:
                dup.add(i)
            seen.add(i)
        if dup:
            issues.append(CsvCheckIssue("ERROR", f"pbs_lines.csv: id duplicati: {sorted(list(dup))[:5]} ..."))

        # parent references exist
        id_set = set(ids)
        bad_parent = 0
        for x in rows:
            p = (x.get("parent_line_id/id") or "").strip()
            if p and p not in id_set:
                bad_parent += 1
        if bad_parent:
            issues.append(CsvCheckIssue("ERROR", f"pbs_lines.csv: {bad_parent} parent_line_id/id non risolti"))

        # release_id coherence (optional strict)
        if release_name:
            wrong = sum(1 for x in rows if (x.get("release_id") or "").strip() != release_name)
            if wrong:
                issues.append(CsvCheckIssue("WARN", f"pbs_lines.csv: {wrong} righe con release_id diverso da release x_name"))

    if not any(i.level == "ERROR" for i in issues):
        issues.append(CsvCheckIssue("INFO", "CSV check OK (struttura importabile)."))
    return issues
