# core/services/odoo_pbs_export.py
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from core.domain.models import PbsDocument, MdpRow


@dataclass(frozen=True)
class OdooPbsExportConfig:
    """
    Config per export CSV Odoo-ready.

    Assunzioni (come da tua descrizione Odoo Studio):
      - x_pbs_release: name, source_filename, source_hash, parsed_at, status
      - x_pbs_line: release_id, sequence, src_row, code, description, rev, qty, level,
                    parent_line_id, expected_bom_key, x_name

    Se i nomi tecnici in Odoo cambiano, modifica SOLO questi campi.
    """
    # --- Release model fields (x_pbs_release) ---
    release_name_field: str = "x_name"
    release_source_filename_field: str = "source_filename"
    release_source_hash_field: str = "source_hash"
    release_parsed_at_field: str = "parsed_at"
    release_status_field: str = "status"

    # --- Lines model fields (x_pbs_line) ---
    line_external_id_field: str = "id"          # Odoo import: External ID
    line_release_m2o_field: str = "release_id"  # Many2one -> x_pbs_release (aggancio per name)

    line_sequence_field: str = "sequence"
    line_src_row_field: str = "src_row"
    line_code_field: str = "code"
    line_description_field: Optional[str] = None  # non usato
    line_x_name_field: str = "x_name"
    line_rev_field: str = "rev"
    line_qty_field: str = "qty"
    line_level_field: str = "level"
    line_parent_m2o_field: str = "parent_line_id"  # Many2one -> x_pbs_line
    line_expected_bom_key_field: str = "expected_bom_key"

    # In CSV import, il parent viene referenziato via external id: parent_line_id/id
    parent_ref_suffix: str = "/id"

    # CSV
    csv_encoding: str = "utf-8"
    csv_dialect: str = "excel"

    # Release defaults
    default_status: str = "draft"

    # Deterministic keys
    expected_bom_key_sep: str = "|"
    line_external_id_prefix: str = "PBS_LINE"   # rende gli External ID leggibili e stabili


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def default_release_name_from_path(pbs_path: Path) -> str:
    """
    Nome release deterministico e *non deduttivo*:
    usa lo stem del file sorgente, normalizzando solo gli spazi.
    """
    stem = pbs_path.stem.strip()
    stem = " ".join(stem.split())
    return stem or "PBS_RELEASE"


def _fmt_dt_odoo(dt: datetime) -> str:
    # Odoo import CSV tipicamente accetta "YYYY-MM-DD HH:MM:SS"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _as_float_str(v: Any) -> str:
    """
    Converti qty in stringa con punto decimale (Odoo-safe).
    Nessuna deduzione: se non parsabile -> "0".
    """
    if v is None:
        return "0"
    try:
        return format(float(v), "g")
    except Exception:
        return "0"


def _make_expected_bom_key(code: str, rev: str, sep: str) -> str:
    return f"{(code or '').strip()}{sep}{(rev or '').strip()}"


def _make_line_external_id(release_name: str, sequence: int, prefix: str) -> str:
    # Deterministico (no UUID, no timestamp)
    safe_release = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in release_name)
    return f"{prefix}__{safe_release}__{sequence:06d}"


def export_pbs_to_odoo_csv(
    pbs: PbsDocument,
    out_dir: str | Path,
    release_name: Optional[str] = None,
    status: Optional[str] = None,
    parsed_at: Optional[datetime] = None,
    config: OdooPbsExportConfig = OdooPbsExportConfig(),
) -> Tuple[Path, Path]:
    """
    Export deterministico PBS -> due CSV:
      - pbs_release.csv
      - pbs_lines.csv

    Input:
      - PbsDocument (path + rows[MdpRow])

    Output:
      - paths dei due CSV generati

    Import in Odoo:
      1) Import pbs_release.csv su x_pbs_release
      2) Import pbs_lines.csv su x_pbs_line:
         - release_id agganciato per name
         - parent_line_id agganciato tramite External ID (id + parent_line_id/id)
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    release_name = release_name or default_release_name_from_path(pbs.path)
    status = status or config.default_status
    parsed_at = parsed_at or datetime.now()

    source_filename = pbs.path.name
    source_hash = sha256_file(pbs.path)

    release_csv = out_dir / "pbs_release.csv"
    lines_csv = out_dir / "pbs_lines.csv"

    # -------------------------
    # 1) RELEASE CSV
    # -------------------------
    release_headers = [
        config.release_name_field,
        config.release_source_filename_field,
        config.release_source_hash_field,
        config.release_parsed_at_field,
        config.release_status_field,
    ]

    with release_csv.open("w", encoding=config.csv_encoding, newline="") as f:
        w = csv.DictWriter(f, fieldnames=release_headers, dialect=config.csv_dialect)
        w.writeheader()
        w.writerow(
            {
                config.release_name_field: release_name,
                config.release_source_filename_field: source_filename,
                config.release_source_hash_field: source_hash,
                config.release_parsed_at_field: _fmt_dt_odoo(parsed_at),
                config.release_status_field: status,
            }
        )

    # -------------------------
    # 2) LINES CSV
    # -------------------------
    parent_ref_field = f"{config.line_parent_m2o_field}{config.parent_ref_suffix}"

    line_headers = [
        config.line_external_id_field,  # "id" (External ID)
        config.line_release_m2o_field,  # release_id (by name)
        config.line_sequence_field,
        config.line_src_row_field,
        config.line_code_field,

        # ONLY ONE descriptive field in your Odoo: x_name
        config.line_x_name_field,

        config.line_rev_field,
        config.line_qty_field,
        config.line_level_field,
        parent_ref_field,  # parent_line_id/id
        config.line_expected_bom_key_field,
    ]

    # Stack deterministico per parent_line_id basato SOLO sul level PBS
    last_id_at_level: Dict[int, str] = {}

    with lines_csv.open("w", encoding=config.csv_encoding, newline="") as f:
        w = csv.DictWriter(f, fieldnames=line_headers, dialect=config.csv_dialect)
        w.writeheader()

        for i, row in enumerate(pbs.rows, start=1):
            seq = i
            line_id = _make_line_external_id(release_name, seq, config.line_external_id_prefix)

            lvl = int(row.level or 0)
            parent_id = last_id_at_level.get(lvl - 1, "") if lvl > 0 else ""

            # aggiorna stack
            last_id_at_level[lvl] = line_id
            # pulisci livelli più profondi quando si risale
            for deeper in list(last_id_at_level.keys()):
                if deeper > lvl:
                    del last_id_at_level[deeper]

            w.writerow(
                {
                    config.line_external_id_field: line_id,
                    config.line_release_m2o_field: release_name,
                    config.line_sequence_field: seq,
                    config.line_src_row_field: int(row.src_row),
                    config.line_code_field: (row.code or "").strip(),

                    # Your Odoo "Description" is x_name (Char)
                    config.line_x_name_field: row.description or "",

                    config.line_rev_field: (row.rev or "").strip(),
                    config.line_qty_field: _as_float_str(row.qty),
                    config.line_level_field: lvl,
                    parent_ref_field: parent_id,
                    config.line_expected_bom_key_field: _make_expected_bom_key(
                        row.code, row.rev, config.expected_bom_key_sep
                    ),
                }
            )

    return release_csv, lines_csv
