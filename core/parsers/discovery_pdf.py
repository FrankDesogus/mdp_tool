from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
import os


PDF_EXT = {".pdf"}


@dataclass(frozen=True)
class DiscoveryPdfResult:
    base_dir: Path
    bom_pdf: Tuple[Path, ...]


# -------------------------------------------------
# helpers
# -------------------------------------------------

def _is_pdf(p: Path) -> bool:
    return p.suffix.lower() in PDF_EXT


def _name_has(p: Path, needle: str) -> bool:
    return needle.lower() in p.name.lower()


def _safe_scandir(dir_path: Path):
    """os.scandir wrapper that won't blow up on permissions / network hiccups."""
    try:
        with os.scandir(dir_path) as it:
            for entry in it:
                yield entry
    except Exception:
        return


def _should_skip_dir(name: str, skip_names: Iterable[str]) -> bool:
    n = (name or "").strip().lower()
    if not n:
        return False
    if n in skip_names:
        return True
    if n.startswith("~$"):  # temp / lock files
        return True
    return False


# -------------------------------------------------
# main discovery (PDF ONLY)
# -------------------------------------------------

def discover_folder_pdf(base_dir: str | Path) -> DiscoveryPdfResult:
    """
    Scansiona la cartella cercando SOLO:
        - BOM PDF (file .pdf con 'BOM' nel nome)

    Versione alleggerita e robusta per share di rete.
    """

    base = Path(base_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Cartella non valida: {base}")

    # ---- knobs conservativi ----
    max_depth = 4
    max_bom_pdf = 10000

    skip_dir_names = {
        ".git", ".svn", "__pycache__", "node_modules",
        "_diagnostics", "_odoo_export",
    }

    bom_pdf: List[Path] = []

    # BFS stack
    stack: List[Tuple[Path, int]] = [(base, 0)]

    while stack:
        cur_dir, depth = stack.pop()
        if depth > max_depth:
            continue

        for entry in _safe_scandir(cur_dir):
            try:
                name = entry.name
            except Exception:
                continue

            # --------------------
            # directories
            # --------------------
            try:
                if entry.is_dir(follow_symlinks=False):
                    if _should_skip_dir(name, skip_dir_names):
                        continue
                    stack.append((Path(entry.path), depth + 1))
                    continue
            except Exception:
                continue

            # --------------------
            # files
            # --------------------
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                p = Path(entry.path)
            except Exception:
                continue

            try:
                # ⭐ QUI la differenza chiave:
                # SOLO PDF con BOM nel nome
                if _is_pdf(p) and _name_has(p, "BOM"):
                    bom_pdf.append(p)
            except Exception:
                continue

            if len(bom_pdf) >= max_bom_pdf:
                break

    bom_pdf.sort(key=lambda x: x.as_posix().lower())

    return DiscoveryPdfResult(
        base_dir=base,
        bom_pdf=tuple(bom_pdf),
    )
