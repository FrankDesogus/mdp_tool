# mdp_tool/core/parsers/discovery.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterable
import os
import re


EXCEL_EXT = {".xls", ".xlsx", ".xlsm"}
PDF_EXT = {".pdf"}


@dataclass(frozen=True)
class DiscoveryResult:
    base_dir: Path
    pbs_candidates: Tuple[Path, ...]
    bom_excel: Tuple[Path, ...]
    bom_pdf: Tuple[Path, ...]


def _is_excel(p: Path) -> bool:
    return p.suffix.lower() in EXCEL_EXT


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
    # hidden-ish / noisy
    if n.startswith("~$"):
        return True
    return False


def _is_rev_folder_name(name: str) -> bool:
    """
    Riconosce cartelle tipo:
      "REV A", "Rev B", "rev 01", ecc.
    """
    return bool(re.match(r"^\s*rev\s+([a-z0-9]+)\s*$", (name or ""), flags=re.IGNORECASE))


def discover_folder(base_dir: str | Path) -> DiscoveryResult:
    print("[DISCOVERY_DEBUG] discovery.py LOADED - build=2026-02-12")

    """
    Scansiona (controllato, adatto a filesystem di rete):
    - PBS: file excel con 'PBS' nel nome
    - BOM Excel: excel con 'PRT LIST' o 'PART LIST' nel nome
    - BOM PDF: pdf con 'BOM' nel nome

    Fix performance/stability:
    - evita rglob("*") su share grandi
    - limita profondità (max_depth)
    - skip directory note
    - gestisce eccezioni IO/permessi senza bloccare
    - early-stop: per non scansionare tutto se non serve
    """
    base = Path(base_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Cartella non valida: {base}")

    # FIX: se l'utente seleziona direttamente una cartella "REV X",
    # risaliamo di 1 livello così troviamo anche REV più aggiornate.
    if _is_rev_folder_name(base.name):
        base = base.parent

    # ---- knobs (keep conservative defaults) ----
    max_depth = 4  # 0=solo base, 1=base+figli, ...
    skip_dir_names = {
        ".git", ".svn", "__pycache__", "node_modules",
        "_diagnostics", "_odoo_export",
    }
    max_bom_excel = 8000
    max_bom_pdf = 8000

    # early-stop: invece di "break" (che rischia di tagliare REV più alte),
    # smettiamo solo di COLLEZIONARE BOM, ma continuiamo a cercare PBS.
    stop_when_found_pbs_and_boms = True
    min_boms_to_stop = 200

    pbs: List[Path] = []
    bom_xls: List[Path] = []
    bom_pdf: List[Path] = []

    stop_collecting_boms = False

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

            # directory handling
            try:
                if entry.is_dir(follow_symlinks=False):
                    if _should_skip_dir(name, skip_dir_names):
                        continue
                    stack.append((Path(entry.path), depth + 1))
                    continue
            except Exception:
                continue

            # file handling
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                p = Path(entry.path)
            except Exception:
                continue

            # collect
            try:
                # PBS: sempre, anche se smettiamo di raccogliere BOM
                if _is_excel(p) and _name_has(p, "PBS"):
                    pbs.append(p)

                # BOM: solo finché non abbiamo deciso di smettere
                elif not stop_collecting_boms:
                    if _is_excel(p) and (_name_has(p, "PRT LIST") or _name_has(p, "PART LIST")):
                        if len(bom_xls) < max_bom_excel:
                            bom_xls.append(p)
                    elif _is_pdf(p) and _name_has(p, "BOM"):
                        if len(bom_pdf) < max_bom_pdf:
                            bom_pdf.append(p)

            except Exception:
                continue

        # early stop condition (senza troncare la ricerca PBS)
        if stop_when_found_pbs_and_boms and len(pbs) >= 1:
            if (len(bom_xls) + len(bom_pdf)) >= min_boms_to_stop:
                stop_collecting_boms = True
                # NON break: continuiamo a scansionare per trovare PBS REV più alte

    # ordinamento stabile
    pbs.sort(key=lambda x: x.as_posix().lower())
    bom_xls.sort(key=lambda x: x.as_posix().lower())
    bom_pdf.sort(key=lambda x: x.as_posix().lower())

    return DiscoveryResult(
        base_dir=base,
        pbs_candidates=tuple(pbs),
        bom_excel=tuple(bom_xls),
        bom_pdf=tuple(bom_pdf),
    )


def _extract_pbs_rev_token(stem_upper: str) -> str:
    """
    Estrae la revisione vicino a 'PBS', in modo mirato per naming tipo:
      ... PBS REV D
      ... PBSREV D
      ... PBS REV_D
      ... PBS_REV D
      ... PBS D
    Ritorna: "D" oppure "02" ecc. Se non trovato: "".
    """
    s = stem_upper

    # Caso più importante: PBS ... REV <token>
    m = re.search(r"\bPBS\b.*?\bREV\b[\s\-_]*([A-Z]|\d{1,4})\b", s)
    if m:
        return m.group(1)

    # Fallback: PBS <token> subito dopo
    m = re.search(r"\bPBS\b[\s\-_]*([A-Z]|\d{1,4})\b", s)
    if m:
        return m.group(1)

    return ""


def _pbs_rank(p: Path) -> tuple:
    """
    Chiave ordinabile (maggiore = più recente), mirata ai file PBS:

    Priorità:
    1) Token vicino a "PBS REV" (numerico: 02 > 01)
    2) Token vicino a "PBS REV" (lettera: D > C > B > A)
    3) mtime come spareggio
    """
    stem = p.stem.upper()
    token = _extract_pbs_rev_token(stem)

    num = -1
    let = -1

    if token.isdigit():
        num = int(token)
    elif len(token) == 1 and "A" <= token <= "Z":
        let = ord(token) - ord("A") + 1

    try:
        mtime = p.stat().st_mtime
    except Exception:
        mtime = 0.0

    return (num, let, mtime)


def choose_single_pbs(discovery: DiscoveryResult) -> Path:
    print("[DISCOVERY_DEBUG] choose_single_pbs called with:", len(discovery.pbs_candidates))
    for p in discovery.pbs_candidates:
        print("   -", p)

    """
    Sceglie automaticamente il PBS più aggiornato tra i candidati.

    Regole:
    - 0 -> errore
    - 1 -> ritorna quello
    - >1 -> sceglie il "migliore" in base a:
        (PBS REV numerico) > (PBS REV lettera) > (mtime)
      Se ancora ambiguo -> errore
    """
    cands = list(discovery.pbs_candidates)

    if len(cands) == 0:
        raise RuntimeError("Nessun file PBS trovato nella cartella.")
    if len(cands) == 1:
        return cands[0]

    ranked = sorted(cands, key=_pbs_rank, reverse=True)
    best = ranked[0]
    best_rank = _pbs_rank(best)

    # Se ancora ambiguo, NON scegliere a caso
    if len(ranked) > 1 and _pbs_rank(ranked[1]) == best_rank:
        msg = (
            "Trovati più file PBS e non riesco a determinare univocamente il più recente.\n"
            "Candidati (con rank):\n"
            + "\n".join(f"- {p} -> {_pbs_rank(p)}" for p in ranked)
        )
        raise RuntimeError(msg)

    return best
