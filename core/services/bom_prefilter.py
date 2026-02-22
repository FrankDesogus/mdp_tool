# core/services/bom_prefilter.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Dict
import re


# -----------------------------
# Regex helpers
# -----------------------------
_REV_RE = re.compile(r"\bREV\s*[-_ .]*([A-Z0-9]+)\b", re.IGNORECASE)
_ALNUM_RE = re.compile(r"[^A-Z0-9]")
_DIGITS_RE = re.compile(r"\d+")

MIN_MATCH_LEN = 6  # <<<< la tua regola: match minimo nel nome (6)


def normalize_rev(rev: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (rev or "").strip().upper())


def extract_rev_from_text(text: str) -> str:
    """Estrae REV da filename o testo generico."""
    if not text:
        return ""
    m = _REV_RE.search(text.upper())
    return m.group(1).upper() if m else ""


def normalize_alnum(s: str) -> str:
    """Upper + solo alfanumerico."""
    return _ALNUM_RE.sub("", (s or "").upper())


def extract_digit_tokens(s: str, *, min_len: int = 1) -> List[str]:
    """
    Estrae token numerici dal testo normalizzato.
    Non forziamo min_len=6 qui: la soglia la applichiamo sul match_len.
    """
    tokens = _DIGITS_RE.findall(normalize_alnum(s))
    return [t for t in tokens if len(t) >= min_len]


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def best_numeric_match_len(a_text: str, b_text: str) -> int:
    """
    Ritorna la migliore lunghezza di match numerico tra i due testi,
    calcolata come max prefisso comune tra qualunque coppia di token numerici.

    Questo è "elastico" e gestisce bene casi tipo:
      - target: 231005117PRTL  vs header: 2310051171A01   (match_len=9)
      - target corto vs filename lungo e viceversa
      - casi dove bastano 6 cifre iniziali per filtrare
    """
    a_tokens = extract_digit_tokens(a_text, min_len=1)
    b_tokens = extract_digit_tokens(b_text, min_len=1)

    best = 0
    for at in a_tokens:
        for bt in b_tokens:
            best = max(best, common_prefix_len(at, bt), common_prefix_len(bt, at))
    return best


# -----------------------------
# Domain models
# -----------------------------
@dataclass(frozen=True, order=True)
class BomTarget:
    """
    BOM attesa dal PBS.

    Manteniamo 'root' per compatibilità/log, ma il prefilter NON dipende da root.

    NEW (per audit Fase 1):
      - expected_parent_code/rev: assembly PBS da cui deriva la PRTL (cioè il PN "atteso")
      - pbs_src_row: riga PBS per riferimento umano (opzionale)
    """
    root: str
    rev: str
    original_code: str

    # ---- NEW fields (defaults keep backwards compatibility) ----
    expected_parent_code: str = ""
    expected_parent_rev: str = ""
    pbs_src_row: int = -1


@dataclass(frozen=True)
class BomFilenameMatch:
    path: Path
    best_target: Optional[BomTarget]
    score: float
    strength: str   # "STRONG" | "MEDIUM" | "NO_MATCH"
    match_len: int = 0
    filename_rev: str = ""
    has_prtl: bool = False


# -----------------------------
# Matching / scoring
# -----------------------------
def classify_filename_match(
    *,
    match_len: int,
    target_rev: str,
    filename_rev: str,
    min_len: int = MIN_MATCH_LEN,
) -> str:
    """
    Regole:
      - se match_len < min_len: NO_MATCH
      - se entrambe le REV esistono e sono diverse: NO_MATCH
      - se REV combacia (ed esiste nel filename): STRONG
      - altrimenti (REV mancante o solo su un lato): MEDIUM  -> apri e verifica su header
    """
    if match_len < min_len:
        return "NO_MATCH"

    tr = normalize_rev(target_rev)
    fr = normalize_rev(filename_rev)

    if tr and fr and tr != fr:
        return "NO_MATCH"

    if tr and fr and tr == fr:
        return "STRONG"

    return "MEDIUM"


def score_for_sort(strength: str, match_len: int) -> float:
    """
    Score solo per ordinare/loggare:
      - STRONG più alto di MEDIUM
      - a parità, match_len più alto vince
    """
    base = 1.0 if strength == "STRONG" else (0.7 if strength == "MEDIUM" else 0.0)
    # normalizziamo un po' il match_len (cap a 12 solo per stabilità)
    return base + min(match_len, 12) / 100.0


def score_bom_path_against_targets(path: Path, targets: Iterable[BomTarget]) -> BomFilenameMatch:
    name = path.stem
    filename_rev = extract_rev_from_text(name)
    has_prtl = "PRTL" in name.upper()

    best_target: Optional[BomTarget] = None
    best_strength = "NO_MATCH"
    best_len = 0

    rank = {"STRONG": 2, "MEDIUM": 1, "NO_MATCH": 0}

    for tgt in targets:
        ml = best_numeric_match_len(tgt.original_code, name)
        strength = classify_filename_match(
            match_len=ml,
            target_rev=tgt.rev,
            filename_rev=filename_rev,
            min_len=MIN_MATCH_LEN,
        )

        if (rank[strength] > rank[best_strength]) or (rank[strength] == rank[best_strength] and ml > best_len):
            best_strength = strength
            best_target = tgt
            best_len = ml

        # se già STRONG con match sufficiente, non serve continuare troppo
        if best_strength == "STRONG" and best_len >= MIN_MATCH_LEN:
            # non break: potresti avere un target STRONG con match_len maggiore
            pass

    return BomFilenameMatch(
        path=path,
        best_target=best_target,
        score=score_for_sort(best_strength, best_len),
        strength=best_strength,
        match_len=best_len,
        filename_rev=normalize_rev(filename_rev),
        has_prtl=has_prtl,
    )


def prefilter_boms_by_filename(
    bom_paths: Sequence[Path],
    targets: Iterable[BomTarget],
    *,
    include_medium: bool = True,
) -> Tuple[List[BomFilenameMatch], List[Path]]:
    """
    Ritorna:
      - lista match (per logging / debug)
      - lista di path da aprire davvero (STRONG + eventualmente MEDIUM)
    """
    tgt_list = list(targets)
    matches: List[BomFilenameMatch] = []
    to_open: List[Path] = []

    for p in bom_paths:
        m = score_bom_path_against_targets(p, tgt_list)
        matches.append(m)
        if m.strength == "STRONG" or (include_medium and m.strength == "MEDIUM"):
            to_open.append(p)

    strength_rank = {"STRONG": 3, "MEDIUM": 2, "NO_MATCH": 0}
    matches.sort(
        key=lambda x: (strength_rank.get(x.strength, 0), x.match_len, x.score, str(x.path).lower()),
        reverse=True,
    )

    to_open = sorted(set(to_open), key=lambda x: str(x).lower())
    return matches, to_open


# -----------------------------
# PBS targets extraction
# -----------------------------
def extract_numeric_root(code: str) -> str:
    """
    Manteniamo una 'root' SOLO per compatibilità/log.
    Non usiamo più questo valore per decidere il match.
    """
    tokens = extract_digit_tokens(code, min_len=MIN_MATCH_LEN)
    return tokens[0] if tokens else ""


def build_targets_from_pbs_with_parents(pbs) -> Set[BomTarget]:
    """
    NEW: come build_targets_from_pbs, ma prova anche a derivare il parent assembly PBS
    per ogni riga ...PRTL usando pbs.rows + level.

    - Se non trova level/rows coerenti, torna comunque targets senza parent (fallback).
    """
    targets: Set[BomTarget] = set()
    rows = getattr(pbs, "rows", []) or []
    if not rows:
        return targets

    # Se non abbiamo 'level' sulle righe, non possiamo calcolare la gerarchia.
    has_level = all(hasattr(r, "level") for r in rows)
    if not has_level:
        # fallback: stesso comportamento della funzione legacy
        for r in rows:
            code = getattr(r, "code", "") or ""
            rev = getattr(r, "rev", "") or ""
            if (code or "").strip().upper().endswith("PRTL"):
                targets.add(
                    BomTarget(
                        root=extract_numeric_root(code),
                        rev=normalize_rev(rev),
                        original_code=(code or "").strip(),
                    )
                )
        return targets

    # Stack per parent derivato dal level
    # Manteniamo lo stack di tuple (level, row)
    stack: List[Tuple[int, object]] = []

    for r in rows:
        code = getattr(r, "code", "") or ""
        rev = getattr(r, "rev", "") or ""
        if not (code or "").strip():
            continue

        level = int(getattr(r, "level", 0) or 0)
        # pop finché non troviamo un parent con level < current
        while stack and stack[-1][0] >= level:
            stack.pop()

        parent_row = stack[-1][1] if stack else None

        # se questa è una PRTL, creiamo il target con expected parent
        if (code or "").strip().upper().endswith("PRTL"):
            parent_code = (getattr(parent_row, "code", "") or "").strip() if parent_row else ""
            parent_rev = (getattr(parent_row, "rev", "") or "").strip() if parent_row else ""
            src_row = int(getattr(r, "src_row", -1) or -1)

            targets.add(
                BomTarget(
                    root=extract_numeric_root(code),
                    rev=normalize_rev(rev),
                    original_code=(code or "").strip(),
                    expected_parent_code=(parent_code or "").strip(),
                    expected_parent_rev=(parent_rev or "").strip(),
                    pbs_src_row=src_row,
                )
            )

        # push current row as potential parent for subsequent lines
        stack.append((level, r))

    return targets


def build_targets_from_pbs(pbs) -> Set[BomTarget]:
    """
    Estrae le BOM target dal PBS.
    Strategia:
      1) Se il parser PBS espone già una lista di part list, usa quella:
         - pbs.part_lists (preferito)
      2) Altrimenti fallback su pbs.rows filtrando i code che finiscono con 'PRTL'

    NEW:
      - se pbs.rows ha 'level', prova a derivare expected_parent_code/rev per le PRTL.
    """
    targets: Set[BomTarget] = set()

    part_lists = getattr(pbs, "part_lists", None)
    if part_lists:
        for item in part_lists:
            code = getattr(item, "code", "") or ""
            rev = getattr(item, "rev", "") or ""
            targets.add(
                BomTarget(
                    root=extract_numeric_root(code),
                    rev=normalize_rev(rev),
                    original_code=(code or "").strip(),
                    # se part_lists non porta parent info, restano vuoti (compatibile)
                )
            )
        return targets

    # NEW: preferisci la versione con parent se possibile
    try:
        targets = build_targets_from_pbs_with_parents(pbs)
        if targets:
            return targets
    except Exception:
        # fallback al comportamento legacy
        pass

    rows = getattr(pbs, "rows", []) or []
    for r in rows:
        code = getattr(r, "code", "") or ""
        rev = getattr(r, "rev", "") or ""
        if (code or "").strip().upper().endswith("PRTL"):
            targets.add(
                BomTarget(
                    root=extract_numeric_root(code),
                    rev=normalize_rev(rev),
                    original_code=(code or "").strip(),
                )
            )

    return targets


# -----------------------------
# BOM header verification (truth source)
# -----------------------------
@dataclass(frozen=True)
class BomHeaderVerification:
    """
    Risultato spiegabile della verifica header BOM contro i targets PBS.
    """
    matched: bool
    reason: str  # "MATCH" | "NO_NUMERIC_MATCH" | "REV_MISSING" | "REV_MISMATCH"
    best_target: Optional[BomTarget]
    best_match_len: int

    header_code: str
    header_rev: str

    suggestion: str = ""  # e.g. "Fix BOM header PN to expected parent PN ..."


def verify_bom_header_with_diagnosis(header_code: str, header_rev: str, targets: Set[BomTarget]) -> BomHeaderVerification:
    """
    Verifica su header BOM (verità assoluta), ma con regola elastica:
      - match numerico >= 6 tra header_code e target.original_code
      - REV: se entrambe presenti devono combaciare; se il target ha rev e header no -> non validare

    NEW:
      - ritorna anche reason + match_len + suggestion (se target ha expected_parent_code).
    """
    hr = normalize_rev(header_rev)

    best: Optional[BomTarget] = None
    best_len = 0

    # flags per motivazioni
    saw_numeric_match = False
    saw_rev_required_but_missing = False
    saw_rev_mismatch = False

    for t in targets:
        ml = best_numeric_match_len(t.original_code, header_code)
        if ml < MIN_MATCH_LEN:
            continue

        saw_numeric_match = True

        # regola REV
        if t.rev:
            if not hr:
                saw_rev_required_but_missing = True
                continue
            if t.rev != hr:
                saw_rev_mismatch = True
                continue

        if ml > best_len:
            best = t
            best_len = ml

    if best is not None:
        # Matched
        suggestion = ""
        # Se abbiamo parent atteso, possiamo suggerire fix header PN in modo "business"
        if (best.expected_parent_code or "").strip():
            exp_parent = (best.expected_parent_code or "").strip()
            if normalize_alnum(header_code) != normalize_alnum(exp_parent):
                suggestion = f"Verifica/correggi header PN: atteso parent PBS '{exp_parent}'."
        return BomHeaderVerification(
            matched=True,
            reason="MATCH",
            best_target=best,
            best_match_len=best_len,
            header_code=(header_code or "").strip(),
            header_rev=(header_rev or "").strip(),
            suggestion=suggestion,
        )

    # Not matched: pick most meaningful reason
    if not saw_numeric_match:
        reason = "NO_NUMERIC_MATCH"
        suggestion = "Header PN non riconducibile a nessun target PBS (match numerico < soglia)."
    elif saw_rev_required_but_missing:
        reason = "REV_MISSING"
        suggestion = "REV richiesta dal PBS ma mancante nell'header BOM: completa l'header REV o allinea la sorgente."
    elif saw_rev_mismatch:
        reason = "REV_MISMATCH"
        suggestion = "REV header BOM diversa da quella attesa dal PBS: allinea REV o verifica naming/filtri."
    else:
        reason = "NO_MATCH"
        suggestion = "Header non validabile contro i target (motivo generico)."

    return BomHeaderVerification(
        matched=False,
        reason=reason,
        best_target=None,
        best_match_len=0,
        header_code=(header_code or "").strip(),
        header_rev=(header_rev or "").strip(),
        suggestion=suggestion,
    )


def verify_bom_header_against_targets(header_code: str, header_rev: str, targets: Set[BomTarget]) -> Optional[BomTarget]:
    """
    (Compatibilità) Verifica su header BOM (verità assoluta), ma con regola elastica:
      - match numerico >= 6 tra header_code e target.original_code
      - REV: se entrambe presenti devono combaciare; se il target ha rev e header no -> non validare
    Ritorna il target migliore (match_len più alto) o None.

    Nota: ora delega alla funzione con diagnosi e ritorna solo best_target.
    """
    v = verify_bom_header_with_diagnosis(header_code, header_rev, targets)
    return v.best_target if v.matched else None
