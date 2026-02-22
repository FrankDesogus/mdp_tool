# mdp_tool/domain/models.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# =========================
# PBS / MDP
# =========================

@dataclass(frozen=True)
class MdpRow:
    src_row: int
    code: str
    description: str
    rev: str
    qty: float
    desc_col: int
    level: int


@dataclass(frozen=True)
class PbsDocument:
    """Contenuto PBS già parsato + metadati di origine."""
    path: Path
    rows: List[MdpRow]


# =========================
# BOM - normalizzazione
# =========================

class BomLineKind(str, Enum):
    MATERIAL = "MATERIAL"
    DOCUMENT_REF = "DOCUMENT_REF"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class NormalizedBomLine:
    pos: str
    internal_code: str
    description: str

    qty: Optional[float]          # può essere None (celle vuote / righe continue)
    unit: str

    kind: BomLineKind = BomLineKind.UNKNOWN

    # NEW: tipo riga (es. "Disegno"/"Drawing"/"Documento"/"DD"...), utile per classificazione robusta
    type: str = ""

    # campi extra (tabella BOM)
    val: str = ""
    rat: str = ""
    tol: str = ""
    refdes: str = ""
    tecn: str = ""
    notes: str = ""

    manufacturer: str = ""
    manufacturer_code: str = ""
    rev: str = ""


@dataclass(frozen=True)
class BomHeader:
    """Testata minima necessaria per linking (strict)."""
    code: str               # PN / CODE assieme
    revision: str
    title: str = ""
    doc_date_iso: str = ""  # opzionale


@dataclass(frozen=True)
class BomDocument:
    """BOM parsata (Excel o PDF) + righe normalizzate + path sorgente."""
    path: Path
    header: BomHeader
    lines: List[NormalizedBomLine]


# =========================
# LINKING PBS -> BOM (strict)
# =========================

class LinkStatus(str, Enum):
    OK = "OK"
    MISSING = "MISSING"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class BomRef:
    path: Path
    code: str
    revision: str


@dataclass(frozen=True)
class LinkResult:
    assembly_code: str
    assembly_rev: str
    status: LinkStatus
    bom: Optional[BomRef] = None
    candidates: Tuple[BomRef, ...] = tuple()
    message: str = ""


def key_code_rev(code: str, rev: str) -> Tuple[str, str]:
    """Chiave normalizzata per match strict (case-insensitive, trim)."""
    return (code or "").strip().upper(), (rev or "").strip().upper()
