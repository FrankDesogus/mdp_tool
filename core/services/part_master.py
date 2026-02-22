from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from core.domain.models import BomDocument, NormalizedBomLine
from core.services.pn_canonical import canonicalize_pn, canonicalize_rev


@dataclass(frozen=True)
class PartInfo:
    code: str
    description: str = ""
    manufacturer: str = ""
    manufacturer_code: str = ""


def canonical_part_key(code: str, rev: str = "") -> str:
    """Canonical key aligned with exploder child_code canonicalization."""
    raw = (code or "").strip()
    if not raw:
        return ""
    rev_n = canonicalize_rev(rev or "")
    return canonicalize_pn(raw, rev=(rev_n or None))


def _field_score(v: str) -> int:
    return len((v or "").strip())


def _best_of(prev: PartInfo, nxt: PartInfo) -> PartInfo:
    """Best-of merge, field by field (never replace non-empty with empty)."""
    return PartInfo(
        code=prev.code,
        description=prev.description if _field_score(prev.description) >= _field_score(nxt.description) else nxt.description,
        manufacturer=prev.manufacturer if _field_score(prev.manufacturer) >= _field_score(nxt.manufacturer) else nxt.manufacturer,
        manufacturer_code=(
            prev.manufacturer_code
            if _field_score(prev.manufacturer_code) >= _field_score(nxt.manufacturer_code)
            else nxt.manufacturer_code
        ),
    )


def build_part_master(boms: Iterable[BomDocument]) -> Dict[str, PartInfo]:
    out: Dict[str, PartInfo] = {}
    for b in boms or []:
        for ln in getattr(b, "lines", []) or []:
            if not isinstance(ln, NormalizedBomLine):
                continue
            key = canonical_part_key(getattr(ln, "internal_code", "") or "", getattr(ln, "rev", "") or "")
            if not key:
                continue
            candidate = PartInfo(
                code=key,
                description=(getattr(ln, "description", "") or "").strip(),
                manufacturer=(getattr(ln, "manufacturer", "") or "").strip(),
                manufacturer_code=(getattr(ln, "manufacturer_code", "") or "").strip(),
            )
            prev = out.get(key)
            out[key] = candidate if prev is None else _best_of(prev, candidate)
    return out


def lookup_part_info(part_master: Dict[str, PartInfo], code: str, rev: str = "") -> Optional[PartInfo]:
    if not part_master:
        return None
    key = canonical_part_key(code, rev)
    if key and key in part_master:
        return part_master[key]
    raw = (code or "").strip()
    if raw:
        return part_master.get(raw)
    return None


def missing_lookup_samples(part_master: Dict[str, PartInfo], codes: Iterable[Tuple[str, str]], sample_size: int = 20) -> List[str]:
    missing: List[str] = []
    for code, rev in codes:
        if lookup_part_info(part_master, code, rev) is None:
            missing.append(canonical_part_key(code, rev) or (code or "").strip())
        if len(missing) >= sample_size:
            break
    return missing
