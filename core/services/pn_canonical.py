# core/services/pn_canonical.py
from __future__ import annotations
import re
from typing import Optional

_SEP_RE = re.compile(r"[\s\-_\/\.]+")
_SUFFIX_2DIG_WITH_SEP_RE = re.compile(r"^([A-Z]{1,3}\d{4,})[\s\-_\/\.]+(\d{2})$")
_SUFFIX_2DIG_COMPACT_RE = re.compile(r"^([A-Z]{1,3}\d{7,})(\d{2})$")


def _normalize_parts(code: str) -> str:
    parts = [p for p in _SEP_RE.split(code) if p]
    if not parts:
        return ""
    return "-".join(parts)

def canonicalize_rev(rev: str) -> str:
    return (rev or "").strip().upper()


def canonicalize_part_number(code: str, suffix: Optional[str] = None) -> str:
    c_raw = (code or "").strip().upper()
    if not c_raw:
        return ""

    c = re.sub(r"\s+", " ", c_raw).strip()
    sfx = canonicalize_rev(suffix or "")

    m = _SUFFIX_2DIG_WITH_SEP_RE.match(c)
    if m:
        base, parsed_sfx = m.group(1), m.group(2)
        return f"{base}-{(sfx or parsed_sfx)}"

    m = _SUFFIX_2DIG_COMPACT_RE.match(c)
    if m:
        base, parsed_sfx = m.group(1), m.group(2)
        return f"{base}-{(sfx or parsed_sfx)}"

    normalized = _normalize_parts(c)
    if not normalized:
        return ""
    if sfx and normalized != sfx:
        return f"{normalized}-{sfx}"
    return normalized

def canonicalize_pn(code: str, *, rev: Optional[str] = None) -> str:
    """
    Canonical PN (STRICT, non ambiguo):
      - uppercase
      - preserva il confine tra base e suffisso a 2 cifre se presente:
          'E0216160 01' -> 'E0216160-01'
      - normalizza separatori interni in modo conservativo (non concatena a caso)
      - NON rimuove mai la rev dal PN (rev è un attributo separato)
    """
    return canonicalize_part_number(code or "", suffix=canonicalize_rev(rev or ""))
