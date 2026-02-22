# core/services/pn_canonical.py
from __future__ import annotations
import re
from typing import Optional

_SEP_RE = re.compile(r"[\s\-_\/\.]+")
_SUFFIX_2DIG_RE = re.compile(r"^([A-Z]{1,3}\d{4,})\s*(\d{2})$")  # E0216160 01, C0020628 03, ecc.

def canonicalize_rev(rev: str) -> str:
    return (rev or "").strip().upper()

def canonicalize_pn(code: str, *, rev: Optional[str] = None) -> str:
    """
    Canonical PN (STRICT, non ambiguo):
      - uppercase
      - preserva il confine tra base e suffisso a 2 cifre se presente:
          'E0216160 01' -> 'E0216160-01'
      - normalizza separatori interni in modo conservativo (non concatena a caso)
      - NON rimuove mai la rev dal PN (rev è un attributo separato)
    """
    c_raw = (code or "").strip().upper()
    if not c_raw:
        return ""

    # compatta whitespace multipli ma non perdere il confine
    c = re.sub(r"\s+", " ", c_raw).strip()

    m = _SUFFIX_2DIG_RE.match(c)
    if m:
        base, suf = m.group(1), m.group(2)
        return f"{base}-{suf}"

    # fallback: elimina separatori, ma usando '-' come separatore standard tra token
    parts = [p for p in _SEP_RE.split(c) if p]
    if not parts:
        return ""

    return "-".join(parts)
