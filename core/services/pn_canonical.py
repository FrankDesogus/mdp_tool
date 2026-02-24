# core/services/pn_canonical.py
from __future__ import annotations
import re
from typing import Optional

_SEP_RE = re.compile(r"[\s\-_\/\.]+")
_SUFFIX_2DIG_WITH_SEP_RE = re.compile(r"^([A-Z]{1,3}\d{4,})[\s\-_\/\.]+(\d{2})$")
_SUFFIX_2DIG_COMPACT_RE = re.compile(r"^([A-Z]{1,3}\d{4,})(\d{2})$")

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

    # caso esplicito con separatore: "E0216160 01" -> "E0216160-01"
    m = _SUFFIX_2DIG_WITH_SEP_RE.match(c)
    if m:
        base, suf = m.group(1), m.group(2)
        return f"{base}-{suf}"

    # caso compatto (senza separatore) consentito SOLO se la rev passata coincide.
    # evita falsi positivi tipo "E0029472" -> "E00294-72".
    r = canonicalize_rev(rev or "")
    m = _SUFFIX_2DIG_COMPACT_RE.match(c)
    if m and r and m.group(2) == r:
        base, suf = m.group(1), m.group(2)
        return f"{base}-{suf}"

    # fallback: elimina separatori, ma usando '-' come separatore standard tra token
    parts = [p for p in _SEP_RE.split(c) if p]
    if not parts:
        return ""

    return "-".join(parts)
