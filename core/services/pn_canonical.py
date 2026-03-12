# core/services/pn_canonical.py
from __future__ import annotations
import os
import logging
import re
from typing import Optional

_SEP_RE = re.compile(r"[\s\-_\/\.]+")
_SUFFIX_2DIG_WITH_SEP_RE = re.compile(r"^([A-Z]{1,3}\d{4,})[\s\-_\/\.]+(\d{2})$")
_SUFFIX_2DIG_COMPACT_RE = re.compile(r"^([A-Z]{1,3}\d{7,})(\d{2})$")
_LOG = logging.getLogger(__name__)
_DEBUG_ENV = "BOM_KEY_DEBUG"
_DEBUG_TARGET = "166104001"


def _key_debug_enabled() -> bool:
    return (os.getenv(_DEBUG_ENV, "") or "").strip() == "1"


def _is_target(*values: str) -> bool:
    return any(_DEBUG_TARGET in ((v or "").strip()) for v in values)


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

    branch = "fallback"
    suffix_appended = False

    m = _SUFFIX_2DIG_WITH_SEP_RE.match(c)
    if m:
        branch = "regex_with_sep"
        base, parsed_sfx = m.group(1), m.group(2)
        out = f"{base}-{(sfx or parsed_sfx)}"
        suffix_appended = bool(sfx)
        if _key_debug_enabled() and _is_target(c_raw, sfx, out):
            _LOG.info(
                "[KEY_DEBUG][canonicalize_part_number] input_code=%s input_suffix=%s output=%s branch=%s suffix_appended=%s",
                c_raw,
                sfx,
                out,
                branch,
                suffix_appended,
            )
        return out

    m = _SUFFIX_2DIG_COMPACT_RE.match(c)
    if m:
        branch = "regex_compact"
        base, parsed_sfx = m.group(1), m.group(2)
        out = f"{base}-{(sfx or parsed_sfx)}"
        suffix_appended = bool(sfx)
        if _key_debug_enabled() and _is_target(c_raw, sfx, out):
            _LOG.info(
                "[KEY_DEBUG][canonicalize_part_number] input_code=%s input_suffix=%s output=%s branch=%s suffix_appended=%s",
                c_raw,
                sfx,
                out,
                branch,
                suffix_appended,
            )
        return out

    normalized = _normalize_parts(c)
    if not normalized:
        return ""
    if sfx and normalized != sfx:
        if normalized.endswith(f"-{sfx}"):
            out = normalized
        else:
            out = f"{normalized}-{sfx}"
            suffix_appended = True
    else:
        out = normalized

    if _key_debug_enabled() and _is_target(c_raw, sfx, out):
        _LOG.info(
            "[KEY_DEBUG][canonicalize_part_number] input_code=%s input_suffix=%s output=%s branch=%s suffix_appended=%s",
            c_raw,
            sfx,
            out,
            branch,
            suffix_appended,
        )
    return out

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
