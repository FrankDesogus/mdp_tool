# core/services/code_aliases.py
from __future__ import annotations

from typing import Dict

from core.services.bom_prefilter import normalize_alnum


# -----------------------------------------------------------------------------
# HARD-CODED ALIASES (external -> internal)
#
# Inserisci qui le corrispondenze usate in azienda.
# Le chiavi/valori vengono normalizzate (upper + strip non-alnum) prima dell'uso.
#
# Esempio:
#   PBS usa "C0103489" (codice esterno)
#   BOM usa "23092100C16AN" (codice interno)
# -----------------------------------------------------------------------------
CODE_ALIASES_RAW: Dict[str, str] = {
    "C0103489": "23092100C16AN",
    "C0103488": "23092100C16CE",
    "C0103487": "25012401C64AE",
    "C0103486": "25050500C59AN"


    # aggiungi qui altre mapping...
}


def _norm(code: str) -> str:
    return normalize_alnum(code or "")


# mapping normalizzato, costruito una volta
_CODE_ALIASES: Dict[str, str] = {_norm(k): _norm(v) for k, v in CODE_ALIASES_RAW.items() if _norm(k) and _norm(v)}


def canonicalize_code(code: str) -> str:
    """
    Converte un codice esterno nel corrispondente codice interno (se presente).
    Se non esiste mapping, ritorna il codice normalizzato.
    """
    c = _norm(code)
    if not c:
        return c
    return _CODE_ALIASES.get(c, c)


def has_alias(code: str) -> bool:
    return _norm(code) in _CODE_ALIASES
