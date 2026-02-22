# core/services/pbs_service.py
from pathlib import Path

from core.domain.models import PbsDocument
from core.parsers.pbs_excel_com import parse_pbs_excel_via_com
import core.parsers.pbs_excel_com as m

# ✅ NEW: alias/canonicalizzazione codici (external -> internal)
from core.services.code_aliases import canonicalize_code

print("PBS PARSER MODULE PATH:", m.__file__)


def _clone_row_with_code(row, new_code: str):
    """
    Ritorna una copia di `row` con field `code` aggiornato.
    Compatibile con:
      - dataclass(frozen=True) -> dataclasses.replace
      - namedtuple -> _replace
      - fallback: costruttore per keyword (code, rev, qty, level, src_row, description)
    """
    # 1) dataclass
    try:
        import dataclasses
        if dataclasses.is_dataclass(row):
            return dataclasses.replace(row, code=new_code)
    except Exception:
        pass

    # 2) namedtuple (o simili)
    try:
        rep = getattr(row, "_replace", None)
        if callable(rep):
            return row._replace(code=new_code)
    except Exception:
        pass

    # 3) fallback: ricostruzione con kwargs (campi "standard" PBS)
    try:
        cls = row.__class__
        kwargs = {
            "code": new_code,
            "rev": getattr(row, "rev", ""),
            "qty": getattr(row, "qty", None),
            "level": getattr(row, "level", 0),
            "src_row": getattr(row, "src_row", -1),
            "description": getattr(row, "description", ""),
        }
        return cls(**kwargs)
    except Exception:
        # Se proprio non riusciamo a clonare, torniamo la riga originale.
        # (Meglio non bloccare il load PBS.)
        return row


def load_pbs(path: Path, sheet_index: int = 1) -> PbsDocument:
    rows = parse_pbs_excel_via_com(str(path), sheet_index=sheet_index)

    # ✅ NEW: canonicalizza i codici PBS senza mutare gli oggetti (code è read-only)
    new_rows = []
    for r in rows:
        try:
            code = getattr(r, "code", None)
            if code:
                new_code = canonicalize_code(code)
                if new_code and str(new_code) != str(code):
                    r = _clone_row_with_code(r, new_code)
        except Exception:
            pass
        new_rows.append(r)

    return PbsDocument(path=path, rows=new_rows)
