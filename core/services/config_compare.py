
# core/services/config_compare.py
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

from core.domain.models import PbsDocument
from core.services.exploder import ExplosionResult, _norm_code, _norm_rev


@dataclass
class ConfigCompletenessReport:
    root_code: str
    root_rev: str

    # Coverage
    pbs_codes: Set[str] = field(default_factory=set)
    exploded_codes: Set[str] = field(default_factory=set)

    # Findings
    pbs_unexploded_assemblies: Set[str] = field(default_factory=set)  # in PBS, ma non esplosi come assembly
    bom_introduced_not_in_pbs: Dict[str, Decimal] = field(default_factory=dict)  # PN in flat, assente PBS
    revision_mismatches: List[str] = field(default_factory=list)

    # Diagnostics
    cycles: List[Tuple[str, ...]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Completeness: PBS unique PN={len(self.pbs_codes)} | flat PN={len(self.bom_introduced_not_in_pbs)+len(self.exploded_codes)} | "
            f"introduced_not_in_pbs={len(self.bom_introduced_not_in_pbs)} | pbs_unexploded_assemblies={len(self.pbs_unexploded_assemblies)} | "
            f"rev_mismatches={len(self.revision_mismatches)} | cycles={len(self.cycles)}"
        )


def compare_explosion_to_pbs(
    *,
    pbs: PbsDocument,
    explosion: ExplosionResult,
) -> ConfigCompletenessReport:
    rep = ConfigCompletenessReport(root_code=explosion.root_code, root_rev=explosion.root_rev)

    # PBS index
    pbs_rev_by_code: Dict[str, str] = {}
    for r in pbs.rows:
        c = _norm_code(r.code)
        if not c:
            continue
        rep.pbs_codes.add(c)
        # Se lo stesso PN appare più volte in PBS con rev diverse, segnala
        rv = _norm_rev(r.rev)
        if c in pbs_rev_by_code and pbs_rev_by_code[c] != rv and rv:
            rep.revision_mismatches.append(f"PBS: PN {c} compare con REV diverse: {pbs_rev_by_code[c]} vs {rv}")
        if rv:
            pbs_rev_by_code[c] = rv

    # Exploded assemblies set (code-only)
    rep.exploded_codes = {c for (c, _r) in explosion.exploded_assemblies}

    # 2a) Mismatch REV per assiemi (unico caso verificabile): PBS rev vs BOM header rev usata
    for (code, bom_rev) in explosion.exploded_assemblies:
        exp_rev = pbs_rev_by_code.get(code, "")
        if exp_rev and _norm_rev(bom_rev) and _norm_rev(bom_rev) != exp_rev:
            rep.revision_mismatches.append(
                f"ASSEMBLY: PN {code} PBS REV {exp_rev} ma BOM header REV {_norm_rev(bom_rev)}"
            )

    # 1) PN introdotti da BOM non presenti in PBS
    for pn, qty in explosion.qty_by_code.items():
        if pn not in rep.pbs_codes:
            rep.bom_introduced_not_in_pbs[pn] = qty

    # 2) Assemblies PBS mai esplosi (definizione robusta): PN in PBS per cui esiste almeno una BOM
    # nel folder, ma che non è mai stato attraversato come parent nell'esplosione.
    for pn in rep.pbs_codes:
        if pn in getattr(explosion, "available_bom_codes", set()) and pn not in rep.exploded_codes:
            rep.pbs_unexploded_assemblies.add(pn)

    # 3) Revision mismatches dalle diagnostiche di esplosione
    for (pn, expected, found) in explosion.rev_mismatch_sub_boms:
        rep.revision_mismatches.append(f"BOM: PN {pn} rev richiesta {expected} ma usata {found}")

    rep.cycles = list(explosion.cycles)
    rep.warnings.extend(explosion.warnings)
    rep.errors.extend(explosion.errors)

    return rep
