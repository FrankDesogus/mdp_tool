from dataclasses import dataclass
from decimal import Decimal
from typing import List

from core.domain.models import MdpRow
from core.domain.models import BomDocument  # adatta import al tuo file

@dataclass(frozen=True)
class PbsParseResult:
    path: str
    rows: List[MdpRow]

@dataclass(frozen=True)
class BomParseResult:
    path: str
    doc: BomDocument
