from pathlib import Path

from core.domain.models import BomDocument, BomHeader, NormalizedBomLine
from core.services.part_master import build_part_master, lookup_part_info


def _doc(lines):
    return BomDocument(path=Path('x.pdf'), header=BomHeader(code='A', revision='01'), lines=lines)


def test_part_master_canonical_lookup_and_best_of_merge():
    docs = [
        _doc([
            NormalizedBomLine(pos='0010', internal_code='E0216160 01', description='RES.SMD 0805', qty=1, unit='NR', manufacturer='', manufacturer_code=''),
            NormalizedBomLine(pos='0020', internal_code='E021616001', description='', qty=1, unit='NR', manufacturer='YAGEO', manufacturer_code='RC0805'),
        ])
    ]

    master = build_part_master(docs)
    info = lookup_part_info(master, 'E0216160-01', '01')

    assert info is not None
    assert info.description == 'RES.SMD 0805'
    assert info.manufacturer == 'YAGEO'
    assert info.manufacturer_code == 'RC0805'
