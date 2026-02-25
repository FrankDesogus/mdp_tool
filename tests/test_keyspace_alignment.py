from pathlib import Path

from core.domain.models import BomDocument, BomHeader, BomLineKind, NormalizedBomLine
from core.services.exploder_pdf import explode_boms_pdf
from app.state.project_context import ProjectContext


def test_flat_totals_keyspace_matches_edges_keyspace_for_revisioned_child() -> None:
    root = BomDocument(
        path=Path("root.pdf"),
        header=BomHeader(code="ROOT0001", revision="01"),
        lines=[
            NormalizedBomLine(
                pos="10",
                internal_code="166104001-04",
                description="child",
                qty=2.0,
                unit="NR",
                kind=BomLineKind.MATERIAL,
                rev="04",
            )
        ],
    )

    res = explode_boms_pdf(root_code="ROOT0001", root_rev="01", boms=[root])

    assert "166104001-04" in res.qty_by_code
    assert "166104001-04-04" not in res.qty_by_code

    _, parents_by_child, _ = ProjectContext._build_indices_from_edges(res.edges)
    assert "166104001-04" in parents_by_child
