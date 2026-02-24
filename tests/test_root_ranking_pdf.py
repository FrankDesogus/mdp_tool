import sys
import types

sys.modules.setdefault("pdfplumber", types.SimpleNamespace())

from core.use_case.analyze_folder_pdf import rank_roots


def test_rank_roots_prefers_folder_match():
    roots = [("E0181296-06", "06"), ("E1111111-01", "01")]
    children_of = {
        "E0181296-06": {"E1111111-01"},
        "E1111111-01": set(),
    }
    header_nodes = set(children_of.keys())

    selected, rows = rank_roots(
        roots=roots,
        folder_hint=("E0181296-06", "06"),
        children_of=children_of,
        header_nodes=header_nodes,
    )

    assert selected == ("E0181296-06", "06")
    assert rows[0]["folder_match"] == 1


def test_rank_roots_prefers_reachable_count_without_folder_hint():
    roots = [("E0000001-01", "01"), ("E0000002-01", "01")]
    children_of = {
        "E0000001-01": {"E0100000-01", "E0100001-01"},
        "E0100000-01": {"E0100002-01"},
        "E0100001-01": set(),
        "E0100002-01": set(),
        "E0000002-01": {"E0200000-01"},
        "E0200000-01": set(),
    }
    header_nodes = set(children_of.keys())

    selected, _ = rank_roots(
        roots=roots,
        folder_hint=None,
        children_of=children_of,
        header_nodes=header_nodes,
    )

    assert selected == ("E0000001-01", "01")
