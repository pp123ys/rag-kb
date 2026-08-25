# tests/test_rrf.py
from ragkb.retriever.rrf import rrf_merge


def test_rrf_prefers_items_ranked_high_in_both_lists():
    a = ["x1", "x2", "x3", "x4"]
    b = ["x4", "x1", "x5"]
    merged = rrf_merge([a, b], k=60)
    assert merged[0] == "x1"          # 两路都在前列
    assert "x4" in merged[:3]
    assert "x5" in merged             # 仅一路命中也能进
    assert len(merged) == len(set(merged))  # 去重


def test_rrf_single_list_preserves_order():
    merged = rrf_merge([["a", "b", "c"]], k=60)
    assert merged == ["a", "b", "c"]
