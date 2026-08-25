# tests/test_eval.py —— 评测指标（Recall@K / MRR）单元测试
from ragkb.eval.metrics import recall_at_k, mrr


def test_recall_at_k():
    assert recall_at_k([[1, 3, 5]], [5], k=3) == 1.0
    assert recall_at_k([[1, 3, 5]], [9], k=3) == 0.0


def test_mrr():
    # gold=1 在第 3 位（index 2）→ 1/(2+1) = 1/3
    # （计划原文写 0.5 是算术错误：1 在位置 3，MRR=1/3；已修正）
    assert abs(mrr([[2, 5, 1]], [1]) - 1 / 3) < 1e-9
    # gold=1 在第 2 位（index 1）→ 1/(1+1) = 0.5
    assert mrr([[2, 1, 5]], [1]) == 0.5


def test_metrics_empty_and_missing():
    # 空输入不除零
    assert recall_at_k([], [], k=3) == 0.0
    assert mrr([], []) == 0.0
    # gold 未出现在结果中 → 该条贡献 0
    assert recall_at_k([[1, 3, 5]], [9], k=3) == 0.0
    assert mrr([[1, 3, 5]], [9]) == 0.0
