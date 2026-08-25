"""检索评测指标：Recall@K 与 MRR（纯标准库，无外部依赖）。"""


def recall_at_k(ranked_lists, gold_ids, k):
    """Recall@K：gold 出现在前 K 位结果中的查询占比。

    Args:
        ranked_lists: list[list]，每条查询的按相关度降序的 chunk id 列表。
        gold_ids: list，与 ranked_lists 位置对应的每条查询的 gold chunk id。
        k: 只看前 K 个结果。

    Returns:
        float in [0, 1]。
    """
    hits = 0
    for ranked, gold in zip(ranked_lists, gold_ids):
        hits += 1 if gold in ranked[:k] else 0
    return hits / max(len(ranked_lists), 1)


def mrr(ranked_lists, gold_ids):
    """MRR：每条查询取第一个 gold 命中位置的倒数，再对查询数取平均。

    Args:
        ranked_lists: list[list]，每条查询的按相关度降序的 chunk id 列表。
        gold_ids: list，与 ranked_lists 位置对应的每条查询的 gold chunk id。

    Returns:
        float in [0, 1]。
    """
    total = 0.0
    for ranked, gold in zip(ranked_lists, gold_ids):
        try:
            total += 1.0 / (ranked.index(gold) + 1)
        except ValueError:
            pass
    return total / max(len(ranked_lists), 1)
