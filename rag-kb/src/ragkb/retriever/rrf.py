# src/ragkb/retriever/rrf.py
def rrf_merge(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion：按排名倒数融合多路 id 列表，返回去重后的排序 id。

    假设每个列表内 id 互不重复（标准 RRF 假设）；同列表重复 id 会重复计分。
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return [item for item, _ in sorted(scores.items(),
                                       key=lambda kv: kv[1], reverse=True)]
