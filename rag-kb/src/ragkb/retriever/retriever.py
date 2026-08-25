# src/ragkb/retriever/retriever.py
import logging

from ragkb.models import Chunk
from ragkb.retriever.rrf import rrf_merge

logger = logging.getLogger(__name__)


class Retriever:
    """双路召回 → RRF 融合 → （可选重排）→ 版本过滤 → top_m。"""

    def __init__(self, indexer, reranker=None, version_filter=None):
        self._indexer = indexer
        self._reranker = reranker      # 可空：跳过重排
        self._version_filter = version_filter  # 可空：不过滤

    def retrieve(self, query: str, query_vec: list[float],
                 top_k: int = 50, top_n: int = 20, top_m: int = 5,
                 must_not_versions: dict[str, str] | None = None,
                 version_filter=None) -> list[Chunk]:
        """双路召回 → RRF 融合 → （可选重排）→ 版本过滤 → top_m。

        单路召回失败时降级为另一路结果并记录告警（§8），不让一路故障
        拖垮整个查询。version_filter 按调用覆盖构造注入的过滤
        （如显式版本查询）；缺省沿用 self._version_filter。
        """
        return [c for c, _ in self.retrieve_scored(
            query, query_vec, top_k=top_k, top_n=top_n, top_m=top_m,
            must_not_versions=must_not_versions, version_filter=version_filter)]

    def retrieve_scored(self, query: str, query_vec: list[float],
                        top_k: int = 50, top_n: int = 20, top_m: int = 5,
                        must_not_versions: dict[str, str] | None = None,
                        version_filter=None) -> list[tuple[Chunk, float | None]]:
        """同 retrieve，但保留重排分数：(Chunk, score) 列表，按分数降序。

        无重排器时 score 为 None（RRF 序，调用方自行决定是否做阈值过滤）。
        供 MCP search 做「相关性不足 → 没有找到」判定。
        """
        try:
            dense_hits = self._indexer.search_dense(
                query_vec, top_k, must_not_versions=must_not_versions)
        except Exception:
            logger.warning("向量路检索失败，降级为仅关键词路", exc_info=True)
            dense_hits = []
        try:
            keyword_hits = self._indexer.search_keyword(
                query, top_k, must_not_versions=must_not_versions)
        except Exception:
            logger.warning("关键词路检索失败，降级为仅向量路", exc_info=True)
            keyword_hits = []

        # RRF：用 chunk_id 融合，再还原 Chunk
        id_to_chunk = {c.chunk_id: c for c in [*dense_hits, *keyword_hits]}
        merged_ids = rrf_merge(
            [[c.chunk_id for c in dense_hits],
             [c.chunk_id for c in keyword_hits]],
            k=60,
        )[:top_n]
        fused = [id_to_chunk[i] for i in merged_ids if i in id_to_chunk]

        # 重排（可选）：空结果直接短路，避免重排器拒绝空输入
        if not fused:
            return []
        scored: list[tuple[Chunk, float | None]]
        if self._reranker is not None:
            scored = self._reranker.rerank_scored(query, fused)
        else:
            scored = [(c, None) for c in fused]

        # 版本过滤（可选）：调用级 version_filter 优先，缺省用构造注入的过滤
        if version_filter is not None:
            scored = [(c, sc) for c, sc in scored if c in version_filter([c for c, _ in scored])]
        elif self._version_filter is not None:
            kept = self._version_filter([c for c, _ in scored])
            kept_ids = {c.chunk_id for c in kept}
            scored = [(c, sc) for c, sc in scored if c.chunk_id in kept_ids]

        return scored[:top_m]
