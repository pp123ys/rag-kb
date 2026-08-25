# src/ragkb/retriever/retriever.py
from ragkb.models import Chunk
from ragkb.retriever.rrf import rrf_merge


class Retriever:
    """双路召回 → RRF 融合 → （可选重排）→ 版本过滤 → top_m。"""

    def __init__(self, indexer, reranker=None, version_filter=None):
        self._indexer = indexer
        self._reranker = reranker      # 可空：跳过重排
        self._version_filter = version_filter  # 可空：不过滤

    def retrieve(self, query: str, query_vec: list[float],
                 top_k: int = 50, top_n: int = 20, top_m: int = 5,
                 must_not_versions: dict[str, str] | None = None,
                 ) -> list[Chunk]:
        dense_hits = self._indexer.search_dense(
            query_vec, top_k, must_not_versions=must_not_versions)
        keyword_hits = self._indexer.search_keyword(
            query, top_k, must_not_versions=must_not_versions)

        # RRF：用 chunk_id 融合，再还原 Chunk
        id_to_chunk = {c.chunk_id: c for c in [*dense_hits, *keyword_hits]}
        merged_ids = rrf_merge(
            [[c.chunk_id for c in dense_hits],
             [c.chunk_id for c in keyword_hits]],
            k=60,
        )[:top_n]
        fused = [id_to_chunk[i] for i in merged_ids if i in id_to_chunk]

        # 重排（可选）
        if self._reranker is not None:
            fused = self._reranker.rerank(query, fused)

        # 版本过滤（可选）
        if self._version_filter is not None:
            fused = self._version_filter(fused)

        return fused[:top_m]
