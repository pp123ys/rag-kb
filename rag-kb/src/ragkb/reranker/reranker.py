import threading

from ragkb.models import Chunk


class Reranker:
    """bge-reranker 精排：对 query 与每个 chunk 打分，返回降序排列的 Chunk 列表。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 reranker=None, device: str = "cpu", cache_dir: str | None = None):
        self.model_name = model_name
        self._reranker = reranker  # 测试注入
        self._device = device
        self._cache_dir = cache_dir  # 模型缓存目录（缺省走 HF 全局缓存）
        self._lock = threading.Lock()

    def _ensure(self):
        if self._reranker is None:
            with self._lock:
                if self._reranker is None:  # double-checked locking
                    from sentence_transformers import CrossEncoder
                    # CrossEncoder 用 cache_dir（SentenceTransformer 用 cache_folder）
                    self._reranker = CrossEncoder(
                        self.model_name, device=self._device, max_length=512,
                        cache_dir=self._cache_dir)

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """返回按相关性降序排列的 Chunk 列表（不含分数）。"""
        return [c for c, _ in self.rerank_scored(query, chunks)]

    def rerank_scored(self, query: str,
                      chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
        """精排并返回 (Chunk, 分数) 对，按分数降序。分数供相关性阈值判定。"""
        if not chunks:
            return []
        self._ensure()
        pairs = [(query, c.text) for c in chunks]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda p: p[1], reverse=True)
        return [(c, float(sc)) for c, sc in ranked]
