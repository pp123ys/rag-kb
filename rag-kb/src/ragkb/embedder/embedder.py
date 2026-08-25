import threading

import numpy as np


class Embedder:
    """BGE-M3 稠密向量封装（sentence-transformers）。"""

    def __init__(self, model_name: str = "BAAI/bge-m3",
                 encoder=None, device: str = "cpu"):
        self.model_name = model_name
        self._encoder = encoder  # 测试注入
        self._device = device
        self._lock = threading.Lock()
        self._dim: int | None = None

    def _ensure(self):
        if self._encoder is None:
            with self._lock:
                if self._encoder is None:  # double-checked locking
                    from sentence_transformers import SentenceTransformer
                    self._encoder = SentenceTransformer(
                        self.model_name, device=self._device)

    def embed(self, texts: list[str]) -> np.ndarray:
        """批量嵌入，返回 (n, dim) float32；空输入返回 (0, dim)。"""
        if not texts:
            return np.empty((0, self._dim or 0), dtype=np.float32)
        self._ensure()
        vectors = np.asarray(
            self._encoder.encode(texts, normalize_embeddings=True,
                                 batch_size=32),
            dtype=np.float32,
        )
        self._dim = vectors.shape[1]
        return vectors
