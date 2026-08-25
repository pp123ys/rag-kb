import numpy as np


class Embedder:
    """BGE-M3 稠密向量封装（sentence-transformers）。"""

    def __init__(self, model_name: str = "BAAI/bge-m3",
                 encoder=None, device: str = "cpu"):
        self.model_name = model_name
        self._encoder = encoder  # 测试注入
        self._device = device

    def _ensure(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(
                self.model_name, device=self._device)

    def embed(self, texts: list[str]) -> np.ndarray:
        """批量嵌入，返回 (n, dim) float32。"""
        self._ensure()
        return np.asarray(
            self._encoder.encode(texts, normalize_embeddings=True,
                                 batch_size=32),
            dtype=np.float32,
        )
