import hashlib
import re

import jieba
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter,
                                  MatchValue, PointStruct, SparseIndexParams,
                                  SparseVector, SparseVectorParams,
                                  VectorParams)

from ragkb.models import Chunk

# 单字符纯字母/数字 token（如 "A"、"100"）对型号/合同号召回有价值，保留；
# 中文单字（如 "的"）与纯标点（"-"）丢弃。
_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """中文分词（jieba），供稀疏向量与 BM25 关键词召回使用。

    保留多字 token（中文词、数字串）以及单字符字母/数字 token，
    以便型号/合同号（如 A-100、HT-2026-001）可召回；
    纯标点/符号串（如 "---"、"！？"）整体丢弃。
    """
    return [w for w in jieba.cut(text)
            if (len(w) > 1 or _ALNUM_RE.fullmatch(w))
            and any(ch.isalnum() for ch in w)]


def _token_index(token: str) -> int:
    """token → 全局稳定索引（跨文档一致）。

    注意：不能按文档局部建 vocab（同一 token 在不同文档会得到不同索引），
    否则 query 的稀疏向量无法与已入库文档对齐。用确定性哈希保证跨文档一致。

    索引必须落在 Qdrant 稀疏向量允许的 u32 范围 [0, 2^32)（服务器 1.9.7
    实测拒绝 ≥ 2^32 的索引），因此取 md5 前 4 字节作 32 位整数——
    相比旧 crc32 的 31 位截断空间翻倍，碰撞概率减半。
    """
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _sparse(text: str) -> SparseVector:
    tokens = tokenize(text)
    counts: dict[int, float] = {}
    for t in tokens:
        idx = _token_index(t)
        counts[idx] = counts.get(idx, 0.0) + 1.0
    if not counts:
        return SparseVector(indices=[0], values=[0.0])
    return SparseVector(indices=sorted(counts),
                        values=[counts[i] for i in sorted(counts)])


class QdrantIndexer:
    """Qdrant 单存储：稠密向量 + 稀疏向量（BM25）+ payload 过滤。"""

    def __init__(self, settings, client: QdrantClient | None = None):
        self._settings = settings
        # check_compatibility=False：客户端 1.19 与 docker-compose 锁定的服务器
        # 1.9.7 存在版本差，默认会每次连接都打兼容性告警；此处显式关闭该检查
        # （服务端协议稳定，版本差不影响本客户端用到的 API）。
        self._client = client or QdrantClient(
            url=settings.qdrant_url, check_compatibility=False)
        self._collection = settings.collection_name
        self._size = settings.vector_size

    def ensure_collection(self):
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._size,
                                        distance=Distance.COSINE),
            sparse_vectors_config={
                "bm25": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                )
            },
        )
        # doc_id/version 是 version_filter 的常用过滤字段，建 keyword 索引提升过滤性能
        self._client.create_payload_index(
            self._collection, field_name="doc_id", field_schema="keyword")
        self._client.create_payload_index(
            self._collection, field_name="version", field_schema="keyword")

    def recreate(self):
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self.ensure_collection()

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]):
        self.ensure_collection()
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks({len(chunks)}) 与 embeddings({len(embeddings)}) 数量不一致")
        points = []
        for chunk, vec in zip(chunks, embeddings, strict=True):
            payload = chunk.metadata()
            points.append(PointStruct(
                id=chunk.chunk_id,  # chunk_id 即 uuid4，直接作点 ID，无 crc32 碰撞
                vector={"": vec, "bm25": _sparse(chunk.text)},
                payload=payload,
            ))
        # 分批 upsert，避免单请求超出 Qdrant 约 32MB 请求体上限
        for i in range(0, len(points), 256):
            self._client.upsert(self._collection, points=points[i:i + 256])

    @staticmethod
    def _version_filter(must_not_versions: dict[str, str] | None):
        """{doc_id: version} → 排除过期版本组合的 Filter。

        用 Filter.must_not 嵌套 Filter 表达「排除 (doc_id AND version) 组合」，
        与检索后过滤（Task 11 version_filter）构成双保险，以检索后过滤为权威。
        """
        if not must_not_versions:
            return None
        return Filter(must_not=[
            Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="version", match=MatchValue(value=version)),
            ])
            for doc_id, version in must_not_versions.items()
        ])

    def search_dense(self, query_vec: list[float], top_k: int,
                     must_not_versions: dict[str, str] | None = None):
        """稠密向量召回。"""
        self.ensure_collection()
        hits = self._client.query_points(
            self._collection,
            query=query_vec,
            query_filter=self._version_filter(must_not_versions),
            limit=top_k,
            with_payload=True,
        ).points
        return self._to_hits(hits)

    def search_keyword(self, query: str, top_k: int,
                       must_not_versions: dict[str, str] | None = None):
        """BM25 关键词召回（稀疏向量最近邻），过滤语义同 search_dense。"""
        self.ensure_collection()
        hits = self._client.query_points(
            self._collection,
            query=_sparse(query),
            using="bm25",
            query_filter=self._version_filter(must_not_versions),
            limit=top_k,
            with_payload=True,
        ).points
        return self._to_hits(hits)

    @staticmethod
    def _to_hits(hits):
        out = []
        for h in hits:
            payload = h.payload or {}
            out.append(Chunk(
                chunk_id=payload.get("chunk_id") or str(h.id),
                doc_id=payload.get("doc_id", ""),
                doc_type=payload.get("doc_type", ""),
                source=payload.get("source", ""),
                text=payload.get("text", "") or "",
                department=payload.get("department", ""),
                version=payload.get("version", ""),
                effective_date=payload.get("effective_date", ""),
                table_id=payload.get("table_id"),
                image_id=payload.get("image_id"),
            ))
        return out
