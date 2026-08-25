import hashlib
import re
import uuid

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

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

# 嵌入式本地模式按绝对路径复用 client：Qdrant 嵌入式单目录有进程级文件锁，
# 同进程内多个 QdrantClient(path=...) 指向同一目录会互斥报错。缓存后：
# 同一路径只建一个 client，多实例（多 router / both 双传输 / 测试多用例）共享，
# 也符合嵌入式语义（单进程单实例）。远程 url 模式不做缓存。
_EMBEDDED_CLIENTS: dict[str, QdrantClient] = {}


def _point_id(chunk_id: str) -> str:
    """chunk_id → 合法 Qdrant 点 ID（Qdrant 仅接受无符号整数或 UUID 字符串）。

    确定性 chunk_id（doc-version-index）不是 UUID，用 uuid5 映射为确定性
    UUID：同一 chunk_id 恒映射到同一点，重跑入库覆盖同一点而非追加（幂等）。
    已是合法 UUID 的 chunk_id 原样透传（兼容既有数据）。
    """
    if _UUID_RE.fullmatch(chunk_id):
        return chunk_id
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


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
    """Qdrant 单存储：稠密向量 + 稀疏向量（BM25）+ payload 过滤。

    支持两种连接方式（由 settings 决定）：
    - 嵌入式（免 Docker）：settings.qdrant_path 非空 → QdrantClient(path=...)，
      数据落盘本地目录，零外部依赖；
    - 远程服务：settings.qdrant_url → QdrantClient(url=...) 连独立 Qdrant 服务。
    """

    def __init__(self, settings, client: QdrantClient | None = None):
        self._settings = settings
        if client is not None:
            self._client = client
        elif settings.qdrant_path:
            # 嵌入式本地模式：数据目录必须存在，否则 Qdrant 拒绝打开；
            # client 按绝对路径复用（见 _EMBEDDED_CLIENTS 注释）
            from pathlib import Path
            path = str(Path(settings.qdrant_path).resolve())
            Path(path).mkdir(parents=True, exist_ok=True)
            if path not in _EMBEDDED_CLIENTS:
                _EMBEDDED_CLIENTS[path] = QdrantClient(path=path)
            self._client = _EMBEDDED_CLIENTS[path]
        else:
            # check_compatibility=False：客户端 1.19 与 docker-compose 锁定的服务器
            # 1.9.7 存在版本差，默认会每次连接都打兼容性告警；此处显式关闭该检查
            # （服务端协议稳定，版本差不影响本客户端用到的 API）。
            self._client = QdrantClient(
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
            # 文本必须随 payload 存储：向量无法重建原文，检索命中后 text 直接读 payload
            payload["text"] = chunk.text
            # 点 ID 由 chunk_id 确定性映射：幂等重入覆盖同一点，不产生重复块
            points.append(PointStruct(
                id=_point_id(chunk.chunk_id),
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

    def fetch(self, chunk_id: str) -> Chunk | None:
        """按 chunk_id 取回 chunk（点 ID 经 _point_id 映射，payload 含原文）。"""
        self.ensure_collection()
        points = self._client.retrieve(
            self._collection, ids=[_point_id(chunk_id)],
            with_payload=True)
        if not points:
            return None
        return self._to_hits(points)[0]

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
