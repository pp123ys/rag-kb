import pytest

from ragkb.indexers.qdrant_indexer import QdrantIndexer, _sparse, _token_index, tokenize
from ragkb.models import Chunk


def _cid(n: int) -> str:
    """确定性 UUID 点 ID（Qdrant 仅接受无符号整数或 UUID 字符串）。"""
    return f"00000000-0000-4000-8000-{n:012d}"


@pytest.fixture
def indexer(settings):
    # 集成测试用小维度向量（2 维）代替真实 BGE-M3 1024 维，集合维度须一致
    settings.vector_size = 2
    return QdrantIndexer(settings)


# ---------- 纯单元测试（不依赖 docker / Qdrant） ----------

def test_tokenize_keeps_alnum_single_chars():
    tokens = tokenize("型号 A 与 B")
    assert "A" in tokens
    assert "B" in tokens
    assert "型号" in tokens


def test_tokenize_drops_punctuation_only():
    assert tokenize("！？，。") == []
    assert tokenize("---") == []
    assert "-" not in tokenize("A-100")


def test_sparse_produces_stable_token_indices_across_texts():
    text_a = "合同编号 HT-2026-001 使用说明"
    text_b = "另一文档中的 HT-2026-001 与 100 系列"
    sa = _sparse(text_a)
    sb = _sparse(text_b)
    ma = dict(zip(sa.indices, sa.values))
    mb = dict(zip(sb.indices, sb.values))
    # 同一 token 在两个文本中必须映射到同一索引（对齐），且落在 Qdrant
    # 稀疏向量允许的 u32 范围 [0, 2^32) 内
    for token in ("HT", "2026", "001"):
        idx = _token_index(token)
        assert 0 <= idx < 2**32
        assert idx in ma and idx in mb


def test_point_id_deterministic_and_qdrant_valid():
    """chunk_id → 点 ID：确定性（幂等重入覆盖同一点）+ 合法 UUID（Qdrant 约束）。"""
    import uuid as _uuid

    from ragkb.indexers.qdrant_indexer import _point_id
    a = _point_id("d1-v1.0-0")
    b = _point_id("d1-v1.0-0")
    c = _point_id("d1-v1.0-1")
    assert a == b
    assert a != c
    _uuid.UUID(a)  # 必须是合法 UUID，否则抛 ValueError
    # 已是合法 UUID 的 chunk_id 原样透传（兼容既有数据）
    assert _point_id(_cid(1)) == _cid(1)


# ---------- fetch(chunk_id) 按点 ID 取回 ----------

class _FakeRecord:
    """模拟 qdrant-client 1.19 retrieve 返回的 Record（含 .id / .payload）。"""

    def __init__(self, id, payload):
        self.id = id
        self.payload = payload


class _FakeRetrieveClient:
    def __init__(self, records):
        self._records = records
        self.calls = []

    def collection_exists(self, name):
        return True

    def retrieve(self, collection, ids, with_payload=True):
        self.calls.append((collection, list(ids), with_payload))
        return self._records


def test_fetch_returns_chunk_by_id(settings):
    from ragkb.indexers.qdrant_indexer import _point_id
    chunk_id = "d1-v1.0-0"
    rec = _FakeRecord(_point_id(chunk_id), {
        "chunk_id": chunk_id, "doc_id": "d1", "doc_type": "pdf",
        "source": "a.pdf", "text": "原文内容", "department": "销售部",
        "version": "v1.0", "effective_date": "2026-01-01",
        "table_id": None, "image_id": None,
    })
    client = _FakeRetrieveClient([rec])
    idx = QdrantIndexer(settings, client=client)
    got = idx.fetch(chunk_id)
    assert got is not None
    assert got.chunk_id == chunk_id
    assert got.text == "原文内容"
    assert got.source == "a.pdf"
    assert got.version == "v1.0"
    # 点 ID 经 _point_id 映射，请求带 payload
    assert client.calls == [("chunks", [_point_id(chunk_id)], True)]


def test_fetch_missing_returns_none(settings):
    idx = QdrantIndexer(settings, client=_FakeRetrieveClient([]))
    assert idx.fetch("d1-v1.0-9") is None


# ---------- 集成测试（需要本地 Qdrant，docker compose） ----------

@pytest.mark.integration
def test_upsert_and_query_dense(indexer):
    indexer.recreate()
    chunks = [
        Chunk(chunk_id=_cid(1), doc_id="d1", doc_type="pdf", source="a.pdf",
              text="产品规格说明", department="销售部", version="v1.0",
              effective_date="2026-01-15"),
        Chunk(chunk_id=_cid(2), doc_id="d1", doc_type="pdf", source="a.pdf",
              text="合同编号 HT-2026-001", version="v1.0",
              effective_date="2026-01-15"),
    ]
    indexer.upsert(chunks, embeddings=[[0.1, 0.2], [0.9, 0.8]])
    hits = indexer.search_dense([0.9, 0.8], top_k=5)
    assert hits[0].chunk_id == _cid(2)


@pytest.mark.integration
def test_version_filter_excludes_expired(indexer):
    indexer.recreate()
    chunks = [
        Chunk(chunk_id=_cid(1), doc_id="d1", doc_type="pdf", source="a.pdf",
              text="旧版本内容", version="v1.0", effective_date="2025-01-01"),
        Chunk(chunk_id=_cid(2), doc_id="d1", doc_type="pdf", source="a.pdf",
              text="新版本内容", version="v2.0", effective_date="2026-06-01"),
    ]
    indexer.upsert(chunks, embeddings=[[0.5, 0.5], [0.5, 0.5]])
    hits = indexer.search_dense(
        [0.5, 0.5], top_k=5, must_not_versions={"d1": "v1.0"})
    ids = {h.chunk_id for h in hits}
    assert _cid(2) in ids and _cid(1) not in ids


@pytest.mark.integration
def test_keyword_bm25_recall(indexer):
    indexer.recreate()
    chunks = [
        Chunk(chunk_id=_cid(1), doc_id="d1", doc_type="pdf", source="a.pdf",
              text="产品型号 A-100 使用说明", version="v1.0",
              effective_date="2026-01-01"),
        Chunk(chunk_id=_cid(2), doc_id="d1", doc_type="pdf", source="a.pdf",
              text="财务报销流程", version="v1.0", effective_date="2026-01-01"),
    ]
    indexer.upsert(chunks, embeddings=[[0.1, 0.1], [0.2, 0.2]])
    hits = indexer.search_keyword("A-100", top_k=5)
    assert hits[0].chunk_id == _cid(1)


@pytest.mark.integration
def test_payload_round_trip(indexer):
    """payload 元数据字段全量往返，text 随 payload 存储（向量无法重建原文）。"""
    indexer.recreate()
    chunk = Chunk(
        chunk_id=_cid(1), doc_id="d1", doc_type="pdf", source="a.pdf",
        text="含全量元数据的文档片段", department="技术部", version="v3.1",
        effective_date="2026-03-20", table_id="tbl-42", image_id="img-7",
    )
    indexer.upsert([chunk], embeddings=[[0.1, 0.9]])
    hits = indexer.search_dense([0.1, 0.9], top_k=5)
    assert len(hits) == 1
    got = hits[0]
    assert got.chunk_id == _cid(1)
    assert got.doc_id == "d1"
    assert got.doc_type == "pdf"
    assert got.source == "a.pdf"
    assert got.department == "技术部"
    assert got.version == "v3.1"
    assert got.effective_date == "2026-03-20"
    assert got.table_id == "tbl-42"
    assert got.image_id == "img-7"
    # round-trip 后 text 必须保留（向量无法重建原文）
    assert got.text == "含全量元数据的文档片段"

    # 关键词召回路同样必须保留 text
    kw_hits = indexer.search_keyword("全量元数据", top_k=5)
    assert len(kw_hits) == 1
    assert kw_hits[0].text == "含全量元数据的文档片段"


@pytest.mark.integration
def test_keyword_filter_excludes_versions(indexer):
    indexer.recreate()
    chunks = [
        Chunk(chunk_id=_cid(1), doc_id="d1", doc_type="pdf", source="a.pdf",
              text="合同履行条款", version="v1.0", effective_date="2025-01-01"),
        Chunk(chunk_id=_cid(2), doc_id="d1", doc_type="pdf", source="a.pdf",
              text="合同履行条款", version="v2.0", effective_date="2026-06-01"),
    ]
    indexer.upsert(chunks, embeddings=[[0.1, 0.1], [0.2, 0.2]])
    hits = indexer.search_keyword(
        "合同履行", top_k=5, must_not_versions={"d1": "v1.0"})
    ids = {h.chunk_id for h in hits}
    assert _cid(2) in ids and _cid(1) not in ids


@pytest.mark.integration
def test_search_empty_collection(indexer):
    indexer.recreate()
    assert indexer.search_dense([0.1, 0.2], top_k=5) == []
    assert indexer.search_keyword("任意关键词", top_k=5) == []


@pytest.mark.integration
def test_fetch_round_trip(indexer):
    """upsert 后按 chunk_id fetch 能取回原文与 source。"""
    indexer.recreate()
    chunk = Chunk(chunk_id=_cid(1), doc_id="d1", doc_type="pdf", source="a.pdf",
                  text="取回原文测试", version="v1.0", effective_date="2026-01-01")
    indexer.upsert([chunk], embeddings=[[0.1, 0.2]])
    got = indexer.fetch(_cid(1))
    assert got is not None
    assert got.chunk_id == _cid(1)
    assert got.text == "取回原文测试"
    assert got.source == "a.pdf"
    # 不存在的 chunk_id → None
    assert indexer.fetch("d1-v9.9-999") is None
