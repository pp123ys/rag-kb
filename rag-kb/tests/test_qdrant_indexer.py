import pytest

from ragkb.indexers.qdrant_indexer import QdrantIndexer
from ragkb.models import Chunk


@pytest.fixture
def indexer(settings):
    # 集成测试用小维度向量（2 维）代替真实 BGE-M3 1024 维，集合维度须一致
    settings.vector_size = 2
    return QdrantIndexer(settings)


@pytest.mark.integration
def test_upsert_and_query_dense(indexer):
    indexer.recreate()
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", doc_type="pdf", source="a.pdf",
              text="产品规格说明", department="销售部", version="v1.0",
              effective_date="2026-01-15"),
        Chunk(chunk_id="c2", doc_id="d1", doc_type="pdf", source="a.pdf",
              text="合同编号 HT-2026-001", version="v1.0",
              effective_date="2026-01-15"),
    ]
    indexer.upsert(chunks, embeddings=[[0.1, 0.2], [0.9, 0.8]])
    hits = indexer.search_dense([0.9, 0.8], top_k=5)
    assert hits[0].chunk_id == "c2"


@pytest.mark.integration
def test_version_filter_excludes_expired(indexer):
    indexer.recreate()
    chunks = [
        Chunk(chunk_id="old", doc_id="d1", doc_type="pdf", source="a.pdf",
              text="旧版本内容", version="v1.0", effective_date="2025-01-01"),
        Chunk(chunk_id="new", doc_id="d1", doc_type="pdf", source="a.pdf",
              text="新版本内容", version="v2.0", effective_date="2026-06-01"),
    ]
    indexer.upsert(chunks, embeddings=[[0.5, 0.5], [0.5, 0.5]])
    hits = indexer.search_dense(
        [0.5, 0.5], top_k=5, must_not_versions={"d1": "v1.0"})
    ids = {h.chunk_id for h in hits}
    assert "new" in ids and "old" not in ids


@pytest.mark.integration
def test_keyword_bm25_recall(indexer):
    indexer.recreate()
    chunks = [
        Chunk(chunk_id="k1", doc_id="d1", doc_type="pdf", source="a.pdf",
              text="产品型号 A-100 使用说明", version="v1.0",
              effective_date="2026-01-01"),
        Chunk(chunk_id="k2", doc_id="d1", doc_type="pdf", source="a.pdf",
              text="财务报销流程", version="v1.0", effective_date="2026-01-01"),
    ]
    indexer.upsert(chunks, embeddings=[[0.1, 0.1], [0.2, 0.2]])
    hits = indexer.search_keyword("A-100", top_k=5)
    assert hits[0].chunk_id == "k1"
