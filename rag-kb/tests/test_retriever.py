# tests/test_retriever.py
from ragkb.models import Chunk
from ragkb.retriever.retriever import Retriever


class _FakeIndexer:
    def __init__(self, dense=None, keyword=None, versions=None):
        self._dense = dense or []
        self._keyword = keyword or []
        self._versions = versions or {}

    def search_dense(self, query_vec, top_k, must_not_versions=None):
        return self._dense[:top_k]

    def search_keyword(self, query, top_k, must_not_versions=None):
        return self._keyword[:top_k]


def _chunk(cid, text, version="v1.0", effective_date="2026-01-01", doc_id="d1"):
    return Chunk(chunk_id=cid, doc_id=doc_id, doc_type="pdf", source="a.pdf",
                 text=text, version=version, effective_date=effective_date)


def test_retriever_merges_both_paths_and_dedupes():
    dense = [_chunk("c1", "向量命中"), _chunk("c2", "两路都中")]
    keyword = [_chunk("c2", "两路都中"), _chunk("c3", "关键词命中")]
    r = Retriever(indexer=_FakeIndexer(dense, keyword),
                  reranker=None, version_filter=lambda hits: hits)
    results = r.retrieve(query="型号 A-100", query_vec=[0.1, 0.2], top_m=5)
    ids = [c.chunk_id for c in results]
    assert ids == ["c2", "c1", "c3"]  # RRF 序


def test_retriever_applies_version_filter():
    dense = [_chunk("old", "旧版", version="v1.0", effective_date="2025-01-01"),
             _chunk("new", "新版", version="v2.0", effective_date="2026-06-01")]
    r = Retriever(indexer=_FakeIndexer(dense),
                  reranker=None,
                  version_filter=lambda hits: [h for h in hits if h.version == "v2.0"])
    results = r.retrieve(query="x", query_vec=[0.1, 0.2], top_m=5)
    assert [c.chunk_id for c in results] == ["new"]
