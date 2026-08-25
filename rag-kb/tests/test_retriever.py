# tests/test_retriever.py
from ragkb.models import Chunk
from ragkb.retriever.retriever import Retriever


class _FakeIndexer:
    def __init__(self, dense=None, keyword=None, versions=None):
        self._dense = dense or []
        self._keyword = keyword or []
        self._versions = versions or {}
        self.calls = []  # 记录 (method, must_not_versions)

    def search_dense(self, query_vec, top_k, must_not_versions=None):
        self.calls.append(("dense", must_not_versions))
        return self._dense[:top_k]

    def search_keyword(self, query, top_k, must_not_versions=None):
        self.calls.append(("keyword", must_not_versions))
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


def test_retriever_passes_must_not_versions_to_both_paths():
    dense = [_chunk("c1", "x")]
    keyword = [_chunk("c1", "x")]
    idx = _FakeIndexer(dense, keyword)
    r = Retriever(indexer=idx, version_filter=lambda hits: hits)
    r.retrieve(query="q", query_vec=[0.1], must_not_versions={"d1": "v1.0"})
    assert idx.calls == [("dense", {"d1": "v1.0"}),
                         ("keyword", {"d1": "v1.0"})]


class _ReverseReranker:
    """假重排：把输入倒序（验证 rerank 输出生效）。"""
    def rerank(self, query, chunks):
        return list(reversed(chunks))

    def rerank_scored(self, query, chunks):
        return [(c, float(len(chunks) - i)) for i, c in enumerate(reversed(chunks))]


def test_retriever_reranker_output_takes_effect():
    dense = [_chunk("c1", "一"), _chunk("c2", "二"), _chunk("c3", "三")]
    r = Retriever(indexer=_FakeIndexer(dense),
                  reranker=_ReverseReranker(),
                  version_filter=lambda hits: hits)
    results = r.retrieve(query="q", query_vec=[0.1], top_m=5)
    assert [c.chunk_id for c in results] == ["c3", "c2", "c1"]


def test_retriever_truncates_top_n_and_top_m():
    dense = [_chunk(f"c{i}", f"内容{i}") for i in range(1, 7)]
    r = Retriever(indexer=_FakeIndexer(dense),
                  version_filter=lambda hits: hits)
    # top_n=3 截断融合结果，top_m=2 截断最终输出
    results = r.retrieve(query="q", query_vec=[0.1], top_n=3, top_m=2)
    assert len(results) == 2


class _NoEmptyReranker:
    """拒绝空输入的重排器：验证空结果短路（不调用 rerank）。"""
    def rerank(self, query, chunks):
        if not chunks:
            raise AssertionError("rerank 不应收到空列表")
        return chunks

    def rerank_scored(self, query, chunks):
        if not chunks:
            raise AssertionError("rerank 不应收到空列表")
        return [(c, 0.5) for c in chunks]


def test_retriever_skips_rerank_on_empty_fused():
    r = Retriever(indexer=_FakeIndexer(),  # 两路都无命中
                  reranker=_NoEmptyReranker(),
                  version_filter=lambda hits: hits)
    results = r.retrieve(query="q", query_vec=[0.1], top_m=5)
    assert results == []


class _DenseBoomIndexer:
    """向量路必炸、关键词路正常：验证单路失败降级（§8）。"""

    def __init__(self, keyword):
        self._keyword = keyword

    def search_dense(self, query_vec, top_k, must_not_versions=None):
        raise RuntimeError("向量路故障")

    def search_keyword(self, query, top_k, must_not_versions=None):
        return self._keyword[:top_k]


def test_retriever_degrades_when_dense_path_fails(caplog):
    keyword = [_chunk("c1", "关键词命中")]
    r = Retriever(indexer=_DenseBoomIndexer(keyword),
                  reranker=None, version_filter=lambda hits: hits)
    results = r.retrieve(query="q", query_vec=[0.1], top_m=5)
    assert [c.chunk_id for c in results] == ["c1"]
    assert any("向量路检索失败" in rec.message for rec in caplog.records)


class _KeywordBoomIndexer(_DenseBoomIndexer):
    def search_dense(self, query_vec, top_k, must_not_versions=None):
        return self._keyword[:top_k]

    def search_keyword(self, query, top_k, must_not_versions=None):
        raise RuntimeError("关键词路故障")


def test_retriever_degrades_when_keyword_path_fails(caplog):
    dense = [_chunk("c1", "向量命中")]
    r = Retriever(indexer=_KeywordBoomIndexer(dense),
                  reranker=None, version_filter=lambda hits: hits)
    results = r.retrieve(query="q", query_vec=[0.1], top_m=5)
    assert [c.chunk_id for c in results] == ["c1"]
    assert any("关键词路检索失败" in rec.message for rec in caplog.records)


def test_retriever_degrades_to_empty_when_both_paths_fail():
    class BothBoomIndexer(_DenseBoomIndexer):
        def search_keyword(self, query, top_k, must_not_versions=None):
            raise RuntimeError("关键词路故障")

    r = Retriever(indexer=BothBoomIndexer([]),
                  reranker=_NoEmptyReranker(),
                  version_filter=lambda hits: hits)
    # 两路都失败 → 空结果，且不触发 rerank（短路）
    assert r.retrieve(query="q", query_vec=[0.1], top_m=5) == []


def test_retriever_per_call_version_filter_overrides_constructor():
    """调用级 version_filter 优先于构造注入的过滤（显式版本查询路径）。"""
    dense = [_chunk("old", "旧版", version="v1.0", effective_date="2025-01-01"),
             _chunk("new", "新版", version="v2.0", effective_date="2026-06-01")]
    r = Retriever(
        indexer=_FakeIndexer(dense),
        reranker=None,
        version_filter=lambda hits: [h for h in hits if h.version == "v2.0"],
    )
    results = r.retrieve(
        query="x", query_vec=[0.1], top_m=5,
        version_filter=lambda hits: [h for h in hits if h.version == "v1.0"])
    assert [c.chunk_id for c in results] == ["old"]
