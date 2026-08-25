import numpy as np
import pytest

from ragkb.mcp_server.server import build_server
from ragkb.models import Chunk


class _FakeEmbedder:
    """确定性向量替身：避免真实 Embedder 触发 bge-m3 模型下载。

    与 test_embedder.py 的 _FakeEncoder 同款模式；search 工具仍完整走
    embed → retrieve → 格式化 全链路，只是向量是固定的零向量。
    """

    def embed(self, texts):
        return np.zeros((len(texts), 4), dtype=np.float32)


class _FakeRetriever:
    def retrieve(self, query, query_vec, top_k=50, top_n=20, top_m=5,
                 must_not_versions=None, **kwargs):
        return [Chunk(chunk_id="c1", doc_id="d1", doc_type="pdf",
                      source="a.pdf:3", text="产品型号 A-100 单价 99 元。",
                      version="v2.0", effective_date="2026-06-01")]

    def retrieve_scored(self, query, query_vec, top_k=50, top_n=20, top_m=5,
                        must_not_versions=None, **kwargs):
        return [(c, 0.9) for c in self.retrieve(
            query, query_vec, top_k=top_k, top_n=top_n, top_m=top_m,
            must_not_versions=must_not_versions)]


class _FakePg:
    def query(self, table_id):
        return [{"型号": "A-100", "单价": "99"}]

    def search_headers(self, keyword):
        return [{"table_id": "t1", "name": "报价表", "source": "a.xlsx:报价"}]

    def versions(self, doc_id):
        # 默认无版本登记：结果侧版本过滤对无记录 doc 放行
        return []


def test_search_tool_returns_sourced_context():
    server = build_server(retriever=_FakeRetriever(), pg=_FakePg(),
                          embedder=_FakeEmbedder())
    result = server._search("A-100 单价多少", top_k=3)
    # 每个 chunk 必须带 source（防幻觉硬性保证）
    assert result["results"][0]["source"] == "a.pdf:3"
    assert "A-100" in result["results"][0]["text"]
    assert "empty" not in result


def test_search_tool_empty_result_reports_reason():
    class EmptyRetriever:
        def retrieve(self, **kw):
            return []

        def retrieve_scored(self, **kw):
            return []

    server = build_server(retriever=EmptyRetriever(), pg=_FakePg(),
                          embedder=_FakeEmbedder())
    result = server._search("不存在的问题", top_k=3)
    assert result["results"] == []
    assert result["empty_reason"] == "no_hits"


def test_retrieve_table_tool_exact_query():
    server = build_server(retriever=_FakeRetriever(), pg=_FakePg())
    rows = server._retrieve_table(table_id="t1")
    assert rows["rows"][0]["型号"] == "A-100"


def test_list_versions_returns_history():
    class FakeVersionStore:
        def versions(self, doc_id):
            return [{"version": "v1.0", "effective_date": "2025-01-01"},
                    {"version": "v2.0", "effective_date": "2026-06-01"}]

    server = build_server(retriever=_FakeRetriever(), pg=_FakePg(),
                          version_store=FakeVersionStore())
    out = server._list_versions("d1")
    assert len(out["versions"]) == 2


class _FakeMinio:
    def get(self, image_id):
        return b"\x89PNG raw"


def test_get_document_returns_base64_image():
    import base64
    server = build_server(retriever=_FakeRetriever(), pg=_FakePg(),
                          minio=_FakeMinio())
    out = server._get_document(image_id="i1")
    assert out["image_id"] == "i1"
    assert base64.b64decode(out["data_base64"]) == b"\x89PNG raw"


def test_retrieve_table_header_search():
    server = build_server(retriever=_FakeRetriever(), pg=_FakePg())
    out = server._retrieve_table(query="型号")
    assert out["tables"][0]["table_id"] == "t1"


class _NoSourceRetriever:
    def retrieve(self, query, query_vec, top_k=50, top_n=20, top_m=5,
                 must_not_versions=None, **kwargs):
        return [Chunk(chunk_id="c9", doc_id="d9", doc_type="pdf",
                      source="", text="无来源内容",
                      version="v1.0", effective_date="2026-01-01")]

    def retrieve_scored(self, query, query_vec, top_k=50, top_n=20, top_m=5,
                        must_not_versions=None, **kwargs):
        return [(c, 0.9) for c in self.retrieve(
            query, query_vec, top_k=top_k, top_n=top_n, top_m=top_m,
            must_not_versions=must_not_versions)]


def test_search_filters_chunks_without_source():
    server = build_server(retriever=_NoSourceRetriever(), pg=_FakePg(),
                          embedder=_FakeEmbedder())
    result = server._search("测试", top_k=3)
    assert result["results"] == []
    assert result["empty_reason"] == "no_hits"


def test_search_below_relevance_threshold_reports_no_hits():
    """防幻觉：重排分数低于 min_relevance_score 的结果判定为「没有找到」。"""

    class _LowScoreRetriever:
        def retrieve_scored(self, query, query_vec, top_k=50, top_n=20,
                            top_m=5, must_not_versions=None, **kwargs):
            return [(Chunk(chunk_id="c1", doc_id="d1", doc_type="pdf",
                           source="a.pdf:3", text="低相关内容",
                           version="v1.0", effective_date="2026-01-01"), 0.02)]

    server = build_server(retriever=_LowScoreRetriever(), pg=_FakePg(),
                          embedder=_FakeEmbedder())
    result = server._search("完全不相关的问题", top_k=3)
    assert result["results"] == []
    assert result["empty_reason"] == "no_hits"


def test_search_above_threshold_returns_score():
    server = build_server(retriever=_FakeRetriever(), pg=_FakePg(),
                          embedder=_FakeEmbedder())
    result = server._search("A-100 单价多少", top_k=3)
    assert result["results"][0]["score"] == 0.9


class _VersionPg:
    def __init__(self, versions_map):
        self._m = versions_map

    def versions(self, doc_id):
        return self._m.get(doc_id, [])


def test_current_version_filter_keeps_only_current():
    from ragkb.mcp_server.server import CurrentVersionFilter

    f = CurrentVersionFilter(_VersionPg({
        "d1": [{"version": "v2.0", "effective_date": "2026-06-01"},
               {"version": "v1.0", "effective_date": "2025-01-01"}]}))
    chunks = [
        Chunk(chunk_id="old", doc_id="d1", doc_type="pdf", source="a.pdf",
              text="旧版", version="v1.0", effective_date="2025-01-01"),
        Chunk(chunk_id="new", doc_id="d1", doc_type="pdf", source="a.pdf",
              text="新版", version="v2.0", effective_date="2026-06-01"),
        Chunk(chunk_id="noreg", doc_id="d2", doc_type="pdf", source="b.pdf",
              text="无登记", version="v1.0", effective_date="2026-01-01"),
    ]
    out = f(chunks)
    assert {c.chunk_id for c in out} == {"new", "noreg"}


def test_current_version_filter_passes_chunks_without_version():
    from ragkb.mcp_server.server import CurrentVersionFilter

    f = CurrentVersionFilter(_VersionPg({}))
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", doc_type="pdf", source="a.pdf",
              text="无版本字段", version="", effective_date=""),
    ]
    assert f(chunks) == chunks


class _TwoVersionRetriever:
    def retrieve(self, query, query_vec, top_k=50, top_n=20, top_m=5,
                 must_not_versions=None, **kwargs):
        return [
            Chunk(chunk_id="old", doc_id="d1", doc_type="pdf", source="a.pdf",
                  text="旧版内容", version="v1.0", effective_date="2025-01-01"),
            Chunk(chunk_id="new", doc_id="d1", doc_type="pdf", source="a.pdf",
                  text="新版内容", version="v2.0", effective_date="2026-06-01"),
        ]

    def retrieve_scored(self, query, query_vec, top_k=50, top_n=20, top_m=5,
                        must_not_versions=None, **kwargs):
        return [(c, 0.9) for c in self.retrieve(
            query, query_vec, top_k=top_k, top_n=top_n, top_m=top_m,
            must_not_versions=must_not_versions)]


class _VersionedPg(_FakePg):
    def versions(self, doc_id):
        return [{"version": "v2.0", "effective_date": "2026-06-01"},
                {"version": "v1.0", "effective_date": "2025-01-01"}]


def test_search_filters_old_versions():
    server = build_server(retriever=_TwoVersionRetriever(), pg=_VersionedPg(),
                          embedder=_FakeEmbedder())
    result = server._search("测试", top_k=3)
    ids = [r["chunk_id"] for r in result["results"]]
    assert "old" not in ids
    assert "new" in ids


def test_search_tool_exposes_version_param():
    """MCP search 工具暴露 version 入参；department 为预留能力不暴露。"""
    server = build_server(retriever=_FakeRetriever(), pg=_FakePg(),
                          embedder=_FakeEmbedder())
    tool = server._tool_manager.get_tool("search")
    assert "version" in tool.parameters["properties"]
    assert "department" not in tool.parameters["properties"]


def test_search_with_version_returns_only_that_version():
    """显式版本查询：即使 PG 当前生效版本是 v2.0，version="v1.0" 也只返回 v1.0。"""
    server = build_server(retriever=_TwoVersionRetriever(), pg=_VersionedPg(),
                          embedder=_FakeEmbedder())
    result = server._search("测试", top_k=3, version="v1.0")
    ids = [r["chunk_id"] for r in result["results"]]
    assert ids == ["old"]
    assert all(r["version"] == "v1.0" for r in result["results"])


class _FakeFetchIndexer:
    """get_document 用的替身：fetch(chunk_id) 返回 Chunk 或 None。"""

    def __init__(self, chunk=None):
        self._chunk = chunk
        self.calls = []

    def fetch(self, chunk_id):
        self.calls.append(chunk_id)
        return self._chunk


def test_get_document_returns_chunk_content():
    chunk = Chunk(chunk_id="c1", doc_id="d1", doc_type="pdf", source="a.pdf:3",
                  text="产品型号 A-100 单价 99 元。", version="v2.0",
                  effective_date="2026-06-01")
    indexer = _FakeFetchIndexer(chunk)
    server = build_server(retriever=_FakeRetriever(), pg=_FakePg(),
                          embedder=_FakeEmbedder(), indexer=indexer)
    out = server._get_document(chunk_id="c1")
    assert indexer.calls == ["c1"]
    assert out["text"] == "产品型号 A-100 单价 99 元。"
    assert out["source"] == "a.pdf:3"


def test_get_document_missing_chunk_returns_note():
    indexer = _FakeFetchIndexer(None)
    server = build_server(retriever=_FakeRetriever(), pg=_FakePg(),
                          embedder=_FakeEmbedder(), indexer=indexer)
    out = server._get_document(chunk_id="nope")
    assert out["chunk_id"] == "nope"
    assert "note" in out


def test_get_document_requires_chunk_or_image():
    server = build_server(retriever=_FakeRetriever(), pg=_FakePg(),
                          embedder=_FakeEmbedder())
    out = server._get_document()
    assert "note" in out


class _FakeIngestPipeline:
    """入库流水线替身：记录入参并返回固定统计，避免触发真实 Embedder 模型加载。"""

    def __init__(self):
        self.calls = []

    def ingest(self, path, doc_id=None, source=None, department="", version="",
               effective_date="", parse_images=True, skip_embed=False):
        self.calls.append({
            "path": path, "doc_id": doc_id, "source": source,
            "department": department, "version": version,
            "effective_date": effective_date, "skip_embed": skip_embed,
        })
        return {"doc_id": doc_id or "gen-1", "chunks": 3, "tables": 1}


def test_ingest_document_routes_to_pipeline():
    pipe = _FakeIngestPipeline()
    server = build_server(retriever=_FakeRetriever(), pg=_FakePg(),
                          embedder=_FakeEmbedder(), ingest_pipeline=pipe)
    out = server._ingest("d:/docs/报价单.pdf", source="报价单.pdf",
                         department="销售部", version="v1.0",
                         effective_date="2026-01-15")
    assert pipe.calls == [{
        "path": "d:/docs/报价单.pdf", "doc_id": None, "source": "报价单.pdf",
        "department": "销售部", "version": "v1.0",
        "effective_date": "2026-01-15", "skip_embed": False,
    }]
    assert out == {"doc_id": "gen-1", "chunks": 3, "tables": 1}


def test_ingest_document_skip_embed_and_defaults():
    pipe = _FakeIngestPipeline()
    server = build_server(retriever=_FakeRetriever(), pg=_FakePg(),
                          embedder=_FakeEmbedder(), ingest_pipeline=pipe)
    out = server._ingest("a.pdf", skip_embed=True)
    assert pipe.calls[0]["skip_embed"] is True
    assert pipe.calls[0]["source"] is None  # 缺省 source 由 pipeline 取文件名
    assert out["doc_id"] == "gen-1"


def test_ingest_document_tool_registered():
    server = build_server(retriever=_FakeRetriever(), pg=_FakePg(),
                          embedder=_FakeEmbedder())
    tool = server._tool_manager.get_tool("ingest_document")
    assert tool is not None
    props = tool.parameters["properties"]
    assert "path" in props and "skip_embed" in props
