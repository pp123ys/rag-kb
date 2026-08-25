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
                 must_not_versions=None):
        return [Chunk(chunk_id="c1", doc_id="d1", doc_type="pdf",
                      source="a.pdf:3", text="产品型号 A-100 单价 99 元。",
                      version="v2.0", effective_date="2026-06-01")]


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
                 must_not_versions=None):
        return [Chunk(chunk_id="c9", doc_id="d9", doc_type="pdf",
                      source="", text="无来源内容",
                      version="v1.0", effective_date="2026-01-01")]


def test_search_filters_chunks_without_source():
    server = build_server(retriever=_NoSourceRetriever(), pg=_FakePg(),
                          embedder=_FakeEmbedder())
    result = server._search("测试", top_k=3)
    assert result["results"] == []
    assert result["empty_reason"] == "no_hits"


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
                 must_not_versions=None):
        return [
            Chunk(chunk_id="old", doc_id="d1", doc_type="pdf", source="a.pdf",
                  text="旧版内容", version="v1.0", effective_date="2025-01-01"),
            Chunk(chunk_id="new", doc_id="d1", doc_type="pdf", source="a.pdf",
                  text="新版内容", version="v2.0", effective_date="2026-06-01"),
        ]


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
