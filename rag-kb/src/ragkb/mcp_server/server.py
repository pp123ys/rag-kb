import argparse
import base64
import logging

from mcp.server.fastmcp import FastMCP

from ragkb.config import get_settings
from ragkb.embedder import Embedder
from ragkb.indexers import QdrantIndexer
from ragkb.indexers.minio_store import MinioImageStore
from ragkb.indexers.pg_table_indexer import PgTableIndexer
from ragkb.reranker import Reranker
from ragkb.retriever import Retriever

logger = logging.getLogger(__name__)


class VersionStore:
    """版本历史：从 PostgreSQL document_versions 表读取。"""

    def __init__(self, pg):
        self._pg = pg

    def versions(self, doc_id: str) -> list[dict]:
        return self._pg.versions(doc_id)


class _QueryRouter:
    """把 MCP 工具请求路由到检索组件（可注入替身供测试）。"""

    def __init__(self, retriever=None, pg=None, embedder=None,
                 indexer=None, version_store=None, minio=None, settings=None):
        self._settings = settings or get_settings()
        self._indexer = indexer or QdrantIndexer(self._settings)
        self._embedder = embedder or Embedder(self._settings.embed_model)
        self._pg = pg or PgTableIndexer(self._settings.pg_dsn)
        # 默认链路接真实重排器（懒加载，零启动成本）；测试注入 retriever 时保持原样
        self._retriever = retriever or Retriever(
            indexer=self._indexer, reranker=Reranker(self._settings.rerank_model))
        self._version_store = version_store or VersionStore(self._pg)
        self._minio = minio or MinioImageStore(self._settings)

    def search(self, query: str, top_k: int = 5,
               version: str | None = None,
               department: str | None = None) -> dict:
        """双路召回 + RRF + 重排，返回带出处上下文。版本过滤与权限过滤为预留能力。"""
        query_vec = self._embedder.embed([query])[0].tolist()
        # 关键字传参：兼容测试替身（**kw）与真实 Retriever 签名
        chunks = self._retriever.retrieve(query=query, query_vec=query_vec,
                                          top_m=top_k)
        if not chunks:
            return {"results": [], "empty_reason": "no_hits"}
        # 防幻觉硬保证（§6.1.1）：无来源的 chunk 一律不返回
        results = [
            {"chunk_id": c.chunk_id, "text": c.text, "source": c.source,
             "doc_type": c.doc_type, "version": c.version,
             "effective_date": c.effective_date}
            for c in chunks if c.source
        ]
        if not results:
            return {"results": [], "empty_reason": "no_hits"}
        return {"results": results}

    def retrieve_table(self, table_id: str = "",
                       query: str | None = None,
                       columns: list[str] | None = None) -> dict:
        if table_id:
            rows = self._pg.query(table_id)
            return {"rows": rows or [], "columns": columns or []}
        if query:
            found = self._pg.search_headers(query)
            return {"tables": found}
        return {"rows": [], "tables": []}

    def get_document(self, chunk_id: str = "", image_id: str = "") -> dict:
        """取回原文块或图片原图。图片经 MinIO 取回，返回 base64 数据。"""
        if image_id:
            data = self._minio.get(image_id)
            return {"image_id": image_id,
                    "data_base64": base64.b64encode(data).decode("ascii")}
        return {"chunk_id": chunk_id, "note": "原文块经 search 结果的 source 定位"}

    def list_versions(self, doc_id: str) -> dict:
        return {"versions": self._version_store.versions(doc_id)}


def build_server(retriever=None, pg=None, embedder=None, indexer=None,
                 version_store=None, minio=None, settings=None):
    """构造 FastMCP 服务，注册四个工具。"""
    router = _QueryRouter(retriever=retriever, pg=pg, embedder=embedder,
                          indexer=indexer, version_store=version_store,
                          minio=minio, settings=settings)

    mcp = FastMCP("ragkb")

    @mcp.tool()
    def search(query: str, top_k: int = 5) -> dict:
        """检索知识库，返回带出处标注的上下文。只返回真实检索结果。"""
        return router.search(query, top_k=top_k)

    @mcp.tool()
    def retrieve_table(table_id: str = "", query: str | None = None) -> dict:
        """按 table_id 精确取表格数据，或按列名/表头模糊查表。"""
        return router.retrieve_table(table_id=table_id, query=query)

    @mcp.tool()
    def get_document(chunk_id: str = "", image_id: str = "") -> dict:
        """取回原文块或图片原图。"""
        return router.get_document(chunk_id=chunk_id, image_id=image_id)

    @mcp.tool()
    def list_versions(doc_id: str) -> dict:
        """查询文档版本历史。"""
        return router.list_versions(doc_id)

    # 供测试直接调用工具逻辑
    mcp._search = router.search
    mcp._retrieve_table = router.retrieve_table
    mcp._get_document = router.get_document
    mcp._list_versions = router.list_versions
    return mcp


# FastMCP.run() 接受的 transport 取值（mcp>=1.0）：stdio | sse | streamable-http。
# CLI 侧保持 plan 规定的 stdio|http|both 语义，内部映射为 streamable-http。
_TRANSPORT = {"stdio": "stdio", "http": "streamable-http"}


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="RAG 知识库 MCP Server")
    ap.add_argument("--transport", default="stdio",
                    choices=["stdio", "http", "both"])
    args = ap.parse_args()

    mcp = build_server()

    if args.transport == "both":
        # both：分别拉起两个实例（stdio 前台 + http 后台）
        import threading
        http = build_server()
        threading.Thread(target=http.run,
                         kwargs={"transport": _TRANSPORT["http"]},
                         daemon=True).start()
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=_TRANSPORT[args.transport])


if __name__ == "__main__":
    main()
