import argparse
import base64
import logging

from mcp.server.fastmcp import FastMCP

from ragkb.config import get_settings
from ragkb.embedder import Embedder
from ragkb.indexers import QdrantIndexer, get_image_store, get_table_indexer
from ragkb.models import Chunk
from ragkb.reranker import Reranker
from ragkb.retriever import Retriever

logger = logging.getLogger(__name__)


class VersionStore:
    """版本历史：从表格/版本存储（PG 或 SQLite）的 document_versions 表读取。"""

    def __init__(self, pg):
        self._pg = pg

    def versions(self, doc_id: str) -> list[dict]:
        return self._pg.versions(doc_id)


class CurrentVersionFilter:
    """结果侧版本过滤：对每个 doc_id 只保留当前生效版本（PG document_versions 为准）。

    检索到的旧版本 chunk 一律剔除，保证只返回当前生效版本内容。
    """

    def __init__(self, pg):
        self._pg = pg

    def __call__(self, chunks: list[Chunk]) -> list[Chunk]:
        # 收集涉及的 doc_id
        doc_ids = {c.doc_id for c in chunks if c.doc_id}
        if not doc_ids:
            return chunks
        # 查询每个 doc 的当前生效版本（PG 已按 effective_date DESC 排序）
        current: dict[str, str] = {}
        for doc_id in doc_ids:
            versions = self._pg.versions(doc_id)
            if versions:
                current[doc_id] = versions[0]["version"]  # 生效日期最新者
        # 保留无版本记录（可能未走 ingest 版本登记）或匹配当前版本的 chunk
        return [c for c in chunks
                if c.doc_id not in current or c.version == current[c.doc_id]]


class _QueryRouter:
    """把 MCP 工具请求路由到检索组件（可注入替身供测试）。"""

    def __init__(self, retriever=None, pg=None, embedder=None,
                 indexer=None, version_store=None, minio=None, settings=None):
        self._settings = settings or get_settings()
        self._indexer = indexer or QdrantIndexer(self._settings)
        self._embedder = embedder or Embedder(
            self._settings.embed_model,
            cache_dir=self._settings.model_cache_dir)
        self._pg = pg if pg is not None else get_table_indexer(self._settings)
        # 当前生效版本过滤（结果侧，存储端 document_versions 为权威）：
        # 注入默认 Retriever（检索链路内过滤），search 返回前再兜底执行一次
        # （覆盖测试注入的自定义 retriever 场景，保证任何检索器结果都过版本闸门）
        self._version_filter = CurrentVersionFilter(self._pg)
        # 默认链路接真实重排器（懒加载，零启动成本）；测试注入 retriever 时保持原样
        self._retriever = retriever or Retriever(
            indexer=self._indexer,
            reranker=Reranker(self._settings.rerank_model,
                              cache_dir=self._settings.model_cache_dir),
            version_filter=self._version_filter,
        )
        self._version_store = version_store or VersionStore(self._pg)
        self._minio = minio if minio is not None else get_image_store(self._settings)

    def search(self, query: str, top_k: int = 5,
               version: str | None = None,
               department: str | None = None) -> dict:
        """双路召回 + RRF + 重排 + 版本过滤，返回带出处上下文。权限过滤为预留能力。

        version 指定版本号时只返回该版本的 chunk（历史版本查询，跳过
        当前生效版本闸门）；缺省时只返回当前生效版本（PG document_versions
        为权威）。重排分数低于 min_relevance_score 的结果判定为「没有找到」
        （防幻觉：检索到但相关性不足时不得强行作答）。
        """
        query_vec = self._embedder.embed([query])[0].tolist()
        if version:
            # 显式版本查询：过滤只保留目标版本，不查 PG 当前生效版本
            vfilter = lambda chunks: [c for c in chunks if c.version == version]
        else:
            vfilter = self._version_filter
        # 关键字传参：兼容测试替身（**kw）与真实 Retriever 签名；
        # version_filter 按调用覆盖 Retriever 构造注入的默认过滤
        scored = self._retriever.retrieve_scored(
            query=query, query_vec=query_vec,
            top_m=top_k, version_filter=vfilter)
        # 结果侧兜底执行版本闸门（覆盖测试注入的自定义 retriever 场景与
        # 未透传 version_filter 的检索器，保证任何检索器结果都过版本闸门）
        kept_ids = {c.chunk_id for c in vfilter([c for c, _ in scored])}
        scored = [(c, sc) for c, sc in scored if c.chunk_id in kept_ids]
        # 相关性阈值（仅当有重排分数时生效；无重排器时不做阈值判定）
        min_score = self._settings.min_relevance_score
        if scored and all(sc is not None for _, sc in scored):
            scored = [(c, sc) for c, sc in scored
                      if sc is not None and sc >= min_score]
        if not scored:
            return {"results": [], "empty_reason": "no_hits"}
        # 防幻觉硬保证（§6.1.1）：无来源的 chunk 一律不返回
        results = [
            {"chunk_id": c.chunk_id, "text": c.text, "source": c.source,
             "doc_type": c.doc_type, "version": c.version,
             "effective_date": c.effective_date, "score": sc}
            for c, sc in scored if c.source
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
        """取回原文块或图片原图。原文经 Qdrant payload 取回，图片经 MinIO 返回 base64。"""
        if image_id:
            data = self._minio.get(image_id)
            return {"image_id": image_id,
                    "data_base64": base64.b64encode(data).decode("ascii")}
        if chunk_id:
            chunk = self._indexer.fetch(chunk_id)
            if chunk:
                return {"chunk_id": chunk_id, "text": chunk.text,
                        "source": chunk.source}
            return {"chunk_id": chunk_id, "note": "未找到该 chunk"}
        return {"note": "需提供 chunk_id 或 image_id"}

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
    def search(query: str, top_k: int = 5, version: str | None = None) -> dict:
        """检索知识库，返回带出处标注的上下文。只返回真实检索结果。

        version 指定版本号时查该版本；缺省时只返回当前生效版本。
        """
        return router.search(query, top_k=top_k, version=version)

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
