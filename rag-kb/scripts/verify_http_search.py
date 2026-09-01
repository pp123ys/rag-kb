"""HTTP 模式检索验证：连接运行中的 ragkb streamable-http 服务并调用 search。

用法: py -3.12 scripts\verify_http_search.py ["查询词" [top_k]]
默认查询 "A-100 单价多少"，search 调用带 300 秒超时（覆盖模型冷启动）。
"""
import asyncio
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

SEARCH_TIMEOUT = 300  # 秒


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "A-100 单价多少"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    url = os.environ.get("RAGKB_MCP_URL", "http://127.0.0.1:8000/mcp")

    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    print(f"连接: {url}  (search 超时上限 {SEARCH_TIMEOUT}s)")
    async with streamable_http_client(url) as (read, write, _sid):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=120)
            tools = await asyncio.wait_for(session.list_tools(), timeout=60)
            print("可用工具:", [t.name for t in tools.tools])
            r = await asyncio.wait_for(
                session.call_tool("search", {"query": query, "top_k": top_k}),
                timeout=SEARCH_TIMEOUT,
            )
            print(f"search({query!r}, top_k={top_k}):")
            for content in r.content:
                print(" ", content.text[:2000])


if __name__ == "__main__":
    asyncio.run(main())
