"""MCP 接入快速验证：stdio 方式连接 ragkb 并调用 search。

用法: py -3.12 scripts\verify_mcp_connection.py
"""
import asyncio
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")


async def main():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command="py",
        args=["-3.12", "-m", "ragkb.mcp_server.server", "--transport", "stdio"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env={
            "HF_ENDPOINT": "https://hf-mirror.com",
            "HF_HOME": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
            "RAGKB_MODEL_CACHE_DIR": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
            # "HF_HUB_OFFLINE": "1",  # 离线/内网环境解除注释
        },
    )
    print("连接方式: stdio")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=120)
            tools = await asyncio.wait_for(session.list_tools(), timeout=60)
            print("可用工具:", [t.name for t in tools.tools])
            # 首次调用需加载 BGE-M3（嵌入）与 bge-reranker（重排），冷启动可能 1-3 分钟
            r = await asyncio.wait_for(
                session.call_tool("search", {"query": "A-100 单价多少", "top_k": 2}),
                timeout=300,
            )
            print("search('A-100 单价多少'):")
            for content in r.content:
                print(" ", content.text[:200])


if __name__ == "__main__":
    asyncio.run(main())
