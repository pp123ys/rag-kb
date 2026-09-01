import asyncio, json, sys
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def search(session, q, n=3):
    res = await session.call_tool("search", {"query": q, "top_k": n})
    if getattr(res, "structuredContent", None):
        return res.structuredContent
    return {"raw": [c.text for c in res.content]}

async def main():
    url = "http://127.0.0.1:8000/mcp"
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for q in ["汇率", "BDI指数", "SCFI", "美元兑人民币"]:
                sc = await search(session, q, 3)
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                print("### QUERY:", q)
                print(json.dumps(sc, ensure_ascii=False, indent=2)[:900])
                print()

asyncio.run(main())
