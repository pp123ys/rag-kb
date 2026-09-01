#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ask.py — 用知识库做问答的 CLI 聊天工具（薄客户端，连现有 ragkb MCP 服务）。

原理：本脚本不加载任何模型、不碰向量库文件，只通过 MCP 协议调用正在运行的
ragkb 服务的 `search` / `ingest_document` 工具。服务端已加载 BGE-M3 + 重排模型，
会完成 双路召回 → RRF 融合 → 重排 → 版本过滤 → 相关性阈值 判定，客户端零负担。

前置：先启动服务（服务端模型已缓存，秒起）：
    python -m ragkb.mcp_server.server --transport http --port 8000

用法：
    python ask.py [--url http://127.0.0.1:8000/mcp] [--top 5]
    python ask.py --once "SCFI指数"       # 非交互，查询后直接退出（便于脚本调用）

交互命令：
    /top <N>        本次对话默认返回条数
    /show <N>       查看上一条结果中第 N 条的完整正文
    /ingest <路径>  直接入库一个文档（调服务端 ingest_document，可带 --version 等）
    /help           帮助
    /quit           退出（也可按 Ctrl+C 或 Ctrl+D）
"""

import argparse
import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

DEFAULT_URL = "http://127.0.0.1:8000/mcp"
SNIPPET_CHARS = 200  # 默认正文截断长度


# ---------- MCP 调用封装 ----------

def _parse_tool_result(res) -> dict:
    """把 MCP 工具返回值解析成 dict。

    优先读 structuredContent；服务端以 JSON 字符串返回时，则解析 content[0].text。
    """
    sc = getattr(res, "structuredContent", None)
    if sc is not None:
        return sc
    if getattr(res, "content", None):
        for block in res.content:
            text = getattr(block, "text", "")
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"raw_text": text}
    return {}


async def call_tool(session, name: str, args: dict) -> dict:
    res = await session.call_tool(name, args)
    return _parse_tool_result(res)


async def connect(url: str):
    """连接并返回已初始化（initialize）的 session。"""
    ctx = streamablehttp_client(url)
    read, write, _ = await ctx.__aenter__()
    session = ClientSession(read, write)
    await session.__aenter__()
    await session.initialize()
    return ctx, session


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " …（全文用 /show <N> 查看）"


def _format_result(idx: int, item: dict, full: bool = False) -> str:
    lines = []
    meta = []
    if item.get("source"):
        meta.append(f"来源: {item['source']}")
    if item.get("doc_type"):
        meta.append(f"类型: {item['doc_type']}")
    if item.get("version"):
        meta.append(f"版本: {item['version']}")
    if item.get("effective_date"):
        meta.append(f"生效: {item['effective_date']}")
    if item.get("score") is not None:
        meta.append(f"相关度: {item['score']:.3f}")
    lines.append(f"[{idx}] " + " | ".join(meta))
    text = item.get("text", "")
    lines.append("    " + (text if full else _snippet(text)).replace("\n", "\n    "))
    return "\n".join(lines)


def _print_results(payload: dict, top_n: int, full_one: int | None = None) -> list[dict]:
    """打印检索结果，返回结果 list 供 /show 使用。"""
    results = (payload or {}).get("results") or []
    if not results:
        print("（知识库中没有找到相关内容）")
        return []
    print(f"命中 {len(results)} 条：")
    for i, item in enumerate(results, 1):
        print(_format_result(i, item, full=(i == full_one)))
        print()
    return results


# ---------- 交互循环 ----------

async def run(url: str, top_n: int, once: str | None = None):
    try:
        ctx, session = await connect(url)
    except Exception as exc:
        print(f"[错误] 无法连接 MCP 服务 {url}: {exc}")
        print("请先启动服务：python -m ragkb.mcp_server.server --transport http --port 8000")
        return 2

    async def do_search(q: str, n: int) -> list[dict]:
        payload = await call_tool(session, "search", {"query": q, "top_k": n})
        if isinstance(payload, dict) and payload.get("raw_text"):
            print(f"[警告] 服务返回了非预期格式: {payload['raw_text'][:200]}")
            return []
        return _print_results(payload, n)

    async def do_ingest(newline):
        tokens = newline.split()[1:]
        path = tokens[0] if tokens else ""
        extra = {}
        i = 1
        while i < len(tokens):
            if tokens[i].startswith("--") and i + 1 < len(tokens):
                key = tokens[i][2:].replace("-", "_")
                if key in ("department", "version", "effective_date", "source"):
                    extra[key] = tokens[i + 1]
                    i += 2
                    continue
            i += 1
        if not path:
            print("用法: /ingest <文件路径> [--department 销售部] [--version v1.0] [--effective-date 2026-01-15]")
            return
        try:
            res = await call_tool(session, "ingest_document", {"path": path, **extra})
            if res:
                doc = res.get("doc_id", "")
                print(f"已入库文档 {doc!r}: chunks={res.get('chunks')}, tables={res.get('tables')}")
        except Exception as exc:
            print(f"[错误] 入库失败: {exc}")

    if once is not None:
        print(f"查询：{once}")
        await do_search(once, top_n)
        await session.__aexit__(None, None, None)
        await ctx.__aexit__(None, None, None)
        return 0

    print("=" * 60)
    print("知识库问答 CLI（薄客户端，连 ragkb MCP 服务）")
    print(f"服务: {url}    默认返回条数: {top_n}")
    print("直接输入问题提问；/help 查看命令；/quit 或 Ctrl+C 退出。")
    print("=" * 60)

    last_results: list[dict] = []
    cur_n = top_n
    while True:
        try:
            newline = await asyncio.get_event_loop().run_in_executor(None, input, "你> ")
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        newline = newline.strip()
        if not newline:
            continue
        if newline.lower() in ("/quit", "/exit", "exit", "quit"):
            print("再见")
            break
        if newline == "/help":
            print("/top <N>     设置返回条数")
            print("/show <N>    查看上一条结果中第 N 条的完整正文")
            print("/ingest <路径> [--department D] [--version V] [--effective-date D] 入库文档")
            print("/help         帮助")
            print("/quit         退出")
            continue
        if newline.startswith("/top"):
            parts = newline.split()
            if len(parts) == 2 and parts[1].isdigit():
                cur_n = int(parts[1])
                print(f"返回条数设为 {cur_n}")
            else:
                print("用法: /top <N>")
            continue
        if newline.startswith("/show"):
            parts = newline.split()
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if 1 <= idx <= len(last_results):
                    print(_format_result(idx, last_results[idx - 1], full=True))
                else:
                    print(f"当前只缓存 {len(last_results)} 条结果")
            else:
                print("用法: /show <N>")
            continue
        if newline.startswith("/ingest"):
            await do_ingest(newline)
            continue
        if newline.startswith("/"):
            print("未知命令（/help 查看帮助）")
            continue
        # 普通问题 → 检索
        try:
            last_results = await do_search(newline, cur_n)
        except Exception as exc:
            print(f"[错误] 搜索失败: {exc}")

    await session.__aexit__(None, None, None)
    await ctx.__aexit__(None, None, None)
    return 0


def main():
    ap = argparse.ArgumentParser(description="知识库问答 CLI（连 ragkb MCP 服务）")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"MCP 服务地址（默认 {DEFAULT_URL}）")
    ap.add_argument("--top", type=int, default=5, help="默认返回条数（默认 5）")
    ap.add_argument("--once", default=None, help="非交互模式：查询该问题后退出")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    # 非终端（管道/脚本）输入时强制 UTF-8，避免 Windows 下中文 stdin 乱码；
    # 真实终端键盘输入走控制台 API，不受影响。
    if not sys.stdin.isatty():
        try:
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        return asyncio.run(run(args.url, args.top, args.once))
    except KeyboardInterrupt:
        print("\n再见")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
