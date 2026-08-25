# Agent 接入 MCP 配置指南

> 本文档说明如何让 agent 通过 MCP 协议调用 RAG 知识库（ragkb）。
> 代码位置：`rag-kb/`；接入配置示例：`rag-kb/mcp-examples/`。

## 1. 前提

默认**免 Docker 嵌入式**运行（向量落 `data/qdrant`、表格/版本落 `data/ragkb.db`、图片落 `data/images`）：

1. `pip install -e ".[dev]"` 已执行（ragkb 可导入）
2. 模型已下载到 `rag-kb/models/`（约 5GB，BGE-M3 + bge-reranker）
3. 文档已入库：`py -3.12 -m ragkb.pipeline.ingest 文档.pdf --department 销售部 --version v1.0 --effective-date 2026-01-15`

> 远程模式（可选）：需要独立服务时 `cd rag-kb && docker compose up -d`
> （Qdrant / PostgreSQL / MinIO），并在 `.env` 设置切换变量（见 `.env.example`）。

> 模型下载：首次嵌入/重排会自动从 HF 拉取；国内网络建议设置
> `HF_ENDPOINT=https://hf-mirror.com`（已在 `rag-kb/.env.example` 备注）。
> 离线环境设置 `HF_HUB_OFFLINE=1` 可强制只用本地 `models/` 缓存。

## 2. 启动 MCP Server

### 2.1 stdio 模式（agent 作为子进程拉起，单 agent 场景）

```powershell
cd D:\text\rag\rag-kb
py -3.12 -m ragkb.mcp_server.server --transport stdio
```

不需要手动启动——配置好下面的 stdio 参数后，agent 会自动拉起。

### 2.2 HTTP 模式（独立服务，多 agent 共享）

```powershell
cd D:\text\rag\rag-kb
py -3.12 -m ragkb.mcp_server.server --transport http
```

监听 `0.0.0.0:8000`，MCP 端点为 `POST http://<host>:8000/mcp`
（streamable HTTP，需先 GET /mcp 建立会话再 POST 初始化，标准 MCP 客户端自动处理）。

### 2.3 两种都支持

```powershell
py -3.12 -m ragkb.mcp_server.server --transport both
```

> 生产建议用 HTTP：一个服务实例供所有 agent 共享，模型只加载一次。
> stdio 每个 agent 进程各拉一份模型（内存 ~2-3GB/实例）。

## 3. 工具清单（agent 可用能力）

| 工具 | 用途 | 关键入参 |
|------|------|----------|
| `ingest_document` | 入库文档（解析→清洗→切块→嵌入→入库） | `path`（必填，server 侧路径）、`source`、`department`、`version`、`effective_date`、`skip_embed` |
| `search` | 语义+关键词混合检索，返回带出处上下文 | `query`（必填）、`top_k`（默认5）、`version`（可选，查指定版本） |
| `retrieve_table` | 表格精确取数 / 按表头查表 | `table_id` 或 `query` |
| `get_document` | 取回原文块 / 图片原图 | `chunk_id` 或 `image_id` |
| `list_versions` | 文档版本历史 | `doc_id` |

**入库示例**（agent 拿到本地文件后调用）：

```json
{
  "path": "D:\\docs\\报价单.pdf",
  "department": "销售部",
  "version": "v1.0",
  "effective_date": "2026-01-15"
}
```

返回 `{"doc_id": "...", "chunks": 12, "tables": 2}`，随后即可用 `search` 检索该文档。
首次调用 `ingest_document` 会加载嵌入模型（约 15s，之后常驻）；`skip_embed=true` 可跳过
嵌入与向量入库（无模型下载环境的降级验证）。

**防幻觉契约**：`search` 返回的每条结果都带 `source`（文件名:页码/Sheet）
与 `score`（重排相关性，低于 0.1 的结果已按「没有找到」过滤，返回空 + `empty_reason`）。
agent 回答必须逐句引用 `source`，检索不到时回答「知识库中没有找到相关内容」。

**建议 agent 采用** `rag-kb/src/ragkb/prompts/agent_prompt.md` 作为系统提示词。

## 4. 各 Agent 接入配置

### 4.1 Claude Desktop / Claude Code（stdio）

配置文件（Windows）：`%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "ragkb": {
      "command": "py",
      "args": ["-3.12", "-m", "ragkb.mcp_server.server", "--transport", "stdio"],
      "env": {
        "HF_ENDPOINT": "https://hf-mirror.com",
        "HF_HOME": "D:\\text\\rag\\rag-kb\\models",
        "RAGKB_MODEL_CACHE_DIR": "D:\\text\\rag\\rag-kb\\models"
      }
    }
  }
}
```

> 注意：stdio 子进程**不继承** agent 进程的环境变量，模型相关配置必须在
> `env` 里显式给出。离线/内网环境务必加 `"HF_HUB_OFFLINE": "1"`，
> 否则模型加载时 sentence-transformers 会联网探测并超时重试（甚至卡住）。

> 注意：`cwd` 需为 `rag-kb/` 才能导入 ragkb。
> Claude Desktop 的 stdio 配置无法直接指定 cwd —— 若遇到 ModuleNotFoundError，
> 用下面的启动脚本方式（command 指向一个 .cmd 包装器）。

**推荐：包装器脚本方式**（解决 cwd 问题，Windows 可用）

`rag-kb/mcp-examples/ragkb-mcp.cmd`：

```bat
@echo off
cd /d D:\text\rag\rag-kb
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=D:\text\rag\rag-kb\models
set RAGKB_MODEL_CACHE_DIR=D:\text\rag\rag-kb\models
rem 离线/内网环境解除下行注释，强制只用本地模型缓存
rem set HF_HUB_OFFLINE=1
py -3.12 -m ragkb.mcp_server.server --transport stdio
```

配置改为：

```json
{
  "mcpServers": {
    "ragkb": {
      "command": "D:\\text\\rag\\rag-kb\\mcp-examples\\ragkb-mcp.cmd",
      "args": []
    }
  }
}
```

### 4.2 自研 Agent（Python，官方 MCP SDK）

依赖：`pip install mcp`

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # 注意：stdio_client 不继承父进程环境变量，HF/模型配置必须显式传入
    params = StdioServerParameters(
        command="py",
        args=["-3.12", "-m", "ragkb.mcp_server.server", "--transport", "stdio"],
        cwd=r"D:\text\rag\rag-kb",
        env={
            "HF_ENDPOINT": "https://hf-mirror.com",
            "HF_HOME": r"D:\text\rag\rag-kb\models",
            "RAGKB_MODEL_CACHE_DIR": r"D:\text\rag\rag-kb\models",
            # "HF_HUB_OFFLINE": "1",  # 离线/内网环境解除注释
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # 入库（agent 拿到本地文件后）
            r = await session.call_tool("ingest_document", {
                "path": r"D:\docs\报价单.pdf",
                "department": "销售部", "version": "v1.0",
                "effective_date": "2026-01-15",
            })
            print(r.content[0].text)
            # 检索
            r = await session.call_tool("search", {"query": "A-100 单价多少", "top_k": 3})
            for content in r.content:
                print(content.text)


asyncio.run(main())
```

### 4.3 自研 Agent（HTTP，多实例共享）

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    url = "http://localhost:8000/mcp"
    async with streamablehttp_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("可用工具:", [t.name for t in tools.tools])
            r = await session.call_tool("search", {"query": "保修期多久"})
            for content in r.content:
                print(content.text)


asyncio.run(main())
```

### 4.4 其他常见 Agent（JSON 配置格式）

大多数 MCP 客户端（Cursor、Windsurf、VS Code Copilot 等）接受统一 JSON 格式：

```json
{
  "mcpServers": {
    "ragkb": {
      "command": "D:\\text\\rag\\rag-kb\\mcp-examples\\ragkb-mcp.cmd",
      "args": [],
      "env": {
        "HF_ENDPOINT": "https://hf-mirror.com",
        "HF_HOME": "D:\\text\\rag\\rag-kb\\models",
        "RAGKB_MODEL_CACHE_DIR": "D:\\text\\rag\\rag-kb\\models"
      }
    }
  }
}
```

> 与 stdio 同理：以上 `env` 必须显式给出（子进程不继承父进程环境）；离线/内网
> 环境加 `"HF_HUB_OFFLINE": "1"`。包装器 `ragkb-mcp.cmd` 已内置这些变量。

HTTP 版（若客户端支持远程 MCP）：

```json
{
  "mcpServers": {
    "ragkb": {
      "url": "http://localhost:8000/mcp",
      "transport": "streamable-http"
    }
  }
}
```

> HTTP 模式下 server 由你自己启动（`py -3.12 -m ragkb.mcp_server.server --transport http`），
> 环境变量在你启动的 shell 里设置即可，无需在 JSON 里重复。

## 5. 快速验证接入是否成功

```powershell
cd D:\text\rag\rag-kb
py -3.12 scripts\verify_mcp_connection.py
```

期望输出：

```
连接方式: stdio
可用工具: ['search', 'retrieve_table', 'get_document', 'list_versions', 'ingest_document']
search('A-100 单价多少'):
  [demo.pdf] v1.0 (score 0.99): A-100 型号的单价为 99 元，保修期一年。
```

## 6. 常见问题

| 问题 | 解决 |
|------|------|
| `ModuleNotFoundError: ragkb` | cwd 不在 rag-kb/；用包装器 .cmd 脚本或显式 cwd |
| 首次调用慢（30s+） | 正常：模型加载到内存；后续调用快 |
| 内存占用 2-3GB | BGE-M3 + reranker 常驻；多 agent 用 HTTP 共享实例 |
| `HF_ENDPOINT` 未设置导致模型下载失败 | 网络受限时设 `https://hf-mirror.com` |
| 检索结果为空 | 确认文档已入库（含嵌入，勿用 `--skip-embed`）；或问题确实超出知识库（防幻觉阈值过滤） |
| `Empty results` + `empty_reason` | 知识库无相关内容或重排分数低于阈值 0.1，属正常防幻觉行为 |
