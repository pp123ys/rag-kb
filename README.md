# rag-kb — 企业 RAG 知识库（MCP 检索服务）

面向 agent 的企业知识库检索增强系统。输入为 PDF / Excel / 邮件等多源异构文档，输出为**带出处标注**的上下文，供 agent 在其工作流中引用；agent 通过 **MCP（Model Context Protocol）协议**调用，回答的生成由 agent 自身的 LLM 完成。

源码位于 [`rag-kb/`](rag-kb/) 子目录，查看详细设计与使用说明请阅读 [rag-kb/README.md](rag-kb/README.md)。

## 核心能力

- **离线索引流水线**：解析（PDF / Excel / CSV / 邮件 / Markdown，含 PaddleOCR 扫描件识别）→ 清洗 → 语义切块（不截断句子，带 overlap）→ BGE-M3 嵌入 → 双路入库
- **在线检索链路**：BGE-M3 稠密向量 + jieba 分词 BM25 稀疏向量双路召回（Qdrant 单存储）→ RRF（k=60）融合 → bge-reranker-v2-m3 重排 → 生效日期版本闸门 → 带出处上下文
- **表格双通道**：表格转 Markdown 参与语义检索 + 结构化行列数据进表格索引（`retrieve_table` 精确取数）
- **版本管理**：默认只检索当前生效版本，支持显式版本号查询历史
- **防幻觉约定**：检索结果强制携带 `source` 出处；相关性不足返回 `empty_reason`；附送 agent 侧系统提示词模板（只依据检索内容回答、逐句标注出处）

## MCP 工具（6 个）

| 工具 | 用途 |
|------|------|
| `ingest_document` | 入库文档（pdf / xlsx / eml / msg / md / csv） |
| `search` | 双路召回 + RRF + 重排，返回带 source 的上下文 |
| `retrieve_table` | 表格精确取数 / 按表头查表 |
| `get_document` | 取回原文块 / 图片原图 |
| `list_versions` | 文档版本历史 |
| `delete_document` | 删除文档（向量 + 表格/版本 + 原图，幂等） |

## 新手快速开始（Windows，零基础版）

> 下面说明按「电脑上什么都没装」的情况写，跟着做就能跑起来。
> 完整设计与使用说明见 [rag-kb/README.md](rag-kb/README.md)。

### 需要安装的（就 2 样）

1. **Python 3.12** — 从 [python.org](https://www.python.org/downloads/) 下载 64 位安装包。
   ⚠️ 安装时**一定勾选 "Add Python to PATH"**（新手最容易漏的一步，漏了后面命令都跑不了）。
2. **Git** — 用于克隆/更新代码（运行项目本身用不到，有就跳过，没有也别慌）。

**不需要装**：Docker（默认嵌入式模式零外部服务）、GPU / CUDA（CPU 即可，较慢）、模型（首次入库时自动下载，约 2GB）。

### 操作步骤

```powershell
cd D:\text\rag\rag-kb

# 1. 创建虚拟环境（隔离依赖，建议新手也做）
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# 若报"禁止运行脚本"错误，先执行一次：
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2. 安装项目依赖（会下载一批库，PaddleOCR/paddle 较大，需几分钟）
pip install -e ".[dev]"

# 3. 首次入库一个文档（首次会自动下载 ~2GB BGE-M3 模型，请耐心等待）
python -m ragkb.pipeline.ingest 你的文档.pdf --department 销售部 --version v1.0 --effective-date 2026-01-15
# 不想等模型下载，可加 --skip-embed 跳过嵌入、仅验证解析链路

# 4. 启动 MCP 服务
python -m ragkb.mcp_server.server --transport http
# 成功后监听 http://127.0.0.1:8000/mcp
```

### 常见坑

- **PowerShell 提示"无法加载脚本"** → 执行上面的 `Set-ExecutionPolicy`，或跳过 `.venv` 直接 `pip install`（不推荐但能跑）。
- **`pip install` 慢/卡在 paddle** → 该包体积大属正常；可加清华镜像：`pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple`。
- **模型下载慢/失败** → 加 `--skip-embed` 跳过；已下载后可设 `HF_HUB_OFFLINE=1` 强制离线加载。
- **项目没有网页界面** → 它是 MCP 服务而非网站，需要接一个 MCP 客户端（如 Claude Desktop / 支持 MCP 的 agent 工具）才能真正调用 `search`、`ingest_document` 等工具。
- **想快速验证环境** → 直接跑自带冒烟脚本：`powershell -ExecutionPolicy Bypass -File scripts\smoke.ps1`。

## 部署

- **嵌入式（默认，免 Docker）**：Qdrant 本地目录 + SQLite + 本地图片存储，零外部服务
- **远程模式**：环境变量一键切换 Qdrant / PostgreSQL / MinIO（`docker compose up -d`）
- **传输**：stdio 与 streamable HTTP 双传输（FastMCP）

## 技术栈

Python 3.12 · FastMCP · Qdrant · SQLite / PostgreSQL · MinIO · BGE-M3（嵌入，1024 维）· bge-reranker-v2-m3（重排）· PaddleOCR · jieba
