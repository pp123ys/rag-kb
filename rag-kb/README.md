# RAG 知识库（ragkb）

为 agent 提供企业知识库检索增强，通过 MCP 协议调用。

## 快速开始

默认**免 Docker 嵌入式**运行，零外部服务：

1. `pip install -e ".[dev]"`
2. 入库：`python -m ragkb.pipeline.ingest 文档.pdf --department 销售部 --version v1.0 --effective-date 2026-01-15`
   （首次入库会下载 BGE-M3 嵌入模型 ~2GB；无模型下载环境的机器可加 `--skip-embed`，
   跳过嵌入与向量入库，仅验证解析→清洗→切块→表格/图片/版本入库链路。）
3. 启动 MCP：`python -m ragkb.mcp_server.server --transport stdio`（或 `--transport http`）

数据落盘项目内 `data/` 目录（向量 `data/qdrant`、表格/版本 `data/ragkb.db`、图片 `data/images`）。

### 可选：远程服务模式（Docker）

需要独立 Qdrant / PostgreSQL / MinIO 时：

1. `docker compose up -d`（Qdrant / PostgreSQL / MinIO）
2. 在 `.env`（参考 `.env.example`）设置切换变量：
   `RAGKB_QDRANT_PATH=`（清空走远程 URL）、`RAGKB_PG_DSN=...`、`RAGKB_MINIO_ENDPOINT=...`

### 模型存储位置

嵌入（BGE-M3）与重排（bge-reranker）模型下载到**项目内 `models/` 目录**（不入 git），
不占用 C 盘用户缓存。可通过环境变量 `RAGKB_MODEL_CACHE_DIR` 改到其他位置。
离线环境设置 `HF_HUB_OFFLINE=1` 可强制只用本地缓存，避免加载时联网探测超时。

## MCP 工具

| 工具 | 用途 |
|------|------|
| `ingest_document` | 入库文档（pdf / xlsx / eml / msg / md / csv） |
| `delete_document` | 删除文档（向量 + 表格/版本 + 原图，幂等） |
| `search` | 双路召回 + RRF + 重排，返回带 source 的上下文（版本/权限过滤为预留能力） |
| `retrieve_table` | 表格精确取数 / 按表头查表 |
| `get_document` | 取回原文块 / 图片原图 |
| `list_versions` | 文档版本历史 |

> Markdown 与 CSV 的表格会自动进表格索引（`retrieve_table` 可查）；正文照常语义检索。

## 防幻觉约定

agent 必须采用 `src/ragkb/prompts/agent_prompt.md` 系统提示词：
只依据检索内容回答、找不到就说没有找到、逐句标注出处。

## 测试

- 单元：`pytest tests/ -q`
- 集成：`pytest tests/ -m integration -q`（Qdrant 集成测试走嵌入式本地目录，免 Docker；
  PostgreSQL / MinIO 用例需 `docker compose up -d` 起对应服务）
- 模型（需下载）：`pytest -m model -q`

## 端到端冒烟

- Linux / CI：`bash scripts/smoke.sh`
- Windows：`powershell -ExecutionPolicy Bypass -File scripts/smoke.ps1`

冒烟依次执行：生成样例 PDF → 入库 → 集成测试 → MCP 工具注册验证。
默认嵌入式（免 Docker）；检测到远端服务在跑时按远程模式入库。
入库步骤默认走真实 BGE-M3 嵌入；若本机没有模型缓存（约 2GB 下载），脚本自动加
`--skip-embed` 跳过嵌入与向量入库（解析→清洗→切块→表格/图片/版本入库照常验证），
避免冒烟被模型下载拖垮；真实嵌入失败时同样降级并给出提示。

## 评测

`eval/eval_set.jsonl` 为离线评测集（Recall@K / MRR），
`python eval/run_eval.py` 跑评测，上线前扩充到 50–100 条。
