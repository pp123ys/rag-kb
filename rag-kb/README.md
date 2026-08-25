# RAG 知识库（ragkb）

为 agent 提供企业知识库检索增强，通过 MCP 协议调用。

## 快速开始

1. `docker compose up -d`（Qdrant / PostgreSQL / MinIO）
2. `pip install -e ".[dev]"`
3. 入库：`python -m ragkb.pipeline.ingest 文档.pdf --department 销售部 --version v1.0 --effective-date 2026-01-15`
   （首次入库会下载 BGE-M3 嵌入模型 ~2GB；无模型下载环境的机器可加 `--skip-embed`，
   跳过嵌入与向量入库，仅验证解析→清洗→切块→表格/图片/版本入库链路。）
4. 启动 MCP：`python -m ragkb.mcp_server.server --transport stdio`（或 `--transport http`）

## MCP 工具

| 工具 | 用途 |
|------|------|
| `search` | 双路召回 + RRF + 重排，返回带 source 的上下文（版本/权限过滤为预留能力） |
| `retrieve_table` | 表格精确取数 / 按表头查表 |
| `get_document` | 取回原文块 / 图片原图 |
| `list_versions` | 文档版本历史 |

## 防幻觉约定

agent 必须采用 `src/ragkb/prompts/agent_prompt.md` 系统提示词：
只依据检索内容回答、找不到就说没有找到、逐句标注出处。

## 测试

- 单元：`pytest tests/ -q`
- 集成（需 docker）：`pytest tests/ -m integration -q`
- 模型（需下载）：`pytest -m model -q`

## 端到端冒烟

- Linux / CI：`bash scripts/smoke.sh`
- Windows：`powershell -ExecutionPolicy Bypass -File scripts/smoke.ps1`

冒烟依次执行：起服务检查 → 生成样例 PDF → 入库 → 集成测试 → MCP 工具注册验证。
入库步骤默认走真实 BGE-M3 嵌入；若本机没有模型缓存（约 2GB 下载），脚本自动加
`--skip-embed` 跳过嵌入与向量入库（解析→清洗→切块→表格/图片/版本入库照常验证），
避免冒烟被模型下载拖垮；真实嵌入失败时同样降级并给出提示。

## 评测

`eval/eval_set.jsonl` 为离线评测集（Recall@K / MRR），
`python eval/run_eval.py` 跑评测，上线前扩充到 50–100 条。
