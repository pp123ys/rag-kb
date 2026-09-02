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

## 部署

- **嵌入式（默认，免 Docker）**：Qdrant 本地目录 + SQLite + 本地图片存储，零外部服务
- **远程模式**：环境变量一键切换 Qdrant / PostgreSQL / MinIO（`docker compose up -d`）
- **传输**：stdio 与 streamable HTTP 双传输（FastMCP）

## 技术栈

Python 3.12 · FastMCP · Qdrant · SQLite / PostgreSQL · MinIO · BGE-M3（嵌入，1024 维）· bge-reranker-v2-m3（重排）· PaddleOCR · jieba
