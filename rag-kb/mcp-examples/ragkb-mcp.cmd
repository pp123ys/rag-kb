@echo off
REM ragkb MCP Server 包装器：固定工作目录与环境变量，供 agent stdio 配置直接引用
cd /d D:\text\rag\rag-kb
set HF_ENDPOINT=https://hf-mirror.com
py -3.12 -m ragkb.mcp_server.server --transport stdio
