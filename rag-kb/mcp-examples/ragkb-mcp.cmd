@echo off
REM ragkb MCP Server 包装器：固定工作目录与环境变量，供 agent stdio 配置直接引用
REM 注意：stdio 子进程不继承父进程环境，模型相关变量必须在此显式设置
cd /d D:\text\rag\rag-kb
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=D:\text\rag\rag-kb\models
set RAGKB_MODEL_CACHE_DIR=D:\text\rag\rag-kb\models
REM 离线/内网环境解除下行注释，强制只用本地模型缓存（避免加载时联网探测卡住）
REM set HF_HUB_OFFLINE=1
py -3.12 -m ragkb.mcp_server.server --transport stdio
