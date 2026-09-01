@echo off
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=D:\text\rag\rag-kb\models
set RAGKB_MODEL_CACHE_DIR=D:\text\rag\rag-kb\models
cd /d D:\text\rag\rag-kb
py -3.12 -m ragkb.mcp_server.server --transport http --host 127.0.0.1 --port 8000