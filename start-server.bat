@echo off
REM RAG MCP Server Launcher - starts the ragkb MCP service for ask.py etc.
REM Modeled after rag-kb\start-server.bat; placed in the project root D:\text\rag

REM Use the script's own folder as the project root (works from any CWD)
set "PROJECT_DIR=%~dp0"
set "RAGKB_DIR=%PROJECT_DIR%rag-kb"

REM The ragkb package lives under rag-kb\src and the model cache under
REM rag-kb\models, so we must cd into rag-kb to resolve module and cache.
cd /d "%RAGKB_DIR%"

set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%RAGKB_DIR%\models
set RAGKB_MODEL_CACHE_DIR=%RAGKB_DIR%\models

echo Starting RAG KB MCP Server...
echo URL: http://127.0.0.1:8000
echo MCP Endpoint: http://127.0.0.1:8000/mcp
echo Press Ctrl+C to stop
echo.

py -3.12 -m ragkb.mcp_server.server --transport http --host 127.0.0.1 --port 8000

pause
