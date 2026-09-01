# RAG MCP Server Launcher
$ErrorActionPreference = "Continue"

Set-Location "D:\text\rag\rag-kb"

$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "D:\text\rag\rag-kb\models"
$env:RAGKB_MODEL_CACHE_DIR = "D:\text\rag\rag-kb\models"

Write-Host "Starting RAG KB MCP Server..."
Write-Host "URL: http://127.0.0.1:8000"
Write-Host "Press Ctrl+C to stop"
Write-Host ""

py -3.12 -m ragkb.mcp_server.server --transport http --host 127.0.0.1 --port 8000
