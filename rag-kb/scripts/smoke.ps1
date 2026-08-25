# scripts/smoke.ps1 —— 端到端冒烟（Windows 本机版）
# 默认嵌入式（免 Docker）；检测到 Qdrant 6333 在跑时走远程模式（需 .env 或脚本内置
# 变量切远程）。依次：模式检查 → 生成样例 PDF → 入库 → 集成测试 → MCP 工具注册。
#
# 用法：powershell -ExecutionPolicy Bypass -File scripts/smoke.ps1
# （Linux / CI 用 scripts/smoke.sh）
$ErrorActionPreference = "Stop"
$OutputEncoding = New-Object System.Text.UTF8Encoding $false

$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    function Test-Port {
        param([int]$Port)
        try {
            $c = New-Object System.Net.Sockets.TcpClient
            $iar = $c.BeginConnect("127.0.0.1", $Port, $null, $null)
            $waited = $iar.AsyncWaitHandle.WaitOne(2000, $false)
            $ok = $waited -and $c.Connected
            $c.Close()
            return $ok
        } catch { return $false }
    }

    Write-Host "==> 1. 运行模式检查"
    $remote = Test-Port -Port 6333
    if ($remote) {
        Write-Host "    （检测到 Qdrant:6333 在跑 → 远程模式；尝试拉起缺失服务）"
        # 注意：不要重定向 docker 的 stderr（PS 5.1 在 $ErrorActionPreference=Stop 下
        # 会把原生 stderr 行变成终止性 NativeCommandError）；让其原样输出即可。
        & docker compose up -d qdrant postgres minio
        if ($LASTEXITCODE -ne 0) {
            Write-Host "    （compose up 未完全成功，可能是端口被既有容器占用；继续做健康检查）"
        }
        $services = @(@("Qdrant", 6333), @("PostgreSQL", 5432), @("MinIO", 9000))
        $deadline = (Get-Date).AddSeconds(30)
        do {
            $allOk = $true
            foreach ($s in $services) { if (-not (Test-Port -Port $s[1])) { $allOk = $false; break } }
            if (-not $allOk) { Start-Sleep -Seconds 1 }
        } while ((-not $allOk) -and ((Get-Date) -lt $deadline))
        foreach ($s in $services) {
            if (Test-Port -Port $s[1]) { Write-Host "    OK: $($s[0]):$($s[1])" }
            else { Write-Host "    错误：$($s[0]):$($s[1]) 不可达" -ForegroundColor Red; $allOk = $false }
        }
        if (-not $allOk) { throw "依赖服务健康检查失败" }
        # 新 config 默认嵌入式：远程模式需显式清空 qdrant_path 并切 PG/MinIO
        $env:RAGKB_QDRANT_PATH = ""
        $env:RAGKB_PG_DSN = "postgresql://ragkb:ragkb@localhost:5432/ragkb"
        $env:RAGKB_MINIO_ENDPOINT = "localhost:9000"
        $env:RAGKB_MINIO_ACCESS_KEY = "ragkb"
        $env:RAGKB_MINIO_SECRET_KEY = "ragkb-secret"
    } else {
        Write-Host "    （未检测到 Qdrant:6333 → 嵌入式模式，免 Docker：向量落 data/qdrant、表格/版本 data/ragkb.db、图片 data/images）"
        Remove-Item Env:RAGKB_QDRANT_PATH -ErrorAction SilentlyContinue
        Remove-Item Env:RAGKB_PG_DSN -ErrorAction SilentlyContinue
        Remove-Item Env:RAGKB_MINIO_ENDPOINT -ErrorAction SilentlyContinue
    }

    Write-Host "==> 1.5 生成样例文档"
    New-Item -ItemType Directory -Force -Path "tests\fixtures" | Out-Null
    $genScript = Join-Path $env:TEMP "ragkb_gen_sample_pdf.py"
    $genCode = @'
import fitz
doc = fitz.open()
page = doc.new_page()
page.insert_text((72, 72), "产品型号 A-100 单价 99 元。保修期一年。", fontname="china-s")
doc.save("tests/fixtures/sample.pdf")
doc.close()
'@
    [System.IO.File]::WriteAllText($genScript, $genCode,
                                   (New-Object System.Text.UTF8Encoding $false))
    & py -3.12 $genScript
    if ($LASTEXITCODE -ne 0) { throw "生成样例 PDF 失败" }
    Remove-Item $genScript -ErrorAction SilentlyContinue
    Write-Host "    OK: tests\fixtures\sample.pdf"

    Write-Host "==> 2. 入库样例文档"
    $hfHome = if ($env:HF_HOME) { $env:HF_HOME } else { Join-Path $HOME ".cache\huggingface" }
    if (Test-Path $hfHome) {
        Write-Host "    （发现模型缓存，走真实 BGE-M3 嵌入）"
        & py -3.12 -m ragkb.pipeline.ingest tests/fixtures/sample.pdf `
            --source sample.pdf --department 销售部 --version v1.0 `
            --effective-date 2026-01-15
        if ($LASTEXITCODE -ne 0) {
            Write-Host "    （真实嵌入失败，降级 --skip-embed 重试：模型缓存可能损坏，需重新下载）"
            & py -3.12 -m ragkb.pipeline.ingest tests/fixtures/sample.pdf `
                --source sample.pdf --department 销售部 --version v1.0 `
                --effective-date 2026-01-15 --skip-embed
        }
    } else {
        Write-Host "    （未发现模型缓存：--skip-embed 跳过 ~2GB 模型下载，仅验证解析/清洗/切块/表格/版本入库）"
        Write-Host "    注：完整链路需先下载模型，例如 py -3.12 -c ""from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"""
        & py -3.12 -m ragkb.pipeline.ingest tests/fixtures/sample.pdf `
            --source sample.pdf --department 销售部 --version v1.0 `
            --effective-date 2026-01-15 --skip-embed
    }
    if ($LASTEXITCODE -ne 0) { throw "入库失败" }
    Write-Host "    OK: 入库完成"

    Write-Host "==> 3. 运行集成测试"
    if ($remote) {
        # 远程模式：全部集成用例（Qdrant / PG / MinIO）
        & py -3.12 -m pytest tests/ -m integration -q
    } else {
        # 嵌入式模式：Qdrant 集成测试走本地嵌入式目录，免 Docker；PG/MinIO 用例跳过
        & py -3.12 -m pytest tests/test_qdrant_indexer.py -m integration -q
    }
    if ($LASTEXITCODE -ne 0) { throw "集成测试失败" }
    Write-Host "    OK: 集成测试通过"

    Write-Host "==> 4. MCP server 冒烟（--help 验证入口可启动、工具模块可加载）"
    & py -3.12 -m ragkb.mcp_server.server --help
    if ($LASTEXITCODE -ne 0) { throw "MCP server --help 失败" }
    Write-Host "    OK: MCP server 入口可用"

    Write-Host "==> 完成"
} finally {
    Pop-Location
}
