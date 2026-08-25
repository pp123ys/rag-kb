#!/usr/bin/env bash
# scripts/smoke.sh —— 端到端冒烟：起服务 → 生成样例 → 入库 → 集成测试 → MCP 工具注册
#
# 用法（Linux / CI）：
#   bash scripts/smoke.sh
# Windows 本机请用 scripts/smoke.ps1（本脚本的 PowerShell 等价版）。
# 依赖：docker compose（Qdrant / PostgreSQL / MinIO）、python（含 ragkb 及其依赖）。
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> 1. 起服务"
# 幂等拉起；失败不立即退出——若端口已被既有容器占用，交由下方健康检查裁决
docker compose up -d qdrant postgres minio \
    || echo "    （compose up 未完全成功，可能是端口被既有容器占用；继续做健康检查）"

echo "==> 1.1 健康检查（Qdrant 6333 / PostgreSQL 5432 / MinIO 9000）"
python - <<'PY'
import socket, sys, time
services = [("Qdrant", 6333), ("PostgreSQL", 5432), ("MinIO", 9000)]

def probe(port):
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=2)
        s.close()
        return True
    except OSError:
        return False

deadline = time.time() + 30
while time.time() < deadline and not all(probe(p) for _, p in services):
    time.sleep(1)
ok = True
for name, port in services:
    if probe(port):
        print(f"    {name}:{port} OK")
    else:
        print(f"    错误：{name}:{port} 不可达", file=sys.stderr)
        ok = False
sys.exit(0 if ok else 1)
PY

echo "==> 1.5 生成样例文档"
mkdir -p tests/fixtures
python - <<'PY'
import fitz
doc = fitz.open()
page = doc.new_page()
page.insert_text((72, 72), "产品型号 A-100 单价 99 元。保修期一年。", fontname="china-s")
doc.save("tests/fixtures/sample.pdf")
doc.close()
PY

echo "==> 2. 入库样例文档"
run_ingest() {
    python -m ragkb.pipeline.ingest tests/fixtures/sample.pdf \
        --source sample.pdf --department 销售部 --version v1.0 \
        --effective-date 2026-01-15 "$@"
}
if [ -d "${HF_HOME:-$HOME/.cache/huggingface}" ]; then
    echo "    （发现模型缓存，走真实 BGE-M3 嵌入）"
    if ! run_ingest; then
        echo "    （真实嵌入失败，降级 --skip-embed 重试：模型缓存可能损坏，需重新下载）"
        run_ingest --skip-embed
    fi
else
    echo "    （未发现模型缓存：--skip-embed 跳过 ~2GB 模型下载，仅验证解析/清洗/切块/表格/版本入库）"
    echo "    注：完整链路需先下载模型，例如 python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')\""
    run_ingest --skip-embed
fi

echo "==> 3. 运行离线评测"
python -m pytest tests/ -m integration -q

echo "==> 4. 启动 MCP server（stdio）验证工具注册"
# 空 stdin：服务读到 EOF 正常退出（rc=0）；若 5s 仍在运行（rc=124）说明启动成功
# 只是未退出——两者都视为通过；其余退出码视为失败。
set +e
timeout 5 python -m ragkb.mcp_server.server --transport stdio < <(printf '')
rc=$?
set -e
if [ "$rc" -eq 0 ] || [ "$rc" -eq 124 ]; then
    echo "MCP server OK"
else
    echo "错误：MCP server 启动失败（exit $rc）" >&2
    exit 1
fi

echo "==> 完成"
