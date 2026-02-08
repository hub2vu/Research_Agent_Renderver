#!/bin/sh
set -e

# Render assigns PORT for the public web service
: "${PORT:=10000}"

# Force internal service URL to localhost (since we're in one container)
export MCP_SERVER_URL="http://127.0.0.1:8000"

# Optional: map PDF/OUTPUT dirs if your code relies on these envs
: "${PDF_DIR:=/data/pdf}"
: "${OUTPUT_DIR:=/data/output}"
export PDF_DIR OUTPUT_DIR

echo "[render-start] Starting MCP server on :8000"
python -m uvicorn mcp.server:app --host 0.0.0.0 --port 8000 &
MCP_PID=$!

# Wait for MCP to be ready before starting Agent (prevents startup race)
echo "[render-start] Waiting for MCP to be ready..."
python - <<'PY'
import socket, time, sys
host, port = '127.0.0.1', 8000
for i in range(60):
    try:
        s = socket.create_connection((host, port), timeout=1)
        s.close()
        print('[render-start] MCP is ready')
        sys.exit(0)
    except OSError:
        time.sleep(0.5)
print('[render-start] MCP did not become ready in time', file=sys.stderr)
sys.exit(1)
PY

echo "[render-start] Starting Agent server on :8001"
python -m uvicorn agent.server:app --host 0.0.0.0 --port 8001 &
AGENT_PID=$!

# Web: run Vite dev server bound to Render's PORT
echo "[render-start] Starting Web UI on :$PORT"
cd /app/web

exec npm run dev -- --host 0.0.0.0 --port "$PORT"
