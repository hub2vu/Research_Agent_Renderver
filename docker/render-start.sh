#!/bin/sh
set -e

# ──────────────────────────────────────────────────────────────
#  Render Unified Startup Script
#
#  Process 1: render_app (MCP server + Web frontend + APIs)
#             on $PORT (Render's public port, default 10000)
#  Process 2: Agent server on localhost:8001 (internal only)
# ──────────────────────────────────────────────────────────────

: "${PORT:=10000}"

# Agent connects to MCP via the render_app's /api mount
export MCP_SERVER_URL="http://127.0.0.1:${PORT}/api"

# Data directories
: "${PDF_DIR:=/data/pdf}"
: "${OUTPUT_DIR:=/data/output}"
export PDF_DIR OUTPUT_DIR

# Memory-saving flags
: "${SKIP_EMBEDDINGS:=true}"
: "${SKIP_AUTO_PROCESS:=true}"
export SKIP_EMBEDDINGS SKIP_AUTO_PROCESS

echo "============================================"
echo "  Research Agent – Render Deployment"
echo "  Public port      : ${PORT}"
echo "  MCP URL          : ${MCP_SERVER_URL}"
echo "  PDF dir          : ${PDF_DIR}"
echo "  Output dir       : ${OUTPUT_DIR}"
echo "  Skip embeddings  : ${SKIP_EMBEDDINGS}"
echo "  Skip auto-process: ${SKIP_AUTO_PROCESS}"
echo "============================================"

# ── 1. Start the unified server (render_app = MCP + Web) ──
echo "[render] Starting unified server on :${PORT} ..."
python -m uvicorn render_app:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --log-level info &
MAIN_PID=$!

# ── 2. Wait for the server to be ready ──
echo "[render] Waiting for server to be ready ..."
for i in $(seq 1 90); do
    if python -c "
import socket, sys
try:
    s = socket.create_connection(('127.0.0.1', ${PORT}), timeout=1)
    s.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
" 2>/dev/null; then
        echo "[render] Server is ready (took ~${i}s)"
        break
    fi
    sleep 1
done

# ── 3. Start Agent server (background, internal only) ──
echo "[render] Starting Agent server on :8001 ..."
python -m uvicorn agent.server:app \
    --host 127.0.0.1 \
    --port 8001 \
    --log-level info &
AGENT_PID=$!

echo "[render] All processes started. Main PID=${MAIN_PID}, Agent PID=${AGENT_PID}"

# ── 4. Keep running until the main process exits ──
# If the main server dies, Render will restart the container.
wait ${MAIN_PID}
