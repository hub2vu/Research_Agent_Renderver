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

echo "[render-start] Starting Agent server on :8001"
python -m uvicorn agent.server:app --host 0.0.0.0 --port 8001 &
AGENT_PID=$!

# Web: run Vite dev server bound to Render's PORT
echo "[render-start] Starting Web UI on :$PORT"
cd /app/web

# Ensure MCP URL is visible to the web build/runtime (Vite only exposes VITE_* to client)
# Your app currently reads MCP_SERVER_URL in container env (used for server-side calls / proxy if any).
# If you need client-side access, consider switching to VITE_MCP_SERVER_URL and updating code.

exec npm run dev -- --host 0.0.0.0 --port "$PORT"
