#!/usr/bin/env python3
"""
Unified Render Deployment Server

Combines the following into a single FastAPI application:
  - MCP tool server  (mounted at /api, so /api/tools/*, /api/health, etc.)
  - Conference data APIs  (/api/neurips/*, /api/iclr/*)
  - Node-colors API  (/api/node-colors)
  - Agent chat proxy  (/api/chat  -> Agent on localhost:8001)
  - Static file serving  (/output/*, /pdf/*)
  - Built web frontend  (/ catch-all SPA)

The Agent server runs as a SEPARATE background process (started by render-start.sh)
because AgentClient uses synchronous `requests` for MCP communication.

Usage:
  uvicorn render_app:app --host 0.0.0.0 --port $PORT
"""

import asyncio
import csv
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse

# ─── Configuration ──────────────────────────────────────────────────────────

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))
PDF_DIR = Path(os.getenv("PDF_DIR", "/data/pdf"))
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
WEB_DIST_DIR = Path("/app/web/dist")

NODE_COLORS_PATH = OUTPUT_DIR / "graph" / "node_colors.json"
AGENT_URL = os.getenv("AGENT_SERVER_URL", "http://127.0.0.1:8001")

# NeurIPS data
NEURIPS_META = DATA_DIR / "embeddings_Neu" / "metadata.csv"
NEURIPS_SIM = DATA_DIR / "embeddings_Neu" / "similarities.json"

# ICLR data
ICLR_META = DATA_DIR / "embeddings_ICLR" / "ICLR2025_accepted_meta.csv"
ICLR_SIM = DATA_DIR / "embeddings_ICLR" / "similarities.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("render-app")

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "graph").mkdir(parents=True, exist_ok=True)

# ─── In-memory caches (avoid re-parsing large CSVs per request) ─────────────

_neurips_cache: Optional[List[Dict[str, str]]] = None
_iclr_cache: Optional[List[Dict[str, str]]] = None


def _parse_csv(file_path: Path) -> List[Dict[str, str]]:
    """Parse a CSV file (with BOM handling) into list of dicts."""
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _paginate(items: list, offset: int, limit: int) -> dict:
    total = len(items)
    if limit > 0:
        limit = min(500, limit)
        sliced = items[offset : offset + limit]
    else:
        sliced = items
        limit = total
    next_offset = min(total, offset + (limit if limit > 0 else total))
    return {
        "papers": sliced,
        "count": len(sliced),
        "total": total,
        "offset": offset,
        "limit": limit,
        "nextOffset": next_offset,
        "hasMore": next_offset < total,
    }


def _read_json(file_path: Path, fallback: Any = None):
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return fallback


# ─── App creation ───────────────────────────────────────────────────────────

app = FastAPI(title="Research Agent (Render)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
#  Custom API routes (defined BEFORE the /api MCP mount so they take priority)
# ═══════════════════════════════════════════════════════════════════════════


# ─── Node Colors ────────────────────────────────────────────────────────────

@app.api_route("/api/node-colors", methods=["GET", "POST", "OPTIONS"])
async def node_colors(request: Request):
    if request.method == "OPTIONS":
        return Response(status_code=204)

    if request.method == "GET":
        data = _read_json(NODE_COLORS_PATH, {"colors": {}, "timestamp": 0})
        return JSONResponse(data)

    # POST – save colors
    body = await request.json()
    data = {
        "colors": body.get("colors", {}),
        "timestamp": int(time.time() * 1000),
    }
    NODE_COLORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NODE_COLORS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return JSONResponse({"success": True, "timestamp": data["timestamp"]})


@app.get("/api/node-colors/check")
async def node_colors_check():
    if NODE_COLORS_PATH.exists():
        mtime = NODE_COLORS_PATH.stat().st_mtime_ns / 1_000_000
        return {"exists": True, "mtime": mtime}
    return {"exists": False, "mtime": 0}


# ─── NeurIPS API ────────────────────────────────────────────────────────────

@app.get("/api/neurips/papers")
async def neurips_papers(offset: int = 0, limit: int = 0):
    global _neurips_cache
    if _neurips_cache is None:
        _neurips_cache = _parse_csv(NEURIPS_META)
    return _paginate(_neurips_cache, offset, limit)


@app.get("/api/neurips/similarities")
async def neurips_similarities():
    return _read_json(
        NEURIPS_SIM, {"edges": [], "message": "similarities.json not found"}
    )


@app.get("/api/neurips/clusters")
async def neurips_clusters(k: int = 15):
    valid_k = max(5, min(30, k))
    path = DATA_DIR / "embeddings_Neu" / f"neurips_clusters_k{valid_k}.json"
    return _read_json(path, {"paper_id_to_cluster": {}, "k": 0})


# ─── ICLR API ──────────────────────────────────────────────────────────────

@app.get("/api/iclr/papers")
async def iclr_papers(offset: int = 0, limit: int = 0):
    global _iclr_cache
    if _iclr_cache is None:
        _iclr_cache = _parse_csv(ICLR_META)
    return _paginate(_iclr_cache, offset, limit)


@app.get("/api/iclr/similarities")
async def iclr_similarities():
    return _read_json(
        ICLR_SIM, {"edges": [], "message": "similarities.json not found"}
    )


@app.get("/api/iclr/clusters")
async def iclr_clusters(k: int = 15):
    valid_k = max(5, min(30, k))
    path = DATA_DIR / "embeddings_ICLR" / f"iclr_clusters_k{valid_k}.json"
    return _read_json(path, {"paper_id_to_cluster": {}, "k": 0})


# ─── Chat proxy  (→ Agent server on :8001) ─────────────────────────────────

@app.post("/api/chat")
async def chat_proxy(request: Request):
    """Forward chat requests to the Agent server running on localhost:8001."""
    import requests as req_lib

    body = await request.json()
    try:
        resp = await asyncio.to_thread(
            req_lib.post,
            f"{AGENT_URL}/chat",
            json=body,
            timeout=120,
        )
        return resp.json()
    except Exception as e:
        return {"response": f"Chat service unavailable. Error: {e}"}


# ═══════════════════════════════════════════════════════════════════════════
#  Static file serving  (order: specific paths → MCP mount → web catch-all)
# ═══════════════════════════════════════════════════════════════════════════

# Serve extracted output files  (e.g. /output/graph/global_graph.json)
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output-files")

# Serve PDF files  (e.g. /pdf/paper.pdf)
app.mount("/pdf", StaticFiles(directory=str(PDF_DIR)), name="pdf-files")

# ─── Mount the MCP FastAPI app at /api ──────────────────────────────────────
# All MCP routes become /api/tools/*, /api/health, /api/paper/*, etc.
# The /api prefix is stripped when forwarding, matching the Vite proxy behavior.

from mcp.server import app as mcp_app  # noqa: E402

app.mount("/api", mcp_app)

# ─── Serve the built web frontend (catch-all SPA; MUST be last) ────────────

if WEB_DIST_DIR.exists() and (WEB_DIST_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST_DIR), html=True), name="web")
    logger.info(f"Serving web frontend from {WEB_DIST_DIR}")
else:
    logger.warning(
        f"Web build not found at {WEB_DIST_DIR}. "
        "Run 'cd web && npm run build' to create it."
    )

    @app.get("/")
    async def fallback_root():
        return HTMLResponse(
            "<h2>Research Agent</h2>"
            "<p>Web frontend not built. API is available at <code>/api</code>.</p>"
        )
