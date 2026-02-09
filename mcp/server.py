#!/usr/bin/env python3
"""
MCP Server Entrypoint

HTTP API server that exposes all registered MCP tools.
Tools are automatically discovered via registry.py
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import json
import shutil
import uuid
from datetime import datetime

# 새로 만든 도구 임포트
from .tools.page_analyzer import interpret_paper_page
from .registry import (
    execute_tool,
    get_all_tools,
    get_openai_tools_schema,
    list_tool_names,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def auto_process_all_pdfs():
    """
    서버 시작 시 자동으로 미처리 PDF를 처리하는 백그라운드 태스크.

    1단계: 텍스트 + 이미지 추출 (이미 처리된 PDF는 스킵)
    2단계: 레퍼런스 추출 (이미 추출된 논문은 스킵)
    3단계: 글로벌 그래프 빌드 (새로 처리된 논문이 있거나, 그래프가 없으면 실행)
    """
    try:
        logger.info("[Auto-Process] PDF 자동 처리 시작...")

        new_papers_processed = 0

        # ── Step 1: 텍스트 + 이미지 추출 ──
        logger.info("[Auto-Process] Step 1/3: 텍스트 및 이미지 추출 중...")
        result = await execute_tool("process_all_pdfs")
        if result["success"]:
            data = result["result"]
            new_papers_processed = data.get("successful", 0)
            logger.info(
                f"[Auto-Process] 텍스트/이미지 추출 완료 - "
                f"성공: {data.get('successful', 0)}, "
                f"스킵: {data.get('skipped', 0)}, "
                f"실패: {data.get('failed', 0)}"
            )
        else:
            logger.warning(f"[Auto-Process] 텍스트/이미지 추출 실패: {result.get('error')}")

        # ── Step 2: 레퍼런스 추출 ──
        logger.info("[Auto-Process] Step 2/3: 레퍼런스 추출 중...")
        result = await execute_tool("extract_all_references", skip_existing=True)
        if result["success"]:
            data = result["result"]
            refs_processed = data.get("processed", 0)
            new_papers_processed += refs_processed
            logger.info(
                f"[Auto-Process] 레퍼런스 추출 완료 - "
                f"처리: {refs_processed}, "
                f"스킵: {data.get('skipped', 0)}, "
                f"실패: {data.get('failed', 0)}"
            )
        else:
            logger.warning(f"[Auto-Process] 레퍼런스 추출 실패: {result.get('error')}")

        # ── Step 3: 글로벌 그래프 빌드 ──
        graph_path = Path(os.getenv("OUTPUT_DIR", "/data/output")) / "graph" / "global_graph.json"
        should_build_graph = new_papers_processed > 0 or not graph_path.exists()

        if should_build_graph:
            # SKIP_EMBEDDINGS=true → 임베딩 없이 빌드 (RAM 절약, Render 무료 티어 등)
            skip_emb = os.getenv("SKIP_EMBEDDINGS", "").lower() in ("1", "true", "yes")
            logger.info(f"[Auto-Process] Step 3/3: 글로벌 그래프 빌드 중... (embeddings={'off' if skip_emb else 'on'})")
            result = await execute_tool("build_global_graph", use_embeddings=not skip_emb)
            if result["success"]:
                meta = result["result"].get("meta", {})
                logger.info(
                    f"[Auto-Process] 글로벌 그래프 빌드 완료 - "
                    f"논문 수: {meta.get('total_papers', 0)}, "
                    f"엣지 수: {meta.get('total_edges', 0)}"
                )
            else:
                logger.warning(f"[Auto-Process] 글로벌 그래프 빌드 실패: {result.get('error')}")
        else:
            logger.info("[Auto-Process] Step 3/3: 새로 처리된 논문 없음, 글로벌 그래프 빌드 스킵")

        logger.info("[Auto-Process] PDF 자동 처리 완료!")

    except Exception as e:
        logger.error(f"[Auto-Process] 자동 처리 중 오류 발생: {str(e)}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: discover tools
    tools = get_all_tools()
    logger.info(f"MCP Server starting with {len(tools)} tools")
    for name in tools:
        logger.info(f"  - {name}")

    # 서버 시작 후 백그라운드에서 미처리 PDF 자동 처리
    task = asyncio.create_task(auto_process_all_pdfs())

    yield

    # Shutdown: 백그라운드 태스크 정리
    if not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info("[Auto-Process] 백그라운드 태스크 취소됨")
    logger.info("MCP Server shutting down")


app = FastAPI(
    title="MCP Research Agent Server",
    description="Model Context Protocol server for research tools",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 주소(localhost:3000 등) 허용
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST 등 모든 방식 허용
    allow_headers=["*"],
)


class ToolRequest(BaseModel):
    """Request body for tool execution."""

    arguments: Dict[str, Any] = {}


class ToolResponse(BaseModel):
    """Response from tool execution."""

    success: bool
    tool: str
    result: Any = None
    error: str = None
    error_type: str = None


# ============== API Endpoints ==============


@app.get("/")
async def root():
    return {
        "service": "MCP Research Agent Server",
        "version": "1.0.0",
        "tools_count": len(list_tool_names()),
        "endpoints": {
            "list_tools": "/tools",
            "tool_schema": "/tools/schema",
            "execute": "/tools/{tool_name}/execute",
            "health": "/health",
        },
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "tools_loaded": len(list_tool_names())}


@app.get("/tools")
async def list_tools():
    tools = get_all_tools()
    return {
        "total": len(tools),
        "tools": [
            {
                "name": name,
                "description": tool.description,
                "category": tool.category,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "description": p.description,
                        "required": p.required,
                    }
                    for p in tool.parameters
                ],
            }
            for name, tool in tools.items()
        ],
    }


@app.get("/tools/schema")
async def get_tools_schema():
    return {"tools": get_openai_tools_schema()}


@app.get("/tools/{tool_name}")
async def get_tool_info(tool_name: str):
    tools = get_all_tools()
    if tool_name not in tools:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

    tool = tools[tool_name]
    return {
        "name": tool.name,
        "description": tool.description,
        "category": tool.category,
        "parameters": [
            {
                "name": p.name,
                "type": p.type,
                "description": p.description,
                "required": p.required,
                "default": p.default,
            }
            for p in tool.parameters
        ],
    }


@app.post("/tools/{tool_name}/execute", response_model=ToolResponse)
async def execute_tool_endpoint(tool_name: str, request: ToolRequest):
    result = await execute_tool(tool_name, **request.arguments)
    return ToolResponse(**result)


# ============== Convenience Endpoints ==============
# These provide direct access to common tools


@app.get("/pdf/list")
async def list_pdfs():
    result = await execute_tool("list_pdfs")
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result["result"]


@app.post("/pdf/extract")
async def extract_pdf(filename: str):
    result = await execute_tool("extract_all", filename=filename)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result["result"]


@app.get("/pdf/process-all")
async def process_all_pdfs():
    result = await execute_tool("process_all_pdfs")
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result["result"]


@app.get("/arxiv/search")
async def search_arxiv(query: str, max_results: int = 10):
    result = await execute_tool("arxiv_search", query=query, max_results=max_results)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result["result"]


@app.get("/web/search")
async def web_search(query: str, max_results: int = 5):
    result = await execute_tool("web_search", query=query, max_results=max_results)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result["result"]


# [파일 저장 경로 정의]
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))
STATUS_DIR = OUTPUT_DIR / "agent_status"
STATUS_DIR.mkdir(parents=True, exist_ok=True)

# .env management (best-effort; primarily for local/dev)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = Path(os.getenv("DOTENV_PATH", str(PROJECT_ROOT / ".env")))


def _read_dotenv(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    data: Dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            v = v.strip()
            # Remove surrounding quotes if present (handles "value" or 'value')
            if len(v) >= 2 and ((v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'"))):
                v = v[1:-1]
            data[k.strip()] = v
    except Exception as e:
        logger.warning(f"Failed to read .env at {path}: {e}")
    return data


def _write_dotenv(path: Path, updates: Dict[str, str]) -> None:
    """
    Update (or append) keys in .env while preserving other lines best-effort.
    Writes UTF-8.
    """
    lines: List[str] = []
    existing = {}
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    else:
        # ensure parent exists
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # index existing keys
    key_to_idx: Dict[str, int] = {}
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _ = s.split("=", 1)
        k = k.strip()
        key_to_idx[k] = i

    # apply updates
    for k, v in updates.items():
        new_line = f"{k}={v}"
        if k in key_to_idx:
            lines[key_to_idx[k]] = new_line
        else:
            lines.append(new_line)

    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


class DiscordConfigRequest(BaseModel):
    discord_webhook_full: str = ""
    discord_webhook_summary: str = ""


@app.get("/config/discord")
async def get_discord_config():
    """
    Returns Discord webhook URLs from process env, falling back to .env file if present.
    Always reads .env file to ensure values are loaded even if process env is empty.
    """
    # Always read .env file first to ensure values are loaded
    dotenv = _read_dotenv(DOTENV_PATH)
    
    # Get from process env, fallback to .env file
    full = os.getenv("DISCORD_WEBHOOK_FULL", "") or dotenv.get("DISCORD_WEBHOOK_FULL", "")
    summary = os.getenv("DISCORD_WEBHOOK_SUMMARY", "") or dotenv.get("DISCORD_WEBHOOK_SUMMARY", "")

    return {
        "success": True,
        "dotenv_path": str(DOTENV_PATH),
        "discord_webhook_full": full,
        "discord_webhook_summary": summary,
    }


@app.post("/config/discord")
async def update_discord_config(req: DiscordConfigRequest):
    """
    Update .env with Discord webhook URLs and also update current process env
    (so running server can use the new values without restart).
    """
    updates = {
        "DISCORD_WEBHOOK_FULL": (req.discord_webhook_full or "").strip(),
        "DISCORD_WEBHOOK_SUMMARY": (req.discord_webhook_summary or "").strip(),
    }
    try:
        _write_dotenv(DOTENV_PATH, updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write .env: {e}")

    # Update runtime environment for immediate use
    os.environ["DISCORD_WEBHOOK_FULL"] = updates["DISCORD_WEBHOOK_FULL"]
    os.environ["DISCORD_WEBHOOK_SUMMARY"] = updates["DISCORD_WEBHOOK_SUMMARY"]

    return {"success": True, "dotenv_path": str(DOTENV_PATH), **updates}


# [리포트 조회 기능]
@app.get("/reports/{paper_id}")
async def get_report_content(paper_id: str):
    logger.info(f"🔍 [API Request] 리포트 요청 ID: {paper_id}")
    target_dir = OUTPUT_DIR / paper_id

    if not target_dir.exists():
        core_id = paper_id
        if "arxiv." in paper_id:
            core_id = paper_id.split("arxiv.")[-1]

        logger.info(
            f"⚠️ 정확한 폴더 없음. 핵심 ID '{core_id}'가 포함된 폴더를 검색합니다..."
        )

        found = False
        try:
            for folder in OUTPUT_DIR.iterdir():
                if folder.is_dir():
                    if core_id in folder.name or folder.name in paper_id:
                        target_dir = folder
                        found = True
                        logger.info(f"✅ 유사 폴더 발견: {target_dir}")
                        break
        except Exception as e:
            logger.error(f"폴더 검색 중 에러 발생: {e}")

        if not found:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Folder not found",
                    "detail": f"Could not find folder for {paper_id}",
                },
            )

    md_file = target_dir / "summary_report.md"
    txt_file = target_dir / "summary_report.txt"
    final_file = None
    if md_file.exists():
        final_file = md_file
    elif txt_file.exists():
        final_file = txt_file

    if final_file:
        try:
            with open(final_file, "r", encoding="utf-8") as f:
                content = f.read()
            return {"content": content}
        except Exception as e:
            return JSONResponse(
                status_code=500, content={"error": f"Read error: {str(e)}"}
            )
    else:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Report file missing",
                "detail": "Folder exists but summary_report.md or .txt is missing.",
            },
        )


class InterpretRequest(BaseModel):
    paper_id: str
    page_num: int


@app.post("/paper/interpret")
async def interpret_page_endpoint(request: InterpretRequest):
    """
    특정 페이지 해석 요청 API
    """
    logger.info(f"🧠 해석 요청: {request.paper_id} - Page {request.page_num}")

    try:
        result = await interpret_paper_page(request.paper_id, request.page_num)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"해석 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Research Agent Pipeline API ==============

async def run_agent_background(job_id: str, tool_args: Dict[str, Any]):
    """Background task to run the research agent."""
    try:
        result = await execute_tool("run_research_agent", **tool_args, job_id=job_id)
        logger.info(f"[Background] Job {job_id} completed: {result.get('success')}")
    except Exception as e:
        logger.error(f"[Background] Job {job_id} failed: {str(e)}")
        # Update status to failed
        try:
            status_file = STATUS_DIR / f"{job_id}.json"
            if status_file.exists():
                with open(status_file, "r", encoding="utf-8") as f:
                    status_data = json.load(f)
                status_data["status"] = "failed"
                status_data["errors"] = status_data.get("errors", []) + [str(e)]
                status_data["updated_at"] = datetime.now().isoformat()
                with open(status_file, "w", encoding="utf-8") as f:
                    json.dump(status_data, f, ensure_ascii=False, indent=2)
        except Exception as save_error:
            logger.error(f"Failed to save error status: {save_error}")


async def run_conference_pipeline_background(job_id: str, tool_args: Dict[str, Any]):
    """Background task to run the conference pipeline."""
    try:
        result = await execute_tool("run_conference_pipeline", **tool_args, job_id=job_id)
        logger.info(f"[Background] Conference pipeline {job_id} completed: {result.get('success')}")
    except Exception as e:
        logger.error(f"[Background] Conference pipeline {job_id} failed: {str(e)}")
        # Update status to failed
        try:
            status_file = STATUS_DIR / f"{job_id}.json"
            if status_file.exists():
                with open(status_file, "r", encoding="utf-8") as f:
                    status_data = json.load(f)
                status_data["status"] = "failed"
                status_data["errors"] = status_data.get("errors", []) + [str(e)]
                status_data["updated_at"] = datetime.now().isoformat()
                with open(status_file, "w", encoding="utf-8") as f:
                    json.dump(status_data, f, ensure_ascii=False, indent=2)
        except Exception as save_error:
            logger.error(f"Failed to save error status: {save_error}")


@app.post("/agent/run")
async def run_agent_pipeline(background_tasks: BackgroundTasks, request: ToolRequest):
    """
    Start the research agent pipeline in the background.
    Returns immediately with job_id.
    """
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Add job_id to arguments
    tool_args = request.arguments.copy()
    
    # Start background task
    background_tasks.add_task(run_agent_background, job_id, tool_args)
    
    logger.info(f"[API] Started background job: {job_id}")
    
    return {
        "success": True,
        "job_id": job_id,
        "message": "Pipeline started in background",
        "status_url": f"/agent/status/{job_id}"
    }


class ConferencePipelineRequest(BaseModel):
    """Request body for conference pipeline."""
    source: str  # 'arxiv' or 'neurips'
    query: str
    top_k: int = 3
    goal: str = "general understanding"
    analysis_mode: str = "quick"
    discord_webhook_full: str = ""
    discord_webhook_summary: str = ""
    profile_path: str = "users/profile.json"


@app.post("/agent/conference-pipeline")
async def run_conference_pipeline(background_tasks: BackgroundTasks, request: ConferencePipelineRequest):
    """
    Start the conference pipeline (arXiv/NeurIPS) in the background.
    Searches, ranks, downloads, extracts, and analyzes papers.
    Returns immediately with job_id.
    """
    # Validate source
    if request.source not in ("arxiv", "neurips"):
        raise HTTPException(status_code=400, detail="Source must be 'arxiv' or 'neurips'")
    
    job_id = f"conf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Build tool arguments
    tool_args = {
        "source": request.source,
        "query": request.query,
        "top_k": min(max(request.top_k, 1), 5),  # Clamp between 1 and 5
        "goal": request.goal,
        "analysis_mode": request.analysis_mode,
        "discord_webhook_full": request.discord_webhook_full,
        "discord_webhook_summary": request.discord_webhook_summary,
        "profile_path": request.profile_path,
    }
    
    # Start background task
    background_tasks.add_task(run_conference_pipeline_background, job_id, tool_args)
    
    logger.info(f"[API] Started conference pipeline: {job_id} (source={request.source}, query='{request.query}')")
    
    return {
        "success": True,
        "job_id": job_id,
        "source": request.source,
        "query": request.query,
        "top_k": tool_args["top_k"],
        "message": f"Conference pipeline started for {request.source}",
        "status_url": f"/agent/status/{job_id}"
    }


@app.get("/agent/status/{job_id}")
async def get_agent_status(job_id: str):
    """
    Get the current status of a running or completed pipeline job.
    """
    status_file = STATUS_DIR / f"{job_id}.json"
    
    if not status_file.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    try:
        with open(status_file, "r", encoding="utf-8") as f:
            status_data = json.load(f)
        
        # If completed, try to get final result
        result_data = None
        if status_data.get("status") == "completed":
            # Try to find the report file
            report_dir = OUTPUT_DIR / "agent_reports"
            if report_dir.exists():
                # Find the most recent report (could be improved with job_id mapping)
                reports = sorted(report_dir.glob("research_report_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
                if reports:
                    result_data = {
                        "report_path": str(reports[0]),
                        "report_exists": True
                    }
        
        return {
            "success": True,
            "job_id": job_id,
            "status": status_data.get("status", "unknown"),
            "current_step": status_data.get("current_step", ""),
            "progress_percent": status_data.get("progress_percent", 0.0),
            "papers": status_data.get("papers", []),
            "current_paper_idx": status_data.get("current_paper_idx", 0),
            "paper_results_count": status_data.get("paper_results_count", 0),
            "reasoning_log_count": status_data.get("reasoning_log_count", 0),
            "errors": status_data.get("errors", []),
            "created_at": status_data.get("created_at"),
            "updated_at": status_data.get("updated_at"),
            "result": result_data
        }
    except Exception as e:
        logger.error(f"Failed to read status file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read status: {str(e)}")


@app.get("/agent/jobs")
async def list_agent_jobs():
    """
    List all agent jobs (running and completed).
    """
    jobs = []
    
    if not STATUS_DIR.exists():
        return {"jobs": []}
    
    try:
        for status_file in STATUS_DIR.glob("*.json"):
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    status_data = json.load(f)
                jobs.append({
                    "job_id": status_data.get("job_id", status_file.stem),
                    "status": status_data.get("status", "unknown"),
                    "goal": status_data.get("goal", ""),
                    "papers_count": len(status_data.get("papers", [])),
                    "progress_percent": status_data.get("progress_percent", 0.0),
                    "created_at": status_data.get("created_at"),
                    "updated_at": status_data.get("updated_at"),
                })
            except Exception as e:
                logger.warning(f"Failed to read {status_file}: {e}")
        
        # Sort by updated_at descending (most recent first)
        jobs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return {"jobs": jobs}
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        return {"jobs": []}


# ============== Rank Filter API Endpoints ==============

@app.get("/rank-filter/profile")
async def get_user_profile(profile_path: str = "users/profile.json"):
    """
    Get user profile from JSON file.
    """
    try:
        from .tools.rank_filter_utils import load_profile
        profile = load_profile(profile_path, tool_name="get_user_profile")
        return {"profile": profile}
    except Exception as e:
        logger.error(f"Failed to load profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load profile: {str(e)}")


class UpdateProfileRequest(BaseModel):
    """Request body for profile update."""
    profile_path: str = "users/profile.json"
    interests: Optional[Dict[str, List[str]]] = None
    keywords: Optional[Dict[str, Any]] = None
    exclude_local_papers: Optional[bool] = None
    purpose: Optional[str] = None
    ranking_mode: Optional[str] = None
    top_k: Optional[int] = None
    include_contrastive: Optional[bool] = None
    contrastive_type: Optional[str] = None
    preferred_authors: Optional[List[str]] = None
    preferred_institutions: Optional[List[str]] = None
    constraints: Optional[Dict[str, Any]] = None


@app.post("/rank-filter/profile")
async def update_user_profile(request: UpdateProfileRequest):
    """
    Update user profile.
    """
    try:
        result = await execute_tool("update_user_profile", **request.model_dump(exclude_none=True))
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result["result"]
    except Exception as e:
        logger.error(f"Failed to update profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {str(e)}")


class ArxivSearchForRankingRequest(BaseModel):
    """Request body for arXiv search for ranking."""
    query: str
    max_results: int = 50


@app.post("/arxiv/search-for-ranking")
async def search_arxiv_for_ranking(request: ArxivSearchForRankingRequest):
    """
    Search arXiv and convert results to PaperInput format for ranking pipeline.
    """
    try:
        # Search arXiv
        search_result = await execute_tool("arxiv_search", query=request.query, max_results=request.max_results)
        if not search_result["success"]:
            raise HTTPException(status_code=500, detail=search_result.get("error"))
        
        # Convert to PaperInput format
        papers = []
        for paper in search_result["result"]["papers"]:
            # Extract paper ID from entry_id (format: "http://arxiv.org/abs/2301.07041")
            paper_id = paper["id"].split("/")[-1] if "/" in paper["id"] else paper["id"]
            
            papers.append({
                "paper_id": paper_id,
                "title": paper["title"],
                "abstract": paper["summary"],
                "authors": paper["authors"],
                "published": paper.get("published"),
                "categories": paper.get("categories", []),
                "pdf_url": paper.get("pdf_url"),
                "github_url": None,  # Will be searched later if needed
            })
        
        return {
            "query": request.query,
            "total_results": len(papers),
            "papers": papers
        }
    except Exception as e:
        logger.error(f"Failed to search arXiv for ranking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search arXiv: {str(e)}")


class ExecutePipelineRequest(BaseModel):
    """Request body for rank-filter pipeline execution."""
    query: str
    max_results: int = 50
    purpose: str = "general"
    ranking_mode: str = "balanced"
    top_k: int = 5
    include_contrastive: bool = False
    contrastive_type: str = "method"
    profile_path: str = "users/profile.json"


@app.post("/rank-filter/execute-pipeline")
async def execute_rank_filter_pipeline(request: ExecutePipelineRequest):
    """
    Execute the full rank and filter pipeline.
    """
    try:
        # Step 1: Search arXiv
        search_result = await execute_tool("arxiv_search", query=request.query, max_results=request.max_results)
        if not search_result["success"]:
            raise HTTPException(status_code=500, detail=search_result.get("error"))
        
        # Convert to PaperInput format
        papers = []
        for paper in search_result["result"]["papers"]:
            paper_id = paper["id"].split("/")[-1] if "/" in paper["id"] else paper["id"]
            papers.append({
                "paper_id": paper_id,
                "title": paper["title"],
                "abstract": paper["summary"],
                "authors": paper["authors"],
                "published": paper.get("published"),
                "categories": paper.get("categories", []),
                "pdf_url": paper.get("pdf_url"),
                "github_url": None,
            })
        
        if not papers:
            return {
                "success": True,
                "ranked_papers": [],
                "message": "No papers found"
            }
        
        # Step 2: Apply hard filters
        filter_result = await execute_tool("apply_hard_filters", papers=papers, profile_path=request.profile_path, purpose=request.purpose)
        if not filter_result["success"]:
            raise HTTPException(status_code=500, detail=filter_result.get("error"))
        
        passed_papers = filter_result["result"].get("passed_papers", [])
        if not passed_papers:
            return {
                "success": True,
                "ranked_papers": [],
                "message": "All papers were filtered out"
            }
        
        # Step 3: Calculate semantic scores
        semantic_result = await execute_tool("calculate_semantic_scores", papers=passed_papers, profile_path=request.profile_path)
        if not semantic_result["success"]:
            raise HTTPException(status_code=500, detail=semantic_result.get("error"))
        
        semantic_scores = semantic_result["result"].get("scores", {})
        
        # Step 4: Evaluate metrics
        metrics_result = await execute_tool("evaluate_paper_metrics", papers=passed_papers, semantic_scores=semantic_scores, profile_path=request.profile_path)
        if not metrics_result["success"]:
            raise HTTPException(status_code=500, detail=metrics_result.get("error"))
        
        metrics_scores = metrics_result["result"].get("scores", {})
        
        # Step 5: Rank and select top K
        rank_result = await execute_tool(
            "rank_and_select_top_k",
            papers=passed_papers,
            semantic_scores=semantic_scores,
            metrics_scores=metrics_scores,
            top_k=request.top_k,
            purpose=request.purpose,
            ranking_mode=request.ranking_mode,
            include_contrastive=request.include_contrastive,
            contrastive_type=request.contrastive_type,
            profile_path=request.profile_path
        )
        if not rank_result["success"]:
            raise HTTPException(status_code=500, detail=rank_result.get("error"))
        
        return {
            "success": True,
            "ranked_papers": rank_result["result"].get("ranked_papers", []),
            "contrastive_paper": rank_result["result"].get("contrastive_paper"),
            "comparison_notes": rank_result["result"].get("comparison_notes", []),
            "summary": rank_result["result"].get("summary", {})
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


# ============== Paper Delete API ==============

PDF_DIR = Path(os.getenv("PDF_DIR", "/data/pdf"))
GRAPH_DIR = OUTPUT_DIR / "graph"
PAPER_GRAPH_DIR = GRAPH_DIR / "paper"


@app.delete("/paper/{paper_id:path}")
async def delete_paper(paper_id: str):
    """
    Delete a paper and all associated data:
    - output/{paper_id}/ folder (extracted text, images, references, reports, notes)
    - PDF file from pdf/ directory (including subdirectories)
    - Paper-specific graph cache & refs cache from output/graph/paper/
    - Node from global_graph.json
    - Node color from node_colors.json
    """
    deleted = []
    errors = []

    # 1) Delete output/{paper_id}/ folder
    paper_output_dir = OUTPUT_DIR / paper_id
    if paper_output_dir.exists() and paper_output_dir.is_dir():
        try:
            shutil.rmtree(str(paper_output_dir))
            deleted.append(f"output/{paper_id}/")
            logger.info(f"[Delete] Removed output folder: {paper_output_dir}")
        except Exception as e:
            errors.append(f"Failed to delete output folder: {e}")
            logger.error(f"[Delete] Failed to remove output folder: {e}")

    # 2) Delete PDF file(s) matching the paper_id (recursive search in subdirectories)
    if PDF_DIR.exists():
        for pdf_file in PDF_DIR.rglob("*.pdf"):
            if paper_id in pdf_file.stem:
                try:
                    rel_path = pdf_file.relative_to(PDF_DIR)
                    pdf_file.unlink()
                    deleted.append(f"pdf/{rel_path}")
                    logger.info(f"[Delete] Removed PDF: {pdf_file}")
                except Exception as e:
                    errors.append(f"Failed to delete PDF {pdf_file.name}: {e}")

    # 3) Delete paper-specific graph cache + refs cache
    safe_id = paper_id.replace("/", "_")
    if PAPER_GRAPH_DIR.exists():
        for cache_file in PAPER_GRAPH_DIR.iterdir():
            if cache_file.is_file() and safe_id in cache_file.stem:
                try:
                    cache_file.unlink()
                    deleted.append(f"graph/paper/{cache_file.name}")
                    logger.info(f"[Delete] Removed graph cache: {cache_file}")
                except Exception as e:
                    errors.append(f"Failed to delete graph cache {cache_file.name}: {e}")

    # 4) Remove node from global_graph.json
    global_graph_file = GRAPH_DIR / "global_graph.json"
    if global_graph_file.exists():
        try:
            with open(global_graph_file, "r", encoding="utf-8") as f:
                graph_data = json.load(f)

            original_node_count = len(graph_data.get("nodes", []))
            # Filter out nodes matching the paper_id
            graph_data["nodes"] = [
                n for n in graph_data.get("nodes", [])
                if n.get("id") != paper_id
            ]
            # Filter out edges referencing the paper_id
            graph_data["edges"] = [
                e for e in graph_data.get("edges", [])
                if e.get("source") != paper_id and e.get("target") != paper_id
            ]

            removed_nodes = original_node_count - len(graph_data["nodes"])
            if removed_nodes > 0:
                # Update meta
                if "meta" in graph_data:
                    graph_data["meta"]["total_papers"] = len(graph_data["nodes"])
                    graph_data["meta"]["total_edges"] = len(graph_data["edges"])

                with open(global_graph_file, "w", encoding="utf-8") as f:
                    json.dump(graph_data, f, ensure_ascii=False, indent=2)
                deleted.append("global_graph.json (node removed)")
                logger.info(f"[Delete] Removed node from global graph: {paper_id}")
        except Exception as e:
            errors.append(f"Failed to update global graph: {e}")

    # 5) Remove color from node_colors.json
    node_colors_file = GRAPH_DIR / "node_colors.json"
    if node_colors_file.exists():
        try:
            with open(node_colors_file, "r", encoding="utf-8") as f:
                colors_data = json.load(f)

            # Try multiple key formats
            keys_to_remove = [
                paper_id,
                f"paper:{paper_id}",
                f"paper:10.48550_arxiv.{paper_id}",
            ]
            removed_any = False
            for key in keys_to_remove:
                if key in colors_data:
                    del colors_data[key]
                    removed_any = True

            if removed_any:
                with open(node_colors_file, "w", encoding="utf-8") as f:
                    json.dump(colors_data, f, ensure_ascii=False, indent=2)
                deleted.append("node_colors.json (color removed)")
        except Exception as e:
            errors.append(f"Failed to update node colors: {e}")

    if not deleted and not errors:
        raise HTTPException(status_code=404, detail=f"No data found for paper: {paper_id}")

    return {
        "success": len(errors) == 0,
        "paper_id": paper_id,
        "deleted": deleted,
        "errors": errors,
    }


# ============== Main (실행 코드는 파일 맨 끝에 딱 한 번만!) ==============


def main():
    """Run the MCP server."""
    import uvicorn

    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    logger.info(f"Starting MCP server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
