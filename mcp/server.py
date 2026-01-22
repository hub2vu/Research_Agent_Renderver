#!/usr/bin/env python3
"""
MCP Server Entrypoint

HTTP API server that exposes all registered MCP tools.
Tools are automatically discovered via registry.py
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from pydantic import BaseModel, field_validator

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: discover tools
    tools = get_all_tools()
    logger.info(f"MCP Server starting with {len(tools)} tools")
    for name in tools:
        logger.info(f"  - {name}")
    yield
    # Shutdown
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
    """API root - service info."""
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
    """Health check endpoint."""
    return {"status": "healthy", "tools_loaded": len(list_tool_names())}


@app.get("/tools")
async def list_tools():
    """List all available tools."""
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
    """Get tools in OpenAI function calling format."""
    return {"tools": get_openai_tools_schema()}


@app.get("/tools/{tool_name}")
async def get_tool_info(tool_name: str):
    """Get information about a specific tool."""
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
    """Execute a specific tool with given arguments."""
    result = await execute_tool(tool_name, **request.arguments)
    return ToolResponse(**result)


# ============== Convenience Endpoints ==============
# These provide direct access to common tools


@app.get("/pdf/list")
async def list_pdfs():
    """List all PDF files."""
    result = await execute_tool("list_pdfs")
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result["result"]


@app.post("/pdf/extract")
async def extract_pdf(filename: str):
    """Extract text and images from a PDF."""
    result = await execute_tool("extract_all", filename=filename)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result["result"]


@app.get("/pdf/process-all")
async def process_all_pdfs():
    """Process all PDFs in the directory."""
    result = await execute_tool("process_all_pdfs")
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result["result"]


@app.get("/arxiv/search")
async def search_arxiv(query: str, max_results: int = 10):
    """Search arXiv for papers."""
    result = await execute_tool("arxiv_search", query=query, max_results=max_results)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result["result"]


@app.get("/web/search")
async def web_search(query: str, max_results: int = 5):
    """Search the web."""
    result = await execute_tool("web_search", query=query, max_results=max_results)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result["result"]


#  [추가] 파일 저장 경로 정의
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))


#  [추가] 웹사이트가 리포트를 달라고 할 때 처리하는 함수
@app.get("/reports/{paper_id}")
async def get_report_content(paper_id: str):
    """
    웹 프론트엔드용 리포트 조회 API
    (ID가 달라도 폴더를 끝까지 찾아내는 강력한 탐색 모드)
    """
    logger.info(f"🔍 [API Request] 리포트 요청 ID: {paper_id}")

    # 1. [1차 시도] 정확한 폴더 찾기
    target_dir = OUTPUT_DIR / paper_id

    # 2. [2차 시도] 정확한 폴더가 없으면, '유사한' 폴더 찾기 (탐정 모드 🕵️‍♂️)
    if not target_dir.exists():
        # ID에서 핵심 숫자만 추출 (예: 10.48550_arxiv.1809.04281 -> 1809.04281)
        core_id = paper_id
        if "arxiv." in paper_id:
            core_id = paper_id.split("arxiv.")[-1]

        logger.info(
            f"⚠️ 정확한 폴더 없음. 핵심 ID '{core_id}'가 포함된 폴더를 검색합니다..."
        )

        # output 폴더 안의 모든 하위 폴더를 하나씩 검사
        found = False
        try:
            for folder in OUTPUT_DIR.iterdir():
                if folder.is_dir():
                    # 요청 ID가 폴더명에 포함되거나, 폴더명이 요청 ID에 포함되면 '찾았다!' 처리
                    if core_id in folder.name or folder.name in paper_id:
                        target_dir = folder
                        found = True
                        logger.info(f"✅ 유사 폴더 발견: {target_dir}")
                        break
        except Exception as e:
            logger.error(f"폴더 검색 중 에러 발생: {e}")

        if not found:
            logger.error(
                f"❌ 실패: {paper_id} 또는 {core_id}와 일치하는 폴더가 없습니다."
            )
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Folder not found",
                    "detail": f"Could not find folder for {paper_id}",
                },
            )

    # 3. 파일 찾기 (.md 우선, 없으면 .txt)
    md_file = target_dir / "summary_report.md"
    txt_file = target_dir / "summary_report.txt"

    final_file = None
    if md_file.exists():
        final_file = md_file
    elif txt_file.exists():
        final_file = txt_file

    # 4. 내용 읽어서 리턴
    if final_file:
        try:
            with open(final_file, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"📤 리포트 전송 완료: {final_file.name}")
            return {"content": content}
        except Exception as e:
            return JSONResponse(
                status_code=500, content={"error": f"Read error: {str(e)}"}
            )
    else:
        logger.warning(f"❌ 폴더는 찾았으나 리포트 파일이 없음: {target_dir}")
        return JSONResponse(
            status_code=404,
            content={
                "error": "Report file missing",
                "detail": "Folder exists but summary_report.md or .txt is missing.",
            },
        )


# ============== Main ==============


def main():
    """Run the MCP server."""
    import uvicorn

    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))

    logger.info(f"Starting MCP server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
