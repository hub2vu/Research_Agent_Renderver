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

from .registry import (
    execute_tool,
    get_all_tools,
    get_openai_tools_schema,
    list_tool_names,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    lifespan=lifespan
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
            "health": "/health"
        }
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
                        "required": p.required
                    }
                    for p in tool.parameters
                ]
            }
            for name, tool in tools.items()
        ]
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
                "default": p.default
            }
            for p in tool.parameters
        ]
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


# ============== Rank Filter Endpoints ==============

class ArxivSearchRequest(BaseModel):
    """Request body for arXiv search."""
    query: str
    max_results: int = 50


@app.post("/arxiv/search-for-ranking")
async def search_arxiv_for_ranking(request: ArxivSearchRequest):
    """
    Search arXiv and convert results to PaperInput format.
    Returns papers with abstract (from summary) and proper paper_id extraction.
    """
    # Search arXiv
    search_result = await execute_tool("arxiv_search", query=request.query, max_results=request.max_results)
    if not search_result["success"]:
        raise HTTPException(status_code=500, detail=search_result.get("error"))
    
    papers_data = search_result["result"].get("papers", [])
    
    # Convert to PaperInput format
    paper_inputs = []
    for paper in papers_data:
        # Extract paper_id from entry_id (e.g., "http://arxiv.org/abs/2501.12345" -> "2501.12345")
        entry_id = paper.get("id", "")
        paper_id = entry_id.split("/")[-1] if "/" in entry_id else entry_id
        
        # Convert summary to abstract
        paper_input = {
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "abstract": paper.get("summary", ""),  # summary -> abstract
            "authors": paper.get("authors", []),
            "published": paper.get("published"),  # Already in YYYY-MM-DD format
            "categories": paper.get("categories", []),
            "pdf_url": paper.get("pdf_url"),
            "github_url": None,  # Default to None
        }
        paper_inputs.append(paper_input)
    
    return {
        "query": request.query,
        "total_results": len(paper_inputs),
        "papers": paper_inputs
    }


class PipelineRequest(BaseModel):
    """Request body for pipeline execution."""
    query: str
    max_results: int = 50
    purpose: str = None
    ranking_mode: str = None
    top_k: int = None
    include_contrastive: bool = None
    contrastive_type: str = None


@app.post("/rank-filter/execute-pipeline")
async def execute_rank_filter_pipeline(request: PipelineRequest):
    """
    Execute the full rank and filter pipeline:
    1. Search arXiv
    2. Convert to PaperInput
    3. Load profile (with purpose, ranking_mode, etc.)
    4. Apply hard filters
    5. Calculate semantic scores
    6. Evaluate paper metrics
    7. Rank and select top-k
    """
    from .tools.rank_filter_utils.loaders import load_profile
    from .tools.rank_filter_utils.path_resolver import resolve_path
    
    try:
        # Step 1: Search arXiv and convert to PaperInput
        search_request = ArxivSearchRequest(query=request.query, max_results=request.max_results)
        search_result = await search_arxiv_for_ranking(search_request)
        papers = search_result["papers"]
        
        if not papers:
            return {
                "success": True,
                "summary": {
                    "input_count": 0,
                    "filtered_count": 0,
                    "scored_count": 0,
                    "output_count": 0,
                },
                "ranked_papers": [],
                "filtered_papers": [],
            }
        
        # Step 2: Load profile
        profile_path = "users/profile.json"
        profile = load_profile(profile_path, tool_name="execute_pipeline")
        
        # Override profile values with request parameters if provided
        purpose = request.purpose if request.purpose is not None else profile.get("purpose", "general")
        ranking_mode = request.ranking_mode if request.ranking_mode is not None else profile.get("ranking_mode", "balanced")
        top_k = request.top_k if request.top_k is not None else profile.get("top_k", 5)
        include_contrastive = request.include_contrastive if request.include_contrastive is not None else profile.get("include_contrastive", False)
        contrastive_type = request.contrastive_type if request.contrastive_type is not None else profile.get("contrastive_type", "method")
        
        # Step 3: Apply hard filters
        filter_result = await execute_tool(
            "apply_hard_filters",
            papers=papers,
            profile_path=profile_path,
            purpose=purpose
        )
        if not filter_result["success"]:
            raise HTTPException(status_code=500, detail=f"Filtering failed: {filter_result.get('error')}")
        
        passed_papers = filter_result["result"].get("passed_papers", [])
        filtered_papers = filter_result["result"].get("filtered_papers", [])
        
        if not passed_papers:
            return {
                "success": True,
                "summary": {
                    "input_count": len(papers),
                    "filtered_count": len(filtered_papers),
                    "scored_count": 0,
                    "output_count": 0,
                    "purpose": purpose,
                    "ranking_mode": ranking_mode,
                },
                "ranked_papers": [],
                "filtered_papers": filtered_papers,
            }
        
        # Step 4: Calculate semantic scores
        semantic_result = await execute_tool(
            "calculate_semantic_scores",
            papers=passed_papers,
            profile_path=profile_path,
            enable_llm_verification=True
        )
        if not semantic_result["success"]:
            raise HTTPException(status_code=500, detail=f"Semantic scoring failed: {semantic_result.get('error')}")
        
        semantic_scores = semantic_result["result"].get("scores", {})
        
        # Step 5: Evaluate paper metrics
        metrics_result = await execute_tool(
            "evaluate_paper_metrics",
            papers=passed_papers,
            semantic_scores=semantic_scores,
            profile_path=profile_path
        )
        if not metrics_result["success"]:
            raise HTTPException(status_code=500, detail=f"Metrics evaluation failed: {metrics_result.get('error')}")
        
        metrics_scores = metrics_result["result"].get("scores", {})
        
        # Step 6: Rank and select top-k
        rank_result = await execute_tool(
            "rank_and_select_top_k",
            papers=passed_papers,
            semantic_scores=semantic_scores,
            metrics_scores=metrics_scores,
            top_k=top_k,
            purpose=purpose,
            ranking_mode=ranking_mode,
            include_contrastive=include_contrastive,
            contrastive_type=contrastive_type,
            profile_path=profile_path
        )
        if not rank_result["success"]:
            raise HTTPException(status_code=500, detail=f"Ranking failed: {rank_result.get('error')}")
        
        return rank_result["result"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@app.get("/rank-filter/profile")
async def get_user_profile(profile_path: str = "users/profile.json"):
    """Get user profile."""
    from .tools.rank_filter_utils.loaders import load_profile
    
    try:
        profile = load_profile(profile_path, tool_name="get_profile")
        return {
            "success": True,
            "profile": profile,
            "profile_path": profile_path
        }
    except Exception as e:
        logger.error(f"Failed to load profile: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load profile: {str(e)}")


class ProfileUpdateRequest(BaseModel):
    """Request body for profile update."""
    profile_path: str = "users/profile.json"
    interests: Dict[str, Any] = None
    keywords: Dict[str, Any] = None
    exclude_local_papers: bool = None
    purpose: str = None
    ranking_mode: str = None
    top_k: int = None
    include_contrastive: bool = None
    contrastive_type: str = None
    preferred_authors: list = None
    preferred_institutions: list = None
    constraints: Dict[str, Any] = None


@app.post("/rank-filter/profile")
async def update_user_profile(request: ProfileUpdateRequest):
    """Update user profile."""
    update_args = {
        "profile_path": request.profile_path
    }
    
    if request.interests is not None:
        update_args["interests"] = request.interests
    if request.keywords is not None:
        update_args["keywords"] = request.keywords
    if request.exclude_local_papers is not None:
        update_args["exclude_local_papers"] = request.exclude_local_papers
    if request.purpose is not None:
        update_args["purpose"] = request.purpose
    if request.ranking_mode is not None:
        update_args["ranking_mode"] = request.ranking_mode
    if request.top_k is not None:
        update_args["top_k"] = request.top_k
    if request.include_contrastive is not None:
        update_args["include_contrastive"] = request.include_contrastive
    if request.contrastive_type is not None:
        update_args["contrastive_type"] = request.contrastive_type
    if request.preferred_authors is not None:
        update_args["preferred_authors"] = request.preferred_authors
    if request.preferred_institutions is not None:
        update_args["preferred_institutions"] = request.preferred_institutions
    if request.constraints is not None:
        update_args["constraints"] = request.constraints
    
    result = await execute_tool("update_user_profile", **update_args)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    
    return result["result"]


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
