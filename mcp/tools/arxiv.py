"""
arXiv Tools

Provides tools for searching and fetching papers from arXiv.
Uses the arxiv Python API.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from ..base import MCPTool, ToolParameter, ExecutionError

# Try to import arxiv library
try:
    import arxiv
    HAS_ARXIV = True
except ImportError:
    HAS_ARXIV = False


class ArxivSearchTool(MCPTool):
    """Search for papers on arXiv."""

    @property
    def name(self) -> str:
        return "arxiv_search"

    @property
    def description(self) -> str:
        return "Search arXiv for academic papers matching a query"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Search query (e.g., 'transformer attention mechanism')",
                required=True
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Maximum number of results to return",
                required=False,
                default=10
            ),
            ToolParameter(
                name="sort_by",
                type="string",
                description="Sort order: 'relevance', 'lastUpdatedDate', 'submittedDate'",
                required=False,
                default="relevance"
            )
        ]

    @property
    def category(self) -> str:
        return "arxiv"

    async def execute(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance"
    ) -> Dict[str, Any]:
        if not HAS_ARXIV:
            raise ExecutionError(
                "arxiv library not installed. Run: pip install arxiv",
                tool_name=self.name
            )

        # Limit max_results to 30 to avoid rate limiting
        # arXiv API has rate limits and pagination causes multiple requests
        # When max_results > 30, arxiv library uses pagination which triggers rate limits
        max_results = min(max_results, 30)

        # Map sort option
        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate
        }
        sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)

        # Add delay before request to avoid rate limiting (arXiv recommends 3 seconds between requests)
        await asyncio.sleep(3)

        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criterion
            )

            results = []
            for paper in search.results():
                results.append({
                    "id": paper.entry_id,
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "summary": paper.summary,
                    "published": paper.published.isoformat() if paper.published else None,
                    "updated": paper.updated.isoformat() if paper.updated else None,
                    "categories": paper.categories,
                    "pdf_url": paper.pdf_url,
                    "primary_category": paper.primary_category
                })

            return {
                "query": query,
                "total_results": len(results),
                "papers": results
            }

        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error
            if "429" in error_str or "rate limit" in error_str.lower():
                raise ExecutionError(
                    f"arXiv API rate limit exceeded. Please wait a few minutes and try again. "
                    f"Error: {error_str}",
                    tool_name=self.name
                )
            raise ExecutionError(f"arXiv search failed: {error_str}", tool_name=self.name)


class ArxivGetPaperTool(MCPTool):
    """Get detailed information about a specific arXiv paper."""

    @property
    def name(self) -> str:
        return "arxiv_get_paper"

    @property
    def description(self) -> str:
        return "Get detailed information about a specific arXiv paper by ID"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="arXiv paper ID (e.g., '2301.07041' or full URL)",
                required=True
            )
        ]

    @property
    def category(self) -> str:
        return "arxiv"

    async def execute(self, paper_id: str) -> Dict[str, Any]:
        if not HAS_ARXIV:
            raise ExecutionError(
                "arxiv library not installed. Run: pip install arxiv",
                tool_name=self.name
            )

        # Extract ID from URL if needed
        if "arxiv.org" in paper_id:
            paper_id = paper_id.split("/")[-1]
            if paper_id.endswith(".pdf"):
                paper_id = paper_id[:-4]

        try:
            search = arxiv.Search(id_list=[paper_id])
            paper = next(search.results(), None)

            if paper is None:
                raise ExecutionError(f"Paper not found: {paper_id}", tool_name=self.name)

            return {
                "id": paper.entry_id,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary,
                "published": paper.published.isoformat() if paper.published else None,
                "updated": paper.updated.isoformat() if paper.updated else None,
                "categories": paper.categories,
                "primary_category": paper.primary_category,
                "pdf_url": paper.pdf_url,
                "doi": paper.doi,
                "links": [{"title": link.title, "href": link.href} for link in paper.links],
                "comment": paper.comment,
                "journal_ref": paper.journal_ref
            }

        except StopIteration:
            raise ExecutionError(f"Paper not found: {paper_id}", tool_name=self.name)
        except Exception as e:
            raise ExecutionError(f"Failed to fetch paper: {str(e)}", tool_name=self.name)


class ArxivDownloadTool(MCPTool):
    """Download a paper from arXiv."""

    @property
    def name(self) -> str:
        return "arxiv_download"

    @property
    def description(self) -> str:
        return "Download a PDF from arXiv to the pdf directory"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="arXiv paper ID (e.g., '2301.07041')",
                required=True
            ),
            ToolParameter(
                name="filename",
                type="string",
                description="Custom filename (optional, without .pdf extension)",
                required=False,
                default=None
            )
        ]

    @property
    def category(self) -> str:
        return "arxiv"

    async def execute(
        self,
        paper_id: str,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        if not HAS_ARXIV:
            raise ExecutionError(
                "arxiv library not installed. Run: pip install arxiv",
                tool_name=self.name
            )

        from pathlib import Path
        pdf_dir = Path(os.getenv("PDF_DIR", "/data/pdf"))
        pdf_dir.mkdir(parents=True, exist_ok=True)

        # Extract ID from URL if needed
        if "arxiv.org" in paper_id:
            paper_id = paper_id.split("/")[-1]
            if paper_id.endswith(".pdf"):
                paper_id = paper_id[:-4]

        try:
            search = arxiv.Search(id_list=[paper_id])
            paper = next(search.results(), None)

            if paper is None:
                raise ExecutionError(f"Paper not found: {paper_id}", tool_name=self.name)

            # Determine filename
            if filename:
                save_filename = f"{filename}.pdf"
            else:
                # Use paper ID as filename
                save_filename = f"{paper_id.replace('/', '_')}.pdf"

            save_path = pdf_dir / save_filename

            # Download the paper
            paper.download_pdf(dirpath=str(pdf_dir), filename=save_filename)

            return {
                "success": True,
                "paper_id": paper_id,
                "title": paper.title,
                "saved_to": str(save_path),
                "filename": save_filename
            }

        except Exception as e:
            raise ExecutionError(f"Download failed: {str(e)}", tool_name=self.name)
