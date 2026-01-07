"""
RAG QA Tools
Simple retrieval from extracted text files without heavy Vector DB.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from ..base import MCPTool, ToolParameter, ExecutionError

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))


class LocalPaperSearchTool(MCPTool):
    """Search for keywords inside a specific paper's extracted text."""

    @property
    def name(self) -> str:
        return "search_in_paper"

    @property
    def description(self) -> str:
        return "Search for specific keywords or phrases inside a processed paper."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="Paper ID or folder name",
                required=True,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="Keyword to search",
                required=True,
            ),
        ]

    async def execute(self, paper_id: str, query: str) -> Dict[str, Any]:
        target_dir = OUTPUT_DIR / paper_id
        text_file = target_dir / "extracted_text.txt"

        if not text_file.exists():
            return {"error": "Paper not processed yet. Run extract_text first."}

        found_lines = []
        try:
            with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if query.lower() in line.lower():
                        # 앞뒤 문맥 포함해서 저장
                        found_lines.append(f"Line {i+1}: {line.strip()}")
                        if len(found_lines) >= 5:
                            break  # 너무 많이 나오면 자름

            return {
                "paper_id": paper_id,
                "matches_count": len(found_lines),
                "snippets": found_lines,
            }
        except Exception as e:
            raise ExecutionError(f"Search failed: {e}", self.name)
