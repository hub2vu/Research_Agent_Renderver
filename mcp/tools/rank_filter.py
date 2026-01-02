"""
Rank and Filter Papers Tool

Provides tools for ranking and filtering research papers based on metadata
(title, authors, abstract) and user profile. This tool analyzes only abstracts;
full PDF text analysis is handled by paper_analyzer.
"""

from typing import Any, Dict, List, Optional

from ..base import MCPTool, ToolParameter, ExecutionError


class RankAndFilterPapersTool(MCPTool):
    """
    Rank and filter papers based on metadata analysis.
    
    This tool acts as a gatekeeper, performing lightweight evaluation based on
    metadata (title, authors, abstract) to pre-select papers before expensive
    PDF full-text analysis. It does NOT analyze PDF content - that is the role
    of paper_analyzer.
    """

    @property
    def name(self) -> str:
        return "rank_and_filter_papers"

    @property
    def description(self) -> str:
        return (
            "Analyzes search result metadata (title, authors, abstract) to evaluate "
            "and filter papers according to user profile. This tool analyzes only "
            "abstracts; PDF full-text analysis is handled by paper_analyzer. "
            "Use this tool after arxiv_search or web_search to pre-select the most "
            "relevant papers before expensive PDF processing."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="papers",
                type="array",
                items_type="object",
                description=(
                    "List of papers returned by search tools (e.g., arxiv_search, web_search). "
                    "Each paper object must have: paper_id (string), title (string), "
                    "abstract (string), authors (array). Optional fields: published (string), "
                    "categories (array), pdf_url (string), github_url (string), "
                    "affiliations (array). Use this after search tools to filter and rank results."
                ),
                required=True
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description=(
                    "Maximum number of papers to return after ranking. Recommended range: 1-20. "
                    "Use smaller values (3-5) for focused research, larger values (10-15) for "
                    "broad exploration or literature reviews."
                ),
                required=False,
                default=5
            ),
            ToolParameter(
                name="profile_path",
                type="string",
                description=(
                    "Path to user profile JSON file containing research interests, keywords, "
                    "preferred authors, and constraints. Relative paths are resolved relative to "
                    "OUTPUT_DIR environment variable. If file doesn't exist, basic scoring only "
                    "will be performed. Use this to personalize paper selection."
                ),
                required=False,
                default="config/profile.json"
            ),
            ToolParameter(
                name="purpose",
                type="string",
                description=(
                    "Current research purpose/goal. Choose 'general' for balanced recommendation, "
                    "'literature_review' for broad coverage and diversity (relaxes year constraints), "
                    "'implementation' to require code availability (github_url required), "
                    "'idea_generation' to prioritize novelty and recent papers with innovative approaches."
                ),
                required=False,
                default="general"
            ),
            ToolParameter(
                name="ranking_mode",
                type="string",
                description=(
                    "Scoring emphasis mode. Choose 'balanced' for equal weight on all factors, "
                    "'novelty' to emphasize recent papers with innovative methodologies, "
                    "'practicality' to emphasize code availability and reproducibility, "
                    "'diversity' to minimize overlap and encourage diverse perspectives."
                ),
                required=False,
                default="balanced"
            ),
            ToolParameter(
                name="include_contrastive",
                type="boolean",
                description=(
                    "If true, replace the last paper in Top-K with a contrastive paper that "
                    "presents different approach/assumption/domain. Use this to gain broader "
                    "perspective on the research topic. Requires contrastive_type to be specified."
                ),
                required=False,
                default=False
            ),
            ToolParameter(
                name="contrastive_type",
                type="string",
                description=(
                    "Type of contrastive paper to include. Choose 'method' for same problem with "
                    "different methodology, 'assumption' for same method with different assumptions, "
                    "'domain' for same technique applied to different domain. Only used when "
                    "include_contrastive is true."
                ),
                required=False,
                default="method"
            ),
            ToolParameter(
                name="history_path",
                type="string",
                description=(
                    "Path to JSON file containing list of already-read paper IDs. Papers in this "
                    "list will be automatically filtered out. Relative paths resolved relative to "
                    "OUTPUT_DIR. If file doesn't exist, treated as empty list. Use this to avoid "
                    "re-analyzing papers you've already reviewed."
                ),
                required=False,
                default="history/read_papers.json"
            ),
            ToolParameter(
                name="local_pdf_dir",
                type="string",
                description=(
                    "Local directory path containing downloaded PDF files. Relative paths resolved "
                    "relative to PDF_DIR environment variable. Used to check which papers are "
                    "already available locally and tag them accordingly. Useful for prioritizing "
                    "papers you already have."
                ),
                required=False,
                default="pdf/"
            ),
            ToolParameter(
                name="enable_llm_verification",
                type="boolean",
                description=(
                    "If true, performs LLM batch verification on borderline papers (embedding score "
                    "0.4-0.7) to improve accuracy. If false, uses embedding scores only (faster but "
                    "less accurate). Set to false when processing large batches or when speed is "
                    "critical. Recommended: true for top_k <= 10, false for top_k > 15."
                ),
                required=False,
                default=True
            )
        ]

    @property
    def category(self) -> str:
        return "ranking"

    async def execute(
        self,
        papers: List[Dict[str, Any]],
        top_k: int = 5,
        profile_path: str = "config/profile.json",
        purpose: str = "general",
        ranking_mode: str = "balanced",
        include_contrastive: bool = False,
        contrastive_type: str = "method",
        history_path: Optional[str] = "history/read_papers.json",
        local_pdf_dir: str = "pdf/",
        enable_llm_verification: bool = True
    ) -> Dict[str, Any]:
        """
        Execute ranking and filtering logic.
        
        TODO: Implement the ranking and filtering algorithm.
        """
        # Placeholder implementation
        raise ExecutionError(
            "rank_and_filter_papers execute() method not yet implemented",
            tool_name=self.name
        )

