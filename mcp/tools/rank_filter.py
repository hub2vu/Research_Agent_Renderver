"""
Rank and Filter Papers Tool

Provides tools for ranking and filtering research papers based on metadata
(title, authors, abstract) and user profile. This tool analyzes only abstracts;
full PDF text analysis is handled by paper_analyzer.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TypedDict

from ..base import MCPTool, ToolParameter, ExecutionError


# Type Definitions

class PaperInput(TypedDict, total=False):
    """Input paper object structure from search tools."""
    paper_id: str  # Required: arXiv ID (e.g., "2501.12345")
    title: str  # Required: Paper title
    abstract: str  # Required: Abstract
    authors: List[str]  # Required: List of authors
    published: str  # Optional: Publication date (YYYY-MM-DD)
    categories: List[str]  # Optional: arXiv categories
    pdf_url: str  # Optional: PDF download link
    github_url: str  # Optional: GitHub repository link
    affiliations: List[str]  # Optional: Author affiliations


class UserProfileInterests(TypedDict):
    """User interests structure."""
    primary: List[str]  # Core research topics
    secondary: List[str]  # Related topics
    exploratory: List[str]  # Exploratory topics


class UserProfileKeywordsExclude(TypedDict):
    """Keywords exclude structure."""
    hard: List[str]  # Must exclude keywords
    soft: List[str]  # Soft penalty keywords


class UserProfileKeywords(TypedDict):
    """Keywords structure."""
    must_include: List[str]  # Must include keywords
    exclude: UserProfileKeywordsExclude  # Exclude keywords


class UserProfileConstraints(TypedDict):
    """User constraints structure."""
    min_year: int  # Minimum publication year
    require_code: bool  # Code availability required


class UserProfile(TypedDict):
    """User profile structure from profile.json."""
    interests: UserProfileInterests
    keywords: UserProfileKeywords
    preferred_authors: List[str]
    preferred_institutions: List[str]
    constraints: UserProfileConstraints


class ScoreBreakdown(TypedDict):
    """Score breakdown structure."""
    semantic_relevance: float  # 0.0-1.0
    must_keywords: float  # 0.0-1.0
    author_trust: float  # 0.0-1.0
    institution_trust: float  # 0.0-1.0
    recency: float  # 0.0-1.0
    practicality: float  # 0.0-1.0


class ScoreInfo(TypedDict):
    """Score information structure."""
    final: float  # Final score (0.0-1.0)
    breakdown: ScoreBreakdown
    soft_penalty: float  # Soft penalty value (negative)
    penalty_keywords: List[str]  # Keywords that caused penalty
    evaluation_method: str  # "embedding+llm" or "embedding_only"


class LocalStatus(TypedDict):
    """Local file status structure."""
    already_downloaded: bool
    local_path: Optional[str]  # Path to local PDF file or None


class ScoredPaper(TypedDict):
    """Scored paper structure after ranking."""
    rank: int  # Ranking position (1-based)
    paper_id: str
    title: str
    authors: List[str]
    published: Optional[str]
    score: ScoreInfo
    tags: List[str]  # e.g., ["SEMANTIC_HIGH_MATCH", "CODE_AVAILABLE"]
    local_status: LocalStatus
    original_data: Dict[str, Any]  # Original paper input data


class FilteredPaper(TypedDict):
    """Filtered paper structure for removed papers."""
    paper_id: str
    title: str
    filter_reason: str  # e.g., "BLACKLIST_KEYWORD:medical", "ALREADY_READ"
    filter_phase: int  # Phase number where filtering occurred


class ContrastiveInfo(TypedDict):
    """Contrastive paper information structure."""
    type: str  # "method", "assumption", or "domain"
    selected_papers_common_traits: List[str]
    this_paper_traits: List[str]
    contrast_dimensions: List[Dict[str, str]]  # e.g., [{"dimension": "architecture", "others": "Transformer", "this": "SSM"}]


class ContrastivePaper(TypedDict):
    """Contrastive paper structure."""
    paper_id: str
    title: str
    authors: List[str]
    published: Optional[str]
    score: ScoreInfo
    tags: List[str]  # Includes "CONTRASTIVE_PICK", "CONTRASTIVE_METHOD", etc.
    contrastive_info: ContrastiveInfo
    original_data: Dict[str, Any]


class ComparisonNote(TypedDict):
    """Comparison note structure."""
    paper_ids: List[str]
    relation: str  # e.g., "similar_approach", "contrastive"
    shared_traits: Optional[List[str]]
    differentiator: Optional[str]  # Brief description of differences
    contrast_point: Optional[str]  # For contrastive relations


class ToolResultSummary(TypedDict):
    """Tool result summary structure."""
    input_count: int  # Number of input papers
    filtered_count: int  # Number of filtered papers
    scored_count: int  # Number of papers that were scored
    output_count: int  # Final number of papers returned
    purpose: str  # Purpose used: "general", "literature_review", etc.
    ranking_mode: str  # Ranking mode used: "balanced", "novelty", etc.
    profile_used: Optional[str]  # Profile path or None
    llm_verification_used: bool
    llm_calls_made: int  # Number of LLM batch calls made


class ToolResult(TypedDict, total=False):
    """Final tool result structure."""
    success: bool
    error: Optional[str]  # Error message if success is False
    summary: ToolResultSummary
    ranked_papers: List[ScoredPaper]
    filtered_papers: List[FilteredPaper]
    contrastive_paper: Optional[ContrastivePaper]  # Only if include_contrastive is True
    comparison_notes: List[ComparisonNote]
    output_path: str  # Path to saved result file
    generated_at: str  # ISO timestamp


# Utility Functions

def resolve_path(path: str, path_type: str = "output") -> Path:
    """
    Resolve a path string to a Path object.
    
    If path is absolute, return it as-is.
    If path is relative:
    - For path_type="output": resolve relative to OUTPUT_DIR environment variable
    - For path_type="pdf": resolve relative to PDF_DIR (or OUTPUT_DIR if PDF_DIR not set)
    - If environment variable not set: resolve relative to current working directory
    
    Args:
        path: Path string (absolute or relative)
        path_type: Type of path - "output" or "pdf"
        
    Returns:
        Path object with resolved path
    """
    path_obj = Path(path)
    
    # If absolute path, return as-is
    if path_obj.is_absolute():
        return path_obj
    
    # Resolve relative path based on path_type
    if path_type == "output":
        base_dir = os.getenv("OUTPUT_DIR")
        if base_dir:
            return Path(base_dir) / path
        else:
            return Path.cwd() / path
    elif path_type == "pdf":
        base_dir = os.getenv("PDF_DIR")
        if base_dir:
            return Path(base_dir) / path
        else:
            # Fallback to OUTPUT_DIR if PDF_DIR not set
            output_dir = os.getenv("OUTPUT_DIR")
            if output_dir:
                return Path(output_dir) / path
            else:
                return Path.cwd() / path
    else:
        # Unknown path_type, use current working directory
        return Path.cwd() / path


def ensure_directory(path: Path) -> None:
    """
    Ensure that the directory exists, creating it if necessary.
    
    If path points to a file, ensures the parent directory exists.
    If path points to a directory, ensures the directory itself exists.
    
    Args:
        path: Path object to ensure directory for
    """
    # If path has an extension or looks like a file, ensure parent directory
    if path.suffix or not path.name:  # Has extension or is empty name
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Assume it's a directory
        path.mkdir(parents=True, exist_ok=True)


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

    def _load_profile(self, profile_path: str) -> UserProfile:
        """
        Load user profile from JSON file.
        
        Args:
            profile_path: Path to profile JSON file
            
        Returns:
            UserProfile object with loaded data or default values
            
        Raises:
            ExecutionError: If file exists but has invalid format
        """
        resolved_path = resolve_path(profile_path, path_type="output")
        
        # If file doesn't exist, return default profile
        if not resolved_path.exists():
            return {
                "interests": {
                    "primary": [],
                    "secondary": [],
                    "exploratory": []
                },
                "keywords": {
                    "must_include": [],
                    "exclude": {
                        "hard": [],
                        "soft": []
                    }
                },
                "preferred_authors": [],
                "preferred_institutions": [],
                "constraints": {
                    "min_year": 2000,  # Default to allow all papers
                    "require_code": False
                }
            }
        
        # Load and validate JSON file
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Validate and build profile structure
            profile: UserProfile = {
                "interests": {
                    "primary": data.get("interests", {}).get("primary", []),
                    "secondary": data.get("interests", {}).get("secondary", []),
                    "exploratory": data.get("interests", {}).get("exploratory", [])
                },
                "keywords": {
                    "must_include": data.get("keywords", {}).get("must_include", []),
                    "exclude": {
                        "hard": data.get("keywords", {}).get("exclude", {}).get("hard", []),
                        "soft": data.get("keywords", {}).get("exclude", {}).get("soft", [])
                    }
                },
                "preferred_authors": data.get("preferred_authors", []),
                "preferred_institutions": data.get("preferred_institutions", []),
                "constraints": {
                    "min_year": data.get("constraints", {}).get("min_year", 2000),
                    "require_code": data.get("constraints", {}).get("require_code", False)
                }
            }
            
            return profile
            
        except json.JSONDecodeError as e:
            raise ExecutionError(
                f"Invalid JSON format in profile file: {resolved_path}. Error: {str(e)}",
                tool_name=self.name
            )
        except Exception as e:
            raise ExecutionError(
                f"Failed to load profile from {resolved_path}: {str(e)}",
                tool_name=self.name
            )

    def _load_history(self, history_path: str) -> Set[str]:
        """
        Load list of already-read paper IDs from JSON file.
        
        Args:
            history_path: Path to history JSON file
            
        Returns:
            Set of paper IDs. Empty set if file doesn't exist.
        """
        resolved_path = resolve_path(history_path, path_type="output")
        
        # If file doesn't exist, return empty set
        if not resolved_path.exists():
            return set()
        
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle different JSON structures:
            # - List of strings: ["2501.12345", "2501.12346"]
            # - List of objects with paper_id: [{"paper_id": "2501.12345"}, ...]
            # - Object with papers key: {"papers": ["2501.12345", ...]}
            paper_ids = set()
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        paper_ids.add(item)
                    elif isinstance(item, dict) and "paper_id" in item:
                        paper_ids.add(item["paper_id"])
            elif isinstance(data, dict):
                if "papers" in data:
                    papers = data["papers"]
                    if isinstance(papers, list):
                        for item in papers:
                            if isinstance(item, str):
                                paper_ids.add(item)
                            elif isinstance(item, dict) and "paper_id" in item:
                                paper_ids.add(item["paper_id"])
                elif "paper_ids" in data:
                    paper_ids = set(data["paper_ids"])
            
            return paper_ids
            
        except json.JSONDecodeError:
            # Invalid JSON, return empty set (don't fail the tool)
            return set()
        except Exception:
            # Any other error, return empty set
            return set()

    def _scan_local_pdfs(self, pdf_dir: str) -> Set[str]:
        """
        Scan local PDF directory and extract paper IDs from filenames.
        
        Args:
            pdf_dir: Path to PDF directory
            
        Returns:
            Set of paper IDs extracted from filenames. Empty set if directory doesn't exist.
        """
        resolved_path = resolve_path(pdf_dir, path_type="pdf")
        
        # If directory doesn't exist, return empty set
        if not resolved_path.exists() or not resolved_path.is_dir():
            return set()
        
        paper_ids = set()
        
        try:
            # Scan for PDF files
            for pdf_file in resolved_path.glob("*.pdf"):
                filename = pdf_file.stem  # Filename without extension
                
                # Extract paper_id from filename
                # Examples: "2501.12345.pdf" -> "2501.12345"
                #           "2501.12345_v2.pdf" -> "2501.12345"
                #           "10.48550_arxiv.2506.07976.pdf" -> "2506.07976" (extract from arxiv ID)
                
                # Try to extract arXiv ID pattern (YYYY.NNNNN)
                # Remove version suffix if present (e.g., "_v2", ".v2")
                clean_id = filename.split("_v")[0].split(".v")[0]
                
                # Handle arxiv.org URL format: "10.48550_arxiv.2506.07976" -> "2506.07976"
                if "arxiv." in clean_id:
                    # Extract the part after "arxiv."
                    parts = clean_id.split("arxiv.")
                    if len(parts) > 1:
                        clean_id = parts[-1]
                
                # Validate format (should be YYYY.NNNNN or similar)
                if clean_id and (clean_id.replace(".", "").replace("/", "_").isdigit() or "." in clean_id):
                    paper_ids.add(clean_id)
            
            return paper_ids
            
        except Exception:
            # Any error, return empty set
            return set()

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

