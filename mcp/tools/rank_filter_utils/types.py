"""
Type definitions for rank and filter papers tool.
"""

from typing import Any, Dict, List, Optional, TypedDict


# Paper Input Types

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


# User Profile Types

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


# Score Types

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

