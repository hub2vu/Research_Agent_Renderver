"""
Rank and filter papers utilities.

This module provides utilities for ranking and filtering research papers
based on metadata and user profiles.
"""

# Type exports
from .types import (
    ComparisonNote,
    ContrastiveInfo,
    ContrastivePaper,
    FilteredPaper,
    LocalStatus,
    PaperInput,
    ScoreBreakdown,
    ScoreInfo,
    ScoredPaper,
    ToolResult,
    ToolResultSummary,
    UserProfile,
    UserProfileConstraints,
    UserProfileInterests,
    UserProfileKeywords,
    UserProfileKeywordsExclude,
)

# Path resolver exports
from .path_resolver import ensure_directory, resolve_path

# Loader exports
from .loaders import load_history, load_profile, scan_local_pdfs

# Filter exports
from .filters import filter_papers

# Score exports
from .scores import (
    _calculate_embedding_scores,
    _calculate_keyword_scores,
    _classify_papers_by_score,
    _verify_with_llm,
    _merge_scores,
    _calculate_dimension_scores,
    _apply_soft_penalty,
    _calculate_final_score,
)

__all__ = [
    # Types
    "PaperInput",
    "UserProfile",
    "UserProfileInterests",
    "UserProfileKeywords",
    "UserProfileKeywordsExclude",
    "UserProfileConstraints",
    "ScoreBreakdown",
    "ScoreInfo",
    "LocalStatus",
    "ScoredPaper",
    "FilteredPaper",
    "ContrastiveInfo",
    "ContrastivePaper",
    "ComparisonNote",
    "ToolResultSummary",
    "ToolResult",
    # Path resolver
    "resolve_path",
    "ensure_directory",
    # Loaders
    "load_profile",
    "load_history",
    "scan_local_pdfs",
    # Filters
    "filter_papers",
    # Scores
    "_calculate_embedding_scores",
    "_calculate_keyword_scores",
    "_classify_papers_by_score",
    "_verify_with_llm",
    "_merge_scores",
    "_calculate_dimension_scores",
    "_apply_soft_penalty",
    "_calculate_final_score",
]

