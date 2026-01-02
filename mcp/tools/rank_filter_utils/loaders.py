"""
Data loading utilities for rank and filter papers tool.
"""

import json
from pathlib import Path
from typing import Set

from .path_resolver import resolve_path
from .types import UserProfile


def load_profile(profile_path: str, tool_name: str = "rank_and_filter_papers") -> UserProfile:
    """
    Load user profile from JSON file.
    
    Args:
        profile_path: Path to profile JSON file
        tool_name: Tool name for error messages (optional)
            
    Returns:
        UserProfile object with loaded data or default values
        
    Raises:
        Exception: If file exists but has invalid format
    """
    # Import here to avoid circular dependency
    from mcp.base import ExecutionError
    
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
            tool_name=tool_name
        )
    except Exception as e:
        raise ExecutionError(
            f"Failed to load profile from {resolved_path}: {str(e)}",
            tool_name=tool_name
        )


def load_history(history_path: str) -> Set[str]:
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


def scan_local_pdfs(pdf_dir: str) -> Set[str]:
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

