"""
Path resolution utilities for rank and filter papers tool.
"""

import os
from pathlib import Path


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

