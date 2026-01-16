"""
NeurIPS Data Adapter

Converts NeurIPS 2025 metadata into PaperInput format for use with rank_filter pipeline.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .rank_filter_utils.types import PaperInput


class NeurIPSAdapter:
    """Adapter for converting NeurIPS metadata to PaperInput format."""
    
    @staticmethod
    def load_neurips_metadata(
        metadata_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load NeurIPS metadata from JSONL or CSV file.
        
        Args:
            metadata_path: Path to metadata file. If None, uses default path.
            
        Returns:
            List of metadata dictionaries
        """
        if metadata_path is None:
            # Try to find metadata file in common locations
            base_paths = [
                Path("data/embeddings_Neu/metadata.jsonl"),
                Path("data/embeddings_Neu/metadata.csv"),
                Path("data/NeurIPS 2025 Events.csv"),
            ]
            
            # Also check relative to workspace root
            workspace_root = Path.cwd()
            for base_path in base_paths:
                full_path = workspace_root / base_path
                if full_path.exists():
                    metadata_path = str(full_path)
                    break
            
            if metadata_path is None:
                raise FileNotFoundError(
                    "NeurIPS metadata file not found. "
                    "Please provide metadata_path or ensure data files are in expected locations."
                )
        
        metadata_path_obj = Path(metadata_path)
        if not metadata_path_obj.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        metadata_list: List[Dict[str, Any]] = []
        
        # Try JSONL first
        if metadata_path_obj.suffix == ".jsonl":
            with open(metadata_path_obj, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        metadata_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        # Try CSV
        elif metadata_path_obj.suffix == ".csv":
            import csv
            with open(metadata_path_obj, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metadata_list.append(dict(row))
        else:
            raise ValueError(f"Unsupported file format: {metadata_path_obj.suffix}")
        
        return metadata_list
    
    @staticmethod
    def to_paper_input(neurips_row: Dict[str, Any]) -> PaperInput:
        """
        Convert a NeurIPS metadata row to PaperInput format.
        
        Args:
            neurips_row: Dictionary containing NeurIPS metadata
            
        Returns:
            PaperInput dictionary
        """
        # Extract paper_id
        paper_id = str(neurips_row.get("paper_id", ""))
        if not paper_id:
            raise ValueError("paper_id is required in NeurIPS metadata")
        
        # Extract title (from 'name' field)
        title = neurips_row.get("name", "").strip()
        
        # Extract abstract
        abstract = neurips_row.get("abstract", "").strip()
        
        # Extract authors (from 'speakers/authors' field)
        authors_str = neurips_row.get("speakers/authors", "")
        authors: List[str] = []
        if authors_str:
            # Split by comma and clean up
            authors = [
                author.strip()
                for author in authors_str.split(",")
                if author.strip()
            ]
        
        # Extract virtualsite_url (for PDF availability check)
        virtualsite_url = neurips_row.get("virtualsite_url", "")
        
        # Build PaperInput
        paper_input: PaperInput = {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "published": "2025-12-01",  # NeurIPS 2025 conference date (fixed)
            "categories": ["NeurIPS 2025"],
            "pdf_url": None,  # virtualsite_url is not a direct PDF link
            "github_url": None,  # Will be extracted later if needed
        }
        
        return paper_input
    
    @staticmethod
    def check_pdf_availability(virtualsite_url: str) -> bool:
        """
        Check if PDF is available based on virtualsite_url.
        
        Args:
            virtualsite_url: Virtual site URL from NeurIPS metadata
            
        Returns:
            True if URL exists (indicating PDF might be available), False otherwise
        """
        return bool(virtualsite_url and virtualsite_url.strip())
    
    @staticmethod
    def load_cluster_map(
        cluster_k: int = 15,
        cluster_file_path: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Load cluster mapping from JSON file.
        
        Args:
            cluster_k: K value for clustering (default: 15)
            cluster_file_path: Path to cluster file. If None, uses default path.
            
        Returns:
            Dictionary mapping paper_id to cluster_id
        """
        if cluster_file_path is None:
            # Get K value from environment variable or use default
            cluster_k = int(os.getenv("NEURIPS_CLUSTER_K", cluster_k))
            
            # Try to find cluster file
            base_paths = [
                Path(f"data/embeddings_Neu/neurips_clusters_k{cluster_k}.json"),
            ]
            
            workspace_root = Path.cwd()
            for base_path in base_paths:
                full_path = workspace_root / base_path
                if full_path.exists():
                    cluster_file_path = str(full_path)
                    break
            
            if cluster_file_path is None:
                # Return empty dict if cluster file not found
                return {}
        
        cluster_path = Path(cluster_file_path)
        if not cluster_path.exists():
            return {}
        
        try:
            with open(cluster_path, "r", encoding="utf-8") as f:
                cluster_data = json.load(f)
            
            # Extract paper_id_to_cluster mapping
            paper_id_to_cluster = cluster_data.get("paper_id_to_cluster", {})
            
            # Convert keys to strings for consistency
            return {
                str(paper_id): int(cluster_id)
                for paper_id, cluster_id in paper_id_to_cluster.items()
            }
        except Exception:
            return {}
