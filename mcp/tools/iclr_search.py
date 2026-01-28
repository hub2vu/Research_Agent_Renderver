"""
ICLR Search Tool

Hybrid search tool for ICLR 2025 papers combining semantic and keyword search.
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ..base import MCPTool, ToolParameter, ExecutionError
from .rank_filter_utils import load_profile, PaperInput

# Configure logging
logger = logging.getLogger(__name__)

# Module-level caches for singleton pattern
_iclr_embedding_model_cache = None
_iclr_embeddings_array_cache = None
_iclr_embeddings_path_cache = None
_iclr_metadata_cache = None
_iclr_metadata_path_cache = None


class ICLRAdapter:
    """Adapter for converting ICLR metadata to PaperInput format."""

    @staticmethod
    def load_iclr_metadata(
        metadata_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load ICLR metadata from CSV file.

        Args:
            metadata_path: Path to metadata file. If None, uses default path.

        Returns:
            List of metadata dictionaries
        """
        if metadata_path is None:
            # Try to find metadata file in common locations
            base_paths = [
                Path("data/embeddings_ICLR/ICLR2025_accepted_meta.csv"),
            ]

            # Also check relative to workspace root and docker paths
            search_roots = [
                Path("/app"),  # Docker default
                Path.cwd(),  # Current working directory
                Path(__file__).parent.parent.parent,  # From mcp/tools to workspace root
            ]

            for root in search_roots:
                for base_path in base_paths:
                    full_path = root / base_path
                    if full_path.exists():
                        metadata_path = str(full_path.resolve())
                        break
                if metadata_path:
                    break

            if metadata_path is None:
                raise FileNotFoundError(
                    "ICLR metadata file not found. "
                    "Please provide metadata_path or ensure data files are in expected locations."
                )

        metadata_path_obj = Path(metadata_path)
        if not metadata_path_obj.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        metadata_list: List[Dict[str, Any]] = []

        with open(metadata_path_obj, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata_list.append(dict(row))

        return metadata_list

    @staticmethod
    def to_paper_input(iclr_row: Dict[str, Any]) -> PaperInput:
        """
        Convert an ICLR metadata row to PaperInput format.

        Args:
            iclr_row: Dictionary containing ICLR metadata

        Returns:
            PaperInput dictionary
        """
        # Extract paper_id
        paper_id = str(iclr_row.get("paper_id", ""))
        if not paper_id:
            raise ValueError("paper_id is required in ICLR metadata")

        # Extract title
        title = iclr_row.get("title", "").strip()

        # Extract abstract
        abstract = iclr_row.get("abstract", "").strip()

        # ICLR metadata doesn't have authors in the provided CSV
        authors: List[str] = []
        authors_str = iclr_row.get("authors", "")
        if authors_str:
            authors = [
                author.strip()
                for author in authors_str.split(",")
                if author.strip()
            ]

        # Build PaperInput
        paper_input: PaperInput = {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "published": "2025-05-01",  # ICLR 2025 conference date
            "categories": ["ICLR 2025"],
            "pdf_url": f"https://openreview.net/pdf?id={paper_id}",
            "github_url": None,
        }

        return paper_input


def _get_iclr_embedding_model():
    """
    Get or create the embedding model for ICLR (singleton pattern).

    ICLR embeddings use bge-large-en-v1.5 model (1024 dimensions).
    """
    global _iclr_embedding_model_cache

    if _iclr_embedding_model_cache is not None:
        return _iclr_embedding_model_cache

    if not HAS_SENTENCE_TRANSFORMERS:
        return None

    # ICLR uses bge-large-en-v1.5 model (1024 dimensions)
    model_name = os.getenv('ICLR_EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5')

    try:
        logger.info(f"Loading ICLR embedding model: {model_name}")
        _iclr_embedding_model_cache = SentenceTransformer(model_name)
        return _iclr_embedding_model_cache
    except Exception as e:
        logger.error(f"Failed to load ICLR embedding model {model_name}: {e}")
        _iclr_embedding_model_cache = None
        return None


def _get_iclr_embeddings_array(embeddings_path: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Get or load ICLR embeddings array (singleton pattern).
    """
    global _iclr_embeddings_array_cache, _iclr_embeddings_path_cache

    if embeddings_path is None:
        if _iclr_embeddings_path_cache is not None:
            embeddings_path = _iclr_embeddings_path_cache
        else:
            search_paths = [
                Path("/app/data/embeddings_ICLR/ICLR2025_accepted_bge_large_en_v1_5.npy"),
                Path("data/embeddings_ICLR/ICLR2025_accepted_bge_large_en_v1_5.npy"),
            ]

            # Also check from current working directory
            workspace_root = Path.cwd()
            search_paths.append(workspace_root / "data" / "embeddings_ICLR" / "ICLR2025_accepted_bge_large_en_v1_5.npy")

            # Try alternative workspace root
            alt_workspace_root = Path(__file__).parent.parent.parent
            search_paths.append(alt_workspace_root / "data" / "embeddings_ICLR" / "ICLR2025_accepted_bge_large_en_v1_5.npy")

            for full_path in search_paths:
                if full_path.exists():
                    embeddings_path = str(full_path.resolve())
                    _iclr_embeddings_path_cache = embeddings_path
                    logger.info(f"Found ICLR embeddings at: {embeddings_path}")
                    break

            if embeddings_path is None:
                logger.error("ICLR embeddings.npy not found")
                return None

    embeddings_path_obj = Path(embeddings_path) if embeddings_path else None
    if embeddings_path_obj is None or not embeddings_path_obj.exists():
        logger.error(f"ICLR embeddings not found at: {embeddings_path}")
        return None

    if (_iclr_embeddings_array_cache is not None
        and _iclr_embeddings_path_cache == embeddings_path):
        return _iclr_embeddings_array_cache

    try:
        logger.info(f"Loading ICLR embeddings from: {embeddings_path}")
        embeddings_array = np.load(embeddings_path)

        if len(embeddings_array.shape) == 1:
            embeddings_array = embeddings_array.reshape(1, -1)

        logger.info(f"Loaded ICLR embeddings: shape={embeddings_array.shape}")

        _iclr_embeddings_array_cache = embeddings_array
        _iclr_embeddings_path_cache = embeddings_path

        return embeddings_array
    except Exception as e:
        logger.error(f"Failed to load ICLR embeddings: {e}")
        return None


def _get_iclr_metadata_cache(metadata_path: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Get or load ICLR metadata (singleton pattern).
    """
    global _iclr_metadata_cache, _iclr_metadata_path_cache

    if metadata_path is None:
        if _iclr_metadata_path_cache is not None:
            metadata_path = _iclr_metadata_path_cache

    if _iclr_metadata_cache is not None and _iclr_metadata_path_cache == metadata_path:
        return _iclr_metadata_cache

    try:
        metadata = ICLRAdapter.load_iclr_metadata(metadata_path)
        _iclr_metadata_cache = metadata
        if metadata_path:
            _iclr_metadata_path_cache = metadata_path
        return metadata
    except Exception as e:
        logger.error(f"Failed to load ICLR metadata: {e}")
        return None


class ICLRSearchTool(MCPTool):
    """Hybrid search tool for ICLR 2025 papers."""

    @property
    def name(self) -> str:
        return "iclr_search"

    @property
    def description(self) -> str:
        return (
            "Search ICLR 2025 papers using hybrid search (semantic + keyword) "
            "with weighted ranking. Returns top papers as PaperInput format."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Search query string",
                required=True
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Maximum number of results to return (default: 100)",
                required=False,
                default=100
            ),
            ToolParameter(
                name="profile_path",
                type="string",
                description="Path to user profile JSON file",
                required=False,
                default="users/profile.json"
            ),
            ToolParameter(
                name="semantic_weight",
                type="number",
                description="Weight for semantic search (default: 0.3)",
                required=False,
                default=0.3
            ),
            ToolParameter(
                name="keyword_weight",
                type="number",
                description="Weight for keyword search (default: 0.7)",
                required=False,
                default=0.7
            ),
            ToolParameter(
                name="metadata_path",
                type="string",
                description="Path to ICLR metadata file (optional)",
                required=False,
                default=None
            ),
            ToolParameter(
                name="embeddings_path",
                type="string",
                description="Path to embeddings.npy file (optional)",
                required=False,
                default=None
            ),
        ]

    @property
    def category(self) -> str:
        return "iclr"

    async def execute(
        self,
        query: str,
        max_results: int = 100,
        profile_path: str = "users/profile.json",
        semantic_weight: float = 0.3,
        keyword_weight: float = 0.7,
        metadata_path: Optional[str] = None,
        embeddings_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute hybrid search for ICLR papers.
        """
        if not query or not query.strip():
            raise ExecutionError("Query cannot be empty", tool_name=self.name)

        # Load user profile
        try:
            profile = load_profile(profile_path, tool_name=self.name)
        except Exception as e:
            raise ExecutionError(
                f"Failed to load profile: {str(e)}",
                tool_name=self.name
            )

        # Load ICLR metadata
        iclr_metadata = _get_iclr_metadata_cache(metadata_path)
        if iclr_metadata is None:
            try:
                iclr_metadata = ICLRAdapter.load_iclr_metadata(metadata_path)
                global _iclr_metadata_cache, _iclr_metadata_path_cache
                _iclr_metadata_cache = iclr_metadata
            except FileNotFoundError as e:
                raise ExecutionError(
                    f"Failed to load ICLR metadata: {str(e)}",
                    tool_name=self.name
                )

        if not iclr_metadata:
            return {
                "query": query,
                "total_results": 0,
                "papers": [],
                "search_stats": {}
            }

        # Convert to PaperInput format
        paper_inputs_all: List[PaperInput] = []
        for row in iclr_metadata:
            try:
                paper_input = ICLRAdapter.to_paper_input(row)
                paper_inputs_all.append(paper_input)
            except Exception:
                continue

        # Perform semantic search
        semantic_scores = await self._semantic_search(
            query=query,
            paper_inputs=paper_inputs_all,
            embeddings_path=embeddings_path
        )

        # Perform keyword search
        keyword_scores = self._keyword_search(
            query=query,
            paper_inputs=paper_inputs_all
        )

        # Combine scores
        combined_scores = self._combine_scores(
            semantic_scores,
            keyword_scores,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight
        )

        # Sort and select top N
        ranked_paper_ids = sorted(
            combined_scores.keys(),
            key=lambda pid: combined_scores[pid],
            reverse=True
        )[:max_results]

        # Build result papers
        selected_papers_map = {
            p.get("paper_id"): p for p in paper_inputs_all
        }
        selected_papers: List[PaperInput] = []
        for pid in ranked_paper_ids:
            if pid in selected_papers_map:
                paper = selected_papers_map[pid].copy()
                paper["semantic_score"] = semantic_scores.get(pid, 0.0)
                paper["keyword_score"] = keyword_scores.get(pid, 0.0)
                paper["combined_score"] = combined_scores.get(pid, 0.0)
                selected_papers.append(paper)

        return {
            "query": query,
            "total_results": len(selected_papers),
            "papers": selected_papers,
            "search_stats": {
                "semantic_matches": len([s for s in semantic_scores.values() if s > 0]),
                "keyword_matches": len([s for s in keyword_scores.values() if s > 0]),
            }
        }

    async def _semantic_search(
        self,
        query: str,
        paper_inputs: List[PaperInput],
        embeddings_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Perform semantic search using pre-computed embeddings."""
        if not HAS_SENTENCE_TRANSFORMERS or not HAS_SKLEARN:
            return {p.get("paper_id", ""): 0.0 for p in paper_inputs}

        embeddings_array = _get_iclr_embeddings_array(embeddings_path)

        if embeddings_array is None:
            logger.warning("ICLR embeddings not available, skipping semantic search")
            return {p.get("paper_id", ""): 0.0 for p in paper_inputs}

        try:
            model = _get_iclr_embedding_model()
            if model is None:
                return {p.get("paper_id", ""): 0.0 for p in paper_inputs}

            # Get query embedding
            query_embedding = model.encode([query])[0]

            # Check dimension match
            if len(query_embedding) != embeddings_array.shape[1]:
                logger.error(
                    f"Dimension mismatch: query={len(query_embedding)}, "
                    f"embeddings={embeddings_array.shape[1]}"
                )
                return {p.get("paper_id", ""): 0.0 for p in paper_inputs}

            # Calculate cosine similarity
            query_embedding_2d = query_embedding.reshape(1, -1)
            similarities = cosine_similarity(query_embedding_2d, embeddings_array)[0]

            # Map to paper IDs
            scores: Dict[str, float] = {}
            for idx, paper in enumerate(paper_inputs):
                if idx < len(similarities):
                    paper_id = paper.get("paper_id", "")
                    score = float(max(0.0, similarities[idx]))
                    scores[paper_id] = score
                else:
                    paper_id = paper.get("paper_id", "")
                    scores[paper_id] = 0.0

            return scores

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {p.get("paper_id", ""): 0.0 for p in paper_inputs}

    def _keyword_search(
        self,
        query: str,
        paper_inputs: List[PaperInput]
    ) -> Dict[str, float]:
        """Perform keyword search on title and abstract."""
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())

        scores: Dict[str, float] = {}

        for paper in paper_inputs:
            paper_id = paper.get("paper_id", "")
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()

            # Exact title match
            if title == query_lower:
                scores[paper_id] = 1.0
                continue

            # Title contains query
            if query_lower in title:
                scores[paper_id] = 0.8
                continue

            # Word matching
            title_matches = sum(1 for word in query_words if word in title)
            abstract_matches = sum(1 for word in query_words if word in abstract)

            # Weighted score
            score = (title_matches * 3.0 + abstract_matches * 1.0) / max(len(query_words), 1)
            score = min(0.5, score / 3.0)

            scores[paper_id] = score

        return scores

    def _combine_scores(
        self,
        semantic_scores: Dict[str, float],
        keyword_scores: Dict[str, float],
        semantic_weight: float = 0.3,
        keyword_weight: float = 0.7
    ) -> Dict[str, float]:
        """Combine semantic and keyword scores."""
        all_paper_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        combined_scores: Dict[str, float] = {}

        for paper_id in all_paper_ids:
            sem_score = semantic_scores.get(paper_id, 0.0)
            kw_score = keyword_scores.get(paper_id, 0.0)

            combined = sem_score * semantic_weight + kw_score * keyword_weight

            # Boost exact matches
            if kw_score >= 1.0:
                combined = min(1.0, combined * 2.0 + 0.3)
            elif kw_score >= 0.8:
                combined = min(1.0, combined * 1.3)

            combined_scores[paper_id] = combined

        return combined_scores
