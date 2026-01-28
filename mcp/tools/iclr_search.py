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
_iclr_embeddings_dimension_cache = None  # Cache for embeddings dimension
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
    global _iclr_embeddings_array_cache, _iclr_embeddings_path_cache, _iclr_embeddings_dimension_cache

    if embeddings_path is None:
        if _iclr_embeddings_path_cache is not None:
            embeddings_path = _iclr_embeddings_path_cache
            logger.debug(f"Using cached ICLR embeddings path: {embeddings_path}")
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

            logger.info(f"Searching for ICLR embeddings.npy in {len(search_paths)} locations...")
            for full_path in search_paths:
                logger.debug(f"  Checking: {full_path} (exists: {full_path.exists()})")
                if full_path.exists():
                    embeddings_path = str(full_path.resolve())
                    _iclr_embeddings_path_cache = embeddings_path
                    logger.info(f"✓ Found ICLR embeddings at: {embeddings_path}")
                    break

            if embeddings_path is None:
                logger.error(
                    f"✗ ICLR embeddings.npy not found in any of the searched locations. "
                    f"Current working directory: {workspace_root}. "
                    f"Searched paths: {[str(p) for p in search_paths]}"
                )
                return None

    embeddings_path_obj = Path(embeddings_path) if embeddings_path else None
    if embeddings_path_obj is None or not embeddings_path_obj.exists():
        logger.error(
            f"✗ ICLR embeddings not found at specified path: {embeddings_path}. "
            f"Absolute path check: {embeddings_path_obj.resolve() if embeddings_path_obj else 'N/A'}"
        )
        return None

    if (_iclr_embeddings_array_cache is not None
        and _iclr_embeddings_path_cache == embeddings_path):
        logger.debug(f"Returning cached ICLR embeddings array (shape: {_iclr_embeddings_array_cache.shape})")
        return _iclr_embeddings_array_cache

    try:
        logger.info(f"Loading ICLR embeddings from: {embeddings_path}")
        embeddings_array = np.load(embeddings_path)

        if len(embeddings_array.shape) == 1:
            embeddings_array = embeddings_array.reshape(1, -1)

        # Cache dimension for model selection
        embedding_dim = embeddings_array.shape[1] if len(embeddings_array.shape) >= 2 else None
        _iclr_embeddings_dimension_cache = embedding_dim

        logger.info(
            f"✓ Successfully loaded ICLR embeddings.npy. "
            f"Shape: {embeddings_array.shape} "
            f"({embeddings_array.shape[0]} papers × {embedding_dim}D), "
            f"Dtype: {embeddings_array.dtype}, "
            f"Size: {embeddings_array.nbytes / (1024 * 1024):.2f} MB"
        )

        _iclr_embeddings_array_cache = embeddings_array
        _iclr_embeddings_path_cache = embeddings_path

        return embeddings_array
    except Exception as e:
        logger.error(f"✗ Exception loading ICLR embeddings: {type(e).__name__}: {e}")
        return None


def _validate_iclr_embeddings_metadata_mapping(
    embeddings_array: np.ndarray,
    metadata_list: List[Dict[str, Any]]
) -> bool:
    """
    Validate that ICLR embeddings array and metadata have matching row counts.

    Args:
        embeddings_array: Loaded embeddings array (should be 2D)
        metadata_list: List of metadata dictionaries

    Returns:
        True if validation passes, False otherwise
    """
    if embeddings_array is None:
        logger.warning("✗ Cannot validate: embeddings_array is None")
        return False

    if metadata_list is None or len(metadata_list) == 0:
        logger.warning("✗ Cannot validate: metadata_list is empty or None")
        return False

    # Ensure embeddings are 2D
    if len(embeddings_array.shape) == 1:
        embeddings_array = embeddings_array.reshape(1, -1)

    embeddings_count = embeddings_array.shape[0]
    metadata_count = len(metadata_list)

    logger.info(
        f"Validating ICLR embeddings-metadata mapping: "
        f"embeddings rows={embeddings_count}, metadata rows={metadata_count}"
    )

    if embeddings_count != metadata_count:
        logger.error(
            f"✗ Index mismatch detected! "
            f"ICLR embeddings.npy has {embeddings_count} rows, "
            f"but metadata has {metadata_count} rows. "
            f"Difference: {abs(embeddings_count - metadata_count)} rows. "
            f"This will cause incorrect paper-embedding mappings."
        )
        return False

    logger.info(
        f"✓ ICLR index mapping validation passed: "
        f"{embeddings_count} embeddings match {metadata_count} metadata entries"
    )
    return True


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

        # Validate embeddings-metadata mapping before semantic search
        embeddings_array = _get_iclr_embeddings_array(embeddings_path)
        if embeddings_array is not None:
            if not _validate_iclr_embeddings_metadata_mapping(embeddings_array, paper_inputs_all):
                logger.warning(
                    "⚠ Continuing with semantic search despite index mismatch. "
                    "Results may be incorrect. Please check ICLR embeddings.npy and metadata alignment."
                )

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

        # Calculate statistics
        search_stats = {
            "semantic_matches": len([s for s in semantic_scores.values() if s > 0]),
            "keyword_matches": len([s for s in keyword_scores.values() if s > 0]),
            "combined_count": len(combined_scores),
            "avg_combined_score": float(np.mean(list(combined_scores.values()))) if combined_scores else 0.0
        }

        return {
            "query": query,
            "total_results": len(selected_papers),
            "papers": selected_papers,
            "search_stats": search_stats,
            "cached": False
        }

    async def _semantic_search(
        self,
        query: str,
        paper_inputs: List[PaperInput],
        embeddings_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Perform semantic search using pre-computed embeddings.

        Args:
            query: Search query
            paper_inputs: List of papers to search
            embeddings_path: Path to embeddings.npy file

        Returns:
            Dictionary mapping paper_id to semantic score (0.0-1.0)
        """
        if not HAS_SENTENCE_TRANSFORMERS or not HAS_SKLEARN:
            logger.warning(
                "✗ SentenceTransformer or scikit-learn not available. "
                "Skipping semantic search."
            )
            return {p.get("paper_id", ""): 0.0 for p in paper_inputs}

        # Get cached embeddings array
        embeddings_array = _get_iclr_embeddings_array(embeddings_path)

        if embeddings_array is None:
            logger.warning(
                "⚠ ICLR embeddings.npy not found or failed to load. "
                "Falling back to on-the-fly embedding computation. "
                "This will be slower and use more resources."
            )
            # Fallback: use SentenceTransformer to compute embeddings on-the-fly
            return await self._semantic_search_fallback(query, paper_inputs)

        try:
            # Get cached embedding model
            model = _get_iclr_embedding_model()
            if model is None:
                logger.warning(
                    "⚠ Failed to load ICLR embedding model. "
                    "Falling back to on-the-fly embedding computation."
                )
                return await self._semantic_search_fallback(query, paper_inputs)

            # Validate index mapping
            if not _validate_iclr_embeddings_metadata_mapping(embeddings_array, paper_inputs):
                logger.error(
                    "✗ Index mismatch between ICLR embeddings and metadata! "
                    "Semantic search results may be incorrect. "
                    "Falling back to on-the-fly computation for safety."
                )
                return await self._semantic_search_fallback(query, paper_inputs)

            logger.debug(f"Computing query embedding for: '{query[:50]}...'")
            # Get query embedding
            query_embedding = model.encode([query])[0]

            # Validate dimensions before computing similarity
            query_dim = query_embedding.shape[0] if hasattr(query_embedding, 'shape') else len(query_embedding)
            embeddings_dim = embeddings_array.shape[1] if len(embeddings_array.shape) >= 2 else None

            logger.debug(
                f"Query embedding dimension: {query_dim}D, "
                f"Embeddings dimension: {embeddings_dim}D"
            )

            # Check dimension mismatch
            if embeddings_dim is not None and query_dim != embeddings_dim:
                logger.error(
                    f"✗ Dimension mismatch detected! "
                    f"Query embedding ({query_dim}D) != paper embeddings ({embeddings_dim}D). "
                    f"This will cause cosine similarity to fail. "
                    f"ICLR uses BAAI/bge-large-en-v1.5 (1024D). "
                    f"Falling back to on-the-fly embedding computation."
                )
                return await self._semantic_search_fallback(query, paper_inputs)

            logger.debug(
                f"Computing cosine similarity: "
                f"query shape={query_embedding.shape}, "
                f"embeddings shape={embeddings_array.shape}"
            )
            # Calculate cosine similarity
            query_embedding_2d = query_embedding.reshape(1, -1)
            similarities = cosine_similarity(query_embedding_2d, embeddings_array)[0]

            logger.debug(
                f"✓ Computed {len(similarities)} similarities. "
                f"Max similarity: {float(np.max(similarities)):.4f}, "
                f"Min similarity: {float(np.min(similarities)):.4f}"
            )

            # Map to paper IDs
            scores: Dict[str, float] = {}
            for idx, paper in enumerate(paper_inputs):
                if idx < len(similarities):
                    paper_id = paper.get("paper_id", "")
                    score = float(max(0.0, similarities[idx]))
                    scores[paper_id] = score
                else:
                    # Handle case where paper_inputs is longer than embeddings
                    logger.warning(
                        f"⚠ Paper at index {idx} (ID: {paper.get('paper_id', 'unknown')}) "
                        f"exceeds embeddings array length {len(similarities)}. "
                        f"Setting score to 0.0"
                    )
                    paper_id = paper.get("paper_id", "")
                    scores[paper_id] = 0.0

            logger.info(
                f"✓ ICLR semantic search completed: "
                f"{len(scores)} scores computed, "
                f"{len([s for s in scores.values() if s > 0])} non-zero scores"
            )

            return scores

        except Exception as e:
            logger.error(
                f"✗ Exception during ICLR semantic search with embeddings.npy: "
                f"{type(e).__name__}: {e}. "
                f"Falling back to on-the-fly computation."
            )
            # Fallback to on-the-fly computation
            return await self._semantic_search_fallback(query, paper_inputs)

    async def _semantic_search_fallback(
        self,
        query: str,
        paper_inputs: List[PaperInput]
    ) -> Dict[str, float]:
        """
        Fallback semantic search using on-the-fly embedding computation.

        Args:
            query: Search query
            paper_inputs: List of papers to search

        Returns:
            Dictionary mapping paper_id to semantic score
        """
        if not HAS_SENTENCE_TRANSFORMERS or not HAS_SKLEARN:
            logger.warning(
                "✗ SentenceTransformer or scikit-learn not available. "
                "Cannot perform fallback semantic search."
            )
            return {p.get("paper_id", ""): 0.0 for p in paper_inputs}

        try:
            logger.info(
                f"🔄 Performing ICLR fallback semantic search: "
                f"computing embeddings for {len(paper_inputs)} papers on-the-fly. "
                f"This may take longer."
            )

            # Get cached embedding model
            model = _get_iclr_embedding_model()
            if model is None:
                logger.error(
                    "✗ Failed to load ICLR embedding model. "
                    "Cannot perform fallback semantic search."
                )
                return {p.get("paper_id", ""): 0.0 for p in paper_inputs}

            # Encode query
            logger.debug(f"Encoding query: '{query[:50]}...'")
            query_embedding = model.encode([query])[0]

            # Encode papers (title + abstract)
            logger.debug(f"Encoding {len(paper_inputs)} ICLR papers...")
            paper_texts = [
                f"{p.get('title', '')} {p.get('abstract', '')}"
                for p in paper_inputs
            ]
            paper_embeddings = model.encode(paper_texts, show_progress_bar=True)

            logger.debug(f"✓ Encoded {len(paper_embeddings)} ICLR paper embeddings")

            # Calculate cosine similarity
            query_embedding_2d = query_embedding.reshape(1, -1)
            similarities = cosine_similarity(query_embedding_2d, paper_embeddings)[0]

            logger.debug(
                f"✓ Computed similarities. "
                f"Max: {float(np.max(similarities)):.4f}, "
                f"Min: {float(np.min(similarities)):.4f}"
            )

            # Map to paper IDs
            scores: Dict[str, float] = {}
            for idx, paper in enumerate(paper_inputs):
                paper_id = paper.get("paper_id", "")
                score = float(max(0.0, similarities[idx]))
                scores[paper_id] = score

            logger.info(
                f"✓ ICLR fallback semantic search completed: "
                f"{len(scores)} scores computed"
            )

            return scores

        except Exception as e:
            logger.error(
                f"✗ Exception during ICLR fallback semantic search: "
                f"{type(e).__name__}: {e}"
            )
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
