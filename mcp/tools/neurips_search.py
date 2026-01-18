"""
NeurIPS Search Tool

Hybrid search tool for NeurIPS 2025 papers combining semantic and keyword search with RRF.
"""

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
from .rank_filter_utils.cache import _generate_cache_key, load_cache, save_cache
from .neurips_adapter import NeurIPSAdapter

# Configure logging
logger = logging.getLogger(__name__)

# Module-level caches for singleton pattern
_embedding_model_cache = None
_embeddings_array_cache = None
_embeddings_path_cache = None
_embeddings_validation_cache = None  # Cache for validation status
_metadata_cache = None
_metadata_path_cache = None


def _get_embedding_model():
    """
    Get or create the embedding model (singleton pattern).
    
    Returns the cached SentenceTransformer model, loading it only once.
    Returns None if sentence-transformers is not available or loading fails.
    """
    global _embedding_model_cache
    if _embedding_model_cache is None and HAS_SENTENCE_TRANSFORMERS:
        try:
            _embedding_model_cache = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            _embedding_model_cache = None
    return _embedding_model_cache


def _get_embeddings_array(embeddings_path: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Get or load embeddings array (singleton pattern).
    
    Args:
        embeddings_path: Path to embeddings.npy file
        
    Returns:
        Cached embeddings array or None if not available
    """
    global _embeddings_array_cache, _embeddings_path_cache, _embeddings_validation_cache
    
    # Find embeddings file if not provided
    if embeddings_path is None:
        if _embeddings_path_cache is not None:
            embeddings_path = _embeddings_path_cache
            logger.debug(f"Using cached embeddings path: {embeddings_path}")
        else:
            # Priority order: Docker container path first, then workspace root
            search_paths = [
                Path("/app/data/embeddings_Neu/embeddings.npy"),  # Docker container path
                Path("data/embeddings_Neu/embeddings.npy"),  # Relative to workspace root
            ]
            
            # Also check from current working directory
            workspace_root = Path.cwd()
            search_paths.append(workspace_root / "data" / "embeddings_Neu" / "embeddings.npy")
            
            # Try alternative workspace root (from mcp/tools to workspace root)
            alt_workspace_root = Path(__file__).parent.parent.parent
            search_paths.append(alt_workspace_root / "data" / "embeddings_Neu" / "embeddings.npy")
            
            logger.info(f"Searching for embeddings.npy in {len(search_paths)} locations...")
            for full_path in search_paths:
                logger.debug(f"  Checking: {full_path} (exists: {full_path.exists()})")
                if full_path.exists():
                    embeddings_path = str(full_path.resolve())
                    _embeddings_path_cache = embeddings_path
                    logger.info(f"✓ Found embeddings.npy at: {embeddings_path}")
                    break
            
            if embeddings_path is None:
                logger.error(
                    f"✗ embeddings.npy not found in any of the searched locations. "
                    f"Current working directory: {workspace_root}. "
                    f"Searched paths: {[str(p) for p in search_paths]}"
                )
                return None
    
    # Validate path
    embeddings_path_obj = Path(embeddings_path) if embeddings_path else None
    if embeddings_path_obj is None or not embeddings_path_obj.exists():
        logger.error(
            f"✗ embeddings.npy not found at specified path: {embeddings_path}. "
            f"Absolute path check: {embeddings_path_obj.resolve() if embeddings_path_obj else 'N/A'}"
        )
        return None
    
    # Return cached array if path matches and validated
    if (_embeddings_array_cache is not None 
        and _embeddings_path_cache == embeddings_path 
        and _embeddings_validation_cache is True):
        logger.debug(f"Returning cached embeddings array (shape: {_embeddings_array_cache.shape})")
        return _embeddings_array_cache
    
    # Load embeddings
    try:
        logger.info(f"Loading embeddings from: {embeddings_path}")
        embeddings_array = np.load(embeddings_path)
        
        # Ensure embeddings are 2D
        if len(embeddings_array.shape) == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        
        logger.info(
            f"✓ Successfully loaded embeddings.npy. "
            f"Shape: {embeddings_array.shape}, "
            f"Dtype: {embeddings_array.dtype}, "
            f"Size: {embeddings_array.nbytes / (1024 * 1024):.2f} MB"
        )
        
        # Cache it
        _embeddings_array_cache = embeddings_array
        _embeddings_path_cache = embeddings_path
        _embeddings_validation_cache = True
        
        return embeddings_array
    except FileNotFoundError as e:
        logger.error(
            f"✗ FileNotFoundError loading embeddings.npy: {e}. "
            f"Path: {embeddings_path}, "
            f"Absolute: {Path(embeddings_path).resolve() if embeddings_path else 'N/A'}"
        )
        return None
    except Exception as e:
        logger.error(
            f"✗ Exception loading embeddings.npy: {type(e).__name__}: {e}. "
            f"Path: {embeddings_path}"
        )
        return None


def _validate_embeddings_metadata_mapping(
    embeddings_array: np.ndarray,
    metadata_list: List[Dict[str, Any]]
) -> bool:
    """
    Validate that embeddings array and metadata have matching row counts.
    
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
        f"Validating embeddings-metadata mapping: "
        f"embeddings rows={embeddings_count}, metadata rows={metadata_count}"
    )
    
    if embeddings_count != metadata_count:
        logger.error(
            f"✗ Index mismatch detected! "
            f"embeddings.npy has {embeddings_count} rows, "
            f"but metadata has {metadata_count} rows. "
            f"Difference: {abs(embeddings_count - metadata_count)} rows. "
            f"This will cause incorrect paper-embedding mappings."
        )
        return False
    
    logger.info(
        f"✓ Index mapping validation passed: "
        f"{embeddings_count} embeddings match {metadata_count} metadata entries"
    )
    return True


def _get_metadata_cache(metadata_path: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Get or load metadata (singleton pattern).
    
    Args:
        metadata_path: Path to metadata file
        
    Returns:
        Cached metadata list or None if not available
    """
    global _metadata_cache, _metadata_path_cache
    
    # Find metadata file if not provided
    if metadata_path is None:
        if _metadata_path_cache is not None:
            metadata_path = _metadata_path_cache
            logger.debug(f"Using cached metadata path: {metadata_path}")
        else:
            base_paths = [
                Path("data/embeddings_Neu/metadata.jsonl"),
                Path("data/embeddings_Neu/metadata.csv"),
                Path("data/NeurIPS 2025 Events.csv"),
            ]
            
            # Priority order: Docker container path first, then workspace root
            search_roots = [
                Path("/app"),  # Docker default
                Path.cwd(),  # Current working directory
                Path(__file__).parent.parent.parent,  # From mcp/tools to workspace root
            ]
            
            logger.info(f"Searching for metadata file in {len(search_roots)} root directories...")
            for root in search_roots:
                for base_path in base_paths:
                    full_path = root / base_path
                    logger.debug(f"  Checking: {full_path} (exists: {full_path.exists()})")
                    if full_path.exists():
                        metadata_path = str(full_path.resolve())
                        _metadata_path_cache = metadata_path
                        logger.info(f"✓ Found metadata file at: {metadata_path}")
                        break
                if metadata_path:
                    break
            
            if metadata_path is None:
                logger.error(
                    f"✗ Metadata file not found in any of the searched locations. "
                    f"Current working directory: {Path.cwd()}. "
                    f"Searched roots: {[str(r) for r in search_roots]}"
                )
                return None
    
    # Validate path
    metadata_path_obj = Path(metadata_path) if metadata_path else None
    if metadata_path_obj is None or not metadata_path_obj.exists():
        logger.error(
            f"✗ Metadata file not found at specified path: {metadata_path}. "
            f"Absolute path check: {metadata_path_obj.resolve() if metadata_path_obj else 'N/A'}"
        )
        return None
    
    # Return cached metadata if path matches
    if _metadata_cache is not None and _metadata_path_cache == metadata_path:
        logger.debug(f"Returning cached metadata ({len(_metadata_cache)} entries)")
        return _metadata_cache
    
    # Load metadata
    try:
        logger.info(f"Loading metadata from: {metadata_path}")
        metadata = NeurIPSAdapter.load_neurips_metadata(metadata_path)
        logger.info(f"✓ Successfully loaded metadata. Total entries: {len(metadata)}")
        _metadata_cache = metadata
        _metadata_path_cache = metadata_path
        return metadata
    except FileNotFoundError as e:
        logger.error(
            f"✗ FileNotFoundError loading metadata: {e}. "
            f"Path: {metadata_path}, "
            f"Absolute: {Path(metadata_path).resolve() if metadata_path else 'N/A'}"
        )
        return None
    except Exception as e:
        logger.error(
            f"✗ Exception loading metadata: {type(e).__name__}: {e}. "
            f"Path: {metadata_path}"
        )
        return None


class NeurIPSSearchTool(MCPTool):
    """Hybrid search tool for NeurIPS 2025 papers."""
    
    @property
    def name(self) -> str:
        return "neurips_search"
    
    @property
    def description(self) -> str:
        return (
            "Search NeurIPS 2025 papers using hybrid search (semantic + keyword) "
            "with RRF (Reciprocal Rank Fusion) ranking. Returns top papers as PaperInput format."
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
                description="Weight for semantic search in weighted average (default: 0.3)",
                required=False,
                default=0.3
            ),
            ToolParameter(
                name="keyword_weight",
                type="number",
                description="Weight for keyword search in weighted average (default: 0.7)",
                required=False,
                default=0.7
            ),
            ToolParameter(
                name="use_cache",
                type="boolean",
                description="Whether to use cache for results (default: True)",
                required=False,
                default=True
            ),
            ToolParameter(
                name="metadata_path",
                type="string",
                description="Path to NeurIPS metadata file (optional, auto-detected if not provided)",
                required=False,
                default=None
            ),
            ToolParameter(
                name="embeddings_path",
                type="string",
                description="Path to embeddings.npy file (optional, auto-detected if not provided)",
                required=False,
                default=None
            ),
        ]
    
    @property
    def category(self) -> str:
        return "neurips"
    
    async def execute(
        self,
        query: str,
        max_results: int = 100,
        profile_path: str = "users/profile.json",
        semantic_weight: float = 0.3,
        keyword_weight: float = 0.7,
        use_cache: bool = True,
        metadata_path: Optional[str] = None,
        embeddings_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute hybrid search for NeurIPS papers.
        
        Returns:
            Dictionary with search results and statistics
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
        
        # Check cache
        if use_cache:
            cache_key = _generate_cache_key(query, profile)
            cached_result = load_cache(cache_key)
            if cached_result is not None:
                return {
                    **cached_result,
                    "cached": True
                }
        
        # Load NeurIPS metadata (using cache)
        neurips_metadata = _get_metadata_cache(metadata_path)
        if neurips_metadata is None:
            # If metadata_path was None and cache didn't find it, 
            # pass None explicitly to let NeurIPSAdapter search
            try:
                neurips_metadata = NeurIPSAdapter.load_neurips_metadata(metadata_path)
                # Cache it for next time
                global _metadata_cache, _metadata_path_cache
                _metadata_cache = neurips_metadata
                if metadata_path or _metadata_path_cache:
                    _metadata_path_cache = metadata_path or _metadata_path_cache
            except FileNotFoundError as e:
                # More helpful error message
                raise ExecutionError(
                    f"Failed to load NeurIPS metadata: {str(e)}. "
                    f"Current working directory: {Path.cwd()}. "
                    f"Looking for files: data/embeddings_Neu/metadata.jsonl, "
                    f"data/embeddings_Neu/metadata.csv, data/NeurIPS 2025 Events.csv",
                    tool_name=self.name
                )
            except Exception as e:
                raise ExecutionError(
                    f"Failed to load NeurIPS metadata: {str(e)}",
                    tool_name=self.name
                )
        
        if not neurips_metadata:
            return {
                "query": query,
                "total_results": 0,
                "papers": [],
                "search_stats": {
                    "semantic_matches": 0,
                    "keyword_matches": 0,
                    "rrf_combined_count": 0,
                    "avg_rrf_score": 0.0
                },
                "cached": False
            }
        
        # Convert to PaperInput format
        paper_inputs_all: List[PaperInput] = []
        for row in neurips_metadata:
            try:
                paper_input = NeurIPSAdapter.to_paper_input(row)
                paper_inputs_all.append(paper_input)
            except Exception:
                # Skip invalid rows
                continue
        
        # Validate embeddings-metadata mapping before semantic search
        embeddings_array = _get_embeddings_array(embeddings_path)
        if embeddings_array is not None:
            if not _validate_embeddings_metadata_mapping(embeddings_array, paper_inputs_all):
                logger.warning(
                    "⚠ Continuing with semantic search despite index mismatch. "
                    "Results may be incorrect. Please check embeddings.npy and metadata alignment."
                )
        
        # Perform semantic search
        semantic_scores = await self._semantic_search_with_embeddings_npy(
            query=query,
            paper_inputs=paper_inputs_all,
            embeddings_path=embeddings_path
        )
        
        # Perform keyword search
        keyword_scores = self._keyword_search(
            query=query,
            paper_inputs=paper_inputs_all
        )
        
        # Combine with weighted average
        combined_scores = self._combine_scores_weighted(
            semantic_scores,
            keyword_scores,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            exact_match_boost=2.0
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
                # Add search scores for reference
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
        
        result = {
            "query": query,
            "total_results": len(selected_papers),
            "papers": selected_papers,
            "search_stats": search_stats,
            "cached": False
        }
        
        # Save to cache
        if use_cache:
            try:
                save_cache(cache_key, result)
            except Exception:
                # Cache save failure shouldn't break the operation
                pass
        
        return result
    
    async def _semantic_search_with_embeddings_npy(
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
            # Fallback: return empty scores
            return {p.get("paper_id", ""): 0.0 for p in paper_inputs}
        
        # Get cached embeddings array
        embeddings_array = _get_embeddings_array(embeddings_path)
        
        if embeddings_array is None:
            logger.warning(
                "⚠ embeddings.npy not found or failed to load. "
                "Falling back to on-the-fly embedding computation. "
                "This will be slower and use more resources."
            )
            # Fallback: use SentenceTransformer to compute embeddings on-the-fly
            return await self._semantic_search_fallback(query, paper_inputs)
        
        try:
            # Get cached embedding model
            model = _get_embedding_model()
            if model is None:
                logger.warning(
                    "⚠ Failed to load embedding model. "
                    "Falling back to on-the-fly embedding computation."
                )
                return await self._semantic_search_fallback(query, paper_inputs)
            
            # Validate index mapping
            if not _validate_embeddings_metadata_mapping(embeddings_array, paper_inputs):
                logger.error(
                    "✗ Index mismatch between embeddings and metadata! "
                    "Semantic search results may be incorrect. "
                    "Falling back to on-the-fly computation for safety."
                )
                return await self._semantic_search_fallback(query, paper_inputs)
            
            logger.debug(f"Computing query embedding for: '{query[:50]}...'")
            # Get query embedding
            query_embedding = model.encode([query])[0]
            
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
                    # Normalize to 0.0-1.0 (cosine similarity is already -1.0 to 1.0)
                    score = float(max(0.0, similarities[idx]))
                    scores[paper_id] = score
                else:
                    # Handle case where paper_inputs is longer than embeddings (shouldn't happen after validation)
                    logger.warning(
                        f"⚠ Paper at index {idx} (ID: {paper.get('paper_id', 'unknown')}) "
                        f"exceeds embeddings array length {len(similarities)}. "
                        f"Setting score to 0.0"
                    )
                    paper_id = paper.get("paper_id", "")
                    scores[paper_id] = 0.0
            
            logger.info(
                f"✓ Semantic search completed: "
                f"{len(scores)} scores computed, "
                f"{len([s for s in scores.values() if s > 0])} non-zero scores"
            )
            
            return scores
            
        except Exception as e:
            logger.error(
                f"✗ Exception during semantic search with embeddings.npy: "
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
                f"🔄 Performing fallback semantic search: "
                f"computing embeddings for {len(paper_inputs)} papers on-the-fly. "
                f"This may take longer."
            )
            
            # Get cached embedding model
            model = _get_embedding_model()
            if model is None:
                logger.error(
                    "✗ Failed to load embedding model. "
                    "Cannot perform fallback semantic search."
                )
                return {p.get("paper_id", ""): 0.0 for p in paper_inputs}
            
            # Encode query
            logger.debug(f"Encoding query: '{query[:50]}...'")
            query_embedding = model.encode([query])[0]
            
            # Encode papers (title + abstract)
            logger.debug(f"Encoding {len(paper_inputs)} papers...")
            paper_texts = [
                f"{p.get('title', '')} {p.get('abstract', '')}"
                for p in paper_inputs
            ]
            paper_embeddings = model.encode(paper_texts, show_progress_bar=True)
            
            logger.debug(f"✓ Encoded {len(paper_embeddings)} paper embeddings")
            
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
                f"✓ Fallback semantic search completed: "
                f"{len(scores)} scores computed"
            )
            
            return scores
            
        except Exception as e:
            logger.error(
                f"✗ Exception during fallback semantic search: "
                f"{type(e).__name__}: {e}"
            )
            return {p.get("paper_id", ""): 0.0 for p in paper_inputs}
    
    def _keyword_search(
        self,
        query: str,
        paper_inputs: List[PaperInput]
    ) -> Dict[str, float]:
        """
        Perform keyword search on title and abstract with hierarchical scoring.
        
        Scoring hierarchy:
        1. Exact title match: 1.0
        2. Title contains/starts with query: 0.8 + coverage
        3. All words in title: 0.6-0.8
        4. Partial word matching: max 0.5
        
        Args:
            query: Search query
            paper_inputs: List of papers to search
            
        Returns:
            Dictionary mapping paper_id to keyword score (0.0-1.0)
        """
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        
        scores: Dict[str, float] = {}
        
        for paper in paper_inputs:
            paper_id = paper.get("paper_id", "")
            title = paper.get("title", "").strip()
            title_lower = title.lower()
            abstract = paper.get("abstract", "").lower()
            
            # 1. Exact title match (highest priority, maximum score)
            if title_lower == query_lower:
                scores[paper_id] = 1.0
                continue
            
            # 2. Title contains query or starts with query (high score with coverage)
            if query_lower in title_lower or title_lower.startswith(query_lower):
                # Calculate coverage: how much of query is covered by title
                title_words = set(title_lower.split())
                matched_words = query_words & title_words
                coverage = len(matched_words) / len(query_words) if query_words else 0
                
                # Title contains full query: very high score
                if query_lower in title_lower:
                    scores[paper_id] = min(1.0, 0.8 + coverage * 0.2)
                else:
                    # Title starts with query
                    scores[paper_id] = 0.7 + coverage * 0.1
                continue
            
            # 3. All query words are in title (moderate-high score)
            title_matches = sum(1 for word in query_words if word in title_lower)
            if title_matches == len(query_words) and query_words:
                # All words matched in title
                scores[paper_id] = 0.6 + (title_matches / max(len(query_words), 1)) * 0.2
                continue
            
            # 4. Partial word matching (low score, max 0.5)
            abstract_matches = sum(1 for word in query_words if word in abstract)
            
            # Weighted score: title 3x, abstract 1x
            score = (title_matches * 3.0 + abstract_matches * 1.0) / max(len(query_words), 1)
            
            # Normalize to 0.0-0.5 for partial matches
            score = min(0.5, score / 3.0)
            
            scores[paper_id] = score
        
        return scores
    
    def _combine_scores_weighted(
        self,
        semantic_scores: Dict[str, float],
        keyword_scores: Dict[str, float],
        semantic_weight: float = 0.3,
        keyword_weight: float = 0.7,
        exact_match_boost: float = 2.0
    ) -> Dict[str, float]:
        """
        Combine semantic and keyword scores using weighted average with exact match boost.
        
        Args:
            semantic_scores: Dictionary mapping paper_id to semantic score
            keyword_scores: Dictionary mapping paper_id to keyword score
            semantic_weight: Weight for semantic search (default: 0.3)
            keyword_weight: Weight for keyword search (default: 0.7)
            exact_match_boost: Multiplier for papers with keyword_score == 1.0 (default: 2.0)
            
        Returns:
            Dictionary mapping paper_id to combined score (0.0-1.0)
        """
        all_paper_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        combined_scores: Dict[str, float] = {}
        
        for paper_id in all_paper_ids:
            sem_score = semantic_scores.get(paper_id, 0.0)
            kw_score = keyword_scores.get(paper_id, 0.0)
            
            # Basic weighted average
            combined = (sem_score * semantic_weight + kw_score * keyword_weight)
            
            # Apply boost for exact matches (keyword_score >= 1.0)
            if kw_score >= 1.0:
                combined = combined * exact_match_boost
                combined = min(1.0, combined)  # Cap at 1.0
            
            combined_scores[paper_id] = combined
        
        return combined_scores
