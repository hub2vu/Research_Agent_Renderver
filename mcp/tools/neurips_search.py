"""
NeurIPS Search Tool

Hybrid search tool for NeurIPS 2025 papers combining semantic and keyword search with RRF.
"""

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

# Module-level caches for singleton pattern
_embedding_model_cache = None
_embeddings_array_cache = None
_embeddings_path_cache = None
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
    global _embeddings_array_cache, _embeddings_path_cache
    
    # Find embeddings file if not provided
    if embeddings_path is None:
        if _embeddings_path_cache is not None:
            embeddings_path = _embeddings_path_cache
        else:
            base_paths = [
                Path("data/embeddings_Neu/embeddings.npy"),
            ]
            
            workspace_root = Path.cwd()
            for base_path in base_paths:
                full_path = workspace_root / base_path
                if full_path.exists():
                    embeddings_path = str(full_path)
                    _embeddings_path_cache = embeddings_path
                    break
    
    if embeddings_path is None or not Path(embeddings_path).exists():
        return None
    
    # Return cached array if path matches
    if _embeddings_array_cache is not None and _embeddings_path_cache == embeddings_path:
        return _embeddings_array_cache
    
    # Load embeddings
    try:
        embeddings_array = np.load(embeddings_path)
        
        # Ensure embeddings are 2D
        if len(embeddings_array.shape) == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        
        # Cache it
        _embeddings_array_cache = embeddings_array
        _embeddings_path_cache = embeddings_path
        
        return embeddings_array
    except Exception:
        return None


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
        else:
            base_paths = [
                Path("data/embeddings_Neu/metadata.jsonl"),
                Path("data/embeddings_Neu/metadata.csv"),
                Path("data/NeurIPS 2025 Events.csv"),
            ]
            
            workspace_root = Path.cwd()
            for base_path in base_paths:
                full_path = workspace_root / base_path
                if full_path.exists():
                    metadata_path = str(full_path)
                    _metadata_path_cache = metadata_path
                    break
    
    if metadata_path is None:
        return None
    
    # Return cached metadata if path matches
    if _metadata_cache is not None and _metadata_path_cache == metadata_path:
        return _metadata_cache
    
    # Load metadata
    try:
        metadata = NeurIPSAdapter.load_neurips_metadata(metadata_path)
        _metadata_cache = metadata
        _metadata_path_cache = metadata_path
        return metadata
    except Exception:
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
                description="Weight for semantic search in RRF (default: 1.0)",
                required=False,
                default=1.0
            ),
            ToolParameter(
                name="keyword_weight",
                type="number",
                description="Weight for keyword search in RRF (default: 1.0)",
                required=False,
                default=1.0
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
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.0,
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
            try:
                neurips_metadata = NeurIPSAdapter.load_neurips_metadata(metadata_path)
                # Cache it for next time
                global _metadata_cache, _metadata_path_cache
                _metadata_cache = neurips_metadata
                if metadata_path:
                    _metadata_path_cache = metadata_path
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
        
        # Combine with RRF
        rrf_scores = self._apply_rrf(
            semantic_scores,
            keyword_scores,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            k=60  # RRF parameter
        )
        
        # Sort and select top N
        ranked_paper_ids = sorted(
            rrf_scores.keys(),
            key=lambda pid: rrf_scores[pid],
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
                paper["rrf_score"] = rrf_scores.get(pid, 0.0)
                selected_papers.append(paper)
        
        # Calculate statistics
        search_stats = {
            "semantic_matches": len([s for s in semantic_scores.values() if s > 0]),
            "keyword_matches": len([s for s in keyword_scores.values() if s > 0]),
            "rrf_combined_count": len(rrf_scores),
            "avg_rrf_score": float(np.mean(list(rrf_scores.values()))) if rrf_scores else 0.0
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
            # Fallback: return empty scores
            return {p.get("paper_id", ""): 0.0 for p in paper_inputs}
        
        # Get cached embeddings array
        embeddings_array = _get_embeddings_array(embeddings_path)
        
        if embeddings_array is None:
            # Fallback: use SentenceTransformer to compute embeddings on-the-fly
            return await self._semantic_search_fallback(query, paper_inputs)
        
        try:
            # Get cached embedding model
            model = _get_embedding_model()
            if model is None:
                return await self._semantic_search_fallback(query, paper_inputs)
            
            # Get query embedding
            query_embedding = model.encode([query])[0]
            
            # Calculate cosine similarity
            query_embedding_2d = query_embedding.reshape(1, -1)
            similarities = cosine_similarity(query_embedding_2d, embeddings_array)[0]
            
            # Map to paper IDs
            scores: Dict[str, float] = {}
            for idx, paper in enumerate(paper_inputs):
                if idx < len(similarities):
                    paper_id = paper.get("paper_id", "")
                    # Normalize to 0.0-1.0 (cosine similarity is already -1.0 to 1.0)
                    score = float(max(0.0, similarities[idx]))
                    scores[paper_id] = score
            
            return scores
            
        except Exception as e:
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
            return {p.get("paper_id", ""): 0.0 for p in paper_inputs}
        
        try:
            # Get cached embedding model
            model = _get_embedding_model()
            if model is None:
                return {p.get("paper_id", ""): 0.0 for p in paper_inputs}
            
            # Encode query
            query_embedding = model.encode([query])[0]
            
            # Encode papers (title + abstract)
            paper_texts = [
                f"{p.get('title', '')} {p.get('abstract', '')}"
                for p in paper_inputs
            ]
            paper_embeddings = model.encode(paper_texts)
            
            # Calculate cosine similarity
            query_embedding_2d = query_embedding.reshape(1, -1)
            similarities = cosine_similarity(query_embedding_2d, paper_embeddings)[0]
            
            # Map to paper IDs
            scores: Dict[str, float] = {}
            for idx, paper in enumerate(paper_inputs):
                paper_id = paper.get("paper_id", "")
                score = float(max(0.0, similarities[idx]))
                scores[paper_id] = score
            
            return scores
            
        except Exception:
            return {p.get("paper_id", ""): 0.0 for p in paper_inputs}
    
    def _keyword_search(
        self,
        query: str,
        paper_inputs: List[PaperInput]
    ) -> Dict[str, float]:
        """
        Perform keyword search on title and abstract.
        
        Args:
            query: Search query
            paper_inputs: List of papers to search
            
        Returns:
            Dictionary mapping paper_id to keyword score
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scores: Dict[str, float] = {}
        
        for paper in paper_inputs:
            paper_id = paper.get("paper_id", "")
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            
            # Calculate keyword matches
            title_matches = sum(1 for word in query_words if word in title)
            abstract_matches = sum(1 for word in query_words if word in abstract)
            
            # Weighted score: title 3x, abstract 1x
            score = (title_matches * 3.0 + abstract_matches * 1.0) / max(len(query_words), 1)
            
            # Normalize to 0.0-1.0
            score = min(1.0, score / 3.0)  # Max possible score is 3.0 (all words in title)
            
            scores[paper_id] = score
        
        return scores
    
    def _apply_rrf(
        self,
        semantic_scores: Dict[str, float],
        keyword_scores: Dict[str, float],
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.0,
        k: int = 60
    ) -> Dict[str, float]:
        """
        Apply Reciprocal Rank Fusion (RRF) to combine semantic and keyword search results.
        
        Args:
            semantic_scores: Dictionary mapping paper_id to semantic score
            keyword_scores: Dictionary mapping paper_id to keyword score
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword search
            k: RRF parameter (default: 60)
            
        Returns:
            Dictionary mapping paper_id to RRF score
        """
        # Rank papers by each method
        semantic_ranked = sorted(
            semantic_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        keyword_ranked = sorted(
            keyword_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create rank dictionaries (1-indexed)
        semantic_ranks: Dict[str, int] = {}
        for rank, (paper_id, _) in enumerate(semantic_ranked, start=1):
            semantic_ranks[paper_id] = rank
        
        keyword_ranks: Dict[str, int] = {}
        for rank, (paper_id, _) in enumerate(keyword_ranked, start=1):
            keyword_ranks[paper_id] = rank
        
        # Calculate RRF scores
        all_paper_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        rrf_scores: Dict[str, float] = {}
        
        for paper_id in all_paper_ids:
            sem_rank = semantic_ranks.get(paper_id, float('inf'))
            kw_rank = keyword_ranks.get(paper_id, float('inf'))
            
            # RRF formula with weights
            rrf_score = (
                semantic_weight / (k + sem_rank) +
                keyword_weight / (k + kw_rank)
            )
            
            rrf_scores[paper_id] = rrf_score
        
        return rrf_scores
