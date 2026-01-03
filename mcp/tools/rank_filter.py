"""
Rank and Filter Papers Tool

Provides tools for ranking and filtering research papers based on metadata
(title, authors, abstract) and user profile. This tool analyzes only abstracts;
full PDF text analysis is handled by paper_analyzer.
"""

from typing import Any, Dict, List, Optional

from ..base import MCPTool, ToolParameter, ExecutionError

from datetime import datetime
from typing import Tuple

from .rank_filter_utils import (
    PaperInput,
    UserProfile,
    FilteredPaper,
    load_profile,
    load_history,
    scan_local_pdfs,
    filter_papers,
    _calculate_embedding_scores,
    _calculate_keyword_scores,
    _classify_papers_by_score,
    _verify_with_llm,
    _merge_scores,
    _calculate_dimension_scores,
    _apply_soft_penalty,
    _calculate_final_score,
    _rank_and_select,
    _select_contrastive_paper,
    _generate_tags,
    _generate_comparison_notes,
    _save_and_format_result,
)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore


class RankAndFilterPapersTool(MCPTool):
    """
    Rank and filter papers based on metadata analysis.
    
    This tool acts as a gatekeeper, performing lightweight evaluation based on
    metadata (title, authors, abstract) to pre-select papers before expensive
    PDF full-text analysis. It does NOT analyze PDF content - that is the role
    of paper_analyzer.
    """
    
    # Class-level embedding model cache
    _embedding_model: Optional[SentenceTransformer] = None
    _model_load_failed: bool = False

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

    def _empty_result(
        self,
        filtered_papers: List[FilteredPaper],
        input_count: int,
        purpose: str,
        ranking_mode: str,
        profile_path: Optional[str]
    ) -> Dict[str, Any]:
        """
        Generate an empty result dictionary when all papers are filtered or input is empty.
        
        Args:
            filtered_papers: List of filtered papers
            input_count: Number of input papers
            purpose: Research purpose
            ranking_mode: Ranking mode
            profile_path: Optional profile path used
            
        Returns:
            Dictionary with empty result structure
        """
        return {
            "success": True,
            "error": None,
            "summary": {
                "input_count": input_count,
                "filtered_count": len(filtered_papers),
                "scored_count": 0,
                "output_count": 0,
                "purpose": purpose,
                "ranking_mode": ranking_mode,
                "profile_used": profile_path,
                "llm_verification_used": False,
                "llm_calls_made": 0
            },
            "ranked_papers": [],
            "filtered_papers": filtered_papers,
            "contrastive_paper": None,
            "comparison_notes": [],
            "output_path": None,
            "generated_at": datetime.now().isoformat()
        }

    def _get_embedding_model(self) -> Optional[SentenceTransformer]:
        """
        Get the embedding model with lazy loading.
        
        Returns the cached model if already loaded, attempts to load it if not,
        or returns None if loading failed previously or library is not available.
        
        Returns:
            SentenceTransformer model instance or None if unavailable/failed
        """
        # If already loaded, return it
        if self._embedding_model is not None:
            return self._embedding_model
        
        # If loading failed before, don't try again
        if self._model_load_failed:
            return None
        
        # Check if library is available
        if not HAS_SENTENCE_TRANSFORMERS:
            self._model_load_failed = True
            return None
        
        # Try to load the model
        try:
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            return self._embedding_model
        except Exception:
            # Mark as failed so we don't try again
            self._model_load_failed = True
            return None

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
        """
        try:
            # Parameter extraction and validation
            if not papers:
                return self._empty_result(
                    [],
                    0,
                    purpose,
                    ranking_mode,
                    profile_path
                )
            
            # Convert input papers to PaperInput format
            paper_inputs: List[PaperInput] = [PaperInput(**p) for p in papers]
            
            # Phase 1: Load data
            profile = load_profile(profile_path, tool_name=self.name)
            history = load_history(history_path or "history/read_papers.json")
            local_pdfs = scan_local_pdfs(local_pdf_dir)
            
            # Phase 2: Filter papers
            passed_papers, filtered_papers = filter_papers(
                paper_inputs, profile, history, purpose
            )
            
            if not passed_papers:
                return self._empty_result(
                    filtered_papers,
                    len(papers),
                    purpose,
                    ranking_mode,
                    profile_path
                )
            
            # Phase 3: Scoring
            # Stage 1: Embedding scores
            embedding_scores = _calculate_embedding_scores(passed_papers, profile)
            high_group, mid_group, low_group = _classify_papers_by_score(passed_papers, embedding_scores)
            
            # Stage 2: LLM verification (conditional)
            llm_results = {}
            llm_calls = 0
            if enable_llm_verification and mid_group:
                llm_results = await _verify_with_llm(mid_group, profile, batch_size=5)
                llm_calls = (len(mid_group) + 4) // 5  # batch_size 5 기준
            
            # Merge scores
            merged_scores = _merge_scores(
                embedding_scores,
                llm_results,
                {p["paper_id"] for p in high_group},
                {p["paper_id"] for p in low_group}
            )
            
            # Phase 4-5: Dimension scores + penalty + weighted final score
            final_scores: Dict[str, Dict[str, Any]] = {}
            for paper in passed_papers:
                paper_id = paper["paper_id"]
                semantic_info = merged_scores.get(paper_id, {"semantic_score": 0.0, "evaluation_method": "unknown", "llm_reason": None})
                
                dim_scores = _calculate_dimension_scores(
                    paper, profile, semantic_info["semantic_score"], local_pdfs
                )
                
                penalty, penalty_kws = _apply_soft_penalty(paper, profile)
                
                final = _calculate_final_score(dim_scores, penalty, purpose, ranking_mode)
                
                final_scores[paper_id] = {
                    "final_score": final,
                    "breakdown": dim_scores,
                    "soft_penalty": penalty,
                    "penalty_keywords": penalty_kws,
                    "evaluation_method": semantic_info["evaluation_method"],
                    "llm_reason": semantic_info.get("llm_reason")
                }
            
            # Phase 6: Ranking and selection
            selected = _rank_and_select(
                passed_papers, final_scores, top_k, ranking_mode, include_contrastive
            )
            
            # Phase 7: Contrastive (conditional)
            contrastive_result = None
            if include_contrastive:
                selected_ids = {p["paper_id"] for p in selected}
                remaining = [p for p in passed_papers if p["paper_id"] not in selected_ids]
                contrastive_result = _select_contrastive_paper(
                    selected, remaining, contrastive_type, final_scores
                )
            
            # Phase 8: Tagging and result formatting
            ranked_results = []
            for rank, paper in enumerate(selected, 1):
                paper_id = paper["paper_id"]
                tags = _generate_tags(paper, final_scores[paper_id], local_pdfs, False, None)
                
                ranked_results.append({
                    "rank": rank,
                    "paper_id": paper_id,
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []),
                    "published": paper.get("published"),
                    "score": {
                        "final": final_scores[paper_id]["final_score"],
                        "breakdown": final_scores[paper_id]["breakdown"],
                        "soft_penalty": final_scores[paper_id]["soft_penalty"],
                        "penalty_keywords": final_scores[paper_id]["penalty_keywords"],
                        "evaluation_method": final_scores[paper_id]["evaluation_method"],
                    },
                    "tags": tags,
                    "local_status": {
                        "already_downloaded": paper_id in local_pdfs,
                        "local_path": f"pdf/{paper_id}.pdf" if paper_id in local_pdfs else None
                    },
                    "original_data": paper
                })
            
            # Contrastive formatting
            contrastive_formatted = None
            if contrastive_result:
                contrastive_paper, contrastive_info = contrastive_result
                paper_id = contrastive_paper["paper_id"]
                
                # Get score info - use default if not in final_scores (shouldn't happen in normal flow, but handle for safety)
                score_info = final_scores.get(paper_id, {
                    "final_score": 0.0,
                    "breakdown": {},
                    "soft_penalty": 0.0,
                    "penalty_keywords": [],
                    "evaluation_method": "unknown"
                })
                
                tags = _generate_tags(contrastive_paper, score_info, local_pdfs, True, contrastive_type)
                
                contrastive_formatted = {
                    "paper_id": paper_id,
                    "title": contrastive_paper.get("title", ""),
                    "authors": contrastive_paper.get("authors", []),
                    "published": contrastive_paper.get("published"),
                    "score": {
                        "final": score_info["final_score"],
                        "breakdown": score_info["breakdown"],
                        "soft_penalty": score_info["soft_penalty"],
                        "penalty_keywords": score_info["penalty_keywords"],
                        "evaluation_method": score_info["evaluation_method"],
                    },
                    "tags": tags,
                    "contrastive_info": contrastive_info,
                    "original_data": contrastive_paper
                }
            
            # Comparison notes
            contrastive_paper_for_notes = contrastive_result[0] if contrastive_result else None
            comparison_notes = _generate_comparison_notes(
                selected,
                final_scores,
                contrastive_paper_for_notes
            )
            
            # Phase 9: Save and return
            summary = {
                "input_count": len(papers),
                "filtered_count": len(filtered_papers),
                "scored_count": len(passed_papers),
                "output_count": len(ranked_results) + (1 if contrastive_formatted else 0),
                "purpose": purpose,
                "ranking_mode": ranking_mode,
                "profile_used": profile_path,
                "llm_verification_used": enable_llm_verification and bool(mid_group),
                "llm_calls_made": llm_calls
            }
            
            return _save_and_format_result(
                ranked_results,
                filtered_papers,
                contrastive_formatted,
                comparison_notes,
                summary,
                profile_path
            )
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "summary": None,
                "ranked_papers": [],
                "filtered_papers": [],
                "contrastive_paper": None,
                "comparison_notes": [],
                "output_path": None,
                "generated_at": datetime.now().isoformat()
            }

