"""
Rank and Filter Papers Tool

- UpdateUserProfileTool
- ApplyHardFiltersTool
- CalculateSemanticScoresTool
- EvaluatePaperMetricsTool
- RankAndSelectTopKTool
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
import json

from ..base import MCPTool, ToolParameter, ExecutionError
from .rank_filter_utils import (
    PaperInput,
    UserProfile,
    FilteredPaper,
    load_profile,
    load_history,
    scan_local_pdfs,
    filter_papers,
    _calculate_embedding_scores,
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
from .rank_filter_utils.rankers import _is_neurips_data
from .rank_filter_utils.path_resolver import resolve_path, ensure_directory


# ========== 1) UpdateUserProfileTool ==========
class UpdateUserProfileTool(MCPTool):
    @property
    def name(self) -> str:
        return "update_user_profile"

    @property
    def description(self) -> str:
        return (
            "Update the user profile (interests, keywords) and toggle exclude_local_papers. "
            "Writes to OUTPUT_DIR/users/profile.json by default."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="profile_path",
                type="string",
                description=(
                    "Profile JSON path resolved under OUTPUT_DIR (default: users/profile.json). "
                    "Contains user interests, keywords, preferred authors/institutions, and constraints "
                    "(including exclude_local_papers toggle). If file doesn't exist, a default profile is created."
                ),
                required=False,
                default="users/profile.json",
            ),
            ToolParameter(
                name="interests",
                type="object",
                description=(
                    "Optional interests object with keys: primary (array of strings), secondary (array of strings), "
                    "exploratory (array of strings). Primary interests have highest weight in semantic scoring, "
                    "secondary have medium weight, exploratory have lower weight. Only provided keys are updated."
                ),
                required=False,
            ),
            ToolParameter(
                name="keywords",
                type="object",
                description=(
                    "Optional keywords object with keys: must_include (array of strings), "
                    "exclude (object with hard and soft arrays). must_include keywords must appear in title/abstract. "
                    "hard exclude keywords cause immediate filtering, soft exclude keywords apply penalty. "
                    "Only provided keys are updated."
                ),
                required=False,
            ),
            ToolParameter(
                name="exclude_local_papers",
                type="boolean",
                description="If true, locally existing PDFs will be treated as already read.",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="purpose",
                type="string",
                description=(
                    "Research purpose/goal. Choose 'general' for balanced recommendation, "
                    "'literature_review' for broad coverage, 'implementation' to require code, "
                    "'idea_generation' to prioritize novelty."
                ),
                required=False,
            ),
            ToolParameter(
                name="ranking_mode",
                type="string",
                description=(
                    "Scoring emphasis mode. Choose 'balanced' for equal weight, "
                    "'novelty' to emphasize recent papers, 'practicality' to emphasize code availability, "
                    "'diversity' to minimize overlap."
                ),
                required=False,
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="Maximum number of papers to return after ranking.",
                required=False,
            ),
            ToolParameter(
                name="include_contrastive",
                type="boolean",
                description="If true, include a contrastive paper that presents different approach.",
                required=False,
            ),
            ToolParameter(
                name="contrastive_type",
                type="string",
                description=(
                    "Type of contrastive paper. Choose 'method' for same problem with different methodology, "
                    "'assumption' for same method with different assumptions, "
                    "'domain' for same technique applied to different domain."
                ),
                required=False,
            ),
            ToolParameter(
                name="preferred_authors",
                type="array",
                items_type="string",
                description="List of preferred author names.",
                required=False,
            ),
            ToolParameter(
                name="preferred_institutions",
                type="array",
                items_type="string",
                description="List of preferred institution names.",
                required=False,
            ),
            ToolParameter(
                name="constraints",
                type="object",
                description=(
                    "Optional constraints object with keys: min_year (integer), require_code (boolean), "
                    "exclude_local_papers (boolean). Only provided keys are updated."
                ),
                required=False,
            ),
        ]

    @property
    def category(self) -> str:
        return "ranking"

    async def execute(
        self,
        profile_path: str = "users/profile.json",
        interests: Optional[Dict[str, List[str]]] = None,
        keywords: Optional[Dict[str, Any]] = None,
        exclude_local_papers: Optional[bool] = None,
        purpose: Optional[str] = None,
        ranking_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        include_contrastive: Optional[bool] = None,
        contrastive_type: Optional[str] = None,
        preferred_authors: Optional[List[str]] = None,
        preferred_institutions: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        profile = load_profile(profile_path, tool_name=self.name)

        # Merge updates
        if interests is not None:
            for k in ["primary", "secondary", "exploratory"]:
                if k in interests:
                    profile["interests"][k] = interests.get(k, []) or []

        if keywords is not None:
            if "must_include" in keywords:
                profile["keywords"]["must_include"] = keywords.get("must_include", []) or []
            if "exclude" in keywords:
                excl = keywords.get("exclude", {}) or {}
                profile["keywords"]["exclude"]["hard"] = excl.get("hard", []) or []
                profile["keywords"]["exclude"]["soft"] = excl.get("soft", []) or []

        if exclude_local_papers is not None:
            profile["constraints"]["exclude_local_papers"] = bool(exclude_local_papers)

        if purpose is not None:
            profile["purpose"] = purpose

        if ranking_mode is not None:
            profile["ranking_mode"] = ranking_mode

        if top_k is not None:
            profile["top_k"] = top_k

        if include_contrastive is not None:
            profile["include_contrastive"] = bool(include_contrastive)

        if contrastive_type is not None:
            profile["contrastive_type"] = contrastive_type

        if preferred_authors is not None:
            if len(preferred_authors) == 0:
                # Remove field if empty array
                if "preferred_authors" in profile:
                    del profile["preferred_authors"]
            else:
                profile["preferred_authors"] = preferred_authors

        if preferred_institutions is not None:
            if len(preferred_institutions) == 0:
                # Remove field if empty array
                if "preferred_institutions" in profile:
                    del profile["preferred_institutions"]
            else:
                profile["preferred_institutions"] = preferred_institutions

        if constraints is not None:
            if "min_year" in constraints:
                profile["constraints"]["min_year"] = constraints["min_year"]
            if "require_code" in constraints:
                profile["constraints"]["require_code"] = bool(constraints["require_code"])
            if "exclude_local_papers" in constraints:
                profile["constraints"]["exclude_local_papers"] = bool(constraints["exclude_local_papers"])

        # Save
        resolved = resolve_path(profile_path, path_type="output")
        ensure_directory(resolved)
        with open(resolved, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

        return {
            "profile_path": str(resolved),
            "updated": True,
            "exclude_local_papers": profile["constraints"]["exclude_local_papers"],
            "purpose": profile.get("purpose", "general"),
            "ranking_mode": profile.get("ranking_mode", "balanced"),
            "top_k": profile.get("top_k", 5),
            "include_contrastive": profile.get("include_contrastive", False),
            "contrastive_type": profile.get("contrastive_type", "method"),
        }


# ========== 2) ApplyHardFiltersTool ==========
class ApplyHardFiltersTool(MCPTool):
    @property
    def name(self) -> str:
        return "apply_hard_filters"

    @property
    def description(self) -> str:
        return (
            "Apply hard filters (already read, blacklist keywords, year). "
            "Analyzes search result metadata (title, authors, abstract) to evaluate "
            "and filter papers according to user profile. "
            "If profile.constraints.exclude_local_papers is true, treat local PDFs as ALREADY_READ."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="papers",
                type="array",
                items_type="object",
                description="Input papers (PaperInput-like objects).",
                required=True,
            ),
            ToolParameter(
                name="profile_path",
                type="string",
                description=(
                    "Profile JSON path resolved under OUTPUT_DIR (default: users/profile.json). "
                    "Contains filtering rules (keywords, year constraints) and exclude_local_papers toggle."
                ),
                required=False,
                default="users/profile.json",
            ),
            ToolParameter(
                name="history_path",
                type="string",
                description=(
                    "History JSON path resolved under OUTPUT_DIR (default: history/read_papers.json). "
                    "Contains list of already-read paper IDs. Papers in this list are filtered out as ALREADY_READ."
                ),
                required=False,
                default="history/read_papers.json",
            ),
            ToolParameter(
                name="local_pdf_dir",
                type="string",
                description="Local PDF directory (resolved under PDF_DIR).",
                required=False,
                default="pdf/",
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
                default="general",
            ),
        ]

    @property
    def category(self) -> str:
        return "ranking"

    async def execute(
        self,
        papers: List[Dict[str, Any]],
        profile_path: str = "users/profile.json",
        history_path: str = "history/read_papers.json",
        local_pdf_dir: str = "pdf/",
        purpose: str = "general",
    ) -> Dict[str, Any]:
        if not papers:
            return {"passed_papers": [], "filtered_papers": [], "filter_summary": {"total": 0, "passed": 0, "filtered": 0}}

        profile = load_profile(profile_path, tool_name=self.name)
        history = load_history(history_path)

        # Treat local PDFs as ALREADY_READ when toggle is True
        exclude_local = profile["constraints"].get("exclude_local_papers", False)
        if exclude_local:
            local_ids = scan_local_pdfs(local_pdf_dir)
            history = set(history).union(local_ids)

        paper_inputs: List[PaperInput] = [PaperInput(**p) for p in papers]

        passed, filtered = filter_papers(paper_inputs, profile, history, purpose)

        return {
            "passed_papers": passed,
            "filtered_papers": filtered,
            "filter_summary": {
                "total": len(papers),
                "passed": len(passed),
                "filtered": len(filtered),
            },
        }


# ========== 3) CalculateSemanticScoresTool ==========
class CalculateSemanticScoresTool(MCPTool):
    @property
    def name(self) -> str:
        return "calculate_semantic_scores"

    @property
    def description(self) -> str:
        return (
            "Calculate hybrid semantic relevance scores for papers using embeddings "
            "and optional LLM verification for borderline cases."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="papers",
                type="array",
                items_type="object",
                description="Papers to score (after filters).",
                required=True,
            ),
            ToolParameter(
                name="profile_path",
                type="string",
                description=(
                    "Profile JSON path resolved under OUTPUT_DIR (default: users/profile.json). "
                    "Contains user interests used for semantic relevance calculation."
                ),
                required=False,
                default="users/profile.json",
            ),
            ToolParameter(
                name="enable_llm_verification",
                type="boolean",
                description=(
                    "If true, performs LLM batch verification on borderline papers (embedding score 0.4-0.7) "
                    "to improve accuracy. If false, uses embedding scores only (faster but less accurate). "
                    "Set to false when processing large batches or when speed is critical. "
                    "Recommended: true for top_k <= 10, false for top_k > 15."
                ),
                required=False,
                default=True,
            ),
        ]

    @property
    def category(self) -> str:
        return "ranking"

    async def execute(
        self,
        papers: List[Dict[str, Any]],
        profile_path: str = "users/profile.json",
        enable_llm_verification: bool = True,
    ) -> Dict[str, Any]:
        if not papers:
            return {"scores": {}}

        profile = load_profile(profile_path, tool_name=self.name)
        paper_inputs: List[PaperInput] = [PaperInput(**p) for p in papers]

        embedding_scores = _calculate_embedding_scores(paper_inputs, profile)
        high_group, mid_group, low_group = _classify_papers_by_score(paper_inputs, embedding_scores)

        llm_results: Dict[str, Tuple[float, str]] = {}
        if enable_llm_verification and mid_group:
            # Use token-efficient LLM verification (key sentences extraction)
            llm_results = await _verify_with_llm(mid_group, profile, batch_size=5, use_key_sentences=True)

        merged = _merge_scores(
            embedding_scores,
            llm_results,
            {p["paper_id"] for p in high_group},
            {p["paper_id"] for p in low_group},
        )

        return {"scores": merged}


# ========== 4) EvaluatePaperMetricsTool ==========
class EvaluatePaperMetricsTool(MCPTool):
    @property
    def name(self) -> str:
        return "evaluate_paper_metrics"

    @property
    def description(self) -> str:
        return (
            "Evaluate metrics for each paper (keywords, authors, institutions, recency, practicality) "
            "using provided semantic scores. Automatically searches for GitHub URLs in extracted PDF text "
            "if github_url is not already provided in the paper data."
        )

    async def _find_github_urls(self, paper_id: str) -> Optional[str]:
        """
        Search for GitHub URLs in extracted PDF text file using CheckGithubLinkTool.
        
        Args:
            paper_id: Paper identifier (filename without .pdf extension)
            
        Returns:
            First GitHub URL found, or None if not found or text file doesn't exist
        """
        # Import here to avoid circular dependency
        from .pdf import CheckGithubLinkTool
        
        # Use CheckGithubLinkTool to find GitHub URLs
        github_tool = CheckGithubLinkTool()
        result = await github_tool.execute(paper_id)
        
        # Extract first URL from result
        github_urls = result.get("github_urls")
        if github_urls and isinstance(github_urls, list) and len(github_urls) > 0:
            return github_urls[0]
        
        return None

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="papers",
                type="array",
                items_type="object",
                description="Papers to evaluate.",
                required=True,
            ),
            ToolParameter(
                name="semantic_scores",
                type="object",
                description="Semantic scores map from calculate_semantic_scores.",
                required=True,
            ),
            ToolParameter(
                name="profile_path",
                type="string",
                description=(
                    "Profile JSON path resolved under OUTPUT_DIR (default: users/profile.json). "
                    "Contains keywords, preferred authors/institutions used for metrics calculation."
                ),
                required=False,
                default="users/profile.json",
            ),
            ToolParameter(
                name="local_pdf_dir",
                type="string",
                description=(
                    "Local PDF directory path resolved under PDF_DIR (default: pdf/). "
                    "Used to check which papers are already available locally for practicality scoring. "
                    "Only used if local_pdfs is not provided."
                ),
                required=False,
                default="pdf/",
            ),
            ToolParameter(
                name="local_pdfs",
                type="array",
                items_type="string",
                description=(
                    "Optional: Pre-scanned set of paper IDs that exist locally (from apply_hard_filters or previous scan). "
                    "If provided, avoids re-scanning the PDF directory. If not provided, will scan local_pdf_dir automatically."
                ),
                required=False,
            ),
            ToolParameter(
                name="neurips_cluster_map",
                type="object",
                description=(
                    "Optional: NeurIPS cluster mapping (paper_id -> cluster_id). "
                    "If provided, enables cluster-based metrics calculation. "
                    "Auto-loaded if papers are from NeurIPS dataset."
                ),
                required=False,
                default=None,
            ),
            ToolParameter(
                name="neurips_selected_clusters",
                type="array",
                items_type="integer",
                description=(
                    "Optional: List of cluster IDs that are already well-represented in results. "
                    "Used for diversity calculation in NeurIPS metrics."
                ),
                required=False,
                default=None,
            ),
        ]

    @property
    def category(self) -> str:
        return "ranking"

    async def execute(
        self,
        papers: List[Dict[str, Any]],
        semantic_scores: Dict[str, Dict[str, Any]],
        profile_path: str = "users/profile.json",
        local_pdf_dir: str = "pdf/",
        local_pdfs: Optional[List[str]] = None,
        neurips_cluster_map: Optional[Dict[str, int]] = None,
        neurips_selected_clusters: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        if not papers:
            return {"scores": {}}

        profile = load_profile(profile_path, tool_name=self.name)
        
        # Use provided local_pdfs or scan if not provided
        if local_pdfs is not None:
            local_pdfs_set: Set[str] = set(local_pdfs)
        else:
            local_pdfs_set = scan_local_pdfs(local_pdf_dir)
        
        paper_inputs: List[PaperInput] = [PaperInput(**p) for p in papers]

        # Automatically enrich papers with GitHub URLs if missing
        for paper in paper_inputs:
            # Only search if github_url is not already provided
            if not paper.get("github_url"):
                paper_id = paper.get("paper_id", "")
                if paper_id:
                    github_url = await self._find_github_urls(paper_id)
                    if github_url:
                        paper["github_url"] = github_url

        # Load NeurIPS cluster data if not provided and papers are from NeurIPS
        if neurips_cluster_map is None:
            is_neurips = _is_neurips_data(paper_inputs)
            if is_neurips:
                try:
                    from .rank_filter_utils.neurips_metrics import load_neurips_cluster_data
                    neurips_cluster_map, cluster_sizes = load_neurips_cluster_data()
                except Exception:
                    neurips_cluster_map = None
                    cluster_sizes = {}
        else:
            # If cluster_map provided, load cluster sizes
            try:
                from .rank_filter_utils.neurips_metrics import load_neurips_cluster_data
                _, cluster_sizes = load_neurips_cluster_data()
            except Exception:
                cluster_sizes = {}
        
        results: Dict[str, Dict[str, Any]] = {}
        ranked_papers_so_far: List[PaperInput] = []  # For diversity calculation
        
        for paper in paper_inputs:
            pid = paper.get("paper_id", "")
            sem = semantic_scores.get(pid, {})
            sem_score = float(sem.get("semantic_score", 0.0))

            breakdown = _calculate_dimension_scores(paper, profile, sem_score, local_pdfs_set)
            penalty, penalty_kws = _apply_soft_penalty(paper, profile)
            
            # Add NeurIPS-specific metrics if cluster data available
            # Only calculates diversity_penalty (cluster-interest alignment removed for simplicity)
            if neurips_cluster_map:
                try:
                    from .rank_filter_utils.neurips_metrics import _calculate_neurips_specific_metrics
                    neurips_metrics = _calculate_neurips_specific_metrics(
                        paper=paper,
                        cluster_map=neurips_cluster_map,
                        cluster_sizes=cluster_sizes,
                        profile=profile,
                        ranked_papers_so_far=ranked_papers_so_far
                    )
                    # Apply diversity penalty to soft_penalty
                    diversity_penalty = neurips_metrics.get("diversity_penalty", 0.0)
                    penalty += diversity_penalty
                except Exception:
                    # If NeurIPS metrics calculation fails, continue without them
                    pass

            results[pid] = {
                "breakdown": breakdown,
                "soft_penalty": penalty,
                "penalty_keywords": penalty_kws,
            }
            
            # Track for diversity calculation (simplified - just add to list)
            ranked_papers_so_far.append(paper)

        return {"scores": results}


# ========== 5) RankAndSelectTopKTool ==========
class RankAndSelectTopKTool(MCPTool):
    @property
    def name(self) -> str:
        return "rank_and_select_top_k"

    @property
    def description(self) -> str:
        return (
            "Combine semantic and metrics scores to compute final scores, rank papers, "
            "and select Top-K. Optionally include a contrastive paper."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="papers",
                type="array",
                items_type="object",
                description="Papers to rank.",
                required=True,
            ),
            ToolParameter(
                name="semantic_scores",
                type="object",
                description="Semantic scores map (from calculate_semantic_scores).",
                required=True,
            ),
            ToolParameter(
                name="metrics_scores",
                type="object",
                description="Metrics scores map (from evaluate_paper_metrics).",
                required=True,
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
                default=5,
            ),
            ToolParameter(
                name="purpose",
                type="string",
                description=(
                    "Research purpose/goal that determines score weight distribution. "
                    "Choose 'general' for balanced recommendation across all factors, "
                    "'literature_review' for broad coverage and diversity (relaxes year constraints), "
                    "'implementation' to prioritize code availability and reproducibility, "
                    "'idea_generation' to prioritize novelty and recent papers with innovative approaches."
                ),
                required=False,
                default="general",
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
                default="balanced",
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
                default=False,
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
                default="method",
            ),
            ToolParameter(
                name="local_pdf_dir",
                type="string",
                description=(
                    "Local PDF directory path resolved under PDF_DIR (default: pdf/). "
                    "Used to check which papers are already available locally for tagging. "
                    "Only used if local_pdfs is not provided."
                ),
                required=False,
                default="pdf/",
            ),
            ToolParameter(
                name="local_pdfs",
                type="array",
                items_type="string",
                description=(
                    "Optional: Pre-scanned set of paper IDs that exist locally (from apply_hard_filters or evaluate_paper_metrics). "
                    "If provided, avoids re-scanning the PDF directory. If not provided, will scan local_pdf_dir automatically."
                ),
                required=False,
            ),
            ToolParameter(
                name="profile_path",
                type="string",
                description=(
                    "Profile JSON path resolved under OUTPUT_DIR (default: users/profile.json). "
                    "Used for saving ranking results. The profile itself is not modified."
                ),
                required=False,
                default="users/profile.json",
            ),
            ToolParameter(
                name="cluster_k",
                type="integer",
                description="K value for NeurIPS clustering (default: 15, from env NEURIPS_CLUSTER_K)",
                required=False,
                default=None
            ),
        ]

    @property
    def category(self) -> str:
        return "ranking"

    async def execute(
        self,
        papers: List[Dict[str, Any]],
        semantic_scores: Dict[str, Dict[str, Any]],
        metrics_scores: Dict[str, Dict[str, Any]],
        top_k: int = 5,
        purpose: str = "general",
        ranking_mode: str = "balanced",
        include_contrastive: bool = False,
        contrastive_type: str = "method",
        local_pdf_dir: str = "pdf/",
        local_pdfs: Optional[List[str]] = None,
        profile_path: str = "users/profile.json",
        cluster_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not papers:
            summary = {
                "input_count": 0,
                "filtered_count": 0,
                "scored_count": 0,
                "output_count": 0,
                "purpose": purpose,
                "ranking_mode": ranking_mode,
                "profile_used": profile_path,
                "llm_verification_used": False,
                "llm_calls_made": 0,
            }
            return _save_and_format_result([], [], None, [], summary, profile_path)

        # Use provided local_pdfs or scan if not provided
        if local_pdfs is not None:
            local_pdfs_set: Set[str] = set(local_pdfs)
        else:
            local_pdfs_set = scan_local_pdfs(local_pdf_dir)
        
        paper_inputs: List[PaperInput] = [PaperInput(**p) for p in papers]
        
        # Extract keyword_score and combined_score from papers (preserved from neurips_search)
        keyword_scores: Dict[str, float] = {}
        combined_scores: Dict[str, float] = {}
        for p in papers:
            pid = p.get("paper_id", "")
            if pid:
                keyword_scores[pid] = float(p.get("keyword_score", 0.0))
                combined_scores[pid] = float(p.get("combined_score", 0.0))
        
        # Build final scores
        final_scores: Dict[str, Dict[str, Any]] = {}
        for p in paper_inputs:
            pid = p.get("paper_id", "")
            sem_info = semantic_scores.get(pid, {})
            sem_score = float(sem_info.get("semantic_score", 0.0))

            met = metrics_scores.get(pid, {})
            breakdown = met.get("breakdown", {})
            soft_penalty = float(met.get("soft_penalty", 0.0))

            # If breakdown not provided, compute minimal one
            if not breakdown:
                profile = load_profile(profile_path, tool_name=self.name)
                breakdown = _calculate_dimension_scores(p, profile, sem_score, local_pdfs_set)
                penalty, _ = _apply_soft_penalty(p, profile)
                soft_penalty = penalty

            # Extract keyword_score for this paper and add to breakdown
            kw_score = keyword_scores.get(pid, 0.0)
            combined_score = combined_scores.get(pid, 0.0)
            
            # Add keyword_score to breakdown for UI display
            breakdown["keyword_score"] = kw_score
            
            # Calculate initial final score
            final = _calculate_final_score(breakdown, soft_penalty, purpose, ranking_mode)
            
            # Boost exact matches (keyword_score >= 1.0) to ensure top ranking
            if kw_score >= 1.0:
                # For exact matches, prioritize the combined_score from neurips_search
                # which already incorporates keyword and semantic scores with proper weighting
                # Ensure it ranks at the top by setting minimum high score
                if combined_score > 0:
                    # Use combined_score as the base, ensure it's high enough
                    final = max(final, combined_score, 0.9)
                else:
                    # Fallback: boost to very high score to ensure top ranking
                    final = max(final, 0.95)
            
            final_scores[pid] = {
                    "final_score": final,
                "breakdown": breakdown,
                "soft_penalty": soft_penalty,
                "penalty_keywords": met.get("penalty_keywords", []),
                "evaluation_method": sem_info.get("evaluation_method", "unknown"),
            }

        # Rank and select
        # Check if NeurIPS data and use cluster quota if available
        is_neurips = _is_neurips_data(paper_inputs)
        cluster_map: Optional[Dict[str, int]] = None
        
        if is_neurips:
            # Try to load cluster map for NeurIPS
            try:
                from .rank_filter_utils.neurips_metrics import load_neurips_cluster_data
                cluster_map, _ = load_neurips_cluster_data(cluster_k=cluster_k)
            except Exception:
                cluster_map = None
        
        if is_neurips and cluster_map:
            # Use cluster quota ranking for NeurIPS
            from .rank_filter_utils.rankers import _rank_with_cluster_quota
            selected = _rank_with_cluster_quota(paper_inputs, final_scores, top_k, cluster_map)
        else:
            # Use standard ranking for arXiv or if cluster map unavailable
            selected = _rank_and_select(paper_inputs, final_scores, top_k, ranking_mode, include_contrastive)

        # Optional contrastive
        contrastive_formatted = None
        contrastive_result = None
        if include_contrastive:
            selected_ids = {p["paper_id"] for p in selected}
            remaining = [p for p in paper_inputs if p["paper_id"] not in selected_ids]
            contrastive_result = _select_contrastive_paper(selected, remaining, contrastive_type, final_scores)
            
            # Fallback: If contrastive paper not found, select one more from remaining
            # to ensure we return top_k papers total
            if not contrastive_result and len(selected) < top_k and remaining:
                # Select the highest scoring paper from remaining
                remaining_with_scores = []
                for paper in remaining:
                    pid = paper.get("paper_id", "")
                    score_info = final_scores.get(pid, {})
                    final_score = score_info.get("final_score", 0.0)
                    remaining_with_scores.append((final_score, paper))
                
                if remaining_with_scores:
                    remaining_with_scores.sort(key=lambda x: x[0], reverse=True)
                    selected.append(remaining_with_scores[0][1])  # Add highest scoring remaining paper

        # Format ranked results
        ranked_results = []
        for rank, paper in enumerate(selected, 1):
            pid = paper["paper_id"]
            tags = _generate_tags(paper, final_scores[pid], local_pdfs_set, False, None)
            ranked_results.append({
                "rank": rank,
                "paper_id": pid,
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "published": paper.get("published"),
                "score": {
                    "final": final_scores[pid]["final_score"],
                    "breakdown": final_scores[pid]["breakdown"],
                    "soft_penalty": final_scores[pid]["soft_penalty"],
                    "penalty_keywords": final_scores[pid].get("penalty_keywords", []),
                    "evaluation_method": final_scores[pid]["evaluation_method"],
                },
                "tags": tags,
                "local_status": {
                    "already_downloaded": pid in local_pdfs_set,
                    "local_path": f"pdf/{pid}.pdf" if pid in local_pdfs_set else None,
                },
                "original_data": paper,
            })

        if contrastive_result:
            contrastive_paper, contrastive_info = contrastive_result
            pid = contrastive_paper["paper_id"]
            score_info = final_scores.get(pid, {
                "final_score": 0.0,
                "breakdown": {},
                "soft_penalty": 0.0,
                "penalty_keywords": [],
                "evaluation_method": "unknown",
            })
            tags = _generate_tags(contrastive_paper, score_info, local_pdfs_set, True, contrastive_type)
            contrastive_formatted = {
                "paper_id": pid,
                "title": contrastive_paper.get("title", ""),
                "authors": contrastive_paper.get("authors", []),
                "published": contrastive_paper.get("published"),
                "score": {
                    "final": score_info["final_score"],
                    "breakdown": score_info["breakdown"],
                    "soft_penalty": score_info["soft_penalty"],
                    "penalty_keywords": score_info.get("penalty_keywords", []),
                    "evaluation_method": score_info["evaluation_method"],
                },
                "tags": tags,
                "contrastive_info": contrastive_info,
                "original_data": contrastive_paper,
            }
            
            # Add contrastive paper to ranked_results as well
            # This ensures we return top_k papers total (top_k-1 regular + 1 contrastive)
            ranked_results.append({
                "rank": len(ranked_results) + 1,
                "paper_id": pid,
                "title": contrastive_paper.get("title", ""),
                "authors": contrastive_paper.get("authors", []),
                "published": contrastive_paper.get("published"),
                "score": {
                    "final": score_info["final_score"],
                    "breakdown": score_info["breakdown"],
                    "soft_penalty": score_info["soft_penalty"],
                    "penalty_keywords": score_info.get("penalty_keywords", []),
                    "evaluation_method": score_info["evaluation_method"],
                },
                "tags": tags,
                "local_status": {
                    "already_downloaded": pid in local_pdfs_set,
                    "local_path": f"pdf/{pid}.pdf" if pid in local_pdfs_set else None,
                },
                "original_data": contrastive_paper,
                "contrastive_info": contrastive_info,  # Include contrastive_info in ranked_results too
            })

        comparison_notes = _generate_comparison_notes(selected, final_scores, contrastive_result[0] if contrastive_result else None)

        summary = {
            "input_count": len(papers),
            "filtered_count": 0,
            "scored_count": len(papers),
            "output_count": len(ranked_results),  # ranked_results already includes contrastive paper if found
            "purpose": purpose,
            "ranking_mode": ranking_mode,
            "profile_used": profile_path,
            "llm_verification_used": False,
            "llm_calls_made": 0,
        }

        return _save_and_format_result(
            ranked_results,
            [],
            contrastive_formatted,
            comparison_notes,
            summary,
            profile_path,
        )
