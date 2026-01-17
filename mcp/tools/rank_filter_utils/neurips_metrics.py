"""
NeurIPS-specific metrics for paper ranking.

Provides cluster-based metrics including Cluster-Interest Alignment
and Cluster Relevance for NeurIPS 2025 papers.
"""

import os
from collections import Counter
from typing import Dict, List, Optional, Set

from .types import PaperInput, UserProfile

# Import NeurIPSAdapter for cluster map loading
try:
    from ...tools.neurips_adapter import NeurIPSAdapter
except ImportError:
    # Fallback if import fails
    NeurIPSAdapter = None


def _calculate_neurips_specific_metrics(
    paper: PaperInput,
    cluster_map: Dict[str, int],
    cluster_sizes: Dict[int, int],
    profile: UserProfile,
    ranked_papers_so_far: List[PaperInput]
) -> Dict[str, float]:
    """
    Calculate NeurIPS-specific metrics for a paper.
    
    Args:
        paper: Paper to evaluate
        cluster_map: Dictionary mapping paper_id to cluster_id
        cluster_sizes: Dictionary mapping cluster_id to number of papers in cluster (unused but kept for API compatibility)
        profile: User profile with interests (unused but kept for API compatibility)
        ranked_papers_so_far: Papers already selected (for diversity calculation)
        
    Returns:
        Dictionary with metric scores:
        - diversity_penalty: Penalty if too many papers from same cluster already selected
    """
    metrics: Dict[str, float] = {
        "diversity_penalty": 0.0
    }
    
    paper_id = paper.get("paper_id", "")
    paper_cluster_id = cluster_map.get(paper_id)
    
    if paper_cluster_id is None:
        # No cluster information available
        return metrics
    
    # Diversity Penalty
    # Penalize if too many papers from the same cluster are already selected
    if ranked_papers_so_far:
        cluster_counts: Dict[int, int] = {}
        for p in ranked_papers_so_far:
            p_id = p.get("paper_id", "")
            c_id = cluster_map.get(p_id)
            if c_id is not None:
                cluster_counts[c_id] = cluster_counts.get(c_id, 0) + 1
        
        # Calculate penalty if this cluster is already over-represented
        current_count = cluster_counts.get(paper_cluster_id, 0)
        # Penalty increases with each additional paper from same cluster
        # Max penalty: 0.3 (30% reduction)
        diversity_penalty = min(0.3, current_count * 0.1)
        metrics["diversity_penalty"] = diversity_penalty
    
    return metrics


def _extract_cluster_keywords(
    cluster_map: Dict[str, int],
    papers: List[PaperInput],
    top_k: int = 10
) -> Dict[int, List[str]]:
    """
    Extract top keywords for each cluster using simple frequency analysis.
    
    Args:
        cluster_map: Dictionary mapping paper_id to cluster_id
        papers: List of all papers
        top_k: Number of top keywords to extract per cluster
        
    Returns:
        Dictionary mapping cluster_id to list of top keywords
    """
    from collections import defaultdict
    import re
    
    # Common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'we', 'they', 'which', 'what', 'who', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'many', 'some', 'any',
        'more', 'most', 'other', 'such', 'only', 'own', 'so', 'than', 'too',
        'very', 'just', 'now', 'then', 'here', 'there', 'where', 'why', 'how'
    }
    
    # Group papers by cluster
    cluster_papers: Dict[int, List[PaperInput]] = defaultdict(list)
    for paper in papers:
        paper_id = paper.get("paper_id", "")
        cluster_id = cluster_map.get(paper_id)
        if cluster_id is not None:
            cluster_papers[cluster_id].append(paper)
    
    # Extract keywords for each cluster
    cluster_keywords: Dict[int, List[str]] = {}
    
    for cluster_id, cluster_paper_list in cluster_papers.items():
        # Collect all text from papers in this cluster
        all_text = []
        for paper in cluster_paper_list:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            all_text.append(f"{title} {abstract}")
        
        combined_text = " ".join(all_text).lower()
        
        # Extract words (3+ characters, alphanumeric)
        words = re.findall(r'\b[a-z]{3,}\b', combined_text)
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Filter out stop words and get top keywords
        keywords = [
            word for word, count in word_freq.most_common(top_k * 2)
            if word not in stop_words
        ][:top_k]
        
        cluster_keywords[cluster_id] = keywords
    
    return cluster_keywords


def load_neurips_cluster_data(
    cluster_k: Optional[int] = None
) -> tuple[Dict[str, int], Dict[int, int]]:
    """
    Load NeurIPS cluster mapping and sizes.
    
    Args:
        cluster_k: K value for clustering (default: from env or 15)
        
    Returns:
        Tuple of (cluster_map, cluster_sizes)
        cluster_map: paper_id -> cluster_id
        cluster_sizes: cluster_id -> count
    """
    if NeurIPSAdapter is None:
        return {}, {}
    
    # Get K value from environment or use default
    if cluster_k is None:
        cluster_k = int(os.getenv("NEURIPS_CLUSTER_K", 15))
    
    # Load cluster map
    cluster_map = NeurIPSAdapter.load_cluster_map(cluster_k=cluster_k)
    
    if not cluster_map:
        return {}, {}
    
    # Calculate cluster sizes
    cluster_sizes: Dict[int, int] = {}
    for cluster_id in cluster_map.values():
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
    
    return cluster_map, cluster_sizes
