"""
Ranking and selection utilities for rank and filter papers tool.
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from .types import ContrastiveInfo, PaperInput

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore

# Try to import sklearn for cosine similarity
try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    cosine_similarity = None  # type: ignore


def _is_neurips_data(papers: List[PaperInput]) -> bool:
    """
    Check if papers are from NeurIPS dataset.
    
    Args:
        papers: List of papers to check
        
    Returns:
        True if papers are from NeurIPS, False otherwise
    """
    if not papers:
        return False
    
    # Check first paper's categories
    first_paper = papers[0]
    categories = first_paper.get("categories", [])
    return "NeurIPS 2025" in categories


def _rank_with_cluster_quota(
    papers: List[PaperInput],
    scores: Dict[str, Dict],  # paper_id -> {final_score, breakdown, ...}
    top_k: int,
    cluster_map: Dict[str, int],
    quota_penalty_ratio: float = 0.75
) -> List[PaperInput]:
    """
    Rank papers using cluster quota system with soft penalty (NeurIPS-specific).
    
    Instead of hard quota that completely excludes papers beyond the quota,
    applies a soft penalty (score reduction) to papers from clusters that exceed
    the recommended quota. This allows excellent papers from popular clusters
    to still be selected while maintaining diversity.
    
    Algorithm:
    1. Sort all papers by initial final_score
    2. Count cluster distribution in top candidates (estimate based on initial ranking)
    3. Apply soft penalty (score reduction) to papers from clusters exceeding quota
    4. Re-sort by adjusted scores and select top_k
    
    Args:
        papers: List of papers to rank
        scores: Dictionary mapping paper_id to score information
        top_k: Number of papers to select
        cluster_map: Dictionary mapping paper_id to cluster_id
        quota_penalty_ratio: Score multiplier for papers exceeding quota (default: 0.75, i.e., 25% reduction)
        
    Returns:
        List of selected papers in ranked order
    """
    import math
    
    if top_k <= 0:
        return []
    
    # Calculate max papers per cluster (20% of top_k)
    max_per_cluster = math.ceil(top_k * 0.2)
    
    # Step 1: Prepare papers with initial scores
    papers_with_scores = []
    for paper in papers:
        paper_id = paper.get("paper_id", "")
        score_info = scores.get(paper_id, {})
        final_score = float(score_info.get("final_score", 0.0))
        papers_with_scores.append((final_score, paper))
    
    # Step 2: Sort by initial score to estimate cluster distribution
    papers_with_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Step 3: Count cluster distribution in top candidates (estimate based on initial ranking)
    # This estimates which clusters are over-represented in top candidates
    cluster_score_counts: Dict[int, int] = {}
    for idx, (final_score, paper) in enumerate(papers_with_scores):
        # Only consider top 2x candidates for estimation (more accurate than full list)
        if idx >= top_k * 2:
            break
        paper_id = paper.get("paper_id", "")
        cluster_id = cluster_map.get(paper_id)
        if cluster_id is not None:
            cluster_score_counts[cluster_id] = cluster_score_counts.get(cluster_id, 0) + 1
    
    # Step 4: Apply soft penalty to papers from clusters exceeding quota
    papers_with_adjusted_scores = []
    for final_score, paper in papers_with_scores:
        paper_id = paper.get("paper_id", "")
        cluster_id = cluster_map.get(paper_id)
        
        adjusted_score = final_score
        
        if cluster_id is not None:
            cluster_count = cluster_score_counts.get(cluster_id, 0)
            # If this cluster already has more papers than quota in top candidates, apply penalty
            if cluster_count > max_per_cluster:
                adjusted_score = final_score * quota_penalty_ratio
        
        papers_with_adjusted_scores.append((adjusted_score, final_score, paper))
    
    # Step 5: Sort by adjusted score (with fallback to original score for ties)
    papers_with_adjusted_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # Step 6: Select top_k papers
    selected: List[PaperInput] = []
    for adjusted_score, original_score, paper in papers_with_adjusted_scores:
        if len(selected) >= top_k:
            break
        selected.append(paper)
    
    return selected


def _rank_and_select(
    papers: List[PaperInput],
    scores: Dict[str, Dict],  # paper_id -> {final_score, breakdown, ...}
    top_k: int,
    ranking_mode: str,
    include_contrastive: bool
) -> List[PaperInput]:
    """
    Rank papers by final_score and select top_k papers.
    
    For "diversity" ranking_mode, applies penalty for papers too similar to
    already selected papers using embedding similarity.
    
    Args:
        papers: List of papers to rank
        scores: Dictionary mapping paper_id to score information
                 (must contain "final_score" key)
        top_k: Number of papers to select
        ranking_mode: Ranking mode ("balanced", "novelty", "practicality", "diversity")
        include_contrastive: If True, select top_k - 1 papers (last slot reserved for contrastive)
        
    Returns:
        List of selected papers in ranked order
    """
    # Determine how many papers to select
    num_to_select = top_k - 1 if include_contrastive else top_k
    
    if num_to_select <= 0:
        return []
    
    if ranking_mode != "diversity":
        # Simple ranking: sort by final_score descending
        papers_with_scores = []
        for paper in papers:
            paper_id = paper.get("paper_id", "")
            score_info = scores.get(paper_id, {})
            final_score = score_info.get("final_score", 0.0)
            papers_with_scores.append((final_score, paper))
        
        # Sort by final_score descending
        papers_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Select top_k
        selected = [paper for _, paper in papers_with_scores[:num_to_select]]
        return selected
    
    # Diversity mode: sequential selection with similarity penalty
    selected: List[PaperInput] = []
    remaining_papers = list(papers)
    
    # Create a dictionary for quick score lookup
    score_lookup: Dict[str, float] = {}
    for paper in papers:
        paper_id = paper.get("paper_id", "")
        score_info = scores.get(paper_id, {})
        score_lookup[paper_id] = score_info.get("final_score", 0.0)
    
    # Try to use embedding model for similarity calculation
    model = None
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            model = None
    
    # Select papers sequentially
    for _ in range(num_to_select):
        if not remaining_papers:
            break
        
        if len(selected) == 0:
            # First paper: select the one with highest score
            remaining_papers.sort(key=lambda p: score_lookup.get(p.get("paper_id", ""), 0.0), reverse=True)
            selected.append(remaining_papers.pop(0))
            continue
        
        # Calculate similarity penalties for remaining papers
        papers_with_adjusted_scores = []
        
        # Get embeddings for already selected papers (if model available)
        selected_embeddings = None
        if model:
            try:
                selected_texts = [
                    f"{p.get('title', '')} {p.get('abstract', '')}"
                    for p in selected
                ]
                selected_embeddings = model.encode(selected_texts)
            except Exception:
                selected_embeddings = None
        
        for paper in remaining_papers:
            paper_id = paper.get("paper_id", "")
            base_score = score_lookup.get(paper_id, 0.0)
            adjusted_score = base_score
            
            # Calculate similarity penalty if model is available
            if model and selected_embeddings is not None:
                try:
                    paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                    paper_embedding = model.encode([paper_text])[0]
                    
                    # Calculate cosine similarity with all selected papers
                    if HAS_SKLEARN and cosine_similarity is not None:
                        paper_embedding_2d = paper_embedding.reshape(1, -1)
                        similarities = cosine_similarity(paper_embedding_2d, selected_embeddings)[0]
                        avg_similarity = float(similarities.mean())
                    else:
                        # Fallback: manual cosine similarity calculation
                        try:
                            import numpy as np
                            similarities = []
                            for sel_emb in selected_embeddings:
                                dot_product = np.dot(paper_embedding, sel_emb)
                                norm_paper = np.linalg.norm(paper_embedding)
                                norm_sel = np.linalg.norm(sel_emb)
                                if norm_paper > 0 and norm_sel > 0:
                                    sim = dot_product / (norm_paper * norm_sel)
                                    similarities.append(sim)
                            avg_similarity = float(np.mean(similarities)) if similarities else 0.0
                        except ImportError:
                            # If numpy is not available, skip similarity penalty
                            avg_similarity = 0.0
                    
                    # Apply penalty if average similarity >= 0.8
                    if avg_similarity >= 0.8:
                        adjusted_score -= 0.2
                
                except Exception:
                    # If embedding calculation fails, use base score
                    pass
            
            papers_with_adjusted_scores.append((adjusted_score, base_score, paper))
        
        # Sort by adjusted score (descending), then by base score if tied
        papers_with_adjusted_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # Select the paper with highest adjusted score
        selected_paper = papers_with_adjusted_scores[0][2]
        selected.append(selected_paper)
        remaining_papers.remove(selected_paper)
    
    return selected


# ============================================================================
# Taxonomy-based contrastive paper selection
# ============================================================================

# Method taxonomy for hierarchical classification
METHOD_TAXONOMY = {
    "Vision": ["CNN", "ViT", "ResNet", "Object Detection", "Segmentation", "YOLO", "Mask R-CNN"],
    "NLP": ["Transformer", "RNN", "LSTM", "BERT", "GPT", "LLM", "Attention", "Language Model"],
    "Learning Paradigm": ["Supervised", "Unsupervised", "Self-supervised", "Reinforcement Learning", "Semi-supervised"],
    "Generative": ["GAN", "VAE", "Diffusion", "Flow-based", "Generative Model"],
    "Architecture": ["Encoder-only", "Decoder-only", "Encoder-Decoder", "Mamba", "RWKV", "SSM"]
}

# Bidirectional opposite concepts mapping
OPPOSITE_CONCEPTS = {
    "supervised": "unsupervised",
    "unsupervised": "supervised",
    "self-supervised": "supervised",
    "generative": "discriminative",
    "discriminative": "generative",
    "transformer": "cnn",
    "cnn": "transformer",
    "attention": "convolution",
    "convolution": "attention",
    "dense": "sparse",
    "sparse": "dense",
    "large-scale": "resource-efficient",
    "resource-efficient": "large-scale",
    "theoretical": "empirical",
    "empirical": "theoretical"
}

# Common stop words for keyword extraction
COMMON_STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'which', 'what', 'who', 'when', 'where', 'why', 'how', 'all',
    'each', 'every', 'many', 'some', 'any', 'more', 'most', 'other',
    'such', 'only', 'own', 'so', 'than', 'too', 'very', 'just', 'now'
}


def _get_concept_roots(text: str) -> Set[str]:
    """
    Extract taxonomy roots (parent concepts) from text using improved matching.
    
    Uses word boundary matching to avoid false positives and handles
    case-insensitive matching properly.
    
    Args:
        text: Lowercase text to search
        
    Returns:
        Set of taxonomy root names found in the text
    """
    roots = set()
    text_lower = text.lower()
    
    # Create word boundary patterns for each taxonomy term
    for root, children in METHOD_TAXONOMY.items():
        for child in children:
            # Use word boundary matching to avoid substring false positives
            # Handle both standalone terms and hyphenated terms
            pattern = r'\b' + re.escape(child.lower()) + r'\b'
            if re.search(pattern, text_lower):
                roots.add(root)
                break  # Found one match in this category, move to next
    
    return roots


def _extract_taxonomy_traits(text: str) -> Set[str]:
    """
    Extract taxonomy-based traits (method terms) from text.
    
    Args:
        text: Lowercase text to extract from
        
    Returns:
        Set of taxonomy method terms found in the text
    """
    traits = set()
    text_lower = text.lower()
    
    for root, children in METHOD_TAXONOMY.items():
        for child in children:
            pattern = r'\b' + re.escape(child.lower()) + r'\b'
            if re.search(pattern, text_lower):
                traits.add(child.lower())
    
    return traits


def _calculate_contrast_score(
    paper: PaperInput,
    selected_traits: Set[str],
    selected_roots: Set[str],
    contrastive_type: str
) -> float:
    """
    Calculate how contrastive a paper is compared to selected papers.
    
    Args:
        paper: Candidate paper
        selected_traits: Set of taxonomy method terms from selected papers
        selected_roots: Set of taxonomy roots from selected papers
        contrastive_type: Type of contrast ("method", "assumption", "domain")
        
    Returns:
        Contrast score between 0.0 and 1.0
    """
    paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    paper_roots = _get_concept_roots(paper_text)
    paper_traits = _extract_taxonomy_traits(paper_text)
    
    score = 0.0
    
    if contrastive_type == "method":
        # Same field (common roots) but different methods
        common_roots = paper_roots.intersection(selected_roots)
        if common_roots:
            # Same field - good for maintaining context
            score += 0.5
            # But methods should be different
            if paper_traits and not paper_traits.intersection(selected_traits):
                score += 0.5  # Different methods - high contrast
        else:
            # Different field entirely - still contrastive but less ideal
            if paper_traits:
                score += 0.3
    
    elif contrastive_type == "assumption":
        # Look for opposite paradigm concepts
        for trait in selected_traits:
            opposite = OPPOSITE_CONCEPTS.get(trait.lower())
            if opposite and opposite in paper_text:
                score = 1.0  # Found clear opposite
                break
        
        # Also check for opposite roots if no direct match
        if score == 0.0:
            # Check if paper has opposite learning paradigms
            paradigm_opposites = {
                "supervised": ["unsupervised", "self-supervised"],
                "unsupervised": ["supervised"],
                "self-supervised": ["supervised"],
                "generative": ["discriminative"],
                "discriminative": ["generative"]
            }
            for sel_trait in selected_traits:
                opposites = paradigm_opposites.get(sel_trait.lower(), [])
                if any(opp in paper_text for opp in opposites):
                    score = 0.8
                    break
    
    elif contrastive_type == "domain":
        # Domain contrast is handled separately via category prefixes
        # This scoring is mainly for ranking within domain candidates
        if not paper_roots.intersection(selected_roots):
            score = 1.0  # Completely different domain
        else:
            score = 0.3  # Some overlap but still different
    
    return min(score, 1.0)


def _select_contrastive_paper(
    selected_papers: List[PaperInput],
    remaining_papers: List[PaperInput],
    contrastive_type: str,
    scores: Dict[str, Dict]
) -> Optional[Tuple[PaperInput, ContrastiveInfo]]:
    """
    Select a contrastive paper based on selected papers and contrastive type.
    
    Uses taxonomy-based approach for better concept matching while maintaining
    compatibility with existing types and functionality.
    
    Args:
        selected_papers: List of already selected papers
        remaining_papers: List of remaining candidate papers
        contrastive_type: Type of contrast ("method", "assumption", "domain")
        scores: Dictionary mapping paper_id to score information
        
    Returns:
        Tuple of (selected paper, contrastive_info) if found, None otherwise
    """
    if not selected_papers or not remaining_papers:
        return None
    
    # Step 1: Extract common traits from selected papers
    # 1.1 Categories frequency analysis
    category_counter = Counter()
    for paper in selected_papers:
        categories = paper.get("categories", [])
        if categories:
            for cat in categories:
                category_counter[cat] += 1
    
    most_common_categories = [cat for cat, count in category_counter.most_common(3)]
    
    # 1.2 Extract taxonomy-based traits and roots
    selected_text = " ".join([
        f"{p.get('title', '')} {p.get('abstract', '')}"
        for p in selected_papers
    ]).lower()
    
    selected_traits = _extract_taxonomy_traits(selected_text)
    selected_roots = _get_concept_roots(selected_text)
    
    # 1.3 Also extract keywords for fallback and display
    words = re.findall(r'\b[a-z]{3,}\b', selected_text)
    word_freq = Counter(words)
    keywords = [
        word for word, count in word_freq.most_common(20)
        if word not in COMMON_STOP_WORDS
    ][:10]
    
    # Combine categories and taxonomy traits for common_traits
    common_traits = most_common_categories + list(selected_traits)[:7]  # Limit taxonomy traits
    
    # Step 2: Filter contrastive candidates based on contrastive_type
    contrastive_candidates: List[PaperInput] = []
    
    if contrastive_type == "method":
        # Find papers in same field (common roots) but with different methods
        for paper in remaining_papers:
            paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            paper_roots = _get_concept_roots(paper_text)
            paper_traits = _extract_taxonomy_traits(paper_text)
            
            # Same field but different methods
            common_roots = paper_roots.intersection(selected_roots)
            if common_roots:
                # Same field - good for maintaining context
                if paper_traits and not paper_traits.intersection(selected_traits):
                    # Different methods - ideal contrastive candidate
                    contrastive_candidates.append(paper)
                elif not paper_traits:
                    # No taxonomy match but same field - still candidate
                    contrastive_candidates.append(paper)
            else:
                # Different field entirely - also a candidate but less ideal
                # Will be scored lower in contrast_score
                contrastive_candidates.append(paper)
    
    elif contrastive_type == "assumption":
        # Find papers with opposite paradigms using OPPOSITE_CONCEPTS
        for paper in remaining_papers:
            paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            
            # Check for opposite concepts
            is_contrastive = False
            for trait in selected_traits:
                opposite = OPPOSITE_CONCEPTS.get(trait.lower())
                if opposite and opposite in paper_text:
                        is_contrastive = True
                        break
            
            # Also check for opposite learning paradigms
            if not is_contrastive:
                paradigm_opposites = {
                    "supervised": ["unsupervised", "self-supervised"],
                    "unsupervised": ["supervised"],
                    "self-supervised": ["supervised"],
                    "generative": ["discriminative"],
                    "discriminative": ["generative"]
                }
                for sel_trait in selected_traits:
                    opposites = paradigm_opposites.get(sel_trait.lower(), [])
                    if any(opp in paper_text for opp in opposites):
                        is_contrastive = True
                        break
            
            if is_contrastive:
                contrastive_candidates.append(paper)
    
    elif contrastive_type == "domain":
        # Completely different arXiv categories (keep existing logic)
        selected_prefixes = set()
        for cat in most_common_categories:
            if '.' in cat:
                prefix = cat.split('.')[0]
                selected_prefixes.add(prefix)
        
        for paper in remaining_papers:
            paper_categories = paper.get("categories", [])
            if paper_categories:
                paper_prefixes = set()
                for cat in paper_categories:
                    if '.' in cat:
                        prefix = cat.split('.')[0]
                        paper_prefixes.add(prefix)
                
                # If no common prefix, it's a different domain
                if not paper_prefixes.intersection(selected_prefixes):
                    contrastive_candidates.append(paper)
    
    # Step 3: Score and rank candidates using hybrid approach
    if not contrastive_candidates:
        return None
    
    candidates_with_scores = []
    for paper in contrastive_candidates:
        paper_id = paper.get("paper_id", "")
        score_info = scores.get(paper_id, {})
        base_score = score_info.get("final_score", 0.0)
        
        # Calculate contrast score
        contrast_score = _calculate_contrast_score(
            paper, selected_traits, selected_roots, contrastive_type
        )
        
        # Hybrid scoring: 40% base_score + 60% contrast_score
        # This prioritizes contrastiveness while maintaining quality
        final_candidate_score = (base_score * 0.4) + (contrast_score * 0.6)
        
        # Only include candidates with some contrast
        if contrast_score > 0:
            candidates_with_scores.append((final_candidate_score, contrast_score, paper))
    
    if not candidates_with_scores:
        return None
    
    # Sort by final candidate score (descending)
    candidates_with_scores.sort(key=lambda x: x[0], reverse=True)
    _, best_contrast_score, selected_paper = candidates_with_scores[0]
    
    # Step 4: Generate contrastive_info (maintain type compatibility)
    selected_paper_categories = selected_paper.get("categories", [])[:3]
    selected_paper_text = f"{selected_paper.get('title', '')} {selected_paper.get('abstract', '')}".lower()
    
    # Extract traits for selected paper
    selected_paper_traits = _extract_taxonomy_traits(selected_paper_text)
    selected_paper_words = re.findall(r'\b[a-z]{3,}\b', selected_paper_text)
    selected_paper_word_freq = Counter(selected_paper_words)
    selected_paper_keywords = [
        word for word, count in selected_paper_word_freq.most_common(10)
        if word not in COMMON_STOP_WORDS
    ][:5]
    
    # Combine categories and traits
    this_paper_traits = selected_paper_categories + list(selected_paper_traits)[:7]
    
    # Generate contrast_dimensions
    contrast_dimensions = []
    
    if contrastive_type == "method":
        # Compare categories
        if most_common_categories and selected_paper_categories:
            contrast_dimensions.append({
                "dimension": "category",
                "others": ", ".join(most_common_categories[:2]),
                "this": ", ".join(selected_paper_categories[:2])
            })
        
        # Compare methodology using taxonomy
        if selected_traits and selected_paper_traits:
            diff_traits = list(selected_paper_traits - selected_traits)[:3]
            if diff_traits:
                contrast_dimensions.append({
                    "dimension": "methodology",
                    "others": ", ".join(list(selected_traits)[:3]),
                    "this": ", ".join(diff_traits[:3])
                })
        elif keywords and selected_paper_keywords:
            # Fallback to keyword comparison
            selected_set = set(keywords[:5])
            paper_set = set(selected_paper_keywords[:5])
            diff_keywords = list(paper_set - selected_set)[:3]
            if diff_keywords:
                contrast_dimensions.append({
                    "dimension": "methodology",
                    "others": ", ".join(list(selected_set)[:3]),
                    "this": ", ".join(diff_keywords)
                })
    
    elif contrastive_type == "assumption":
        # Compare paradigms using OPPOSITE_CONCEPTS
        selected_text_lower = selected_text.lower()
        found_opposite = False
        
        for trait in selected_traits:
            opposite = OPPOSITE_CONCEPTS.get(trait.lower())
            if opposite and opposite in selected_paper_text:
                contrast_dimensions.append({
                    "dimension": "paradigm",
                    "others": trait,
                    "this": opposite
                })
                found_opposite = True
                break
        
        # Fallback to keyword-based comparison
        if not found_opposite:
            paradigm_keywords_map = {
                "supervised": "unsupervised",
                "unsupervised": "supervised",
                "generative": "discriminative",
                "discriminative": "generative",
                "transformer": "cnn/recurrent",
                "cnn": "transformer/attention"
            }
            
            for keyword, opposite in paradigm_keywords_map.items():
                if keyword in selected_text_lower and opposite in selected_paper_text:
                    contrast_dimensions.append({
                        "dimension": "paradigm",
                        "others": keyword,
                        "this": opposite
                    })
                    break
    
    elif contrastive_type == "domain":
        # Compare category prefixes
        if most_common_categories and selected_paper_categories:
            selected_prefixes_list = [cat.split('.')[0] for cat in most_common_categories if '.' in cat]
            paper_prefixes_list = [cat.split('.')[0] for cat in selected_paper_categories if '.' in cat]
            if selected_prefixes_list and paper_prefixes_list and selected_prefixes_list[0] != paper_prefixes_list[0]:
                contrast_dimensions.append({
                    "dimension": "domain",
                    "others": selected_prefixes_list[0],
                    "this": paper_prefixes_list[0]
                })
    
    # If no contrast_dimensions found, add at least one generic one
    if not contrast_dimensions:
        if most_common_categories and selected_paper_categories:
            contrast_dimensions.append({
                "dimension": "category",
                "others": most_common_categories[0] if most_common_categories else "N/A",
                "this": selected_paper_categories[0] if selected_paper_categories else "N/A"
            })
        else:
            # Last resort: use contrast score as dimension
            contrast_dimensions.append({
                "dimension": contrastive_type,
                "others": "selected papers",
                "this": "contrastive approach"
            })
    
    # Build contrastive_info matching ContrastiveInfo TypedDict exactly
    contrastive_info: ContrastiveInfo = {
        "type": contrastive_type,
        "selected_papers_common_traits": common_traits[:10],  # Limit to 10
        "this_paper_traits": this_paper_traits[:10],  # Limit to 10
        "contrast_dimensions": contrast_dimensions
    }
    
    return (selected_paper, contrastive_info)

