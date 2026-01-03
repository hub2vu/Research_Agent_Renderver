"""
Ranking and selection utilities for rank and filter papers tool.
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

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


def _select_contrastive_paper(
    selected_papers: List[PaperInput],
    remaining_papers: List[PaperInput],
    contrastive_type: str,
    scores: Dict[str, Dict]
) -> Optional[Tuple[PaperInput, ContrastiveInfo]]:
    """
    Select a contrastive paper based on selected papers and contrastive type.
    
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
    
    # 1.2 Extract keywords from title + abstract (simple TF-based)
    all_text = " ".join([
        f"{p.get('title', '')} {p.get('abstract', '')}"
        for p in selected_papers
    ]).lower()
    
    # Extract words (simple approach: 3+ character words, exclude common stop words)
    common_stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'which', 'what', 'who', 'when', 'where', 'why', 'how', 'all',
        'each', 'every', 'many', 'some', 'any', 'more', 'most', 'other',
        'such', 'only', 'own', 'so', 'than', 'too', 'very', 'just', 'now'
    }
    
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    word_freq = Counter(words)
    # Filter out stop words and get top keywords
    keywords = [
        word for word, count in word_freq.most_common(20)
        if word not in common_stop_words
    ][:10]  # Top 10 keywords
    
    common_traits = most_common_categories + keywords
    
    # Step 2: Filter contrastive candidates based on contrastive_type
    contrastive_candidates: List[PaperInput] = []
    
    if contrastive_type == "method":
        # Different categories or different methodology keywords
        # Exclude papers with same top categories
        for paper in remaining_papers:
            paper_categories = set(paper.get("categories", []))
            selected_categories_set = set(most_common_categories)
            
            # If paper has different categories, it's a candidate
            if not paper_categories.intersection(selected_categories_set):
                contrastive_candidates.append(paper)
                continue
            
            # Or check for different methodology keywords
            paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            # Look for methodology contrast keywords
            method_keywords = {
                'supervised', 'unsupervised', 'semi-supervised', 'self-supervised',
                'generative', 'discriminative', 'transformer', 'cnn', 'rnn', 'lstm',
                'reinforcement', 'graph', 'attention', 'convolution', 'recurrent'
            }
            paper_method_keywords = method_keywords.intersection(set(re.findall(r'\b[a-z-]+\b', paper_text)))
            selected_method_keywords = method_keywords.intersection(set(keywords))
            
            # If paper has different method keywords, it's a candidate
            if paper_method_keywords and not paper_method_keywords.intersection(selected_method_keywords):
                contrastive_candidates.append(paper)
    
    elif contrastive_type == "assumption":
        # Opposite paradigm keywords (supervised/unsupervised, generative/discriminative, etc.)
        paradigm_pairs = [
            (['supervised'], ['unsupervised', 'self-supervised']),
            (['unsupervised', 'self-supervised'], ['supervised']),
            (['generative'], ['discriminative', 'classification']),
            (['discriminative', 'classification'], ['generative']),
            (['transformer', 'attention'], ['cnn', 'convolution', 'recurrent', 'lstm']),
            (['cnn', 'convolution'], ['transformer', 'attention', 'recurrent']),
        ]
        
        selected_text = " ".join([
            f"{p.get('title', '')} {p.get('abstract', '')}"
            for p in selected_papers
        ]).lower()
        
        selected_paradigm_groups = []
        for group, _ in paradigm_pairs:
            if any(keyword in selected_text for keyword in group):
                selected_paradigm_groups.extend(group)
        
        for paper in remaining_papers:
            paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            
            # Check if paper has opposite paradigm
            is_contrastive = False
            for group, opposites in paradigm_pairs:
                if any(keyword in selected_text for keyword in group):
                    if any(opposite in paper_text for opposite in opposites):
                        is_contrastive = True
                        break
                elif any(keyword in selected_text for keyword in opposites):
                    if any(opp in paper_text for opp in group):
                        is_contrastive = True
                        break
            
            if is_contrastive:
                contrastive_candidates.append(paper)
    
    elif contrastive_type == "domain":
        # Completely different arXiv categories
        # Extract primary category prefix (e.g., "cs" from "cs.LG")
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
    
    # Step 3: Select the highest scored paper from contrastive candidates
    if not contrastive_candidates:
        return None
    
    # Sort candidates by score
    candidates_with_scores = []
    for paper in contrastive_candidates:
        paper_id = paper.get("paper_id", "")
        score_info = scores.get(paper_id, {})
        final_score = score_info.get("final_score", 0.0)
        candidates_with_scores.append((final_score, paper))
    
    candidates_with_scores.sort(key=lambda x: x[0], reverse=True)
    selected_paper = candidates_with_scores[0][1]
    
    # Step 4: Generate contrastive_info
    selected_paper_categories = selected_paper.get("categories", [])[:3]
    selected_paper_text = f"{selected_paper.get('title', '')} {selected_paper.get('abstract', '')}".lower()
    selected_paper_words = re.findall(r'\b[a-z]{3,}\b', selected_paper_text)
    selected_paper_word_freq = Counter(selected_paper_words)
    selected_paper_keywords = [
        word for word, count in selected_paper_word_freq.most_common(10)
        if word not in common_stop_words
    ][:5]
    
    this_paper_traits = selected_paper_categories + selected_paper_keywords
    
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
        # Compare methodology
        if keywords and selected_paper_keywords:
            # Find contrasting keywords
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
        # Compare paradigms
        paradigm_keywords_map = {
            "supervised": "unsupervised",
            "unsupervised": "supervised",
            "generative": "discriminative",
            "discriminative": "generative",
            "transformer": "cnn/recurrent",
            "cnn": "transformer/attention"
        }
        
        selected_text_lower = " ".join([
            f"{p.get('title', '')} {p.get('abstract', '')}"
            for p in selected_papers
        ]).lower()
        
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
            selected_prefixes = [cat.split('.')[0] for cat in most_common_categories if '.' in cat]
            paper_prefixes = [cat.split('.')[0] for cat in selected_paper_categories if '.' in cat]
            if selected_prefixes and paper_prefixes and selected_prefixes[0] != paper_prefixes[0]:
                contrast_dimensions.append({
                    "dimension": "domain",
                    "others": selected_prefixes[0],
                    "this": paper_prefixes[0]
                })
    
    # If no contrast_dimensions found, add at least one generic one
    if not contrast_dimensions:
        if most_common_categories and selected_paper_categories:
            contrast_dimensions.append({
                "dimension": "category",
                "others": most_common_categories[0] if most_common_categories else "N/A",
                "this": selected_paper_categories[0] if selected_paper_categories else "N/A"
            })
    
    contrastive_info: ContrastiveInfo = {
        "type": contrastive_type,
        "selected_papers_common_traits": common_traits[:10],  # Limit to 10
        "this_paper_traits": this_paper_traits[:10],  # Limit to 10
        "contrast_dimensions": contrast_dimensions
    }
    
    return (selected_paper, contrastive_info)

