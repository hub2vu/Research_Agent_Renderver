"""
Formatting utilities for rank and filter papers tool.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from .types import ComparisonNote, PaperInput

# Try to import sentence-transformers for similarity calculation
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


def _generate_tags(
    paper: PaperInput,
    scores: Dict,
    local_pdfs: Set[str],
    is_contrastive: bool = False,
    contrastive_type: Optional[str] = None
) -> List[str]:
    """
    Generate tags for a paper based on scores and metadata.
    
    Args:
        paper: Paper input object
        scores: Dictionary containing score information with "breakdown" key
        local_pdfs: Set of paper IDs available locally
        is_contrastive: Whether this paper is a contrastive pick
        contrastive_type: Type of contrast ("method", "assumption", "domain") if is_contrastive
        
    Returns:
        List of tag strings
    """
    tags: List[str] = []
    
    paper_id = paper.get("paper_id", "")
    github_url = paper.get("github_url", "")
    
    # Extract breakdown scores
    breakdown = scores.get("breakdown", {})
    semantic_relevance = breakdown.get("semantic_relevance", 0.0)
    author_trust = breakdown.get("author_trust", 0.0)
    institution_trust = breakdown.get("institution_trust", 0.0)
    recency = breakdown.get("recency", 0.0)
    must_keywords = breakdown.get("must_keywords", 0.0)
    
    evaluation_method = scores.get("evaluation_method", "")
    
    # Positive tags
    if semantic_relevance >= 0.7:
        tags.append("SEMANTIC_HIGH_MATCH")
    
    if author_trust == 1.0:
        tags.append("PREFERRED_AUTHOR")
    
    if institution_trust > 0:
        tags.append("PREFERRED_INSTITUTION")
    
    if github_url and github_url.strip():
        tags.append("CODE_AVAILABLE")
    
    if recency >= 0.85:
        tags.append("VERY_RECENT")
    
    if paper_id in local_pdfs:
        tags.append("ALREADY_DOWNLOADED")
    
    if must_keywords == 1.0:
        tags.append("MUST_KEYWORD_MATCH")
    
    if "llm" in evaluation_method.lower():
        tags.append("LLM_VERIFIED")
    
    # Warning tags
    if not github_url or not github_url.strip():
        tags.append("NO_CODE")
    
    if recency < 0.3:
        tags.append("OLDER_PAPER")
    
    # Contrastive tags
    if is_contrastive:
        tags.append("CONTRASTIVE_PICK")
        if contrastive_type == "method":
            tags.append("CONTRASTIVE_METHOD")
        elif contrastive_type == "assumption":
            tags.append("CONTRASTIVE_ASSUMPTION")
        elif contrastive_type == "domain":
            tags.append("CONTRASTIVE_DOMAIN")
    
    return tags


def _generate_comparison_notes(
    ranked_papers: List[PaperInput],
    scores: Dict[str, Dict],
    contrastive_paper: Optional[PaperInput] = None
) -> List[ComparisonNote]:
    """
    Generate comparison notes between papers.
    
    Args:
        ranked_papers: List of ranked papers
        scores: Dictionary mapping paper_id to score information
        contrastive_paper: Optional contrastive paper
        
    Returns:
        List of ComparisonNote objects
    """
    notes: List[ComparisonNote] = []
    
    # Step 1: Find pairs with high embedding similarity (>= 0.7)
    if len(ranked_papers) >= 2 and HAS_SENTENCE_TRANSFORMERS:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Generate embeddings for all ranked papers
            paper_texts = []
            paper_ids = []
            for paper in ranked_papers:
                paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                paper_texts.append(paper_text)
                paper_ids.append(paper.get("paper_id", ""))
            
            # Batch encode
            embeddings = model.encode(paper_texts)
            
            # Calculate pairwise similarities
            if HAS_SKLEARN and cosine_similarity is not None:
                similarity_matrix = cosine_similarity(embeddings)
                
                # Find pairs with similarity >= 0.7
                for i in range(len(ranked_papers)):
                    for j in range(i + 1, len(ranked_papers)):
                        similarity = float(similarity_matrix[i][j])
                        
                        if similarity >= 0.7:
                            paper1_id = paper_ids[i]
                            paper2_id = paper_ids[j]
                            paper1 = ranked_papers[i]
                            paper2 = ranked_papers[j]
                            
                            # Extract shared traits (simple keyword extraction)
                            shared_traits = _extract_shared_keywords(
                                paper1.get("title", ""),
                                paper2.get("title", "")
                            )
                            
                            # Generate differentiator (simple: different keywords in titles)
                            differentiator = _extract_differentiator(
                                paper1.get("title", ""),
                                paper2.get("title", "")
                            )
                            
                            notes.append({
                                "paper_ids": [paper1_id, paper2_id],
                                "relation": "similar_approach",
                                "shared_traits": shared_traits,
                                "differentiator": differentiator,
                                "contrast_point": None
                            })
            
            elif HAS_SENTENCE_TRANSFORMERS:
                # Fallback: manual cosine similarity
                try:
                    import numpy as np
                    
                    for i in range(len(ranked_papers)):
                        for j in range(i + 1, len(ranked_papers)):
                            emb1 = embeddings[i]
                            emb2 = embeddings[j]
                            
                            dot_product = np.dot(emb1, emb2)
                            norm1 = np.linalg.norm(emb1)
                            norm2 = np.linalg.norm(emb2)
                            
                            if norm1 > 0 and norm2 > 0:
                                similarity = float(dot_product / (norm1 * norm2))
                                
                                if similarity >= 0.7:
                                    paper1_id = paper_ids[i]
                                    paper2_id = paper_ids[j]
                                    paper1 = ranked_papers[i]
                                    paper2 = ranked_papers[j]
                                    
                                    shared_traits = _extract_shared_keywords(
                                        paper1.get("title", ""),
                                        paper2.get("title", "")
                                    )
                                    
                                    differentiator = _extract_differentiator(
                                        paper1.get("title", ""),
                                        paper2.get("title", "")
                                    )
                                    
                                    notes.append({
                                        "paper_ids": [paper1_id, paper2_id],
                                        "relation": "similar_approach",
                                        "shared_traits": shared_traits,
                                        "differentiator": differentiator,
                                        "contrast_point": None
                                    })
                
                except ImportError:
                    # numpy not available, skip similarity calculation
                    pass
        
        except Exception:
            # If embedding calculation fails, skip similarity-based notes
            pass
    
    # Step 2: Add contrastive relations if contrastive_paper exists
    if contrastive_paper:
        contrastive_id = contrastive_paper.get("paper_id", "")
        ranked_paper_ids = [p.get("paper_id", "") for p in ranked_papers]
        
        for ranked_id in ranked_paper_ids:
            notes.append({
                "paper_ids": [ranked_id, contrastive_id],
                "relation": "contrastive",
                "shared_traits": None,
                "differentiator": None,
                "contrast_point": "different_approach"
            })
    
    return notes


def _extract_shared_keywords(title1: str, title2: str) -> List[str]:
    """
    Extract shared keywords from two titles (simple approach).
    
    Args:
        title1: First title
        title2: Second title
        
    Returns:
        List of shared keywords (max 5)
    """
    # Simple word extraction (3+ characters, lowercase)
    words1 = set(re.findall(r'\b[a-z]{3,}\b', title1.lower()))
    words2 = set(re.findall(r'\b[a-z]{3,}\b', title2.lower()))
    
    # Common stop words to exclude
    stop_words = {
        'the', 'and', 'for', 'with', 'from', 'that', 'this', 'have', 'been',
        'using', 'based', 'approach', 'method', 'learning', 'network'
    }
    
    shared = words1.intersection(words2) - stop_words
    
    # Return top 5 shared keywords
    return list(shared)[:5]


def _extract_differentiator(title1: str, title2: str) -> Optional[str]:
    """
    Extract different keywords from two titles to create a simple differentiator.
    
    Args:
        title1: First title
        title2: Second title
        
    Returns:
        Simple differentiator string or None
    """
    words1 = set(re.findall(r'\b[a-z]{3,}\b', title1.lower()))
    words2 = set(re.findall(r'\b[a-z]{3,}\b', title2.lower()))
    
    stop_words = {
        'the', 'and', 'for', 'with', 'from', 'that', 'this', 'have', 'been',
        'using', 'based', 'approach', 'method', 'learning', 'network'
    }
    
    diff1 = words1 - words2 - stop_words
    diff2 = words2 - words1 - stop_words
    
    # Take a few different keywords from each
    diff_keywords = list(diff1)[:2] + list(diff2)[:2]
    
    if diff_keywords:
        return "_vs_".join(diff_keywords[:3])
    
    return None

