"""
Scoring utilities for rank and filter papers tool.
"""

from typing import Dict, List

from .types import PaperInput, UserProfile

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


def _calculate_embedding_scores(
    papers: List[PaperInput],
    profile: UserProfile
) -> Dict[str, float]:
    """
    Calculate embedding-based relevance scores for papers.
    
    Uses sentence-transformers to compute semantic similarity between
    user interests and paper abstracts. Falls back to keyword-based
    scoring if embeddings are unavailable.
    
    Args:
        papers: List of papers to score
        profile: User profile with interests
        
    Returns:
        Dictionary mapping paper_id to embedding score (0.0-1.0)
    """
    # Try to use embedding model
    if HAS_SENTENCE_TRANSFORMERS and HAS_SKLEARN:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Build interests text with weights
            # primary: 1.0, secondary: 0.7, exploratory: 0.4
            # Repeat keywords proportionally to reflect weights
            interests_text_parts = []
            
            # Primary interests (weight 1.0) - repeat 3 times to give highest weight
            for interest in profile["interests"]["primary"]:
                for _ in range(3):  # Repeat 3 times for weight 1.0
                    interests_text_parts.append(interest)
            
            # Secondary interests (weight 0.7) - repeat 2 times
            for interest in profile["interests"]["secondary"]:
                for _ in range(2):  # Repeat 2 times for weight ~0.67
                    interests_text_parts.append(interest)
            
            # Exploratory interests (weight 0.4) - repeat 1 time
            for interest in profile["interests"]["exploratory"]:
                interests_text_parts.append(interest)  # Repeat 1 time for weight 0.33
            
            # Combine interests into a single text
            interests_text = " ".join(interests_text_parts)
            
            # If no interests, return zero scores
            if not interests_text.strip():
                return {paper["paper_id"]: 0.0 for paper in papers}
            
            # Generate interests embedding
            interests_embedding = model.encode([interests_text])[0]
            
            # Generate paper embeddings (batch processing)
            paper_texts = []
            paper_ids = []
            for paper in papers:
                paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                paper_texts.append(paper_text)
                paper_ids.append(paper["paper_id"])
            
            # Batch encode all papers
            paper_embeddings = model.encode(paper_texts)
            
            # Calculate cosine similarity
            # Reshape interests_embedding for sklearn compatibility
            interests_embedding_2d = interests_embedding.reshape(1, -1)
            similarities = cosine_similarity(interests_embedding_2d, paper_embeddings)[0]
            
            # Convert to dictionary and normalize to 0.0-1.0 (cosine similarity is already -1.0 to 1.0)
            # Normalize: (similarity + 1) / 2 to get 0.0-1.0 range
            scores = {}
            for i, paper_id in enumerate(paper_ids):
                # Normalize cosine similarity to 0.0-1.0 range
                normalized_score = (similarities[i] + 1.0) / 2.0
                scores[paper_id] = max(0.0, min(1.0, normalized_score))  # Clamp to [0, 1]
            
            return scores
            
        except Exception:
            # If embedding fails, fall back to keyword scoring
            pass
    
    # Fallback to keyword-based scoring
    return _calculate_keyword_scores(papers, profile)


def _calculate_keyword_scores(
    papers: List[PaperInput],
    profile: UserProfile
) -> Dict[str, float]:
    """
    Calculate keyword-based relevance scores for papers (fallback method).
    
    Uses simple term frequency (TF) style counting of interest keywords
    appearing in paper abstracts.
    
    Args:
        papers: List of papers to score
        profile: User profile with interests
        
    Returns:
        Dictionary mapping paper_id to keyword score (0.0-1.0)
    """
    # Collect all interest keywords with weights
    interest_keywords = {}
    
    # Primary interests: weight 1.0
    for keyword in profile["interests"]["primary"]:
        keyword_lower = keyword.lower()
        interest_keywords[keyword_lower] = interest_keywords.get(keyword_lower, 0.0) + 1.0
    
    # Secondary interests: weight 0.7
    for keyword in profile["interests"]["secondary"]:
        keyword_lower = keyword.lower()
        interest_keywords[keyword_lower] = interest_keywords.get(keyword_lower, 0.0) + 0.7
    
    # Exploratory interests: weight 0.4
    for keyword in profile["interests"]["exploratory"]:
        keyword_lower = keyword.lower()
        interest_keywords[keyword_lower] = interest_keywords.get(keyword_lower, 0.0) + 0.4
    
    # If no keywords, return zero scores
    if not interest_keywords:
        return {paper["paper_id"]: 0.0 for paper in papers}
    
    # Calculate scores for each paper
    scores = {}
    max_score = 0.0
    
    for paper in papers:
        paper_id = paper["paper_id"]
        abstract = paper.get("abstract", "").lower()
        title = paper.get("title", "").lower()
        text = f"{title} {abstract}"
        
        # Count keyword occurrences with weights
        score = 0.0
        for keyword, weight in interest_keywords.items():
            # Count occurrences of keyword in text
            count = text.count(keyword)
            score += count * weight
        
        scores[paper_id] = score
        max_score = max(max_score, score)
    
    # Normalize scores to 0.0-1.0 range
    if max_score > 0:
        normalized_scores = {
            paper_id: min(1.0, score / max_score)
            for paper_id, score in scores.items()
        }
        return normalized_scores
    else:
        # All scores are zero
        return {paper_id: 0.0 for paper_id in scores.keys()}

