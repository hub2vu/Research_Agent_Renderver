"""
Scoring utilities for rank and filter papers tool.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from .types import PaperInput, UserProfile

# Try to import OpenAI for LLM verification
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None  # type: ignore

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


def _classify_papers_by_score(
    papers: List[PaperInput],
    scores: Dict[str, float]
) -> Tuple[List[PaperInput], List[PaperInput], List[PaperInput]]:
    """
    Classify papers into three groups based on their scores.
    
    Args:
        papers: List of papers to classify
        scores: Dictionary mapping paper_id to score
        
    Returns:
        Tuple of (high_group, mid_group, low_group) where:
        - high_group: papers with score >= 0.7
        - mid_group: papers with 0.4 <= score < 0.7
        - low_group: papers with score < 0.4
    """
    high_group: List[PaperInput] = []
    mid_group: List[PaperInput] = []
    low_group: List[PaperInput] = []
    
    for paper in papers:
        paper_id = paper["paper_id"]
        score = scores.get(paper_id, 0.0)
        
        if score >= 0.7:
            high_group.append(paper)
        elif score >= 0.4:
            mid_group.append(paper)
        else:
            low_group.append(paper)
    
    return (high_group, mid_group, low_group)


def _verify_with_llm(
    papers: List[PaperInput],
    profile: UserProfile,
    batch_size: int = 5
) -> Dict[str, Tuple[float, str]]:
    """
    Verify paper relevance using LLM batch evaluation.
    
    Args:
        papers: List of papers to verify
        profile: User profile with interests
        batch_size: Number of papers per LLM batch call
        
    Returns:
        Dictionary mapping paper_id to (llm_score, reason) tuple.
        If LLM verification fails for a paper, returns (0.0, "") for that paper.
    """
    if not HAS_OPENAI:
        # If OpenAI is not available, return empty results
        return {paper["paper_id"]: (0.0, "") for paper in papers}
    
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    if not api_key:
        # If API key is not available, return empty results
        return {paper["paper_id"]: (0.0, "") for paper in papers}
    
    try:
        client = OpenAI(api_key=api_key)
    except Exception:
        return {paper["paper_id"]: (0.0, "") for paper in papers}
    
    results: Dict[str, Tuple[float, str]] = {}
    
    # Split papers into batches
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        
        # Build prompt
        primary_interests = ", ".join(profile["interests"]["primary"]) if profile["interests"]["primary"] else "None"
        secondary_interests = ", ".join(profile["interests"]["secondary"]) if profile["interests"]["secondary"] else "None"
        
        prompt = f"""사용자 연구 관심사:

- Primary: {primary_interests}
- Secondary: {secondary_interests}

다음 논문들이 위 관심사와 얼마나 관련있는지 평가하세요.
각 논문에 대해 0.0~1.0 점수와 한 줄 판단 근거를 JSON으로 제공하세요.

논문 목록:

"""
        
        for idx, paper in enumerate(batch, start=1):
            paper_id = paper.get("paper_id", "")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")[:500]  # Limit abstract length
            prompt += f"{idx}. [ID: {paper_id}] 제목: {title} / 초록: {abstract}\n\n"
        
        prompt += """응답 형식:
[{{"paper_id": "...", "relevance": 0.75, "reason": "..."}}, ...]
"""
        
        try:
            # Call LLM
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a research paper evaluation assistant. Always respond with valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            # Try to extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            evaluations = json.loads(content)
            
            # Process evaluations
            if isinstance(evaluations, list):
                for eval_item in evaluations:
                    if isinstance(eval_item, dict):
                        paper_id = eval_item.get("paper_id", "")
                        relevance = eval_item.get("relevance", 0.0)
                        reason = eval_item.get("reason", "")
                        
                        # Clamp relevance to [0.0, 1.0]
                        relevance = max(0.0, min(1.0, float(relevance)))
                        
                        results[paper_id] = (relevance, reason)
            
        except (json.JSONDecodeError, KeyError, ValueError, AttributeError) as e:
            # If parsing fails, keep existing embedding scores (will be handled by caller)
            # Mark all papers in this batch as failed
            for paper in batch:
                paper_id = paper.get("paper_id", "")
                if paper_id not in results:
                    results[paper_id] = (0.0, "")
            continue
        except Exception as e:
            # Any other error, mark batch as failed
            for paper in batch:
                paper_id = paper.get("paper_id", "")
                if paper_id not in results:
                    results[paper_id] = (0.0, "")
            continue
    
    # Ensure all papers have results
    for paper in papers:
        paper_id = paper.get("paper_id", "")
        if paper_id not in results:
            results[paper_id] = (0.0, "")
    
    return results


def _merge_scores(
    embedding_scores: Dict[str, float],
    llm_results: Dict[str, Tuple[float, str]],
    high_group_ids: Set[str],
    low_group_ids: Set[str]
) -> Dict[str, Dict]:
    """
    Merge embedding scores and LLM verification results.
    
    Args:
        embedding_scores: Dictionary mapping paper_id to embedding score (0.0-1.0)
        llm_results: Dictionary mapping paper_id to (llm_score, reason) tuple
        high_group_ids: Set of paper IDs in high group (score >= 0.7)
        low_group_ids: Set of paper IDs in low group (score < 0.4)
        
    Returns:
        Dictionary mapping paper_id to {
            "semantic_score": float,
            "evaluation_method": str,
            "llm_reason": Optional[str]
        }
    """
    merged_scores: Dict[str, Dict] = {}
    
    # Get all paper IDs from embedding_scores
    all_paper_ids = set(embedding_scores.keys())
    
    # mid_group_ids = all papers not in high or low groups
    mid_group_ids = all_paper_ids - high_group_ids - low_group_ids
    
    for paper_id in all_paper_ids:
        embedding_score = embedding_scores.get(paper_id, 0.0)
        
        if paper_id in high_group_ids:
            # High group: use embedding score as-is
            merged_scores[paper_id] = {
                "semantic_score": embedding_score,
                "evaluation_method": "embedding_high",
                "llm_reason": None
            }
        elif paper_id in low_group_ids:
            # Low group: use embedding score as-is
            merged_scores[paper_id] = {
                "semantic_score": embedding_score,
                "evaluation_method": "embedding_low",
                "llm_reason": None
            }
        else:
            # Mid group: use LLM score if available, otherwise embedding score
            if paper_id in llm_results and llm_results[paper_id][0] > 0.0:
                llm_score, llm_reason = llm_results[paper_id]
                merged_scores[paper_id] = {
                    "semantic_score": llm_score,
                    "evaluation_method": "embedding+llm",
                    "llm_reason": llm_reason if llm_reason else None
                }
            else:
                # LLM verification failed or not available, use embedding score
                merged_scores[paper_id] = {
                    "semantic_score": embedding_score,
                    "evaluation_method": "embedding_only",
                    "llm_reason": None
                }
    
    return merged_scores


def _calculate_dimension_scores(
    paper: PaperInput,
    profile: UserProfile,
    semantic_score: float,
    local_pdfs: Set[str]
) -> Dict[str, float]:
    """
    Calculate 6-dimensional scores for a paper.
    
    Args:
        paper: Paper input object
        profile: User profile with keywords, preferred authors/institutions
        semantic_score: Pre-calculated semantic relevance score (0.0-1.0)
        local_pdfs: Set of paper IDs available locally
        
    Returns:
        Dictionary with 6 dimension scores:
        {
            "semantic_relevance": float,
            "must_keywords": float,
            "author_trust": float,
            "institution_trust": float,
            "recency": float,
            "practicality": float
        }
    """
    paper_id = paper.get("paper_id", "")
    title = paper.get("title", "").lower()
    abstract = paper.get("abstract", "").lower()
    authors = paper.get("authors", [])
    affiliations = paper.get("affiliations", [])
    published = paper.get("published", "")
    github_url = paper.get("github_url", "")
    
    scores: Dict[str, float] = {}
    
    # 1. semantic_relevance: Use provided semantic_score
    scores["semantic_relevance"] = semantic_score
    
    # 2. must_keywords: Ratio of must_include keywords found
    must_keywords = profile["keywords"]["must_include"]
    if not must_keywords:
        scores["must_keywords"] = 1.0  # No requirements = perfect score
    else:
        text_to_check = f"{title} {abstract}"
        found_count = 0
        for keyword in must_keywords:
            if keyword.lower() in text_to_check:
                found_count += 1
        scores["must_keywords"] = found_count / len(must_keywords)
    
    # 3. author_trust: Check if any preferred author is in authors list
    preferred_authors = profile["preferred_authors"]
    if not preferred_authors:
        scores["author_trust"] = 1.0  # No preferences = perfect score
    else:
        # Normalize author names for comparison (case-insensitive)
        paper_authors_lower = [a.lower().strip() for a in authors]
        preferred_authors_lower = [a.lower().strip() for a in preferred_authors]
        
        # Check if any preferred author matches (exact or substring)
        found = False
        for pref_author in preferred_authors_lower:
            for paper_author in paper_authors_lower:
                if pref_author in paper_author or paper_author in pref_author:
                    found = True
                    break
            if found:
                break
        
        scores["author_trust"] = 1.0 if found else 0.0
    
    # 4. institution_trust: Check preferred_institutions against affiliations
    preferred_institutions = profile["preferred_institutions"]
    if not preferred_institutions:
        scores["institution_trust"] = 1.0  # No preferences = perfect score
    elif not affiliations:
        scores["institution_trust"] = 0.0  # No affiliations = no trust
    else:
        # Normalize for comparison (case-insensitive)
        affiliations_lower = [aff.lower().strip() for aff in affiliations]
        preferred_inst_lower = [inst.lower().strip() for inst in preferred_institutions]
        
        # Check if any preferred institution is found (substring match)
        found = False
        for pref_inst in preferred_inst_lower:
            for aff in affiliations_lower:
                if pref_inst in aff or aff in pref_inst:
                    found = True
                    break
            if found:
                break
        
        scores["institution_trust"] = 1.0 if found else 0.0
    
    # 5. recency: Calculate based on published date
    if not published:
        scores["recency"] = 0.1  # No date = very old
    else:
        try:
            # Parse date (YYYY-MM-DD format)
            pub_date = datetime.strptime(published, "%Y-%m-%d")
            now = datetime.now()
            time_diff = now - pub_date
            
            days_diff = time_diff.days
            
            if days_diff <= 14:  # 2 weeks
                scores["recency"] = 1.0
            elif days_diff <= 30:  # 1 month
                scores["recency"] = 0.85
            elif days_diff <= 90:  # 3 months
                scores["recency"] = 0.7
            elif days_diff <= 180:  # 6 months
                scores["recency"] = 0.5
            elif days_diff <= 365:  # 1 year
                scores["recency"] = 0.3
            else:  # More than 1 year
                scores["recency"] = 0.1
        except (ValueError, TypeError):
            # Invalid date format
            scores["recency"] = 0.1
    
    # 6. practicality: github_url + local_pdfs
    # Note: github_url is used for scoring (bonus points) only, not for filtering.
    # Filtering based on code requirement is disabled in filters.py (Phase 4 commented out)
    # because there's no tool to verify github_url existence yet. Will be uncommented after team discussion.
    practicality = 0.0
    if github_url and github_url.strip():
        practicality += 0.5
    if paper_id in local_pdfs:
        practicality += 0.3
    scores["practicality"] = min(1.0, practicality)  # Clamp to max 1.0
    
    return scores

