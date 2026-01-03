"""
Filtering utilities for rank and filter papers tool.
"""

from typing import List, Tuple

from .types import FilteredPaper, PaperInput, UserProfile


def filter_papers(
    papers: List[PaperInput],
    profile: UserProfile,
    history: set,
    purpose: str
) -> Tuple[List[PaperInput], List[FilteredPaper]]:
    """
    Filter papers based on profile, history, and purpose.
    
    Filtering phases (in order):
    1. ALREADY_READ: Remove papers already in history
    2. BLACKLIST_KEYWORD: Remove papers with hard exclude keywords in title/abstract
    3. TOO_OLD: Remove papers older than min_year (relaxed for literature_review)
    4. NO_CODE_REQUIRED: (TODO - commented out, to be discussed with team)
    
    Args:
        papers: List of input papers to filter
        profile: User profile with keywords and constraints
        history: Set of already-read paper IDs
        purpose: Research purpose ("general", "literature_review", etc.)
        
    Returns:
        Tuple of (filtered papers that passed, list of filtered papers with reasons)
    """
    passed_papers: List[PaperInput] = []
    filtered_papers: List[FilteredPaper] = []
    
    # Phase 1: ALREADY_READ - Remove papers in history
    remaining_papers = []
    for paper in papers:
        paper_id = paper.get("paper_id", "")
        if paper_id in history:
            filtered_papers.append({
                "paper_id": paper_id,
                "title": paper.get("title", ""),
                "filter_reason": "ALREADY_READ",
                "filter_phase": 1
            })
        else:
            remaining_papers.append(paper)
    
    # Phase 2: BLACKLIST_KEYWORD - Remove papers with hard exclude keywords
    hard_keywords = profile["keywords"]["exclude"]["hard"]
    phase2_remaining = []
    for paper in remaining_papers:
        title = paper.get("title", "").lower()
        abstract = paper.get("abstract", "").lower()
        text_to_check = f"{title} {abstract}"
        
        # Check if any hard keyword is in title or abstract
        should_filter = False
        matched_keyword = None
        for keyword in hard_keywords:
            if keyword.lower() in text_to_check:
                should_filter = True
                matched_keyword = keyword
                break
        
        if should_filter:
            filtered_papers.append({
                "paper_id": paper.get("paper_id", ""),
                "title": paper.get("title", ""),
                "filter_reason": f"BLACKLIST_KEYWORD:{matched_keyword}",
                "filter_phase": 2
            })
        else:
            phase2_remaining.append(paper)
    
    # Phase 3: TOO_OLD - Remove papers older than min_year
    min_year = profile["constraints"]["min_year"]
    # Relax min_year by 5 years for literature_review
    if purpose == "literature_review":
        min_year = min_year - 5
    
    phase3_remaining = []
    for paper in phase2_remaining:
        published = paper.get("published")
        if published:
            # Extract year from YYYY-MM-DD format
            try:
                year = int(published.split("-")[0])
                if year < min_year:
                    filtered_papers.append({
                        "paper_id": paper.get("paper_id", ""),
                        "title": paper.get("title", ""),
                        "filter_reason": f"TOO_OLD:{year}",
                        "filter_phase": 3
                    })
                    continue
            except (ValueError, IndexError):
                # If date format is invalid, skip this filter
                pass
        
        phase3_remaining.append(paper)
    
    # Phase 4: NO_CODE_REQUIRED (TODO - commented out for now)
    # TODO: To be discussed with team before implementation
    # If purpose is "implementation" or profile.constraints.require_code is True,
    # filter out papers without github_url
    # 
    # phase4_remaining = []
    # require_code = (purpose == "implementation" or profile["constraints"]["require_code"])
    # for paper in phase3_remaining:
    #     if require_code and not paper.get("github_url"):
    #         filtered_papers.append({
    #             "paper_id": paper.get("paper_id", ""),
    #             "title": paper.get("title", ""),
    #             "filter_reason": "NO_CODE_REQUIRED",
    #             "filter_phase": 4
    #         })
    #     else:
    #         phase4_remaining.append(paper)
    # 
    # passed_papers = phase4_remaining
    
    # For now, all papers from phase 3 pass
    passed_papers = phase3_remaining
    
    return (passed_papers, filtered_papers)

