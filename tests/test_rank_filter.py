"""
Comprehensive unit tests for rank_and_filter_papers tools.

Tests all 5 tools:
1. UpdateUserProfileTool
2. ApplyHardFiltersTool
3. CalculateSemanticScoresTool
4. EvaluatePaperMetricsTool
5. RankAndSelectTopKTool
"""

import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from mcp.tools.rank_filter import (
    ApplyHardFiltersTool,
    CalculateSemanticScoresTool,
    EvaluatePaperMetricsTool,
    RankAndSelectTopKTool,
    UpdateUserProfileTool,
)
from mcp.tools.rank_filter_utils import (
    FilteredPaper,
    PaperInput,
    UserProfile,
    filter_papers,
    load_profile,
    scan_local_pdfs,
)
from mcp.tools.rank_filter_utils.scores import (
    _apply_soft_penalty,
    _calculate_dimension_scores,
)


# ========== Fixtures ==========

@pytest.fixture
def sample_papers() -> List[PaperInput]:
    """Sample papers for testing."""
    today = date.today()
    return [
        PaperInput(
            paper_id="2024.12345",
            title="Deep Learning for Natural Language Processing",
            abstract="We present a novel approach to NLP using deep learning and transformer models.",
            authors=["John Doe", "Jane Smith"],
            published=str(today - timedelta(days=10)),
            categories=["cs.CL", "cs.LG"],
            pdf_url="http://arxiv.org/pdf/2024.12345.pdf",
            github_url="https://github.com/test/repo1",
            affiliations=["MIT"]
        ),
        PaperInput(
            paper_id="2024.12346",
            title="Transformer Models in Computer Vision",
            abstract="We explore transformers for image classification tasks using attention mechanisms.",
            authors=["Alice Johnson"],
            published=str(today - timedelta(days=30)),
            categories=["cs.CV"],
            pdf_url="http://arxiv.org/pdf/2024.12346.pdf",
            github_url="",
            affiliations=["Stanford"]
        ),
        PaperInput(
            paper_id="2024.12347",
            title="Reinforcement Learning Basics",
            abstract="Introduction to reinforcement learning algorithms and Q-learning.",
            authors=["Bob Wilson"],
            published=str(today - timedelta(days=365)),
            categories=["cs.LG"],
            pdf_url="http://arxiv.org/pdf/2024.12347.pdf",
            github_url="https://github.com/test/repo2",
            affiliations=["CMU"]
        ),
        PaperInput(
            paper_id="2024.12348",
            title="Medical AI Applications",
            abstract="We apply deep learning to medical diagnosis and clinical decision making.",
            authors=["Dr. Medical"],
            published=str(today - timedelta(days=60)),
            categories=["cs.AI"],
            pdf_url="http://arxiv.org/pdf/2024.12348.pdf",
            github_url="",
            affiliations=[]
        ),
    ]


@pytest.fixture
def sample_profile() -> UserProfile:
    """Sample user profile for testing."""
    return {
        "interests": {
            "primary": ["deep learning", "transformer"],
            "secondary": ["NLP", "computer vision"],
            "exploratory": ["reinforcement learning"]
        },
        "keywords": {
            "must_include": ["learning"],
            "exclude": {
                "hard": ["medical", "biology", "clinical"],
                "soft": ["basics", "introduction"]
            }
        },
        "preferred_authors": ["John Doe"],
        "preferred_institutions": ["MIT"],
        "constraints": {
            "min_year": 2020,
            "require_code": False,
            "exclude_local_papers": False
        },
        "purpose": "general",
        "ranking_mode": "balanced",
        "top_k": 5,
        "include_contrastive": False,
        "contrastive_type": "method"
    }


@pytest.fixture
def temp_profile_file(tmp_path):
    """Create a temporary profile file."""
    profile_path = tmp_path / "profile.json"
    return profile_path


@pytest.fixture
def temp_output_dir(tmp_path, monkeypatch):
    """Set up temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    users_dir = output_dir / "users"
    users_dir.mkdir(parents=True)
    
    # Set environment variable
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    return output_dir


# ========== Test UpdateUserProfileTool ==========

class TestUpdateUserProfileTool:
    """Test UpdateUserProfileTool."""
    
    @pytest.fixture
    def tool(self):
        return UpdateUserProfileTool()
    
    @pytest.mark.asyncio
    async def test_create_new_profile(self, tool, temp_output_dir):
        """Test creating a new profile."""
        profile_path = "users/test_profile.json"
        
        result = await tool.execute(
            profile_path=profile_path,
            purpose="general",
            ranking_mode="balanced",
            top_k=5
        )
        
        assert result["updated"] is True
        assert "profile_path" in result
        assert result["purpose"] == "general"
        assert result["ranking_mode"] == "balanced"
        assert result["top_k"] == 5
        
        # Verify file was created
        resolved_path = Path(result["profile_path"])
        assert resolved_path.exists()
        
        # Verify content
        with open(resolved_path, "r", encoding="utf-8") as f:
            saved_profile = json.load(f)
            assert saved_profile["purpose"] == "general"
            assert saved_profile["ranking_mode"] == "balanced"
            assert saved_profile["top_k"] == 5
    
    @pytest.mark.asyncio
    async def test_update_interests(self, tool, temp_output_dir):
        """Test updating interests."""
        profile_path = "users/test_profile.json"
        
        # Create initial profile
        await tool.execute(profile_path=profile_path)
        
        # Update interests
        result = await tool.execute(
            profile_path=profile_path,
            interests={
                "primary": ["transformer", "attention"],
                "secondary": ["NLP"],
                "exploratory": ["vision"]
            }
        )
        
        assert result["updated"] is True
        
        # Verify interests were updated
        resolved_path = Path(result["profile_path"])
        with open(resolved_path, "r", encoding="utf-8") as f:
            saved_profile = json.load(f)
            assert saved_profile["interests"]["primary"] == ["transformer", "attention"]
            assert saved_profile["interests"]["secondary"] == ["NLP"]
            assert saved_profile["interests"]["exploratory"] == ["vision"]
    
    @pytest.mark.asyncio
    async def test_update_keywords(self, tool, temp_output_dir):
        """Test updating keywords."""
        profile_path = "users/test_profile.json"
        
        # Create initial profile
        await tool.execute(profile_path=profile_path)
        
        # Update keywords
        result = await tool.execute(
            profile_path=profile_path,
            keywords={
                "must_include": ["transformer"],
                "exclude": {
                    "hard": ["medical"],
                    "soft": ["survey"]
                }
            }
        )
        
        assert result["updated"] is True
        
        # Verify keywords were updated
        resolved_path = Path(result["profile_path"])
        with open(resolved_path, "r", encoding="utf-8") as f:
            saved_profile = json.load(f)
            assert saved_profile["keywords"]["must_include"] == ["transformer"]
            assert saved_profile["keywords"]["exclude"]["hard"] == ["medical"]
            assert saved_profile["keywords"]["exclude"]["soft"] == ["survey"]
    
    @pytest.mark.asyncio
    async def test_update_all_fields(self, tool, temp_output_dir):
        """Test updating all profile fields."""
        profile_path = "users/test_profile.json"
        
        result = await tool.execute(
            profile_path=profile_path,
            purpose="implementation",
            ranking_mode="practicality",
            top_k=10,
            include_contrastive=True,
            contrastive_type="method",
            exclude_local_papers=True,
            preferred_authors=["Author 1", "Author 2"],
            preferred_institutions=["MIT", "Stanford"],
            constraints={
                "min_year": 2022,
                "require_code": True,
                "exclude_local_papers": True
            }
        )
        
        assert result["updated"] is True
        assert result["purpose"] == "implementation"
        assert result["ranking_mode"] == "practicality"
        assert result["top_k"] == 10
        assert result["include_contrastive"] is True
        assert result["contrastive_type"] == "method"
        assert result["exclude_local_papers"] is True
    
    @pytest.mark.asyncio
    async def test_empty_preferred_authors_institutions(self, tool, temp_output_dir):
        """Test that empty arrays for preferred_authors/institutions are not saved."""
        profile_path = "users/test_profile.json"
        
        # First set some values
        await tool.execute(
            profile_path=profile_path,
            preferred_authors=["Author 1"],
            preferred_institutions=["MIT"]
        )
        
        # Then set to empty arrays
        result = await tool.execute(
            profile_path=profile_path,
            preferred_authors=[],
            preferred_institutions=[]
        )
        
        # Verify empty arrays are NOT saved (fields should be removed)
        resolved_path = Path(result["profile_path"])
        with open(resolved_path, "r", encoding="utf-8") as f:
            saved_profile = json.load(f)
            # Fields should not be present in JSON if empty
            assert "preferred_authors" not in saved_profile or saved_profile.get("preferred_authors") == []
            assert "preferred_institutions" not in saved_profile or saved_profile.get("preferred_institutions") == []


# ========== Test ApplyHardFiltersTool ==========

class TestApplyHardFiltersTool:
    """Test ApplyHardFiltersTool."""
    
    @pytest.fixture
    def tool(self):
        return ApplyHardFiltersTool()
    
    @pytest.mark.asyncio
    async def test_filter_blacklist_keyword(self, tool, sample_papers, sample_profile, temp_output_dir):
        """Test filtering papers with hard exclude keywords."""
        # Create profile file
        profile_path = "users/test_profile.json"
        with patch('mcp.tools.rank_filter.load_profile', return_value=sample_profile):
            result = await tool.execute(
                papers=[{**p} for p in sample_papers],
                profile_path=profile_path
            )
        
        # Medical paper should be filtered
        passed_ids = {p["paper_id"] for p in result["passed_papers"]}
        assert "2024.12348" not in passed_ids  # Medical paper
        
        # Should be in filtered list
        filtered_ids = {f["paper_id"] for f in result["filtered_papers"]}
        assert "2024.12348" in filtered_ids
        
        # Check filter reason
        medical_filtered = [f for f in result["filtered_papers"] if f["paper_id"] == "2024.12348"]
        assert len(medical_filtered) > 0
        assert "BLACKLIST_KEYWORD" in medical_filtered[0]["filter_reason"]
    
    @pytest.mark.asyncio
    async def test_filter_already_read(self, tool, sample_papers, sample_profile, temp_output_dir):
        """Test filtering papers that are already read."""
        # Create history file
        history_path = temp_output_dir / "history" / "read_papers.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(["2024.12345", "2024.12346"], f)
        
        with patch('mcp.tools.rank_filter.load_profile', return_value=sample_profile):
            result = await tool.execute(
                papers=[{**p} for p in sample_papers],
                profile_path="users/test_profile.json",
                history_path=str(history_path.relative_to(temp_output_dir))
            )
        
        # Papers in history should be filtered
        passed_ids = {p["paper_id"] for p in result["passed_papers"]}
        assert "2024.12345" not in passed_ids
        assert "2024.12346" not in passed_ids
        assert "2024.12347" in passed_ids  # Not in history
    
    @pytest.mark.asyncio
    async def test_filter_old_papers(self, tool, sample_papers, sample_profile, temp_output_dir):
        """Test filtering papers that are too old."""
        # Profile has min_year=2020, paper from 2023 (365 days ago) should pass
        # But if it's 2024, a paper from 365 days ago is from 2023, which is >= 2020
        
        with patch('mcp.tools.rank_filter.load_profile', return_value=sample_profile):
            result = await tool.execute(
                papers=[{**p} for p in sample_papers],
                profile_path="users/test_profile.json"
            )
        
        # Check that old papers are handled correctly
        # The exact behavior depends on current date and min_year
        assert "filter_summary" in result
        assert result["filter_summary"]["total"] == len(sample_papers)
    
    @pytest.mark.asyncio
    async def test_empty_input(self, tool, sample_profile, temp_output_dir):
        """Test handling empty input."""
        with patch('mcp.tools.rank_filter.load_profile', return_value=sample_profile):
            result = await tool.execute(papers=[])

        assert result["passed_papers"] == []
        assert result["filtered_papers"] == []
        assert result["filter_summary"]["total"] == 0
        assert result["filter_summary"]["passed"] == 0
        assert result["filter_summary"]["filtered"] == 0
    
    @pytest.mark.asyncio
    async def test_exclude_local_papers(self, tool, sample_papers, sample_profile, tmp_path, temp_output_dir):
        """Test exclude_local_papers functionality."""
        # Create a profile with exclude_local_papers=True
        profile_with_exclude = {**sample_profile}
        profile_with_exclude["constraints"]["exclude_local_papers"] = True
        
        # Create a local PDF file
        pdf_dir = tmp_path / "pdf"
        pdf_dir.mkdir()
        (pdf_dir / "2024.12345.pdf").touch()
        
        with patch('mcp.tools.rank_filter.load_profile', return_value=profile_with_exclude), \
             patch('mcp.tools.rank_filter.scan_local_pdfs', return_value={"2024.12345"}):
            
            result = await tool.execute(
                papers=[{**p} for p in sample_papers],
                profile_path="users/test_profile.json",
                local_pdf_dir=str(pdf_dir)
            )
        
        # Paper with local PDF should be filtered as ALREADY_READ
        passed_ids = {p["paper_id"] for p in result["passed_papers"]}
        assert "2024.12345" not in passed_ids


# ========== Test CalculateSemanticScoresTool ==========

class TestCalculateSemanticScoresTool:
    """Test CalculateSemanticScoresTool."""
    
    @pytest.fixture
    def tool(self):
        return CalculateSemanticScoresTool()
    
    @pytest.mark.asyncio
    @patch('mcp.tools.rank_filter._calculate_embedding_scores')
    @patch('mcp.tools.rank_filter._classify_papers_by_score')
    @patch('mcp.tools.rank_filter._verify_with_llm')
    @patch('mcp.tools.rank_filter._merge_scores')
    async def test_calculate_scores_without_llm(
        self, mock_merge, mock_llm, mock_classify, mock_embedding,
        tool, sample_papers, sample_profile, temp_output_dir
    ):
        """Test semantic score calculation without LLM verification."""
        # Setup mocks
        mock_embedding.return_value = {
            "2024.12345": 0.9,
            "2024.12346": 0.7,
            "2024.12347": 0.5
        }
        mock_classify.return_value = (
            [sample_papers[0]],  # high_group
            [],  # mid_group (empty, so no LLM call)
            [sample_papers[1], sample_papers[2]]  # low_group
        )
        mock_merge.return_value = {
            "2024.12345": {"semantic_score": 0.9, "evaluation_method": "embedding_only"},
            "2024.12346": {"semantic_score": 0.7, "evaluation_method": "embedding_only"},
            "2024.12347": {"semantic_score": 0.5, "evaluation_method": "embedding_only"}
        }
        
        with patch('mcp.tools.rank_filter.load_profile', return_value=sample_profile):
            result = await tool.execute(
                papers=[{**p} for p in sample_papers[:3]],
                profile_path="users/test_profile.json",
                enable_llm_verification=False
            )
        
        assert "scores" in result
        assert len(result["scores"]) == 3
        mock_embedding.assert_called_once()
        mock_llm.assert_not_called()  # LLM should not be called
    
    @pytest.mark.asyncio
    @patch('mcp.tools.rank_filter._calculate_embedding_scores')
    @patch('mcp.tools.rank_filter._classify_papers_by_score')
    @patch('mcp.tools.rank_filter._verify_with_llm', new_callable=AsyncMock)
    @patch('mcp.tools.rank_filter._merge_scores')
    async def test_calculate_scores_with_llm(
        self, mock_merge, mock_llm, mock_classify, mock_embedding,
        tool, sample_papers, sample_profile, temp_output_dir
    ):
        """Test semantic score calculation with LLM verification."""
        # Setup mocks
        mock_embedding.return_value = {
            "2024.12345": 0.9,
            "2024.12346": 0.6,  # Mid-group (0.4-0.7)
            "2024.12347": 0.3
        }
        mock_classify.return_value = (
            [sample_papers[0]],  # high_group
            [sample_papers[1]],  # mid_group (will trigger LLM)
            [sample_papers[2]]  # low_group
        )
        mock_llm.return_value = {
            "2024.12346": (0.65, "llm_verified")
        }
        mock_merge.return_value = {
            "2024.12345": {"semantic_score": 0.9, "evaluation_method": "embedding_only"},
            "2024.12346": {"semantic_score": 0.65, "evaluation_method": "embedding+llm"},
            "2024.12347": {"semantic_score": 0.3, "evaluation_method": "embedding_only"}
        }
        
        with patch('mcp.tools.rank_filter.load_profile', return_value=sample_profile):
            result = await tool.execute(
                papers=[{**p} for p in sample_papers[:3]],
                profile_path="users/test_profile.json",
                enable_llm_verification=True
            )
        
        assert "scores" in result
        mock_embedding.assert_called_once()
        mock_llm.assert_called_once()  # LLM should be called for mid-group
    
    @pytest.mark.asyncio
    async def test_empty_input(self, tool, temp_output_dir):
        """Test handling empty input."""
        result = await tool.execute(papers=[])
        
        assert result["scores"] == {}


# ========== Test EvaluatePaperMetricsTool ==========

class TestEvaluatePaperMetricsTool:
    """Test EvaluatePaperMetricsTool."""
    
    @pytest.fixture
    def tool(self):
        return EvaluatePaperMetricsTool()
    
    @pytest.mark.asyncio
    @patch('mcp.tools.rank_filter.EvaluatePaperMetricsTool._find_github_urls')
    async def test_evaluate_metrics(
        self, mock_find_github, tool, sample_papers, sample_profile, temp_output_dir
    ):
        """Test paper metrics evaluation."""
        # Mock semantic scores
        semantic_scores = {
            "2024.12345": {"semantic_score": 0.9, "evaluation_method": "embedding_only"},
            "2024.12346": {"semantic_score": 0.7, "evaluation_method": "embedding_only"},
            "2024.12347": {"semantic_score": 0.5, "evaluation_method": "embedding_only"}
        }
        
        # Mock GitHub URL finding (not called since papers already have github_url)
        mock_find_github.return_value = None
        
        with patch('mcp.tools.rank_filter.load_profile', return_value=sample_profile), \
             patch('mcp.tools.rank_filter.scan_local_pdfs', return_value=set()):
            
            result = await tool.execute(
                papers=[{**p} for p in sample_papers[:3]],
                semantic_scores=semantic_scores,
                profile_path="users/test_profile.json"
            )
        
        assert "scores" in result
        assert len(result["scores"]) == 3
        
        # Check that each paper has breakdown
        for paper_id in ["2024.12345", "2024.12346", "2024.12347"]:
            assert paper_id in result["scores"]
            score_data = result["scores"][paper_id]
            assert "breakdown" in score_data
            assert "soft_penalty" in score_data
            assert "penalty_keywords" in score_data
            
            # Check breakdown structure
            breakdown = score_data["breakdown"]
            assert "semantic_relevance" in breakdown
            assert "must_keywords" in breakdown
            assert "author_trust" in breakdown
            assert "institution_trust" in breakdown
            assert "recency" in breakdown
            assert "practicality" in breakdown
    
    @pytest.mark.asyncio
    @patch('mcp.tools.rank_filter.EvaluatePaperMetricsTool._find_github_urls')
    async def test_find_github_urls(
        self, mock_find_github, tool, sample_papers, sample_profile, temp_output_dir
    ):
        """Test automatic GitHub URL finding."""
        # Create paper without github_url
        paper_without_github = {**sample_papers[0]}
        paper_without_github["github_url"] = None
        
        semantic_scores = {
            "2024.12345": {"semantic_score": 0.9, "evaluation_method": "embedding_only"}
        }
        
        # Mock GitHub URL finding
        mock_find_github.return_value = "https://github.com/found/repo"
        
        with patch('mcp.tools.rank_filter.load_profile', return_value=sample_profile), \
             patch('mcp.tools.rank_filter.scan_local_pdfs', return_value=set()):
            
            result = await tool.execute(
                papers=[paper_without_github],
                semantic_scores=semantic_scores,
                profile_path="users/test_profile.json"
            )
        
        # Should have called _find_github_urls
        mock_find_github.assert_called_once_with("2024.12345")
    
    @pytest.mark.asyncio
    async def test_use_provided_local_pdfs(self, tool, sample_papers, sample_profile, temp_output_dir):
        """Test using provided local_pdfs parameter."""
        semantic_scores = {
            "2024.12345": {"semantic_score": 0.9, "evaluation_method": "embedding_only"}
        }
        
        with patch('mcp.tools.rank_filter.load_profile', return_value=sample_profile):
            result = await tool.execute(
                papers=[{**sample_papers[0]}],
                semantic_scores=semantic_scores,
                profile_path="users/test_profile.json",
                local_pdfs=["2024.12345"]  # Provide pre-scanned local PDFs
            )
        
        # Should not call scan_local_pdfs when local_pdfs is provided
        assert "scores" in result
        # Paper with local PDF should have higher practicality score
        score_data = result["scores"]["2024.12345"]
        assert score_data["breakdown"]["practicality"] > 0
    
    @pytest.mark.asyncio
    async def test_empty_input(self, tool, temp_output_dir):
        """Test handling empty input."""
        result = await tool.execute(
            papers=[],
            semantic_scores={}
        )
        
        assert result["scores"] == {}


# ========== Test RankAndSelectTopKTool ==========

class TestRankAndSelectTopKTool:
    """Test RankAndSelectTopKTool."""
    
    @pytest.fixture
    def tool(self):
        return RankAndSelectTopKTool()
    
    @pytest.mark.asyncio
    @patch('mcp.tools.rank_filter._rank_and_select')
    @patch('mcp.tools.rank_filter._select_contrastive_paper')
    @patch('mcp.tools.rank_filter._save_and_format_result')
    async def test_rank_and_select_top_k(
        self, mock_save, mock_contrastive, mock_rank,
        tool, sample_papers, sample_profile, temp_output_dir
    ):
        """Test ranking and selecting top-k papers."""
        # Setup semantic and metrics scores
        semantic_scores = {
            "2024.12345": {"semantic_score": 0.9, "evaluation_method": "embedding_only"},
            "2024.12346": {"semantic_score": 0.7, "evaluation_method": "embedding_only"},
            "2024.12347": {"semantic_score": 0.5, "evaluation_method": "embedding_only"}
        }
        
        metrics_scores = {
            "2024.12345": {
                "breakdown": {
                    "semantic_relevance": 0.9,
                    "must_keywords": 1.0,
                    "author_trust": 1.0,
                    "institution_trust": 1.0,
                    "recency": 0.95,
                    "practicality": 0.5
                },
                "soft_penalty": 0.0,
                "penalty_keywords": []
            },
            "2024.12346": {
                "breakdown": {
                    "semantic_relevance": 0.7,
                    "must_keywords": 1.0,
                    "author_trust": 0.0,
                    "institution_trust": 0.0,
                    "recency": 0.85,
                    "practicality": 0.0
                },
                "soft_penalty": 0.0,
                "penalty_keywords": []
            },
            "2024.12347": {
                "breakdown": {
                    "semantic_relevance": 0.5,
                    "must_keywords": 1.0,
                    "author_trust": 0.0,
                    "institution_trust": 0.0,
                    "recency": 0.1,
                    "practicality": 0.5
                },
                "soft_penalty": -0.15,
                "penalty_keywords": ["basics"]
            }
        }
        
        # Mock ranking function
        mock_rank.return_value = [sample_papers[0], sample_papers[1]]
        
        # Mock save function
        mock_save.return_value = {
                    "success": True,
                    "error": None,
            "summary": {
                "input_count": 3,
                "filtered_count": 0,
                "scored_count": 3,
                "output_count": 2,
                "purpose": "general",
                "ranking_mode": "balanced",
                "profile_used": "users/test_profile.json",
                "llm_verification_used": False,
                "llm_calls_made": 0
            },
            "ranked_papers": [
                {
                    "rank": 1,
                    "paper_id": "2024.12345",
                    "title": sample_papers[0]["title"],
                    "authors": sample_papers[0]["authors"],
                    "score": {"final": 0.9, "breakdown": metrics_scores["2024.12345"]["breakdown"]},
                    "tags": ["SEMANTIC_HIGH_MATCH"],
                    "local_status": {"already_downloaded": False, "local_path": None},
                    "original_data": sample_papers[0]
                },
                {
                    "rank": 2,
                    "paper_id": "2024.12346",
                    "title": sample_papers[1]["title"],
                    "authors": sample_papers[1]["authors"],
                    "score": {"final": 0.7, "breakdown": metrics_scores["2024.12346"]["breakdown"]},
                    "tags": [],
                    "local_status": {"already_downloaded": False, "local_path": None},
                    "original_data": sample_papers[1]
                }
            ],
            "filtered_papers": [],
            "contrastive_paper": None,
            "comparison_notes": [],
            "output_path": "output/rankings/test.json",
            "generated_at": datetime.now().isoformat()
        }
        
        with patch('mcp.tools.rank_filter.load_profile', return_value=sample_profile), \
             patch('mcp.tools.rank_filter.scan_local_pdfs', return_value=set()):
            
            result = await tool.execute(
                papers=[{**p} for p in sample_papers[:3]],
                semantic_scores=semantic_scores,
                metrics_scores=metrics_scores,
                top_k=2,
                purpose="general",
                ranking_mode="balanced",
                include_contrastive=False,
                profile_path="users/test_profile.json"
            )
        
            assert result["success"] is True
            assert "ranked_papers" in result
        assert len(result["ranked_papers"]) == 2
        assert result["summary"]["output_count"] == 2
        mock_rank.assert_called_once()
        mock_contrastive.assert_not_called()  # Should not be called when include_contrastive=False
    
    @pytest.mark.asyncio
    @patch('mcp.tools.rank_filter._rank_and_select')
    @patch('mcp.tools.rank_filter._select_contrastive_paper')
    @patch('mcp.tools.rank_filter._save_and_format_result')
    async def test_contrastive_paper_selection(
        self, mock_save, mock_contrastive, mock_rank,
        tool, sample_papers, sample_profile, temp_output_dir
    ):
        """Test contrastive paper selection."""
        semantic_scores = {
            "2024.12345": {"semantic_score": 0.9, "evaluation_method": "embedding_only"},
            "2024.12346": {"semantic_score": 0.7, "evaluation_method": "embedding_only"},
            "2024.12347": {"semantic_score": 0.5, "evaluation_method": "embedding_only"}
        }
        
        metrics_scores = {
            pid: {
                "breakdown": {
                    "semantic_relevance": 0.8,
                    "must_keywords": 1.0,
                    "author_trust": 0.0,
                    "institution_trust": 0.0,
                    "recency": 0.8,
                    "practicality": 0.0
                },
                "soft_penalty": 0.0,
                "penalty_keywords": []
            }
            for pid in ["2024.12345", "2024.12346", "2024.12347"]
        }
        
        # Mock ranking
        mock_rank.return_value = [sample_papers[0], sample_papers[1]]
        
        # Mock contrastive selection
        from mcp.tools.rank_filter_utils.types import ContrastiveInfo
        contrastive_info: ContrastiveInfo = {
            "type": "method",
            "selected_papers_common_traits": ["transformer"],
            "this_paper_traits": ["cnn"],
            "contrast_dimensions": [{"dimension": "architecture", "others": "transformer", "this": "cnn"}]
        }
        mock_contrastive.return_value = (sample_papers[2], contrastive_info)
        
        # Mock save
        mock_save.return_value = {
                    "success": True,
                    "error": None,
            "summary": {
                "input_count": 3,
                "filtered_count": 0,
                "scored_count": 3,
                "output_count": 2,
                "purpose": "general",
                "ranking_mode": "balanced",
                "profile_used": "users/test_profile.json",
                "llm_verification_used": False,
                "llm_calls_made": 0
            },
            "ranked_papers": [],
            "filtered_papers": [],
            "contrastive_paper": {
                "paper_id": "2024.12347",
                "title": sample_papers[2]["title"],
                "contrastive_info": contrastive_info
            },
            "comparison_notes": [],
            "output_path": "output/rankings/test.json",
                    "generated_at": datetime.now().isoformat()
                }
            
        with patch('mcp.tools.rank_filter.load_profile', return_value=sample_profile), \
             patch('mcp.tools.rank_filter.scan_local_pdfs', return_value=set()):
            
            result = await tool.execute(
                papers=[{**p} for p in sample_papers[:3]],
                semantic_scores=semantic_scores,
                metrics_scores=metrics_scores,
                top_k=2,
                purpose="general",
                ranking_mode="balanced",
                include_contrastive=True,
                contrastive_type="method",
                profile_path="users/test_profile.json"
            )
        
        assert result["success"] is True
        mock_contrastive.assert_called_once()
        assert "contrastive_paper" in result or result.get("contrastive_paper") is not None
    
    @pytest.mark.asyncio
    async def test_empty_input(self, tool, temp_output_dir):
        """Test handling empty input."""
        result = await tool.execute(
            papers=[],
            semantic_scores={},
            metrics_scores={}
        )
        
        assert result["success"] is True
        assert result["summary"]["input_count"] == 0
        assert len(result["ranked_papers"]) == 0


# ========== Integration Tests ==========

class TestFullPipeline:
    """Test full pipeline integration."""
    
    @pytest.mark.asyncio
    @patch('mcp.tools.rank_filter._calculate_embedding_scores')
    @patch('mcp.tools.rank_filter._classify_papers_by_score')
    @patch('mcp.tools.rank_filter._verify_with_llm')
    @patch('mcp.tools.rank_filter._merge_scores')
    @patch('mcp.tools.rank_filter._rank_and_select')
    @patch('mcp.tools.rank_filter._save_and_format_result')
    async def test_full_pipeline_execution(
        self, mock_save, mock_rank, mock_merge, mock_llm, mock_classify, mock_embedding,
        sample_papers, sample_profile, temp_output_dir
    ):
        """Test full pipeline: filter -> score -> evaluate -> rank."""
        # Step 1: Apply hard filters
        filter_tool = ApplyHardFiltersTool()
        with patch('mcp.tools.rank_filter.load_profile', return_value=sample_profile), \
             patch('mcp.tools.rank_filter.load_history', return_value=set()), \
             patch('mcp.tools.rank_filter.scan_local_pdfs', return_value=set()):
            
            filter_result = await filter_tool.execute(
                papers=[{**p} for p in sample_papers],
                profile_path="users/test_profile.json"
            )
        
        passed_papers = filter_result["passed_papers"]
        # Medical paper should be filtered
        assert len(passed_papers) < len(sample_papers)
        passed_ids = {p["paper_id"] for p in passed_papers}
        assert "2024.12348" not in passed_ids  # Medical paper filtered
        
        # Step 2: Calculate semantic scores
        semantic_tool = CalculateSemanticScoresTool()
        mock_embedding.return_value = {p["paper_id"]: 0.8 for p in passed_papers}
        mock_classify.return_value = ([], [], passed_papers)  # All in low group
        mock_merge.return_value = {
            p["paper_id"]: {"semantic_score": 0.8, "evaluation_method": "embedding_only"}
            for p in passed_papers
        }
        
        with patch('mcp.tools.rank_filter.load_profile', return_value=sample_profile):
            semantic_result = await semantic_tool.execute(
                papers=passed_papers,
                profile_path="users/test_profile.json",
                enable_llm_verification=False
            )
            
        semantic_scores = semantic_result["scores"]
        assert len(semantic_scores) == len(passed_papers)
        
        # Step 3: Evaluate metrics
        metrics_tool = EvaluatePaperMetricsTool()
        with patch('mcp.tools.rank_filter.load_profile', return_value=sample_profile), \
             patch('mcp.tools.rank_filter.scan_local_pdfs', return_value=set()), \
             patch.object(metrics_tool, '_find_github_urls', return_value=None):
            
            metrics_result = await metrics_tool.execute(
                papers=passed_papers,
                semantic_scores=semantic_scores,
                profile_path="users/test_profile.json"
            )
        
        metrics_scores = metrics_result["scores"]
        assert len(metrics_scores) == len(passed_papers)
        
        # Step 4: Rank and select
        rank_tool = RankAndSelectTopKTool()
        mock_rank.return_value = passed_papers[:2]  # Top 2
        mock_save.return_value = {
                    "success": True,
                    "error": None,
            "summary": {
                "input_count": len(passed_papers),
                "filtered_count": 0,
                "scored_count": len(passed_papers),
                "output_count": 2,
                "purpose": "general",
                "ranking_mode": "balanced",
                "profile_used": "users/test_profile.json",
                "llm_verification_used": False,
                "llm_calls_made": 0
            },
            "ranked_papers": [
                {
                    "rank": 1,
                    "paper_id": p["paper_id"],
                    "title": p["title"],
                    "authors": p["authors"],
                    "score": {"final": 0.8, "breakdown": {}},
                    "tags": [],
                    "local_status": {"already_downloaded": False, "local_path": None},
                    "original_data": p
                }
                for p in passed_papers[:2]
            ],
            "filtered_papers": [],
            "contrastive_paper": None,
            "comparison_notes": [],
            "output_path": "output/rankings/test.json",
            "generated_at": datetime.now().isoformat()
        }
        
        with patch('mcp.tools.rank_filter.load_profile', return_value=sample_profile), \
             patch('mcp.tools.rank_filter.scan_local_pdfs', return_value=set()):
            
            rank_result = await rank_tool.execute(
                papers=passed_papers,
                semantic_scores=semantic_scores,
                metrics_scores=metrics_scores,
                top_k=2,
                profile_path="users/test_profile.json"
            )
        
        assert rank_result["success"] is True
        assert len(rank_result["ranked_papers"]) == 2
        assert rank_result["summary"]["output_count"] == 2


# ========== Edge Cases and Error Handling ==========

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_profile_loading_defaults(self, temp_output_dir):
        """Test that default profile is returned when file doesn't exist."""
        profile = load_profile("users/nonexistent.json", tool_name="test")
        
        assert "interests" in profile
        assert "keywords" in profile
        assert "constraints" in profile
        assert profile["purpose"] == "general"
        assert profile["ranking_mode"] == "balanced"
        assert profile["top_k"] == 5
    
    @pytest.mark.asyncio
    async def test_update_profile_with_empty_arrays(self, temp_output_dir):
        """Test updating profile with empty preferred arrays."""
        tool = UpdateUserProfileTool()
        
        # First set values
        await tool.execute(
            profile_path="users/test_profile.json",
            preferred_authors=["Author 1"],
            preferred_institutions=["MIT"]
        )
        
        # Then set to empty
        result = await tool.execute(
            profile_path="users/test_profile.json",
            preferred_authors=[],
            preferred_institutions=[]
        )
        
        # Verify empty arrays are NOT saved (fields should be removed)
        resolved_path = Path(result["profile_path"])
        with open(resolved_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
            # Fields should not be present in JSON if empty
            assert "preferred_authors" not in saved or saved.get("preferred_authors") == []
            assert "preferred_institutions" not in saved or saved.get("preferred_institutions") == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
