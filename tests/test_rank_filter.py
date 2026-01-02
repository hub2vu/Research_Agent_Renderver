"""
Unit tests for rank_and_filter_papers tool.
"""

import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from mcp.tools.rank_filter import RankAndFilterPapersTool
from mcp.tools.rank_filter_utils import (
    FilteredPaper,
    PaperInput,
    UserProfile,
    filter_papers,
    load_profile,
    resolve_path,
)
from mcp.tools.rank_filter_utils.scores import (
    _apply_soft_penalty,
    _calculate_dimension_scores,
)


@pytest.fixture
def sample_papers() -> List[PaperInput]:
    """Sample papers for testing."""
    today = date.today()
    return [
        PaperInput(
            paper_id="2024.12345",
            title="Deep Learning for Natural Language Processing",
            abstract="We present a novel approach to NLP using deep learning.",
            authors=["John Doe", "Jane Smith"],
            published=str(today - timedelta(days=10)),
            categories=["cs.CL", "cs.LG"],
            github_url="https://github.com/test/repo1",
            affiliations=["MIT"]
        ),
        PaperInput(
            paper_id="2024.12346",
            title="Transformer Models in Computer Vision",
            abstract="We explore transformers for image classification tasks.",
            authors=["Alice Johnson"],
            published=str(today - timedelta(days=30)),
            categories=["cs.CV"],
            github_url="",
            affiliations=["Stanford"]
        ),
        PaperInput(
            paper_id="2024.12347",
            title="Reinforcement Learning Basics",
            abstract="Introduction to reinforcement learning algorithms.",
            authors=["Bob Wilson"],
            published=str(today - timedelta(days=365)),
            categories=["cs.LG"],
            github_url="https://github.com/test/repo2",
            affiliations=["CMU"]
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
                "hard": ["medical", "biology"],
                "soft": ["basics", "introduction"]
            }
        },
        "preferred_authors": ["John Doe"],
        "preferred_institutions": ["MIT"],
        "constraints": {
            "min_year": 2020,
            "require_code": False
        }
    }


@pytest.fixture
def tool():
    """Create a RankAndFilterPapersTool instance."""
    return RankAndFilterPapersTool()


class TestFiltering:
    """Test filtering functionality."""
    
    def test_filter_blacklist_keyword(self, sample_papers, sample_profile):
        """Test that papers with hard exclude keywords are filtered out."""
        # Add a paper with blacklisted keyword
        blacklisted_paper = PaperInput(
            paper_id="2024.99999",
            title="Medical AI Applications",
            abstract="We apply deep learning to medical diagnosis.",
            authors=["Dr. Medical"],
            published="2024-01-01",
            categories=["cs.AI"],
            github_url="",
            affiliations=[]
        )
        
        papers = sample_papers + [blacklisted_paper]
        history = set()
        
        passed, filtered = filter_papers(papers, sample_profile, history, "general")
        
        # Blacklisted paper should be filtered
        paper_ids = {p["paper_id"] for p in passed}
        assert "2024.99999" not in paper_ids
        
        # Should be in filtered list with correct reason
        filtered_ids = {f["paper_id"] for f in filtered}
        assert "2024.99999" in filtered_ids
        
        filtered_reasons = {f["filter_reason"] for f in filtered if f["paper_id"] == "2024.99999"}
        assert any("BLACKLIST_KEYWORD" in reason for reason in filtered_reasons)
    
    def test_filter_already_read(self, sample_papers, sample_profile):
        """Test that papers in history are filtered out."""
        history = {"2024.12345", "2024.12346"}
        
        passed, filtered = filter_papers(sample_papers, sample_profile, history, "general")
        
        # Papers in history should be filtered
        paper_ids = {p["paper_id"] for p in passed}
        assert "2024.12345" not in paper_ids
        assert "2024.12346" not in paper_ids
        assert "2024.12347" in paper_ids  # Not in history
        
        # Should be in filtered list with ALREADY_READ reason
        filtered_reasons = {
            f["filter_reason"] for f in filtered
            if f["paper_id"] in ["2024.12345", "2024.12346"]
        }
        assert all(reason == "ALREADY_READ" for reason in filtered_reasons)


class TestScoring:
    """Test scoring functionality."""
    
    def test_score_calculation(self, sample_papers, sample_profile):
        """Test dimension score calculation accuracy."""
        paper = sample_papers[0]  # Recent paper with preferred author
        local_pdfs = set()
        
        # Mock semantic score
        semantic_score = 0.8
        
        scores = _calculate_dimension_scores(paper, sample_profile, semantic_score, local_pdfs)
        
        # Check all dimension scores are present
        assert "semantic_relevance" in scores
        assert "must_keywords" in scores
        assert "author_trust" in scores
        assert "institution_trust" in scores
        assert "recency" in scores
        assert "practicality" in scores
        
        # Check specific values
        assert scores["semantic_relevance"] == 0.8
        assert scores["must_keywords"] == 1.0  # "learning" in title/abstract
        assert scores["author_trust"] == 1.0  # "John Doe" is preferred
        assert scores["institution_trust"] == 1.0  # "MIT" is preferred
        assert scores["recency"] >= 0.85  # Recent (within 2 weeks)
        assert scores["practicality"] == 0.5  # Has github_url
    
    def test_soft_penalty(self, sample_papers, sample_profile):
        """Test soft penalty application."""
        # Paper with soft exclude keyword
        paper = sample_papers[2]  # "Reinforcement Learning Basics" - has "basics"
        
        penalty, penalty_keywords = _apply_soft_penalty(paper, sample_profile)
        
        # Should have penalty
        assert penalty < 0
        assert "basics" in penalty_keywords or "introduction" in penalty_keywords
        
        # Paper without soft keywords should have no penalty
        paper_no_penalty = sample_papers[0]
        penalty2, keywords2 = _apply_soft_penalty(paper_no_penalty, sample_profile)
        assert penalty2 == 0.0
        assert len(keywords2) == 0


class TestRanking:
    """Test ranking functionality."""
    
    @patch('mcp.tools.rank_filter_utils.rankers.HAS_SENTENCE_TRANSFORMERS', False)
    def test_ranking_order(self, sample_papers, sample_profile):
        """Test that papers are ranked by final score."""
        from mcp.tools.rank_filter_utils.rankers import _rank_and_select
        
        # Create mock scores
        scores = {
            "2024.12345": {
                "final_score": 0.9,
                "breakdown": {},
                "soft_penalty": 0.0,
                "penalty_keywords": [],
                "evaluation_method": "embedding_only"
            },
            "2024.12346": {
                "final_score": 0.7,
                "breakdown": {},
                "soft_penalty": 0.0,
                "penalty_keywords": [],
                "evaluation_method": "embedding_only"
            },
            "2024.12347": {
                "final_score": 0.5,
                "breakdown": {},
                "soft_penalty": 0.0,
                "penalty_keywords": [],
                "evaluation_method": "embedding_only"
            },
        }
        
        ranked = _rank_and_select(sample_papers, scores, top_k=2, ranking_mode="balanced", include_contrastive=False)
        
        # Should be sorted by score descending
        assert len(ranked) == 2
        assert ranked[0]["paper_id"] == "2024.12345"  # Highest score
        assert ranked[1]["paper_id"] == "2024.12346"  # Second highest
    
    @patch('mcp.tools.rank_filter_utils.rankers.HAS_SENTENCE_TRANSFORMERS', False)
    def test_diversity_mode(self, sample_papers, sample_profile):
        """Test diversity mode similarity penalty."""
        from mcp.tools.rank_filter_utils.rankers import _rank_and_select
        
        # Create mock scores (all same score to test diversity logic)
        scores = {
            "2024.12345": {
                "final_score": 0.8,
                "breakdown": {},
                "soft_penalty": 0.0,
                "penalty_keywords": [],
                "evaluation_method": "embedding_only"
            },
            "2024.12346": {
                "final_score": 0.8,
                "breakdown": {},
                "soft_penalty": 0.0,
                "penalty_keywords": [],
                "evaluation_method": "embedding_only"
            },
            "2024.12347": {
                "final_score": 0.8,
                "breakdown": {},
                "soft_penalty": 0.0,
                "penalty_keywords": [],
                "evaluation_method": "embedding_only"
            },
        }
        
        ranked = _rank_and_select(sample_papers, scores, top_k=2, ranking_mode="diversity", include_contrastive=False)
        
        # Should select papers (diversity logic will be applied if embeddings available)
        assert len(ranked) == 2


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_input(self, tool):
        """Test handling of empty input."""
        result = await tool.execute(papers=[])
        
        assert result["success"] is True
        assert result["error"] is None
        assert len(result["ranked_papers"]) == 0
        assert result["summary"]["input_count"] == 0
    
    @pytest.mark.asyncio
    @patch('mcp.tools.rank_filter_utils.loaders.resolve_path')
    async def test_no_profile(self, mock_resolve_path, tool, sample_papers):
        """Test behavior when profile doesn't exist."""
        # Mock profile path to not exist
        mock_path = Mock(spec=Path)
        mock_path.exists.return_value = False
        mock_resolve_path.return_value = mock_path
        
        # Mock other functions
        with patch('mcp.tools.rank_filter_utils.loaders.load_history', return_value=set()), \
             patch('mcp.tools.rank_filter_utils.loaders.scan_local_pdfs', return_value=set()), \
             patch('mcp.tools.rank_filter_utils.filters.filter_papers', return_value=(sample_papers, [])), \
             patch('mcp.tools.rank_filter_utils.scores._calculate_embedding_scores', return_value={}), \
             patch('mcp.tools.rank_filter_utils.scores._classify_papers_by_score', return_value=([], [], [])), \
             patch('mcp.tools.rank_filter_utils.rankers._rank_and_select', return_value=[]), \
             patch('mcp.tools.rank_filter_utils.formatters._save_and_format_result', return_value={"success": True}):
            
            result = await tool.execute(papers=[{**p} for p in sample_papers])
            
            # Should still succeed with default profile
            assert result["success"] is True
    
    def test_path_resolution(self, tmp_path):
        """Test path resolution logic."""
        # Test absolute path
        abs_path = Path("/absolute/path/to/file.json")
        resolved = resolve_path(str(abs_path), "output")
        assert resolved == abs_path
        
        # Test relative path with environment variable
        with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
            resolved = resolve_path("relative/path.json", "output")
            assert resolved == tmp_path / "relative/path.json"
        
        # Test relative path without environment variable (should use cwd)
        with patch.dict(os.environ, {}, clear=True):
            resolved = resolve_path("relative/path.json", "output")
            assert resolved.is_absolute() or resolved == Path.cwd() / "relative/path.json"


class TestIntegration:
    """Integration tests with mocked external dependencies."""
    
    @pytest.mark.asyncio
    @patch('mcp.tools.rank_filter_utils.scores._calculate_embedding_scores')
    @patch('mcp.tools.rank_filter_utils.scores._verify_with_llm')
    async def test_full_pipeline(self, mock_llm, mock_embedding, tool, sample_profile):
        """Test full pipeline with mocked embeddings and LLM."""
        # Create 10 mock papers
        today = date.today()
        mock_papers = []
        for i in range(10):
            paper = PaperInput(
                paper_id=f"2024.{12000 + i}",
                title=f"Research Paper {i}: Deep Learning Applications",
                abstract=f"This paper discusses deep learning and machine learning techniques for problem {i}.",
                authors=[f"Author {i}", f"CoAuthor {i}"],
                published=str(today - timedelta(days=i * 30)),
                categories=["cs.LG", "cs.AI"],
                github_url=f"https://github.com/test/repo{i}" if i % 2 == 0 else "",
                affiliations=["MIT"] if i % 3 == 0 else ["Stanford"]
            )
            mock_papers.append(paper)
        
        # Setup mocks
        embedding_scores = {f"2024.{12000 + i}": 0.9 - (i * 0.05) for i in range(10)}
        mock_embedding.return_value = embedding_scores
        mock_llm.return_value = {}
        
        # Mock file operations
        with patch('mcp.tools.rank_filter_utils.loaders.load_profile', return_value=sample_profile), \
             patch('mcp.tools.rank_filter_utils.loaders.load_history', return_value=set()), \
             patch('mcp.tools.rank_filter_utils.loaders.scan_local_pdfs', return_value=set()), \
             patch('mcp.tools.rank_filter_utils.formatters._save_and_format_result') as mock_save:
            
            # Mock _save_and_format_result to return a proper result structure
            def mock_save_result(ranked_papers, filtered_papers, contrastive_paper, comparison_notes, summary, profile_path):
                return {
                    "success": True,
                    "error": None,
                    "summary": summary,
                    "ranked_papers": ranked_papers,
                    "filtered_papers": filtered_papers,
                    "contrastive_paper": contrastive_paper,
                    "comparison_notes": comparison_notes,
                    "output_path": f"rankings/{summary.get('output_count', 0)}_papers.json",
                    "generated_at": datetime.now().isoformat()
                }
            
            mock_save.side_effect = mock_save_result
            
            top_k = 5
            result = await tool.execute(
                papers=[{**p} for p in mock_papers],
                top_k=top_k,
                enable_llm_verification=False
            )
            
            # Verify results
            assert result["success"] is True
            assert "ranked_papers" in result
            assert len(result["ranked_papers"]) <= top_k
            
            # Verify each paper has required fields
            for paper in result["ranked_papers"]:
                assert "rank" in paper
                assert "paper_id" in paper
                assert "title" in paper
                assert "authors" in paper
                assert "score" in paper
                assert "tags" in paper
                assert isinstance(paper["tags"], list)
                assert "local_status" in paper
                assert "original_data" in paper
            
            # Verify output_path exists
            assert "output_path" in result
            assert result["output_path"] is not None
            
            # Verify mocks were called
            mock_embedding.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('mcp.tools.rank_filter_utils.scores._calculate_embedding_scores')
    @patch('mcp.tools.rank_filter_utils.scores._verify_with_llm')
    @patch('mcp.tools.rank_filter_utils.rankers._select_contrastive_paper')
    async def test_contrastive_selection(self, mock_contrastive, mock_llm, mock_embedding, tool, sample_papers, sample_profile):
        """Test contrastive paper selection."""
        # Setup mocks
        mock_embedding.return_value = {
            "2024.12345": 0.8,
            "2024.12346": 0.7,
            "2024.12347": 0.6
        }
        mock_llm.return_value = {}
        
        # Mock contrastive paper result
        from mcp.tools.rank_filter_utils.types import ContrastiveInfo
        
        contrastive_info: ContrastiveInfo = {
            "type": "method",
            "selected_papers_common_traits": ["transformer", "attention"],
            "this_paper_traits": ["cnn", "convolution"],
            "contrast_dimensions": [
                {
                    "dimension": "architecture",
                    "others": "transformer",
                    "this": "cnn"
                }
            ]
        }
        
        contrastive_paper = PaperInput(
            paper_id="2024.99999",
            title="CNN-Based Image Classification",
            abstract="We use CNN instead of transformers for image classification.",
            authors=["CNN Author"],
            published=str(date.today() - timedelta(days=60)),
            categories=["cs.CV"],
            github_url="",
            affiliations=["Other University"]
        )
        
        mock_contrastive.return_value = (contrastive_paper, contrastive_info)
        
        # Mock file operations
        with patch('mcp.tools.rank_filter_utils.loaders.load_profile', return_value=sample_profile), \
             patch('mcp.tools.rank_filter_utils.loaders.load_history', return_value=set()), \
             patch('mcp.tools.rank_filter_utils.loaders.scan_local_pdfs', return_value=set()), \
             patch('mcp.tools.rank_filter_utils.formatters._save_and_format_result') as mock_save:
            
            # Mock _save_and_format_result to return a proper result structure
            def mock_save_result(ranked_papers, filtered_papers, contrastive_paper, comparison_notes, summary, profile_path):
                return {
                    "success": True,
                    "error": None,
                    "summary": summary,
                    "ranked_papers": ranked_papers,
                    "filtered_papers": filtered_papers,
                    "contrastive_paper": contrastive_paper,
                    "comparison_notes": comparison_notes,
                    "output_path": "rankings/test.json",
                    "generated_at": datetime.now().isoformat()
                }
            
            mock_save.side_effect = mock_save_result
            
            result = await tool.execute(
                papers=[{**p} for p in sample_papers],
                top_k=2,
                include_contrastive=True,
                contrastive_type="method",
                enable_llm_verification=False
            )
            
            # Verify results
            assert result["success"] is True
            assert "contrastive_paper" in result
            assert result["contrastive_paper"] is not None
            
            # Verify contrastive_info structure
            contrastive = result["contrastive_paper"]
            assert "paper_id" in contrastive
            assert "title" in contrastive
            assert "score" in contrastive
            assert "tags" in contrastive
            assert "contrastive_info" in contrastive
            assert "original_data" in contrastive
            
            # Verify contrastive_info fields
            info = contrastive["contrastive_info"]
            assert "type" in info
            assert info["type"] == "method"
            assert "selected_papers_common_traits" in info
            assert isinstance(info["selected_papers_common_traits"], list)
            assert "this_paper_traits" in info
            assert isinstance(info["this_paper_traits"], list)
            assert "contrast_dimensions" in info
            assert isinstance(info["contrast_dimensions"], list)
            
            # Verify contrastive paper was selected
            mock_contrastive.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

