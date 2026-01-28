"""
Research Agent Tool

LLM-Orchestrated Research Agent that dynamically decides analysis steps.
Uses existing tools (generate_report, extract_paper_sections, analyze_section)
and makes intelligent decisions based on paper content.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from ..base import MCPTool, ToolParameter, ExecutionError
from ..registry import execute_tool

logger = logging.getLogger("mcp.tools.research_agent")
logger.setLevel(logging.INFO)

# OpenAI client
aclient = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Paths
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/output"))
PDF_DIR = Path(os.getenv("PDF_DIR", "data/pdf"))


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ReasoningEntry:
    """A single reasoning/decision entry by the agent."""
    timestamp: str
    paper_id: str
    context: str          # What the agent observed
    decision: str         # What the agent decided
    rationale: str        # Why the agent made this decision
    action_taken: str     # The tool/action executed


@dataclass
class PaperAnalysisResult:
    """Analysis results for a single paper."""
    paper_id: str
    title: str
    summary_report: str                      # From generate_report
    selected_sections: List[str]             # Sections chosen by LLM
    section_analyses: Dict[str, str]         # section_title -> analysis
    selection_reasoning: str                 # Why these sections were chosen


@dataclass
class AgentState:
    """State management for the research agent."""
    goal: str
    papers: List[str]
    analysis_mode: str = "quick"             # quick, standard, deep
    current_paper_idx: int = 0
    paper_results: List[PaperAnalysisResult] = field(default_factory=list)
    reasoning_log: List[ReasoningEntry] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def log_reasoning(
        self,
        paper_id: str,
        context: str,
        decision: str,
        rationale: str,
        action_taken: str
    ):
        """Add a reasoning entry to the log."""
        entry = ReasoningEntry(
            timestamp=datetime.now().isoformat(),
            paper_id=paper_id,
            context=context,
            decision=decision,
            rationale=rationale,
            action_taken=action_taken
        )
        self.reasoning_log.append(entry)
        logger.info(f"[Agent Decision] {paper_id}: {decision}")


# =============================================================================
# LLM Orchestrator
# =============================================================================

class LLMOrchestrator:
    """Handles LLM-based decision making for the agent."""
    
    SECTION_SELECTION_PROMPT = """You are a Research Agent analyzing an academic paper. 
Your task is to identify the most critical sections for deep analysis based on the paper's summary.

## Paper Summary Report
{summary_report}

## Available Sections (from table of contents)
{sections_list}

## User's Analysis Goal
{goal}

## Analysis Mode
{mode} (quick=2 sections, standard=3-4 sections, deep=5+ sections)

## Instructions
1. Read the summary report carefully to understand the paper's main contributions
2. Identify which sections contain the core technical contributions
3. Consider the user's goal when selecting sections
4. For "quick" mode, select exactly 2 most important sections
5. Avoid selecting Introduction, Conclusion, or Related Work unless specifically relevant to the goal

## Response Format (JSON)
{{
    "reasoning": "Your detailed explanation of why you selected these sections based on the summary",
    "selected_sections": ["Section Title 1", "Section Title 2"],
    "section_relevance": {{
        "Section Title 1": "Why this section is important",
        "Section Title 2": "Why this section is important"
    }}
}}

Respond ONLY with valid JSON, no additional text."""

    @staticmethod
    async def select_sections(
        summary_report: str,
        sections: List[Dict],
        goal: str,
        mode: str = "quick"
    ) -> Dict[str, Any]:
        """
        Use LLM to select which sections to analyze based on the summary report.
        
        Args:
            summary_report: The generated summary from generate_report
            sections: List of sections from extract_paper_sections
            goal: User's analysis goal
            mode: Analysis depth mode (quick/standard/deep)
            
        Returns:
            Dict with 'reasoning', 'selected_sections', 'section_relevance'
        """
        # Format sections for prompt
        sections_list = "\n".join([
            f"- {s.get('title', 'Unknown')} (Level {s.get('level', 1)}, Page {s.get('page', '?')})"
            for s in sections
        ])
        
        prompt = LLMOrchestrator.SECTION_SELECTION_PROMPT.format(
            summary_report=summary_report[:8000],  # Truncate if too long
            sections_list=sections_list,
            goal=goal,
            mode=mode
        )
        
        try:
            response = await aclient.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research assistant. Always respond with valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Validate response structure
            if "selected_sections" not in result:
                result["selected_sections"] = []
            if "reasoning" not in result:
                result["reasoning"] = "No reasoning provided"
            if "section_relevance" not in result:
                result["section_relevance"] = {}
                
            return result
            
        except Exception as e:
            logger.error(f"LLM section selection failed: {e}")
            # Fallback: select first 2 non-intro sections
            fallback_sections = [
                s.get("title") for s in sections
                if s.get("title") and "introduction" not in s.get("title", "").lower()
            ][:2]
            
            return {
                "reasoning": f"Fallback selection due to error: {str(e)}",
                "selected_sections": fallback_sections,
                "section_relevance": {}
            }

    @staticmethod
    async def generate_executive_summary(
        paper_results: List[PaperAnalysisResult],
        goal: str
    ) -> str:
        """Generate an executive summary across all analyzed papers."""
        
        papers_info = "\n\n".join([
            f"### {r.title}\n{r.summary_report[:2000]}..."
            for r in paper_results
        ])
        
        prompt = f"""You are a Research Agent. Generate a concise executive summary 
that synthesizes the key insights from all analyzed papers.

## User's Goal
{goal}

## Papers Analyzed
{papers_info}

## Instructions
1. Provide a 3-5 sentence summary highlighting the most important findings
2. Focus on insights relevant to the user's goal
3. Note any connections or contrasts between the papers
4. Write in Korean

## Response Format
Write the executive summary directly, no JSON needed.
"""
        
        try:
            response = await aclient.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            return f"분석된 논문 {len(paper_results)}편에 대한 요약 (자동 생성 실패)"


# =============================================================================
# Final Report Generator
# =============================================================================

class FinalReportGenerator:
    """Generates the final analysis report including agent reasoning."""
    
    @staticmethod
    def generate(
        state: AgentState,
        executive_summary: str
    ) -> str:
        """
        Generate the complete final report.
        
        Includes:
        - Executive summary
        - Per-paper summary reports
        - Per-paper deep analyses
        - Agent decision log
        """
        report_parts = []
        
        # Header
        report_parts.append("# 📚 Research Analysis Report")
        report_parts.append(f"\n생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report_parts.append(f"\n분석 목표: {state.goal}")
        report_parts.append(f"\n분석 모드: {state.analysis_mode}")
        report_parts.append(f"\n분석 논문 수: {len(state.paper_results)}")
        
        # Executive Summary
        report_parts.append("\n\n---\n")
        report_parts.append("## 📋 Executive Summary")
        report_parts.append(f"\n{executive_summary}")
        
        # Per-paper sections
        for i, result in enumerate(state.paper_results, 1):
            report_parts.append(f"\n\n---\n")
            report_parts.append(f"## 📄 Paper {i}: {result.title}")
            
            # Summary Report
            report_parts.append("\n### 1. 종합 요약 보고서")
            report_parts.append(f"\n{result.summary_report}")
            
            # Deep Analysis
            report_parts.append("\n### 2. 심층 분석")
            
            # Why these sections
            report_parts.append("\n#### 🤖 Agent의 섹션 선택 근거")
            report_parts.append(f"\n> {result.selection_reasoning}")
            report_parts.append(f"\n\n선택된 섹션: {', '.join(result.selected_sections)}")
            
            # Section analyses
            for section_title, analysis in result.section_analyses.items():
                report_parts.append(f"\n\n#### 📖 {section_title}")
                report_parts.append(f"\n{analysis}")
        
        # Agent Decision Log
        report_parts.append("\n\n---\n")
        report_parts.append("## 🧠 Agent Decision Log")
        report_parts.append("\n이 리포트는 AI Agent가 다음과 같은 판단 과정을 거쳐 생성되었습니다:\n")
        
        report_parts.append("\n| 시간 | 논문 | 결정 | 근거 |")
        report_parts.append("\n|------|------|------|------|")
        
        for entry in state.reasoning_log:
            time_short = entry.timestamp.split("T")[1][:8] if "T" in entry.timestamp else entry.timestamp
            decision_short = entry.decision[:50] + "..." if len(entry.decision) > 50 else entry.decision
            rationale_short = entry.rationale[:80] + "..." if len(entry.rationale) > 80 else entry.rationale
            report_parts.append(f"\n| {time_short} | {entry.paper_id[:15]} | {decision_short} | {rationale_short} |")
        
        # Errors if any
        if state.errors:
            report_parts.append("\n\n---\n")
            report_parts.append("## ⚠️ 처리 중 발생한 오류")
            for err in state.errors:
                report_parts.append(f"\n- {err}")
        
        # Footer
        report_parts.append("\n\n---\n")
        report_parts.append("\n*이 리포트는 LLM-Orchestrated Research Agent에 의해 자동 생성되었습니다.*")
        
        return "".join(report_parts)


# =============================================================================
# Main Research Agent Tool
# =============================================================================

class ResearchAgentTool(MCPTool):
    """
    LLM-Orchestrated Research Agent.
    
    Analyzes papers by:
    1. Generating summary reports (generate_report)
    2. Using LLM to select key sections based on summaries
    3. Performing deep analysis on selected sections (analyze_section)
    4. Compiling everything into a comprehensive report with reasoning
    """
    
    @property
    def name(self) -> str:
        return "run_research_agent"
    
    @property
    def description(self) -> str:
        return (
            "LLM-orchestrated research agent that analyzes papers intelligently. "
            "First generates summary reports, then uses LLM to select key sections, "
            "and performs deep analysis. Includes agent reasoning in the final report."
        )
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_ids",
                type="array",
                items_type="string",
                description="List of paper IDs to analyze (max 3)",
                required=True
            ),
            ToolParameter(
                name="goal",
                type="string",
                description="User's analysis goal (e.g., 'understand the methodology', 'implementation details')",
                required=False,
                default="general understanding"
            ),
            ToolParameter(
                name="analysis_mode",
                type="string",
                description="Analysis depth: 'quick' (2 sections), 'standard' (3-4), 'deep' (5+)",
                required=False,
                default="quick"
            ),
            ToolParameter(
                name="slack_webhook_full",
                type="string",
                description="Slack webhook URL for full report channel (optional)",
                required=False,
                default=""
            ),
            ToolParameter(
                name="slack_webhook_summary",
                type="string",
                description="Slack webhook URL for summary notification channel (optional)",
                required=False,
                default=""
            ),
            ToolParameter(
                name="source",
                type="string",
                description="Paper source: 'arxiv', 'neurips', 'local'",
                required=False,
                default="local"
            )
        ]
    
    @property
    def category(self) -> str:
        return "agent"
    
    async def execute(
        self,
        paper_ids: List[str],
        goal: str = "general understanding",
        analysis_mode: str = "quick",
        slack_webhook_full: str = "",
        slack_webhook_summary: str = "",
        source: str = "local",
        **kwargs
    ) -> Dict[str, Any]:
        """Execute the research agent pipeline."""
        
        # Limit papers to 3
        paper_ids = paper_ids[:3]
        
        # Initialize state
        state = AgentState(
            goal=goal,
            papers=paper_ids,
            analysis_mode=analysis_mode
        )
        
        logger.info(f"[Research Agent] Starting analysis of {len(paper_ids)} papers")
        logger.info(f"[Research Agent] Goal: {goal}, Mode: {analysis_mode}")
        
        # Process each paper
        for idx, paper_id in enumerate(paper_ids):
            state.current_paper_idx = idx
            logger.info(f"[Research Agent] Processing paper {idx + 1}/{len(paper_ids)}: {paper_id}")
            
            try:
                result = await self._process_single_paper(paper_id, state)
                state.paper_results.append(result)
            except Exception as e:
                error_msg = f"Paper {paper_id} 처리 중 오류: {str(e)}"
                state.errors.append(error_msg)
                logger.error(error_msg)
        
        # Generate executive summary
        executive_summary = ""
        if state.paper_results:
            executive_summary = await LLMOrchestrator.generate_executive_summary(
                state.paper_results,
                goal
            )
        
        # Generate final report
        final_report = FinalReportGenerator.generate(state, executive_summary)
        
        # Save report
        report_path = OUTPUT_DIR / "agent_reports"
        report_path.mkdir(parents=True, exist_ok=True)
        report_filename = f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_file = report_path / report_filename
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(final_report)
        
        logger.info(f"[Research Agent] Report saved to {report_file}")
        
        # Send notifications to Slack
        notification_results = {}
        
        # Send full report to first Slack channel
        if slack_webhook_full:
            try:
                full_message = f"📚 *Research Analysis Report*\n\n"
                full_message += f"*Goal:* {goal}\n"
                full_message += f"*Papers Analyzed:* {len(state.paper_results)}\n"
                full_message += f"*Analysis Mode:* {analysis_mode}\n"
                full_message += f"*Generated:* {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                full_message += f"---\n\n"
                full_message += final_report
                
                slack_full_result = await execute_tool(
                    "send_slack_notification",
                    webhook_url=slack_webhook_full,
                    message=full_message
                )
                notification_results["slack_full"] = slack_full_result
            except Exception as e:
                notification_results["slack_full"] = {"success": False, "error": str(e)}
        
        # Send summary to second Slack channel
        if slack_webhook_summary:
            try:
                slack_summary = f"📚 *Research Report Generated*\n\n"
                slack_summary += f"*Goal:* {goal}\n"
                slack_summary += f"*Papers:* {len(state.paper_results)}\n\n"
                slack_summary += f"*Executive Summary:*\n{executive_summary}"
                
                slack_summary_result = await execute_tool(
                    "send_slack_notification",
                    webhook_url=slack_webhook_summary,
                    message=slack_summary
                )
                notification_results["slack_summary"] = slack_summary_result
            except Exception as e:
                notification_results["slack_summary"] = {"success": False, "error": str(e)}
        
        return {
            "success": True,
            "papers_analyzed": len(state.paper_results),
            "report_path": str(report_file),
            "executive_summary": executive_summary,
            "reasoning_log_count": len(state.reasoning_log),
            "notifications": notification_results,
            "errors": state.errors if state.errors else None
        }
    
    async def _process_single_paper(
        self,
        paper_id: str,
        state: AgentState
    ) -> PaperAnalysisResult:
        """Process a single paper through the agent pipeline."""
        
        # Clean up paper_id (remove .pdf extension if present)
        clean_paper_id = paper_id.replace(".pdf", "")
        
        # Phase 1: Check if extraction exists, if not extract
        text_path = OUTPUT_DIR / clean_paper_id / "extracted_text.json"
        if not text_path.exists():
            logger.info(f"[Agent] Extracting text from {paper_id}")
            
            # Try to find PDF
            pdf_filename = f"{paper_id}" if paper_id.endswith(".pdf") else f"{paper_id}.pdf"
            
            extract_result = await execute_tool("extract_all", filename=pdf_filename)
            
            if not extract_result.get("success"):
                raise ExecutionError(
                    f"Text extraction failed: {extract_result.get('error')}",
                    tool_name=self.name
                )
            
            state.log_reasoning(
                paper_id=clean_paper_id,
                context="PDF 파일 발견",
                decision="텍스트 추출 실행",
                rationale="분석을 위해 PDF에서 텍스트를 먼저 추출해야 함",
                action_taken="extract_all"
            )
        
        # Phase 2: Generate summary report
        logger.info(f"[Agent] Generating summary report for {clean_paper_id}")
        
        report_result = await execute_tool("generate_report", paper_id=clean_paper_id)
        
        if not report_result.get("success"):
            raise ExecutionError(
                f"Report generation failed: {report_result.get('error')}",
                tool_name=self.name
            )
        
        summary_report = report_result.get("result", {}).get("content", "")
        
        state.log_reasoning(
            paper_id=clean_paper_id,
            context="텍스트 추출 완료",
            decision="종합 요약 보고서 생성",
            rationale="논문의 전체 구조와 핵심 기여를 파악하기 위해 먼저 요약 생성",
            action_taken="generate_report"
        )
        
        # Extract title from summary (first line usually contains title)
        title_match = re.search(r"(?:제목|Title)[:\s]*(.+?)(?:\n|$)", summary_report, re.IGNORECASE)
        paper_title = title_match.group(1).strip() if title_match else clean_paper_id
        
        # Phase 3: Extract sections (table of contents)
        logger.info(f"[Agent] Extracting sections for {clean_paper_id}")
        
        sections_result = await execute_tool("extract_paper_sections", paper_id=clean_paper_id)
        
        sections = []
        if sections_result.get("success"):
            sections = sections_result.get("result", {}).get("sections", [])
        
        if not sections:
            # Fallback: create generic sections
            sections = [
                {"title": "Abstract", "level": 1, "page": 1},
                {"title": "Introduction", "level": 1, "page": 1},
                {"title": "Method", "level": 1, "page": 3},
                {"title": "Experiments", "level": 1, "page": 6},
                {"title": "Conclusion", "level": 1, "page": 10}
            ]
        
        # Phase 4: LLM decides which sections to analyze
        logger.info(f"[Agent] LLM selecting sections for {clean_paper_id}")
        
        llm_decision = await LLMOrchestrator.select_sections(
            summary_report=summary_report,
            sections=sections,
            goal=state.goal,
            mode=state.analysis_mode
        )
        
        selected_sections = llm_decision.get("selected_sections", [])
        selection_reasoning = llm_decision.get("reasoning", "")
        
        state.log_reasoning(
            paper_id=clean_paper_id,
            context=f"목차 추출 완료: {len(sections)}개 섹션 발견",
            decision=f"분석할 섹션 선택: {', '.join(selected_sections)}",
            rationale=selection_reasoning,
            action_taken="LLM section selection"
        )
        
        # Phase 5: Analyze selected sections
        section_analyses = {}
        
        for section_title in selected_sections:
            logger.info(f"[Agent] Analyzing section: {section_title}")
            
            # Find next section for boundary
            next_section = ""
            for i, s in enumerate(sections):
                if s.get("title") == section_title and i + 1 < len(sections):
                    next_section = sections[i + 1].get("title", "")
                    break
            
            analysis_result = await execute_tool(
                "analyze_section",
                paper_id=clean_paper_id,
                section_title=section_title,
                next_section_title=next_section
            )
            
            if analysis_result.get("success"):
                analysis_text = analysis_result.get("result", {}).get("analysis_text", "")
                section_analyses[section_title] = analysis_text
                
                state.log_reasoning(
                    paper_id=clean_paper_id,
                    context=f"섹션 '{section_title}' 분석 중",
                    decision="심층 분석 완료",
                    rationale=llm_decision.get("section_relevance", {}).get(section_title, "핵심 섹션"),
                    action_taken=f"analyze_section({section_title})"
                )
            else:
                section_analyses[section_title] = f"분석 실패: {analysis_result.get('error', 'Unknown error')}"
        
        return PaperAnalysisResult(
            paper_id=clean_paper_id,
            title=paper_title,
            summary_report=summary_report,
            selected_sections=selected_sections,
            section_analyses=section_analyses,
            selection_reasoning=selection_reasoning
        )
