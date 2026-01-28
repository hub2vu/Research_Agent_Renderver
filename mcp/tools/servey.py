"""
Survey Generation Tools (Pro Version)
Generates high-quality survey papers using a Two-Pass (Plan -> Write) approach.
"""

import os
import logging
from typing import Any, Dict, List
from openai import AsyncOpenAI

from ..base import MCPTool, ToolParameter, ExecutionError

logger = logging.getLogger("mcp.tools.survey")
logger.setLevel(logging.INFO)

aclient = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class GenerateClusterSurveyTool(MCPTool):
    """
    Two-Pass Survey Generator:
    1. Plan: Analyze papers to create a Taxonomy.
    2. Write: Generate the full survey based on the Taxonomy.
    """

    @property
    def name(self) -> str:
        return "generate_cluster_survey"

    @property
    def description(self) -> str:
        return "Write a professional survey paper using a Plan-then-Write strategy."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="topic", type="string", description="Survey topic", required=True
            ),
            ToolParameter(
                name="papers_context",
                type="string",
                description="Paper details",
                required=True,
            ),
            ToolParameter(
                name="language",
                type="string",
                description="Output language",
                required=False,
                default="Korean",
            ),
        ]

    @property
    def category(self) -> str:
        return "analysis"

    async def execute(
        self, topic: str, papers_context: str, language: str = "Korean"
    ) -> Dict[str, Any]:
        logger.info(f"Generating Pro Survey for: {topic}")

        try:
            # ====================================================
            # PHASE 1: The Architect (Taxonomy & Structure Plan)
            # ====================================================
            logger.info("Phase 1: Planning Taxonomy...")

            plan_prompt = f"""
            You are a Senior Editor at a top AI journal.
            Task: Analyze the provided papers and design a **Taxonomy (Classification System)** for a survey paper on "{topic}".

            **Source Papers:**
            {papers_context}

            **Goal:**
            Group these papers into 3-4 distinct, logical categories based on their **Methodology** or **Problem Definition**.
            Do NOT just list them. Find the underlying structure.

            **Output Format:**
            Just return the categories and which papers belong to them.
            1. Category Name A: [Paper IDs...] - Brief rationale
            2. Category Name B: [Paper IDs...] - Brief rationale
            ...
            """

            plan_response = await aclient.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": plan_prompt}],
                temperature=0.2,  # 기획은 냉철하게
            )
            taxonomy_plan = plan_response.choices[0].message.content

            # ====================================================
            # PHASE 2: The Writer (Deep Dive & Synthesis)
            # ====================================================
            logger.info("Phase 2: Writing Full Paper...")

            write_prompt = f"""
            You are a Distinguished Researcher writing a Survey Paper.
            
            **Topic:** {topic}
            **Language:** {language}
            
            **Instructions:**
            Write a comprehensive survey paper following the **Taxonomy Plan** below.
            Your tone must be highly academic, critical, and insightful.
            
            **Taxonomy Plan (Use this structure):**
            {taxonomy_plan}

            **Source Papers:**
            {papers_context}

            **Required Structure (Markdown):**
            
            # {topic}: A Comprehensive Survey

            ## 1. Introduction
            - Define the problem scope.
            - Explain why this classification (Taxonomy) matters.

            ## 2. Taxonomy & Categorization (Main Body)
            - For EACH category in the plan:
              - **Define** the approach conceptually.
              - **Synthesize** the papers (Don't just list summaries. Connect them: "While Paper A proposed X, Paper B improved it by Y").
              - **Cite** specific papers using [Paper X].

            ## 3. Comparative Analysis
            - Create a **Markdown Table** comparing key methods.
            - Columns: [Method/Category, Core Technique, Pros, Cons].

            ## 4. Challenges & Future Directions
            - Identify open problems NOT solved by these papers.
            - Propose specific future research directions.

            **Style Guidelines:**
            - Use **bold** for emphasis.
            - If possible, include a textual representation of a diagram (e.g., Mermaid or ASCII) to explain the taxonomy.
            """

            final_response = await aclient.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": write_prompt}],
                temperature=0.4,  # 글쓰기는 약간 창의적으로
            )

            return {
                "topic": topic,
                "status": "success",
                "survey_content": final_response.choices[0].message.content,
                "taxonomy_debug": taxonomy_plan,  # 디버깅용 (필요시 확인 가능)
            }

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise ExecutionError(f"Survey generation failed: {str(e)}", self.name)
