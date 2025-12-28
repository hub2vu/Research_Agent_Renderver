"""
Agent Planner Module

Responsible for creating execution plans from goals.
This module is NEVER executed directly - only imported by client.py

NO side-effects allowed - planning is pure reasoning.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .memory import Plan, PlanStep

logger = logging.getLogger(__name__)


class Planner:
    """
    Creates execution plans from user goals.

    The planner analyzes the goal and available tools to create
    a sequence of steps that will achieve the goal.
    """

    def __init__(self, available_tools: List[Dict[str, Any]]):
        """
        Initialize the planner with available tools.

        Args:
            available_tools: List of tool schemas in OpenAI format
        """
        self.available_tools = available_tools
        self.tool_map = {
            tool["function"]["name"]: tool["function"]
            for tool in available_tools
        }

    def get_planning_prompt(self, goal: str, context: Dict[str, Any] = None) -> str:
        """
        Generate a prompt for the LLM to create a plan.

        Args:
            goal: The user's goal or request
            context: Additional context (e.g., previous results)

        Returns:
            A prompt string for the LLM
        """
        tools_description = self._format_tools_for_prompt()

        context_section = ""
        if context:
            context_section = f"""
Current Context:
{json.dumps(context, indent=2)}
"""

        return f"""You are a research assistant planner. Create a step-by-step plan to achieve the following goal.

Goal: {goal}
{context_section}
Available Tools:
{tools_description}

Create a plan as a JSON array of steps. Each step should have:
- step_number: Sequential number starting from 1
- tool_name: Name of the tool to use
- description: What this step accomplishes
- arguments: Dictionary of arguments for the tool
- depends_on: List of step numbers this step depends on (empty if none)

Important:
- Only use tools that are available
- Be specific with tool arguments
- Consider dependencies between steps
- Keep the plan focused and efficient

Respond with ONLY the JSON array, no additional text.

Example format:
[
  {{"step_number": 1, "tool_name": "list_pdfs", "description": "List available PDFs", "arguments": {{}}, "depends_on": []}},
  {{"step_number": 2, "tool_name": "extract_all", "description": "Extract content from paper", "arguments": {{"filename": "paper.pdf"}}, "depends_on": [1]}}
]
"""

    def _format_tools_for_prompt(self) -> str:
        """Format tools as a readable list for the prompt."""
        lines = []
        for name, func in self.tool_map.items():
            params = func.get("parameters", {}).get("properties", {})
            param_str = ", ".join(
                f"{p}: {info.get('type', 'any')}"
                for p, info in params.items()
            )
            lines.append(f"- {name}({param_str}): {func.get('description', 'No description')}")
        return "\n".join(lines)

    def parse_plan_response(self, response: str, goal: str) -> Optional[Plan]:
        """
        Parse the LLM's response into a Plan object.

        Args:
            response: The LLM's response (should be JSON)
            goal: The original goal

        Returns:
            A Plan object, or None if parsing fails
        """
        try:
            # Try to extract JSON from the response
            response = response.strip()

            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            steps_data = json.loads(response)

            if not isinstance(steps_data, list):
                logger.error("Plan response is not a list")
                return None

            steps = []
            for step_data in steps_data:
                step = PlanStep(
                    step_number=step_data.get("step_number", len(steps) + 1),
                    tool_name=step_data.get("tool_name", ""),
                    description=step_data.get("description", ""),
                    arguments=step_data.get("arguments", {}),
                    depends_on=step_data.get("depends_on", [])
                )
                steps.append(step)

            return Plan(goal=goal, steps=steps)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create plan: {e}")
            return None

    def validate_plan(self, plan: Plan) -> List[str]:
        """
        Validate a plan for correctness.

        Args:
            plan: The plan to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not plan.steps:
            errors.append("Plan has no steps")
            return errors

        step_numbers = set()
        for step in plan.steps:
            # Check for duplicate step numbers
            if step.step_number in step_numbers:
                errors.append(f"Duplicate step number: {step.step_number}")
            step_numbers.add(step.step_number)

            # Check if tool exists
            if step.tool_name not in self.tool_map:
                errors.append(f"Unknown tool in step {step.step_number}: {step.tool_name}")

            # Check dependencies exist
            for dep in step.depends_on:
                if dep not in step_numbers and dep >= step.step_number:
                    errors.append(
                        f"Step {step.step_number} depends on non-existent step {dep}"
                    )

        return errors

    def create_simple_plan(self, tool_name: str, arguments: Dict[str, Any]) -> Plan:
        """
        Create a simple single-step plan.

        Useful for direct tool invocations without complex planning.
        """
        step = PlanStep(
            step_number=1,
            tool_name=tool_name,
            description=f"Execute {tool_name}",
            arguments=arguments
        )
        return Plan(goal=f"Execute {tool_name}", steps=[step])

    def get_next_steps(self, plan: Plan) -> List[PlanStep]:
        """
        Get the next steps that can be executed.

        Returns steps whose dependencies are all completed.
        """
        completed_steps = {
            step.step_number
            for step in plan.steps
            if step.status == "completed"
        }

        next_steps = []
        for step in plan.steps:
            if step.status != "pending":
                continue

            # Check if all dependencies are completed
            if all(dep in completed_steps for dep in step.depends_on):
                next_steps.append(step)

        return next_steps

    def is_plan_complete(self, plan: Plan) -> bool:
        """Check if all steps in the plan are completed."""
        return all(step.status == "completed" for step in plan.steps)

    def is_plan_failed(self, plan: Plan) -> bool:
        """Check if any step in the plan has failed."""
        return any(step.status == "failed" for step in plan.steps)
