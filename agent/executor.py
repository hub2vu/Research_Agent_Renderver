"""
Agent Executor Module

Converts plans into MCP tool calls.
This module is NEVER executed directly - only imported by client.py

The executor bridges the Agent layer (pure reasoning) and MCP layer (side-effects).
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .memory import Memory, Plan, PlanStep

logger = logging.getLogger(__name__)


class ExecutionResult:
    """Result of executing a plan step."""

    def __init__(
        self,
        step: PlanStep,
        success: bool,
        result: Any = None,
        error: str = None,
        duration_ms: float = None
    ):
        self.step = step
        self.success = success
        self.result = result
        self.error = error
        self.duration_ms = duration_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step.step_number,
            "tool_name": self.step.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms
        }


class Executor:
    """
    Executes plans by calling MCP tools.

    The executor:
    1. Takes a plan from the Planner
    2. Resolves step dependencies
    3. Calls MCP tools via the provided tool_caller
    4. Records results in Memory
    """

    def __init__(
        self,
        tool_caller: Callable[[str, Dict[str, Any]], Any],
        memory: Memory
    ):
        """
        Initialize the executor.

        Args:
            tool_caller: Async function to call MCP tools.
                         Signature: async (tool_name, arguments) -> result
            memory: Memory instance for recording results
        """
        self.tool_caller = tool_caller
        self.memory = memory

    async def execute_step(self, step: PlanStep) -> ExecutionResult:
        """
        Execute a single plan step.

        Args:
            step: The step to execute

        Returns:
            ExecutionResult with success status and result/error
        """
        logger.info(f"Executing step {step.step_number}: {step.tool_name}")
        step.status = "running"

        start_time = time.time()

        try:
            # Call the MCP tool
            result = await self.tool_caller(step.tool_name, step.arguments)

            duration_ms = (time.time() - start_time) * 1000

            # Check if the result indicates success
            if isinstance(result, dict) and "success" in result:
                success = result["success"]
                if not success:
                    error = result.get("error", "Unknown error")
                    step.status = "failed"
                    step.result = error

                    self.memory.add_tool_result(
                        step.tool_name,
                        step.arguments,
                        error,
                        success=False,
                        duration_ms=duration_ms
                    )

                    return ExecutionResult(
                        step=step,
                        success=False,
                        error=error,
                        duration_ms=duration_ms
                    )
                else:
                    result = result.get("result", result)

            # Success
            step.status = "completed"
            step.result = result

            self.memory.add_tool_result(
                step.tool_name,
                step.arguments,
                result,
                success=True,
                duration_ms=duration_ms
            )

            return ExecutionResult(
                step=step,
                success=True,
                result=result,
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error = str(e)

            step.status = "failed"
            step.result = error

            self.memory.add_tool_result(
                step.tool_name,
                step.arguments,
                error,
                success=False,
                duration_ms=duration_ms
            )

            logger.error(f"Step {step.step_number} failed: {error}")

            return ExecutionResult(
                step=step,
                success=False,
                error=error,
                duration_ms=duration_ms
            )

    async def execute_plan(
        self,
        plan: Plan,
        on_step_complete: Callable[[ExecutionResult], None] = None
    ) -> List[ExecutionResult]:
        """
        Execute an entire plan.

        Steps are executed in dependency order. Steps with no dependencies
        or whose dependencies are complete can run concurrently.

        Args:
            plan: The plan to execute
            on_step_complete: Optional callback for each completed step

        Returns:
            List of ExecutionResults for all steps
        """
        self.memory.set_plan(plan)
        plan.status = "running"
        results = []

        while True:
            # Get steps that are ready to execute
            ready_steps = self._get_ready_steps(plan)

            if not ready_steps:
                # Check if we're done or stuck
                if self._is_plan_complete(plan):
                    plan.status = "completed"
                    break
                elif self._has_failed_steps(plan):
                    plan.status = "failed"
                    break
                else:
                    # No ready steps but plan not complete - shouldn't happen
                    logger.error("Plan execution stuck - no ready steps")
                    plan.status = "failed"
                    break

            # Execute ready steps (could be parallel in future)
            for step in ready_steps:
                result = await self.execute_step(step)
                results.append(result)

                self.memory.update_plan_step(
                    step.step_number,
                    step.status,
                    step.result
                )

                if on_step_complete:
                    on_step_complete(result)

                # If step failed and is critical, stop
                if not result.success:
                    logger.warning(f"Step {step.step_number} failed, continuing...")

        return results

    def _get_ready_steps(self, plan: Plan) -> List[PlanStep]:
        """Get steps whose dependencies are all completed."""
        completed = {
            step.step_number
            for step in plan.steps
            if step.status == "completed"
        }

        ready = []
        for step in plan.steps:
            if step.status != "pending":
                continue
            if all(dep in completed for dep in step.depends_on):
                ready.append(step)

        return ready

    def _is_plan_complete(self, plan: Plan) -> bool:
        """Check if all steps are completed."""
        return all(
            step.status in ("completed", "failed")
            for step in plan.steps
        )

    def _has_failed_steps(self, plan: Plan) -> bool:
        """Check if any steps have failed."""
        return any(step.status == "failed" for step in plan.steps)

    async def execute_single_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute a single tool without a full plan.

        Convenience method for simple, direct tool calls.
        """
        step = PlanStep(
            step_number=1,
            tool_name=tool_name,
            description=f"Direct call to {tool_name}",
            arguments=arguments
        )
        return await self.execute_step(step)
