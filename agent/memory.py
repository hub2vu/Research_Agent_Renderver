"""
Agent Memory Module

Stores conversation history, intermediate results, and state.
This module is NEVER executed directly - only imported by client.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class MessageRole(Enum):
    """Role of a message in the conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in the conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI message format."""
        return {
            "role": self.role.value,
            "content": self.content
        }


@dataclass
class ToolCall:
    """Record of a tool invocation."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[float] = None


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    step_number: int
    tool_name: str
    description: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None


@dataclass
class Plan:
    """An execution plan."""
    goal: str
    steps: List[PlanStep]
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, running, completed, failed


class Memory:
    """
    Agent memory - stores all state.

    Responsibilities:
    - Conversation history
    - Tool call history
    - Current plan
    - Intermediate results
    - Context for the LLM
    """

    def __init__(self, system_prompt: str = None):
        self.messages: List[Message] = []
        self.tool_calls: List[ToolCall] = []
        self.current_plan: Optional[Plan] = None
        self.context: Dict[str, Any] = {}

        # Add system message if provided
        if system_prompt:
            self.add_message(MessageRole.SYSTEM, system_prompt)

    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> Message:
        """Add a message to the conversation history."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        return message

    def add_user_message(self, content: str) -> Message:
        """Add a user message."""
        return self.add_message(MessageRole.USER, content)

    def add_assistant_message(self, content: str) -> Message:
        """Add an assistant message."""
        return self.add_message(MessageRole.ASSISTANT, content)

    def add_tool_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
        duration_ms: float = None
    ) -> ToolCall:
        """Record a tool call and its result."""
        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            success=success,
            duration_ms=duration_ms
        )
        self.tool_calls.append(tool_call)

        # Also add as a tool message for context
        result_str = str(result) if not isinstance(result, str) else result
        self.add_message(
            MessageRole.TOOL,
            result_str,
            metadata={"tool_name": tool_name, "success": success}
        )

        return tool_call

    def set_plan(self, plan: Plan) -> None:
        """Set the current execution plan."""
        self.current_plan = plan

    def update_plan_step(
        self,
        step_number: int,
        status: str,
        result: Any = None
    ) -> None:
        """Update a step in the current plan."""
        if self.current_plan is None:
            return

        for step in self.current_plan.steps:
            if step.step_number == step_number:
                step.status = status
                step.result = result
                break

    def get_context_messages(self, max_messages: int = 50) -> List[Dict]:
        """Get messages in OpenAI format for context."""
        messages = self.messages[-max_messages:]
        return [msg.to_openai_format() for msg in messages]

    def get_recent_tool_calls(self, n: int = 10) -> List[ToolCall]:
        """Get the n most recent tool calls."""
        return self.tool_calls[-n:]

    def set_context(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.context.get(key, default)

    def clear_context(self) -> None:
        """Clear all context variables."""
        self.context = {}

    def reset(self, keep_system: bool = True) -> None:
        """Reset the memory, optionally keeping the system message."""
        system_message = None
        if keep_system and self.messages:
            for msg in self.messages:
                if msg.role == MessageRole.SYSTEM:
                    system_message = msg
                    break

        self.messages = []
        self.tool_calls = []
        self.current_plan = None
        self.context = {}

        if system_message:
            self.messages.append(system_message)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the memory state."""
        return {
            "total_messages": len(self.messages),
            "total_tool_calls": len(self.tool_calls),
            "successful_tool_calls": len([t for t in self.tool_calls if t.success]),
            "has_plan": self.current_plan is not None,
            "plan_status": self.current_plan.status if self.current_plan else None,
            "context_keys": list(self.context.keys())
        }
