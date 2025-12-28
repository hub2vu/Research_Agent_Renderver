"""
MCP Tool Base Classes and Decorators

Provides common wrapper, validation, and error handling for all MCP tools.
"""

import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """Complete definition of an MCP tool."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None
    category: str = "general"


class MCPToolError(Exception):
    """Base exception for MCP tool errors."""
    def __init__(self, message: str, tool_name: str = None, details: Dict = None):
        self.message = message
        self.tool_name = tool_name
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(MCPToolError):
    """Raised when tool input validation fails."""
    pass


class ExecutionError(MCPToolError):
    """Raised when tool execution fails."""
    pass


class MCPTool(ABC):
    """
    Abstract base class for MCP tools.

    All tools must inherit from this class and implement:
    - name: Tool identifier
    - description: What the tool does
    - parameters: List of ToolParameter definitions
    - execute(): The actual tool logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass

    @property
    def parameters(self) -> List[ToolParameter]:
        """List of parameters the tool accepts."""
        return []

    @property
    def category(self) -> str:
        """Category for grouping tools."""
        return "general"

    def validate(self, **kwargs) -> Dict[str, Any]:
        """
        Validate input parameters.
        Returns validated/normalized parameters.
        Raises ValidationError if validation fails.
        """
        validated = {}

        for param in self.parameters:
            value = kwargs.get(param.name)

            if value is None:
                if param.required:
                    raise ValidationError(
                        f"Missing required parameter: {param.name}",
                        tool_name=self.name
                    )
                value = param.default

            validated[param.name] = value

        return validated

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with given parameters.
        This method should contain the actual tool logic.
        """
        pass

    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Public entry point: validate and execute.
        Returns standardized response format.
        """
        try:
            validated = self.validate(**kwargs)
            result = await self.execute(**validated)
            return {
                "success": True,
                "tool": self.name,
                "result": result
            }
        except ValidationError as e:
            logger.error(f"Validation error in {self.name}: {e.message}")
            return {
                "success": False,
                "tool": self.name,
                "error": e.message,
                "error_type": "validation"
            }
        except ExecutionError as e:
            logger.error(f"Execution error in {self.name}: {e.message}")
            return {
                "success": False,
                "tool": self.name,
                "error": e.message,
                "error_type": "execution"
            }
        except Exception as e:
            logger.exception(f"Unexpected error in {self.name}")
            return {
                "success": False,
                "tool": self.name,
                "error": str(e),
                "error_type": "unexpected"
            }

    def to_definition(self) -> ToolDefinition:
        """Convert tool to ToolDefinition for registry."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            handler=self.run,
            category=self.category
        )

    def to_openai_schema(self) -> Dict:
        """Convert tool to OpenAI function calling schema."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


# Registry for function-based tools
_function_tools: Dict[str, ToolDefinition] = {}


def tool(
    name: str,
    description: str,
    parameters: List[ToolParameter] = None,
    category: str = "general"
):
    """
    Decorator to register a function as an MCP tool.

    Usage:
        @tool(
            name="my_tool",
            description="Does something useful",
            parameters=[
                ToolParameter("input", "string", "The input value")
            ]
        )
        async def my_tool(input: str) -> str:
            return f"Processed: {input}"
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(**kwargs):
            try:
                result = await func(**kwargs)
                return {
                    "success": True,
                    "tool": name,
                    "result": result
                }
            except Exception as e:
                logger.exception(f"Error in tool {name}")
                return {
                    "success": False,
                    "tool": name,
                    "error": str(e),
                    "error_type": "execution"
                }

        # Register the tool
        _function_tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or [],
            handler=wrapper,
            category=category
        )

        return wrapper

    return decorator


def get_function_tools() -> Dict[str, ToolDefinition]:
    """Get all registered function-based tools."""
    return _function_tools.copy()
