"""
MCP Tool Registry

Single Source of Truth (SSOT) for tool discovery and collection.
Automatically discovers and registers all tools from mcp/tools/ directory.
"""

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Type

from .base import MCPTool, ToolDefinition, get_function_tools

logger = logging.getLogger(__name__)

# Global registry
_tool_registry: Dict[str, ToolDefinition] = {}
_initialized: bool = False


def _discover_tools() -> None:
    """
    Discover and register all tools from mcp/tools/ directory.
    This is the ONLY place where tools are collected.
    """
    global _tool_registry, _initialized

    if _initialized:
        return

    tools_package = "mcp.tools"
    tools_path = Path(__file__).parent / "tools"

    if not tools_path.exists():
        logger.warning(f"Tools directory not found: {tools_path}")
        _initialized = True
        return

    # Import all modules in mcp/tools/
    for _, module_name, _ in pkgutil.iter_modules([str(tools_path)]):
        if module_name.startswith("_"):
            continue

        try:
            full_module_name = f"{tools_package}.{module_name}"
            module = importlib.import_module(full_module_name)
            logger.debug(f"Loaded tool module: {full_module_name}")

            # Find all MCPTool subclasses in the module
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, MCPTool)
                    and obj is not MCPTool
                    and not inspect.isabstract(obj)
                ):
                    try:
                        instance = obj()
                        definition = instance.to_definition()
                        _tool_registry[definition.name] = definition
                        logger.info(f"Registered tool: {definition.name} ({module_name})")
                    except Exception as e:
                        logger.error(f"Failed to instantiate tool {name}: {e}")

        except Exception as e:
            logger.error(f"Failed to load tool module {module_name}: {e}")

    # Also collect function-based tools (decorated with @tool)
    for name, definition in get_function_tools().items():
        if name not in _tool_registry:
            _tool_registry[name] = definition
            logger.info(f"Registered function tool: {name}")

    _initialized = True
    logger.info(f"Tool discovery complete. Total tools: {len(_tool_registry)}")


def get_all_tools() -> Dict[str, ToolDefinition]:
    """
    Get all registered tools.
    This is the public API for accessing tools.
    """
    _discover_tools()
    return _tool_registry.copy()


def get_tool(name: str) -> Optional[ToolDefinition]:
    """
    Get a specific tool by name.
    Returns None if tool not found.
    """
    _discover_tools()
    return _tool_registry.get(name)


def get_tools_by_category(category: str) -> Dict[str, ToolDefinition]:
    """Get all tools in a specific category."""
    _discover_tools()
    return {
        name: tool
        for name, tool in _tool_registry.items()
        if tool.category == category
    }


def list_tool_names() -> List[str]:
    """Get list of all registered tool names."""
    _discover_tools()
    return list(_tool_registry.keys())


def get_openai_tools_schema() -> List[Dict]:
    """
    Get all tools in OpenAI function calling format.
    Used by the Agent layer to configure LLM.
    """
    _discover_tools()
    schemas = []

    for name, definition in _tool_registry.items():
        properties: Dict[str, Dict] = {}
        required: List[str] = []

        for param in definition.parameters:
            prop: Dict[str, Dict] = {
                "description": param.description
            }

            # 🔥 핵심: array 타입 처리
            if param.type == "array":
                prop["type"] = "array"
                prop["items"] = {
                    "type": getattr(param, "items_type", "string")
                }
            else:
                prop["type"] = param.type

            # default 값이 있으면 포함 (OpenAI schema 허용)
            if getattr(param, "default", None) is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": definition.description,
                "parameters": {
                    "type": "object",
                    "properties": properties
                }
            }
        }

        # required가 있을 때만 포함 (빈 리스트는 넣지 않는 게 안전)
        if required:
            schema["function"]["parameters"]["required"] = required

        schemas.append(schema)

    return schemas


async def execute_tool(name: str, **kwargs) -> Dict:
    """
    Execute a tool by name with given arguments.
    Returns standardized response format.
    """
    tool = get_tool(name)

    if tool is None:
        return {
            "success": False,
            "tool": name,
            "error": f"Tool not found: {name}",
            "error_type": "not_found"
        }

    if tool.handler is None:
        return {
            "success": False,
            "tool": name,
            "error": f"Tool has no handler: {name}",
            "error_type": "no_handler"
        }

    return await tool.handler(**kwargs)


def reset_registry() -> None:
    """Reset the registry (mainly for testing)."""
    global _tool_registry, _initialized
    _tool_registry = {}
    _initialized = False
