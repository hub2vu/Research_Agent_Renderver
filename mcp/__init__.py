"""
MCP (Model Context Protocol) Layer

This layer handles actual side-effects: file I/O, API calls, etc.
All tools are auto-discovered via registry.py
"""

from .registry import get_all_tools, get_tool
from .base import MCPTool, tool

__all__ = ["get_all_tools", "get_tool", "MCPTool", "tool"]
