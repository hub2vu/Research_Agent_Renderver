"""
Agent Layer

This layer handles thinking, planning, and decision-making.
NO side-effects allowed - all actions go through MCP layer.
"""

from .client import AgentClient

__all__ = ["AgentClient"]
