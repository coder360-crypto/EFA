"""
Base Environment Package

Provides the foundation for all environment implementations:
- Abstract Environment base class
- Tools registry for managing environment-specific tools
"""

from .environment import Environment
from .tools_registry import ToolsRegistry

__all__ = [
    'Environment',
    'ToolsRegistry'
]
