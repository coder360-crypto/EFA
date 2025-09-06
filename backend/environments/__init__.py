"""
Environments Module

This module provides environment management for different contexts:
- Base environment interface and tools registry
- Nextcloud environment with cloud storage tools
- Custom environment template
- Environment manager for dynamic switching
"""

from .environment_manager import EnvironmentManager
from .base_environment.environment import Environment
from .base_environment.tools_registry import ToolsRegistry

__all__ = [
    'EnvironmentManager',
    'Environment', 
    'ToolsRegistry'
]
