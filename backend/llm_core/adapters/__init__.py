"""
LLM Adapters Module

Contains implementation of the LLMInterface for OpenRouter:
- OpenRouter (Unified access to multiple LLM providers)
"""

from .openrouter_adapter import OpenRouterAdapter

__all__ = [
    'OpenRouterAdapter'
]
