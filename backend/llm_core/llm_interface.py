"""
Abstract LLM Interface

Defines the standard interface that all LLM adapters must implement.
This ensures consistent behavior across different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass


@dataclass
class LLMMessage:
    """Standard message format for LLM interactions"""
    role: str  # 'system', 'user', 'assistant'
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Standard response format from LLM"""
    content: str
    usage: Optional[Dict[str, int]] = None  # tokens used
    metadata: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


@dataclass
class LLMConfig:
    """Configuration for LLM adapters"""
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    additional_params: Optional[Dict[str, Any]] = None


class LLMInterface(ABC):
    """Abstract interface for all LLM adapters"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM
        
        Args:
            messages: List of messages in the conversation
            **kwargs: Additional parameters specific to the adapter
            
        Returns:
            LLMResponse object with the generated content
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM
        
        Args:
            messages: List of messages in the conversation
            **kwargs: Additional parameters specific to the adapter
            
        Yields:
            Chunks of generated text
        """
        pass
    
    @abstractmethod
    async def get_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        Get embeddings for the provided texts
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional parameters specific to the adapter
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate the LLM configuration
        
        Returns:
            True if configuration is valid, False otherwise
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the LLM service is healthy and accessible
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.config.model_name,
            'provider': self.__class__.__name__,
            'config': self.config.__dict__
        }
