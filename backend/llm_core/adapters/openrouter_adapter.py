"""
OpenRouter LLM Adapter

Implementation of LLMInterface for OpenRouter API.
OpenRouter provides unified access to multiple LLM providers with automatic routing,
rate limiting, and credit management.
"""

from typing import List, Optional, AsyncGenerator, Dict, Any
import aiohttp
import asyncio
import logging
import json
from datetime import datetime

from ..llm_interface import LLMInterface, LLMMessage, LLMResponse, LLMConfig


class OpenRouterAdapter(LLMInterface):
    """OpenRouter implementation of LLMInterface"""
    
    def __init__(self, config: LLMConfig, api_key: Optional[str] = None, site_url: Optional[str] = None, app_name: Optional[str] = None):
        super().__init__(config)
        self.api_key = api_key
        self.site_url = site_url or "https://github.com/yourusername/efa"
        self.app_name = app_name or "EFA Backend"
        self.base_url = "https://openrouter.ai/api/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
        # OpenRouter-specific settings
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
            "Content-Type": "application/json"
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
        return self.session
    
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate response using OpenRouter API"""
        try:
            session = await self._get_session()
            
            # Convert messages to OpenRouter format
            openrouter_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Prepare request parameters
            request_params = {
                "model": self.config.model_name,
                "messages": openrouter_messages,
                "temperature": self.config.temperature,
            }
            
            # Add optional parameters
            if self.config.max_tokens:
                request_params["max_tokens"] = self.config.max_tokens
            if self.config.top_p:
                request_params["top_p"] = self.config.top_p
            if self.config.frequency_penalty:
                request_params["frequency_penalty"] = self.config.frequency_penalty
            if self.config.presence_penalty:
                request_params["presence_penalty"] = self.config.presence_penalty
            if self.config.stop_sequences:
                request_params["stop"] = self.config.stop_sequences
            
            # Add OpenRouter-specific parameters
            if self.config.additional_params:
                # OpenRouter supports provider-specific routing
                if "provider" in self.config.additional_params:
                    request_params["provider"] = self.config.additional_params["provider"]
                if "route" in self.config.additional_params:
                    request_params["route"] = self.config.additional_params["route"]
                # Add other additional parameters
                for key, value in self.config.additional_params.items():
                    if key not in ["provider", "route"]:
                        request_params[key] = value
            
            # Override with any kwargs
            request_params.update(kwargs)
            
            # Make API call
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=request_params
            ) as response:
                if response.status == 402:
                    error_data = await response.json()
                    raise Exception(f"Insufficient credits: {error_data.get('error', {}).get('message', 'Payment required')}")
                elif response.status == 429:
                    error_data = await response.json()
                    raise Exception(f"Rate limit exceeded: {error_data.get('error', {}).get('message', 'Too many requests')}")
                elif response.status != 200:
                    error_data = await response.json()
                    raise Exception(f"OpenRouter API error {response.status}: {error_data.get('error', {}).get('message', 'Unknown error')}")
                
                response_data = await response.json()
            
            # Extract response data
            choice = response_data["choices"][0]
            content = choice["message"]["content"] or ""
            
            # Extract usage information
            usage = response_data.get("usage", {})
            
            return LLMResponse(
                content=content,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                },
                finish_reason=choice.get("finish_reason"),
                metadata={
                    "model": response_data.get("model", self.config.model_name),
                    "id": response_data.get("id"),
                    "created": response_data.get("created"),
                    "provider": response_data.get("provider"),
                    "native_tokens": usage.get("native_tokens"),
                    "cost": response_data.get("cost")
                }
            )
            
        except Exception as e:
            self.logger.error(f"OpenRouter generation failed: {e}")
            raise
    
    async def generate_stream(self, messages: List[LLMMessage], **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response using OpenRouter API"""
        try:
            session = await self._get_session()
            
            # Convert messages to OpenRouter format
            openrouter_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Prepare request parameters
            request_params = {
                "model": self.config.model_name,
                "messages": openrouter_messages,
                "temperature": self.config.temperature,
                "stream": True
            }
            
            # Add optional parameters
            if self.config.max_tokens:
                request_params["max_tokens"] = self.config.max_tokens
            if self.config.top_p:
                request_params["top_p"] = self.config.top_p
            if self.config.frequency_penalty:
                request_params["frequency_penalty"] = self.config.frequency_penalty
            if self.config.presence_penalty:
                request_params["presence_penalty"] = self.config.presence_penalty
            if self.config.stop_sequences:
                request_params["stop"] = self.config.stop_sequences
            
            # Add OpenRouter-specific parameters
            if self.config.additional_params:
                if "provider" in self.config.additional_params:
                    request_params["provider"] = self.config.additional_params["provider"]
                if "route" in self.config.additional_params:
                    request_params["route"] = self.config.additional_params["route"]
                for key, value in self.config.additional_params.items():
                    if key not in ["provider", "route"]:
                        request_params[key] = value
            
            # Override with any kwargs
            request_params.update(kwargs)
            
            # Make streaming API call
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=request_params
            ) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise Exception(f"OpenRouter streaming failed {response.status}: {error_data.get('error', {}).get('message', 'Unknown error')}")
                
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        if data_str == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            if "choices" in chunk_data and chunk_data["choices"]:
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
                    
        except Exception as e:
            self.logger.error(f"OpenRouter streaming failed: {e}")
            raise
    
    async def get_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Get embeddings - OpenRouter doesn't directly provide embeddings API
        You would need to use a specific model that supports embeddings
        """
        # OpenRouter doesn't have a dedicated embeddings endpoint like OpenAI
        # You would need to use models that support embeddings through the chat interface
        # or use a dedicated embedding model through the completions endpoint
        raise NotImplementedError(
            "OpenRouter doesn't provide a dedicated embeddings API. "
            "Use a model that supports embeddings through the chat interface or "
            "use a dedicated embedding service."
        )
    
    def validate_config(self) -> bool:
        """Validate OpenRouter configuration"""
        try:
            if not self.api_key:
                self.logger.error("OpenRouter API key is required")
                return False
            
            # Check parameter ranges (similar to OpenAI)
            if not (0.0 <= self.config.temperature <= 2.0):
                self.logger.error("Temperature must be between 0.0 and 2.0")
                return False
            
            if self.config.top_p and not (0.0 <= self.config.top_p <= 1.0):
                self.logger.error("Top-p must be between 0.0 and 1.0")
                return False
            
            if self.config.frequency_penalty and not (-2.0 <= self.config.frequency_penalty <= 2.0):
                self.logger.error("Frequency penalty must be between -2.0 and 2.0")
                return False
            
            if self.config.presence_penalty and not (-2.0 <= self.config.presence_penalty <= 2.0):
                self.logger.error("Presence penalty must be between -2.0 and 2.0")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Config validation failed: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check OpenRouter API health and get account info"""
        try:
            session = await self._get_session()
            
            # Check API key status and credits
            async with session.get(f"{self.base_url}/key") as response:
                if response.status == 401:
                    self.logger.error("Invalid OpenRouter API key")
                    return False
                elif response.status != 200:
                    self.logger.error(f"OpenRouter API health check failed: {response.status}")
                    return False
                
                key_info = await response.json()
                data = key_info.get("data", {})
                
                # Log account information
                usage = data.get("usage", 0)
                limit = data.get("limit")
                is_free_tier = data.get("is_free_tier", False)
                
                self.logger.info(f"OpenRouter account status:")
                self.logger.info(f"  Credits used: {usage}")
                self.logger.info(f"  Credit limit: {limit if limit is not None else 'Unlimited'}")
                self.logger.info(f"  Free tier: {is_free_tier}")
                
                # Check if account has negative balance
                if limit is not None and usage >= limit:
                    self.logger.warning("OpenRouter credit limit reached")
                    return False
                
                return True
                
        except Exception as e:
            self.logger.error(f"OpenRouter health check failed: {e}")
            return False
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter"""
        try:
            session = await self._get_session()
            
            async with session.get(f"{self.base_url}/models") as response:
                if response.status != 200:
                    raise Exception(f"Failed to get models: HTTP {response.status}")
                
                models_data = await response.json()
                return models_data.get("data", [])
                
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return []
    
    async def get_providers(self) -> List[Dict[str, Any]]:
        """Get list of available providers from OpenRouter"""
        try:
            session = await self._get_session()
            
            async with session.get(f"{self.base_url}/providers") as response:
                if response.status != 200:
                    raise Exception(f"Failed to get providers: HTTP {response.status}")
                
                providers_data = await response.json()
                return providers_data.get("data", [])
                
        except Exception as e:
            self.logger.error(f"Failed to get available providers: {e}")
            return []
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get detailed account information including credits and limits"""
        try:
            session = await self._get_session()
            
            async with session.get(f"{self.base_url}/key") as response:
                if response.status != 200:
                    raise Exception(f"Failed to get account info: HTTP {response.status}")
                
                account_data = await response.json()
                return account_data.get("data", {})
                
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            return {}
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model_name': self.config.model_name,
            'provider': 'OpenRouter',
            'config': self.config.__dict__,
            'api_base': self.base_url,
            'site_url': self.site_url,
            'app_name': self.app_name
        }
