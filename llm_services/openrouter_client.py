# fixed_openrouter_client.py
"""
Fixed OpenRouter LLM Client - No Streaming Support
"""

import os
import asyncio
import httpx
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Model information structure"""
    id: str
    name: str
    description: str
    context_length: int
    max_completion_tokens: int
    pricing: Dict[str, float]
    is_free: bool
    supports_tools: bool
    provider: str

class OpenRouterClient:
    """Fixed OpenRouter LLM Client - No Streaming"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1"):
        """Initialize OpenRouter client"""
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key parameter")
        
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-username/your-repo",
            "X-Title": "MCP OpenRouter Client"
        }
        
        # Initialize persistent client
        self._client: Optional[httpx.AsyncClient] = None
        
        # Comprehensive model catalog
        self.models = self._initialize_models()
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create persistent HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                follow_redirects=True
            )
        return self._client
    
    async def close(self):
        """Close HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def _initialize_models(self) -> Dict[str, ModelInfo]:
        """Initialize comprehensive model catalog"""
        return {
            # FREE MODELS
            "meta-llama/llama-3.2-3b-instruct:free": ModelInfo(
                id="meta-llama/llama-3.2-3b-instruct:free",
                name="Llama 3.2 3B Instruct (Free)",
                description="Meta's Llama 3.2 3B instruction-tuned model - free tier",
                context_length=131072,
                max_completion_tokens=4096,
                pricing={"prompt": 0.0, "completion": 0.0},
                is_free=True,
                supports_tools=True,
                provider="Meta"
            ),
            "meta-llama/llama-3.2-1b-instruct:free": ModelInfo(
                id="meta-llama/llama-3.2-1b-instruct:free",
                name="Llama 3.2 1B Instruct (Free)",
                description="Meta's Llama 3.2 1B instruction-tuned model - free tier",
                context_length=131072,
                max_completion_tokens=4096,
                pricing={"prompt": 0.0, "completion": 0.0},
                is_free=True,
                supports_tools=True,
                provider="Meta"
            ),
            "mistralai/mistral-7b-instruct:free": ModelInfo(
                id="mistralai/mistral-7b-instruct:free",
                name="Mistral 7B Instruct (Free)",
                description="Mistral's 7B instruction-tuned model - free tier",
                context_length=32768,
                max_completion_tokens=4096,
                pricing={"prompt": 0.0, "completion": 0.0},
                is_free=True,
                supports_tools=False,
                provider="Mistral AI"
            ),
            "nousresearch/hermes-3-llama-3.1-405b:free": ModelInfo(
                id="nousresearch/hermes-3-llama-3.1-405b:free",
                name="Hermes 3 Llama 3.1 405B (Free)",
                description="Nous Research's Hermes 3 based on Llama 3.1 405B - free tier",
                context_length=131072,
                max_completion_tokens=4096,
                pricing={"prompt": 0.0, "completion": 0.0},
                is_free=True,
                supports_tools=True,
                provider="Nous Research"
            ),
            "huggingfaceh4/zephyr-7b-beta:free": ModelInfo(
                id="huggingfaceh4/zephyr-7b-beta:free",
                name="Zephyr 7B Beta (Free)",
                description="Hugging Face's Zephyr 7B beta model - free tier",
                context_length=32768,
                max_completion_tokens=4096,
                pricing={"prompt": 0.0, "completion": 0.0},
                is_free=True,
                supports_tools=False,
                provider="Hugging Face"
            ),
            
            # PREMIUM MODELS
            "openai/gpt-4o": ModelInfo(
                id="openai/gpt-4o",
                name="GPT-4o",
                description="OpenAI's GPT-4o with vision and tool calling",
                context_length=128000,
                max_completion_tokens=4096,
                pricing={"prompt": 0.005, "completion": 0.015},
                is_free=False,
                supports_tools=True,
                provider="OpenAI"
            ),
            "openai/gpt-4o-mini": ModelInfo(
                id="openai/gpt-4o-mini",
                name="GPT-4o Mini",
                description="OpenAI's smaller, faster GPT-4o model",
                context_length=128000,
                max_completion_tokens=16384,
                pricing={"prompt": 0.00015, "completion": 0.0006},
                is_free=False,
                supports_tools=True,
                provider="OpenAI"
            ),
            "anthropic/claude-3.5-sonnet": ModelInfo(
                id="anthropic/claude-3.5-sonnet",
                name="Claude 3.5 Sonnet",
                description="Anthropic's most capable model with advanced reasoning",
                context_length=200000,
                max_completion_tokens=4096,
                pricing={"prompt": 0.003, "completion": 0.015},
                is_free=False,
                supports_tools=True,
                provider="Anthropic"
            ),
            "anthropic/claude-3-haiku": ModelInfo(
                id="anthropic/claude-3-haiku",
                name="Claude 3 Haiku",
                description="Anthropic's fastest model for simple tasks",
                context_length=200000,
                max_completion_tokens=4096,
                pricing={"prompt": 0.00025, "completion": 0.00125},
                is_free=False,
                supports_tools=True,
                provider="Anthropic"
            ),
            "google/gemini-pro-1.5": ModelInfo(
                id="google/gemini-pro-1.5",
                name="Gemini Pro 1.5",
                description="Google's latest Gemini Pro model with large context",
                context_length=1000000,
                max_completion_tokens=8192,
                pricing={"prompt": 0.00125, "completion": 0.005},
                is_free=False,
                supports_tools=True,
                provider="Google"
            )
        }
    
    def get_free_models(self) -> List[ModelInfo]:
        """Get all free models"""
        return [model for model in self.models.values() if model.is_free]
    
    def get_models_with_tools(self) -> List[ModelInfo]:
        """Get all models that support tool calling"""
        return [model for model in self.models.values() if model.supports_tools]
    
    def get_model_by_provider(self, provider: str) -> List[ModelInfo]:
        """Get models by provider"""
        return [model for model in self.models.values() if model.provider.lower() == provider.lower()]
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "meta-llama/llama-3.2-3b-instruct:free",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None
    ) -> Dict:
        """Create chat completion with persistent client"""
        
        # Validate model
        if model not in self.models:
            logger.warning(f"Unknown model: {model}. Using default free model.")
            model = "meta-llama/llama-3.2-3b-instruct:free"
        
        model_info = self.models[model]
        
        # Check tool support
        if tools and not model_info.supports_tools:
            logger.warning(f"Model {model} doesn't support tools. Removing tools from request.")
            tools = None
            tool_choice = None
        
        # Prepare request
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        
        if max_tokens:
            payload["max_tokens"] = min(max_tokens, model_info.max_completion_tokens)
        
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        
        try:
            client = await self._get_client()
            
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            # Try to recreate client on error
            if self._client:
                try:
                    await self._client.aclose()
                except:
                    pass
                self._client = None
            raise
    
    async def simple_chat(
        self, 
        prompt: str, 
        model: str = "meta-llama/llama-3.2-3b-instruct:free",
        system_message: Optional[str] = None
    ) -> str:
        """Simple chat interface with error handling"""
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.chat_completion(messages, model=model)
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Simple chat error: {str(e)}")
            return f"Error: {str(e)}"
    
    def list_models(self, free_only: bool = False, with_tools: bool = False) -> None:
        """Print available models"""
        models_to_show = list(self.models.values())
        
        if free_only:
            models_to_show = [m for m in models_to_show if m.is_free]
        
        if with_tools:
            models_to_show = [m for m in models_to_show if m.supports_tools]
        
        print(f"\n{'='*80}")
        print(f"Available Models ({'Free Only' if free_only else 'All'}) ({'With Tools' if with_tools else ''})")
        print(f"{'='*80}")
        
        for model in models_to_show:
            status = "🆓 FREE" if model.is_free else f"💰 ${model.pricing['prompt']:.4f}/$1K prompt"
            tools_support = "🔧 Tools" if model.supports_tools else "❌ No Tools"
            
            print(f"\n📱 {model.name}")
            print(f"   ID: {model.id}")
            print(f"   Provider: {model.provider}")
            print(f"   Status: {status}")
            print(f"   Features: {tools_support}")
            print(f"   Context: {model.context_length:,} tokens")
            print(f"   Description: {model.description}")

# Usage examples
async def test_client():
    """Test the client"""
    print("🔧 Testing OpenRouter Client...")
    
    async with OpenRouterClient() as client:  # Use context manager
        try:
            # Test simple chat
            response = await client.simple_chat(
                "What is 2+2? Give a brief answer.",
                model="meta-llama/llama-3.2-3b-instruct:free"
            )
            print(f"✅ Llama 3.2 1B Response: {response}")
            
            # Test another model
            response2 = await client.simple_chat(
                "Hello! How are you?",
                model="meta-llama/llama-3.2-3b-instruct:free"
            )
            print(f"✅ Llama 3.2 3B Response: {response2}")
            
            # Test with tools
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            }
                        }
                    }
                }
            ]
            
            response3 = await client.chat_completion(
                messages=[{"role": "user", "content": "What's the weather like in New York?"}],
                model="deepseek/deepseek-chat-v3-0324:free",
                tools=tools
            )
            print(f"✅ Tools Response: {response3}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Test the client
    async def main():
        await test_client()
    
    asyncio.run(main())