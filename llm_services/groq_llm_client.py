
# groq_llm_router.py
"""
Fixed Groq LLM Router - Ultra-fast inference with proper error handling
"""

import os
import asyncio
import httpx
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GroqModelInfo:
    """Groq model information structure"""
    id: str
    name: str
    description: str
    context_length: int
    max_completion_tokens: int
    developer: str
    supports_tools: bool
    supports_vision: bool = False
    speed_tier: str = "ultra_fast"  # All Groq models are ultra-fast

class GroqClient:
    """Fixed Groq LLM Router - Ultra-fast inference client"""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.groq.com/openai/v1"):
        """Initialize Groq client"""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY env var or pass api_key parameter")

        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Initialize persistent client
        self._client: Optional[httpx.AsyncClient] = None

        # Verified Groq model catalog (based on current API documentation)
        self.models = self._initialize_models()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create persistent HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
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

    def _initialize_models(self) -> Dict[str, GroqModelInfo]:
        """Initialize verified Groq model catalog (updated Sept 2025)"""
        return {
            # PRODUCTION MODELS (Verified from Groq docs)
            "llama-3.3-70b-versatile": GroqModelInfo(
                id="llama-3.3-70b-versatile",
                name="Llama 3.3 70B Versatile",
                description="Meta's latest Llama 3.3 70B model - production ready",
                context_length=131072,
                max_completion_tokens=32768,
                developer="Meta",
                supports_tools=True,
                supports_vision=False
            ),
            "llama-3.1-8b-instant": GroqModelInfo(
                id="llama-3.1-8b-instant",
                name="Llama 3.1 8B Instant",
                description="Fast and efficient Llama 3.1 8B model",
                context_length=131072,
                max_completion_tokens=8192,
                developer="Meta",
                supports_tools=True,
                supports_vision=False
            ),
            "llama-3.1-70b-versatile": GroqModelInfo(
                id="llama-3.1-70b-versatile",
                name="Llama 3.1 70B Versatile",
                description="High-performance Llama 3.1 70B model",
                context_length=131072,
                max_completion_tokens=8192,
                developer="Meta",
                supports_tools=True,
                supports_vision=False
            ),
            "llama3-70b-8192": GroqModelInfo(
                id="llama3-70b-8192",
                name="Llama 3 70B",
                description="Meta's Llama 3 70B model",
                context_length=8192,
                max_completion_tokens=8192,
                developer="Meta",
                supports_tools=False,
                supports_vision=False
            ),
            "llama3-8b-8192": GroqModelInfo(
                id="llama3-8b-8192",
                name="Llama 3 8B",
                description="Meta's Llama 3 8B model",
                context_length=8192,
                max_completion_tokens=8192,
                developer="Meta",
                supports_tools=False,
                supports_vision=False
            ),
            "mixtral-8x7b-32768": GroqModelInfo(
                id="mixtral-8x7b-32768",
                name="Mixtral 8x7B",
                description="Mistral AI's efficient mixture-of-experts model",
                context_length=32768,
                max_completion_tokens=32768,
                developer="Mistral AI",
                supports_tools=True,
                supports_vision=False
            ),
            "gemma2-9b-it": GroqModelInfo(
                id="gemma2-9b-it",
                name="Gemma 2 9B Instruct",
                description="Google's latest Gemma 2 model with improved performance",
                context_length=8192,
                max_completion_tokens=8192,
                developer="Google",
                supports_tools=True,
                supports_vision=False
            ),
            "llama-guard-3-8b": GroqModelInfo(
                id="llama-guard-3-8b",
                name="Llama Guard 3 8B",
                description="Meta's safety model for content moderation",
                context_length=8192,
                max_completion_tokens=8192,
                developer="Meta",
                supports_tools=False,
                supports_vision=False
            ),

            # TOOL-OPTIMIZED MODELS
            "llama3-groq-70b-8192-tool-use-preview": GroqModelInfo(
                id="llama3-groq-70b-8192-tool-use-preview",
                name="Llama 3 Groq 70B Tool Use",
                description="Llama 3 70B optimized for tool usage on Groq",
                context_length=8192,
                max_completion_tokens=8192,
                developer="Meta/Groq",
                supports_tools=True,
                supports_vision=False
            ),
            "llama3-groq-8b-8192-tool-use-preview": GroqModelInfo(
                id="llama3-groq-8b-8192-tool-use-preview",
                name="Llama 3 Groq 8B Tool Use",
                description="Llama 3 8B optimized for tool usage on Groq",
                context_length=8192,
                max_completion_tokens=8192,
                developer="Meta/Groq",
                supports_tools=True,
                supports_vision=False
            ),

            # PREVIEW MODELS (may change)
            "llama-3.2-1b-preview": GroqModelInfo(
                id="llama-3.2-1b-preview",
                name="Llama 3.2 1B (Preview)",
                description="Ultra-lightweight Llama 3.2 model",
                context_length=131072,
                max_completion_tokens=8192,
                developer="Meta",
                supports_tools=True,
                supports_vision=False
            ),
            "llama-3.2-3b-preview": GroqModelInfo(
                id="llama-3.2-3b-preview",
                name="Llama 3.2 3B (Preview)",
                description="Compact Llama 3.2 model for efficient inference",
                context_length=131072,
                max_completion_tokens=8192,
                developer="Meta",
                supports_tools=True,
                supports_vision=False
            ),
            "llama-3.2-11b-vision-preview": GroqModelInfo(
                id="llama-3.2-11b-vision-preview",
                name="Llama 3.2 11B Vision (Preview)",
                description="Llama 3.2 with vision capabilities for multimodal tasks",
                context_length=131072,
                max_completion_tokens=8192,
                developer="Meta",
                supports_tools=True,
                supports_vision=True
            ),
            "llama-3.2-90b-vision-preview": GroqModelInfo(
                id="llama-3.2-90b-vision-preview",
                name="Llama 3.2 90B Vision (Preview)",
                description="Large Llama 3.2 model with advanced vision understanding",
                context_length=131072,
                max_completion_tokens=8192,
                developer="Meta",
                supports_tools=True,
                supports_vision=True
            ),

            # REASONING MODELS
            "deepseek-r1-distill-llama-70b": GroqModelInfo(
                id="deepseek-r1-distill-llama-70b",
                name="DeepSeek R1 Distill Llama 70B",
                description="DeepSeek's reasoning model distilled on Llama 70B",
                context_length=131072,
                max_completion_tokens=8192,
                developer="DeepSeek",
                supports_tools=True,
                supports_vision=False
            )
        }

    async def get_available_models(self) -> Dict[str, Any]:
        """Fetch current models from Groq API"""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.base_url}/models",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Could not fetch models from API: {str(e)}")
            return {"data": []}

    def get_models_by_developer(self, developer: str) -> List[GroqModelInfo]:
        """Get models by developer"""
        return [model for model in self.models.values() 
                if model.developer.lower() == developer.lower()]

    def get_models_with_tools(self) -> List[GroqModelInfo]:
        """Get all models that support tool calling"""
        return [model for model in self.models.values() if model.supports_tools]

    def get_models_with_vision(self) -> List[GroqModelInfo]:
        """Get all models that support vision"""
        return [model for model in self.models.values() if model.supports_vision]

    def get_production_models(self) -> List[GroqModelInfo]:
        """Get production-ready models (non-preview)"""
        return [model for model in self.models.values() 
                if "preview" not in model.id.lower()]

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False
    ) -> Dict:
        """Create chat completion with Groq API - with better error handling"""

        # Validate model
        if model not in self.models:
            logger.warning(f"Unknown model: {model}. Using default model.")
            model = "llama-3.1-8b-instant"

        model_info = self.models[model]

        # Check tool support
        if tools and not model_info.supports_tools:
            logger.warning(f"Model {model} doesn't support tools. Removing tools from request.")
            tools = None
            tool_choice = None

        # Prepare request with proper parameter names
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }

        # Use max_completion_tokens instead of max_tokens for newer models
        if max_tokens:
            payload["max_completion_tokens"] = min(max_tokens, model_info.max_completion_tokens)

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

            # Enhanced error handling
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Groq API error {response.status_code}: {error_text}")

                # Try to parse error for more details
                try:
                    error_json = response.json()
                    error_message = error_json.get("error", {}).get("message", error_text)
                    raise Exception(f"Groq API error: {error_message}")
                except json.JSONDecodeError:
                    raise Exception(f"Groq API error {response.status_code}: {error_text}")

            return response.json()

        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
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
        model: str = "llama-3.1-8b-instant",
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

    def list_models(self, filter_by: Optional[str] = None, production_only: bool = False) -> None:
        """Print available models with optional filtering"""
        models_to_show = list(self.models.values())

        if production_only:
            models_to_show = self.get_production_models()

        if filter_by:
            filter_lower = filter_by.lower()
            models_to_show = [m for m in models_to_show 
                            if filter_lower in m.name.lower() or 
                            filter_lower in m.developer.lower() or
                            filter_lower in m.id.lower()]

        print(f"\n{'='*80}")
        title = f"Groq Models"
        if production_only:
            title += " (Production Only)"
        if filter_by:
            title += f" (Filtered: {filter_by})"
        print(title)
        print(f"{'='*80}")

        # Group by developer
        by_developer = {}
        for model in models_to_show:
            if model.developer not in by_developer:
                by_developer[model.developer] = []
            by_developer[model.developer].append(model)

        for developer, models in by_developer.items():
            print(f"\n🏢 {developer} Models:")
            print("-" * 50)

            for model in models:
                tools_support = "🔧 Tools" if model.supports_tools else "❌ No Tools"
                vision_support = "👁️ Vision" if model.supports_vision else ""
                preview_flag = "🚧 Preview" if "preview" in model.id.lower() else "✅ Production"
                features = f"{tools_support} {vision_support} {preview_flag}".strip()

                print(f"\n⚡ {model.name}")
                print(f"   ID: {model.id}")
                print(f"   Features: {features}")
                print(f"   Context: {model.context_length:,} tokens")
                print(f"   Max Output: {model.max_completion_tokens:,} tokens")
                print(f"   Description: {model.description}")

    def get_recommended_model(self, task_type: str = "general") -> str:
        """Get recommended model for specific task types"""
        recommendations = {
            "general": "llama-3.1-8b-instant",
            "reasoning": "deepseek-r1-distill-llama-70b",
            "coding": "llama-3.1-70b-versatile", 
            "fast": "llama-3.1-8b-instant",
            "tools": "llama3-groq-70b-8192-tool-use-preview",
            "vision": "llama-3.2-11b-vision-preview",
            "lightweight": "llama-3.2-1b-preview",
            "balanced": "llama-3.3-70b-versatile",
            "production": "llama-3.1-70b-versatile"
        }

        return recommendations.get(task_type.lower(), "llama-3.1-8b-instant")


# Usage examples and testing
async def test_groq_client():
    """Test the Groq client with better error handling"""
    print("🚀 Testing Fixed Groq Client...")

    async with GroqClient() as client:
        try:
            # Test simple chat with production model
            print("\n1. Testing simple chat...")
            response = await client.simple_chat(
                "What is 2+2? Give a brief answer.",
                model="llama3-groq-70b-8192-tool-use-preview"
            )
            print(f"✅ Llama 3.1 8B Response: {response}")

            # Test with different model
            print("\n2. Testing another model...")
            response2 = await client.simple_chat(
                "Hello! How are you?",
                model="llama-3.3-70b-versatile",
                system_message="You are a helpful assistant that responds concisely."
            )
            print(f"✅ Llama 3.3 70B Response: {response2}")

            # Test with tools using a model that supports them
            print("\n3. Testing tools...")
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state/country"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]

            response3 = await client.chat_completion(
                messages=[{"role": "user", "content": "What's the weather like in New York?"}],
                model="deepseek-r1-distill-llama-70b",
                tools=tools
            )
            print(f"✅ Tools Response: {response3.get('choices', [{}])[0].get('message', {})}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")


async def demo_groq_models():
    """Demo Groq model capabilities"""
    print("🔧 Fixed Groq Model Demo")

    client = GroqClient()

    # Show production models only
    client.list_models(production_only=True)

    print(f"\n🛠️ Models with Tool Support: {len(client.get_models_with_tools())}")
    print(f"👁️ Models with Vision Support: {len(client.get_models_with_vision())}")
    print(f"✅ Production Models: {len(client.get_production_models())}")

    print("\n💡 Recommended Models by Task:")
    tasks = ["general", "reasoning", "coding", "fast", "tools", "vision", "lightweight", "production"]
    for task in tasks:
        recommended = client.get_recommended_model(task)
        print(f"   {task.title()}: {recommended}")

    # Try to fetch current models from API
    print("\n🔍 Fetching current models from API...")
    try:
        api_models = await client.get_available_models()
        model_count = len(api_models.get("data", []))
        print(f"Found {model_count} models available via API")
    except Exception as e:
        print(f"Could not fetch API models: {str(e)}")


if __name__ == "__main__":
    async def main():
        await demo_groq_models()
        print("\n" + "="*50)
        await test_groq_client()

    asyncio.run(main())
