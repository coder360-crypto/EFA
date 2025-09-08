"""
Corrected MCP LLM Client with proper SSE handling
Handles both JSON and Server-Sent Events responses from MCP servers
"""

import os
import json
import asyncio
import httpx
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
import uuid
import re

# Import LLM clients
from openrouter_client import OpenRouterClient
from groq_llm_client import GroqClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPServerConfig:
    """MCP Server configuration"""
    name: str
    url: str
    timeout: int = 30
    session_id: Optional[str] = None

@dataclass
class MCPTool:
    """MCP Tool representation"""
    name: str
    description: str
    parameters: Dict[str, Any]
    server_name: str

class MCPClient:
    """MCP Client with SSE and JSON response handling"""
    
    def __init__(self, global_session_id: str):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.available_tools: Dict[str, MCPTool] = {}
        self._client: Optional[httpx.AsyncClient] = None
        self.global_session_id = global_session_id
        self.initialized_servers = set()
        
        # Add your server - note the port should match your actual server
        self.add_server(MCPServerConfig(
            name="ai_agent_tools",
            url="http://localhost:8002"  # Changed from 8000 to match your server
        ))
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client
    
    async def close(self):
        """Close HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    def add_server(self, config: MCPServerConfig):
        """Add MCP server"""
        self.servers[config.name] = config
    
    def _parse_sse_response(self, text: str) -> Dict[str, Any]:
        """Parse Server-Sent Events response"""
        logger.debug(f"Parsing SSE response: {text[:200]}...")
        
        # Handle SSE format: "data: {json}\n\n"
        lines = text.strip().split('\n')
        json_data = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('data: '):
                json_str = line[6:]  # Remove "data: " prefix
                try:
                    json_data = json.loads(json_str)
                    break
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse JSON from line: {line}")
                    continue
        
        if json_data is None:
            # Fallback: try to find JSON in the entire response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    json_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        
        if json_data is None:
            raise ValueError(f"Could not extract JSON from SSE response: {text}")
        
        return json_data
    
    async def _send_mcp_request(self, server: MCPServerConfig, method: str, params: Dict = None, is_notification: bool = False) -> Dict[str, Any]:
        """Send MCP request with proper SSE/JSON response handling"""
        if params is None:
            params = {}
        
        # Build the message according to MCP protocol
        if is_notification:
            message = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params
            }
        else:
            message = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": method,
                "params": params
            }
        
        # Use the MCP endpoint
        url = f"{server.url}/mcp"
        
        # Base headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        # Add session header if server is initialized
        if server.name in self.initialized_servers and server.session_id:
            headers["Mcp-Session-Id"] = server.session_id
        
        client = await self._get_client()
        
        logger.debug(f"Sending MCP {method} to {server.name}")
        logger.debug(f"URL: {url}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Payload: {json.dumps(message, indent=2)}")
        
        try:
            response = await client.post(url, headers=headers, json=message)
            
            # Log response details
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            logger.debug(f"Response content-type: {response.headers.get('content-type', 'unknown')}")
            
            response.raise_for_status()
            
            # For notifications, we might not get a response body
            if is_notification:
                return {"success": True}
            
            # Get response text
            response_text = response.text
            logger.debug(f"Raw response: {response_text[:500]}...")
            
            # Determine response format and parse accordingly
            content_type = response.headers.get('content-type', '').lower()
            
            if 'text/event-stream' in content_type or response_text.strip().startswith('data:'):
                # Handle SSE response
                logger.debug("Parsing as SSE response")
                result = self._parse_sse_response(response_text)
            else:
                # Handle regular JSON response
                logger.debug("Parsing as JSON response")
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {response_text}")
                    raise ValueError(f"Invalid JSON response: {str(e)}")
            
            logger.debug(f"Parsed response from {server.name}: {json.dumps(result, indent=2)}")
            
            if "error" in result:
                raise Exception(f"MCP Error: {result['error']}")
            
            # Extract session ID from response headers or body
            session_id = None
            if method == "initialize":
                # Check headers first
                if "mcp-session-id" in response.headers:
                    session_id = response.headers["mcp-session-id"]
                # Check response body
                elif "result" in result and isinstance(result["result"], dict):
                    if "sessionId" in result["result"]:
                        session_id = result["result"]["sessionId"]
                    elif "session_id" in result["result"]:
                        session_id = result["result"]["session_id"]
                
                if session_id:
                    logger.info(f"Received session ID: {session_id}")
                    return result, session_id
            
            return result
            
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_body = e.response.text
                error_detail = f" - {error_body}"
            except:
                pass
            logger.error(f"HTTP error {e.response.status_code} from {server.name}{error_detail}")
            raise
        except Exception as e:
            logger.error(f"Error sending MCP request to {server.name}: {str(e)}")
            raise
    
    async def connect_to_server(self, server_name: str) -> bool:
        """Connect to MCP server following correct lifecycle"""
        if server_name not in self.servers:
            logger.error(f"Server {server_name} not found in configuration")
            return False
        
        server = self.servers[server_name]
        
        try:
            # Step 1: Initialize MCP session
            logger.info(f"Step 1: Initializing MCP session for {server_name}")
            
            init_params = {
                "protocolVersion": "2024-11-05",  # Using standard version
                "capabilities": {
                    "tools": {"listChanged": True},
                    "roots": {"listChanged": True}
                },
                "clientInfo": {
                    "name": "MCP LLM Client",
                    "version": "1.0.0"
                }
            }
            
            result = await self._send_mcp_request(server, "initialize", init_params)
            
            # Handle session ID extraction
            if isinstance(result, tuple):
                init_response, session_id = result
                server.session_id = session_id
                logger.info(f"Session initialized with ID from response: {server.session_id}")
            else:
                # Fallback to using global session ID
                server.session_id = self.global_session_id
                logger.info(f"Session initialized with global ID: {server.session_id}")
            
            # Step 2: Mark server as initialized BEFORE sending notification
            self.initialized_servers.add(server_name)
            
            # Step 3: Send initialized notification (critical step)
            logger.info(f"Step 2: Sending initialized notification to {server_name}")
            await self._send_mcp_request(server, "notifications/initialized", {}, is_notification=True)
            logger.info(f"Initialized notification sent to {server_name}")
            
            # Step 4: Load tools (now that session is fully established)
            logger.info(f"Step 3: Loading tools from {server_name}")
            await self._load_tools(server_name)
            
            logger.info(f"Successfully connected to {server_name}")
            return True
                
        except Exception as e:
            logger.error(f"Connection to {server_name} failed: {str(e)}")
            # Clean up on failure
            if server_name in self.initialized_servers:
                self.initialized_servers.remove(server_name)
            server.session_id = None
            return False
    
    async def _load_tools(self, server_name: str):
        """Load tools from server"""
        if server_name not in self.initialized_servers:
            raise Exception(f"Server {server_name} not initialized")
        
        try:
            result = await self._send_mcp_request(self.servers[server_name], "tools/list")
            
            if "result" not in result:
                logger.warning(f"No result field in tools/list response from {server_name}")
                return
            
            tools_data = result["result"]
            
            # Handle different response formats
            if isinstance(tools_data, dict) and "tools" in tools_data:
                tools_list = tools_data["tools"]
            elif isinstance(tools_data, list):
                tools_list = tools_data
            else:
                logger.warning(f"Unexpected tools data format from {server_name}: {tools_data}")
                return
            
            # Process tools
            tools_loaded = 0
            for tool in tools_list:
                try:
                    tool_key = f"{server_name}:{tool['name']}"
                    self.available_tools[tool_key] = MCPTool(
                        name=tool['name'],
                        description=tool.get('description', 'No description'),
                        parameters=tool.get('inputSchema', {}),
                        server_name=server_name
                    )
                    tools_loaded += 1
                    logger.debug(f"Loaded tool: {tool['name']}")
                    
                except KeyError as e:
                    logger.warning(f"Tool missing required field {e}: {tool}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing tool {tool}: {str(e)}")
                    continue
                
            logger.info(f"Successfully loaded {tools_loaded} tools from {server_name}")
                    
        except Exception as e:
            logger.error(f"Failed to load tools from {server_name}: {str(e)}")
            raise
    
    async def call_tool(self, tool_key: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool with SSE response handling"""
        if tool_key not in self.available_tools:
            raise ValueError(f"Tool not found: {tool_key}")
        
        tool = self.available_tools[tool_key]
        server = self.servers[tool.server_name]
        
        if tool.server_name not in self.initialized_servers:
            raise Exception(f"Server {tool.server_name} not initialized")
        
        params = {
            "name": tool.name,
            "arguments": arguments
        }
        
        logger.info(f"Calling tool {tool.name} with arguments: {arguments}")
        result = await self._send_mcp_request(server, "tools/call", params)
        
        if "result" in result:
            return result["result"]
        
        raise Exception(f"Invalid response from tool call: {result}")
    
    def get_tools_for_llm(self) -> List[Dict]:
        """Format tools for LLM"""
        llm_tools = []
        
        for tool_key, tool in self.available_tools.items():
            llm_tool = {
                "type": "function",
                "function": {
                    "name": tool_key.replace(":", "_"),
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            llm_tools.append(llm_tool)
        
        return llm_tools

class MCPLLMAgent:
    """MCP + LLM Agent with SSE support"""
    
    def __init__(self, 
                 provider: str = "openrouter", 
                 model: str = None):
        self.agent_id = str(uuid.uuid4())
        self.provider = provider.lower()
        self.llm = None
        self.mcp = MCPClient(self.agent_id)
        
        # Set default models based on provider
        if model is None:
            if self.provider == "groq":
                self.model = "llama-3.1-8b-instant"
            else:  # openrouter
                self.model = "nousresearch/hermes-3-llama-3.1-405b:free"
        else:
            self.model = model
            
        self.conversation_history = []
        self._llm_initialized = False
    
    async def _get_llm(self):
        """Get LLM client with proper initialization"""
        if self.llm is None:
            try:
                if self.provider == "groq":
                    self.llm = GroqClient()
                    logger.info(f"Initialized Groq client with model: {self.model}")
                else:  # openrouter
                    self.llm = OpenRouterClient()
                    logger.info(f"Initialized OpenRouter client with model: {self.model}")
                
                # Test the connection
                await self._test_llm_connection()
                self._llm_initialized = True
                logger.info(f"{self.provider.upper()} LLM client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize {self.provider.upper()} LLM client: {str(e)}")
                raise
        return self.llm
    
    async def _test_llm_connection(self):
        """Test LLM connection with a simple request"""
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            await self.llm.chat_completion(
                messages=test_messages,
                model=self.model,
                max_tokens=10
            )
        except Exception as e:
            logger.error(f"LLM connection test failed: {str(e)}")
            raise
    
    async def connect_to_servers(self, server_names: List[str]) -> Dict[str, bool]:
        """Connect to MCP servers with proper lifecycle"""
        results = {}
        logger.info(f"Connecting to MCP servers with agent ID: {self.agent_id[:8]}")
        
        for server_name in server_names:
            try:
                logger.info(f"Attempting to connect to {server_name}")
                success = await self.mcp.connect_to_server(server_name)
                results[server_name] = success
                if success:
                    logger.info(f"✅ Connected to {server_name}")
                else:
                    logger.error(f"❌ Failed to connect to {server_name}")
            except Exception as e:
                logger.error(f"❌ Error connecting to {server_name}: {str(e)}")
                results[server_name] = False
        
        return results
    
    def _get_system_prompt(self) -> str:
        """Get enhanced system prompt"""
        tools_info = ""
        if self.mcp.available_tools:
            tools_info = "\n\nAVAILABLE TOOLS:\n"
            for tool_key, tool in self.mcp.available_tools.items():
                params_info = ""
                if tool.parameters and "properties" in tool.parameters:
                    props = tool.parameters["properties"]
                    param_list = [f"{k}: {v.get('type', 'any')}" for k, v in props.items()]
                    params_info = f" (Parameters: {', '.join(param_list)})"
                
                tools_info += f"• {tool.name}: {tool.description}{params_info}\n"
        
        return f"""You are an AI assistant with access to MCP tools for file management and operations.

TOOL USAGE INSTRUCTIONS:
1. Use tools proactively when user requests involve file operations, directory listings, or system tasks
2. Always explain what you're doing and why
3. Format tool outputs in a user-friendly way
4. Handle tool errors gracefully

AGENT DETAILS:
- Agent ID: {self.agent_id[:8]}
- Session ID: {self.agent_id[:8]}

use tools/list tool for available tools

Be helpful and use tools when they would benefit the user's request."""

    async def chat(self, message: str, use_tools: bool = True) -> str:
        """Chat with LLM using MCP tools"""
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        messages.extend(self.conversation_history)
        
        # Get available tools
        tools = None
        if use_tools and self.mcp.available_tools:
            tools = self.mcp.get_tools_for_llm()
            logger.info(f"Available tools for LLM: {len(tools)}")
        
        try:
            # Initialize LLM if needed
            llm = await self._get_llm()
            
            # Make initial LLM request
            response = await llm.chat_completion(
                messages=messages,
                model=self.model,
                tools=tools,
                temperature=0.7
            )
            
            choice = response["choices"][0]
            message_resp = choice["message"]
            
            # Handle tool calls
            if message_resp.get("tool_calls"):
                logger.info(f"LLM requested {len(message_resp['tool_calls'])} tool calls")
                tool_results = []
                
                for tool_call in message_resp["tool_calls"]:
                    function = tool_call["function"]
                    function_name = function["name"]
                    tool_key = function_name.replace("_", ":", 1)
                    
                    logger.info(f"Calling tool: {tool_key}")
                    
                    try:
                        arguments = json.loads(function["arguments"])
                        result = await self.mcp.call_tool(tool_key, arguments)
                        
                        tool_results.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(result, indent=2)
                        })
                        
                        logger.info(f"Tool {tool_key} executed successfully")
                        
                    except Exception as e:
                        error_msg = f"Error calling {tool_key}: {str(e)}"
                        logger.error(error_msg)
                        
                        tool_results.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error: {str(e)}"
                        })
                
                # Add tool call and results to conversation
                self.conversation_history.append(message_resp)
                messages.extend([message_resp] + tool_results)
                
                # Get final response with tool results
                final_response = await llm.chat_completion(
                    messages=messages,
                    model=self.model,
                    temperature=0.7
                )
                
                response_text = final_response["choices"][0]["message"]["content"]
                
            else:
                response_text = message_resp["content"]
            
            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            return response_text
            
        except Exception as e:
            error_msg = f"Error in chat: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def list_tools(self):
        """List available tools"""
        if not self.mcp.available_tools:
            print("❌ No tools available")
            return
        
        print(f"\n🔧 AVAILABLE TOOLS ({len(self.mcp.available_tools)}) - Agent ID: {self.agent_id[:8]}")
        print("=" * 70)
        
        for tool_key, tool in self.mcp.available_tools.items():
            server_name = tool.server_name
            server = self.mcp.servers.get(server_name)
            session_id = server.session_id[:8] if server and server.session_id else "None"
            
            print(f"📍 {tool.name}")
            print(f"   Description: {tool.description}")
            print(f"   Server: {server_name} (Session: {session_id})")
            
            if tool.parameters and "properties" in tool.parameters:
                props = tool.parameters["properties"]
                if props:
                    print("   Parameters:")
                    for param_name, param_info in props.items():
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "No description")
                        required = param_name in tool.parameters.get("required", [])
                        req_mark = " *" if required else ""
                        print(f"     • {param_name}{req_mark}: {param_type} - {param_desc}")
            print()

    def get_provider_info(self):
        """Get provider information"""
        return {
            "provider": self.provider,
            "model": self.model,
            "agent_id": self.agent_id,
            "initialized": self._llm_initialized,
            "mcp_servers_connected": len(self.mcp.initialized_servers),
            "tools_available": len(self.mcp.available_tools)
        }
    
    async def close(self):
        """Close all connections"""
        logger.info(f"Closing agent {self.agent_id[:8]}")
        if self.llm:
            try:
                await self.llm.close()
                self.llm = None
                logger.info("LLM client closed")
            except Exception as e:
                logger.error(f"Error closing LLM client: {str(e)}")
        
        try:
            await self.mcp.close()
            logger.info("MCP client closed")
        except Exception as e:
            logger.error(f"Error closing MCP client: {str(e)}")

async def main():
    """Test the corrected MCP client with SSE support"""
    print("🚀 MCP LLM Client with SSE Support")
    print("Handles both JSON and Server-Sent Events responses")
    print("=" * 55)
    
    try:
        # Initialize agent
        agent = MCPLLMAgent(provider="groq", model="llama-3.3-70b-versatile")
        
        print(f"\n🚀 Starting Agent - ID: {agent.agent_id[:8]}")
        print(f"📡 Provider: {agent.provider.upper()}")
        print(f"🤖 Model: {agent.model}")
        print("\n📋 MCP Session Lifecycle:")
        print("   1. Initialize session (handle SSE/JSON)")
        print("   2. Send initialized notification")
        print("   3. Load tools")
        print("\nConnecting to MCP servers...")
        
        # Connect to servers
        results = await agent.connect_to_servers(["ai_agent_tools"])
        
        if any(results.values()):
            print("\n✅ MCP servers connected successfully!")
            agent.list_tools()
            
            print(f"\n🎯 Agent ready! Try asking about files or directories.")
            print("💡 Commands: 'tools', 'info', 'quit', 'clear'")
            
            while True:
                try:
                    user_input = input(f"\n🗣️  You: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit']:
                        break
                    
                    if user_input.lower() in ['tools', 'list tools']:
                        agent.list_tools()
                        continue
                        
                    if user_input.lower() in ['info', 'provider info']:
                        info = agent.get_provider_info()
                        print(f"\n📊 PROVIDER INFO:")
                        for key, value in info.items():
                            print(f"   {key.title()}: {value}")
                        continue
                    
                    if user_input.lower() == 'clear':
                        agent.conversation_history.clear()
                        print("🧹 Conversation history cleared")
                        continue
                    
                    if user_input:
                        print("🤖 Assistant: ", end="")
                        response = await agent.chat(user_input)
                        print(response)
                        
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Unexpected error: {str(e)}")
        
        else:
            print("❌ Failed to connect to MCP servers")
            print("Make sure your MCP server is running on http://localhost:8002")
            print("\n🔍 Debug steps:")
            print("1. Verify server is running: curl http://localhost:8002/health")
            print("2. Check MCP endpoint: curl http://localhost:8002/mcp")
            print("3. Check response format (JSON vs SSE)")
    
    except Exception as e:
        print(f"❌ Initialization error: {str(e)}")
        return
    
    finally:
        if 'agent' in locals():
            await agent.close()
        print("🔚 Agent shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())