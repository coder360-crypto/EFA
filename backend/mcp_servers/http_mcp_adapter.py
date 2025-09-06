"""
HTTP MCP Adapter

Implementation of MCP adapter that communicates with remote MCP servers over HTTP.
Supports JSON-RPC protocol for MCP communication.
"""

from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
import json
import logging
from datetime import datetime

from .base_adapter import MCPAdapter, MCPTool, MCPResponse, MCPServerInfo, MCPToolType, MCPRequest


class HTTPMCPAdapter(MCPAdapter):
    """HTTP implementation of MCP adapter"""
    
    def __init__(
        self,
        server_name: str,
        base_url: str,
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(server_name)
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.headers = headers or {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
        # Default headers for MCP communication
        self.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    async def connect(self, **kwargs) -> bool:
        """
        Connect to the remote MCP server
        
        Args:
            **kwargs: Additional connection parameters
                - api_key: API key for authentication
                - auth_token: Authentication token
                - custom_headers: Additional headers
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Update headers with authentication if provided
            if 'api_key' in kwargs:
                self.headers['X-API-Key'] = kwargs['api_key']
            if 'auth_token' in kwargs:
                self.headers['Authorization'] = f"Bearer {kwargs['auth_token']}"
            if 'custom_headers' in kwargs:
                self.headers.update(kwargs['custom_headers'])
            
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
            
            # Test connection with server info request
            server_info = await self.get_server_info()
            if server_info:
                self.connected = True
                self.server_info = server_info
                self.logger.info(f"Connected to HTTP MCP server: {self.server_name}")
                return True
            else:
                await self.disconnect()
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to HTTP MCP server: {e}")
            await self.disconnect()
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from the remote MCP server
        
        Returns:
            True if disconnection successful
        """
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.connected = False
            self.server_info = None
            self.logger.info(f"Disconnected from HTTP MCP server: {self.server_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during disconnection: {e}")
            return False
    
    async def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        **kwargs
    ) -> MCPResponse:
        """
        Call a tool on the remote MCP server
        
        Args:
            tool_name: Name of the tool to call
            parameters: Parameters for the tool
            **kwargs: Additional call parameters
        
        Returns:
            MCP response with result or error
        """
        if not self.session:
            return self.create_response(
                request_id=self.generate_request_id(),
                error={
                    "code": "NOT_CONNECTED",
                    "message": "Not connected to MCP server"
                }
            )
        
        request = self.create_request(
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": parameters
            }
        )
        
        try:
            # Send JSON-RPC request
            rpc_request = {
                "jsonrpc": "2.0",
                "id": request.id,
                "method": request.method,
                "params": request.params
            }
            
            async with self.session.post(
                f"{self.base_url}/rpc",
                json=rpc_request
            ) as response:
                
                if response.status != 200:
                    return self.create_response(
                        request_id=request.id,
                        error={
                            "code": "HTTP_ERROR",
                            "message": f"HTTP {response.status}: {response.reason}"
                        }
                    )
                
                response_data = await response.json()
                
                # Handle JSON-RPC response
                if "error" in response_data:
                    return self.create_response(
                        request_id=request.id,
                        error=response_data["error"]
                    )
                else:
                    return self.create_response(
                        request_id=request.id,
                        result=response_data.get("result"),
                        metadata={
                            "tool_name": tool_name,
                            "executed_at": datetime.now().isoformat(),
                            "server": self.server_name
                        }
                    )
                    
        except asyncio.TimeoutError:
            return self.create_response(
                request_id=request.id,
                error={
                    "code": "TIMEOUT",
                    "message": f"Request timed out after {self.timeout} seconds"
                }
            )
        except Exception as e:
            self.logger.error(f"Tool call failed: {e}")
            return self.create_response(
                request_id=request.id,
                error={
                    "code": "REQUEST_ERROR",
                    "message": str(e)
                }
            )
    
    async def list_tools(self) -> List[MCPTool]:
        """
        List available tools on the remote MCP server
        
        Returns:
            List of available tools
        """
        if not self.session:
            return []
        
        request = self.create_request(
            method="tools/list",
            params={}
        )
        
        try:
            # Send JSON-RPC request
            rpc_request = {
                "jsonrpc": "2.0",
                "id": request.id,
                "method": request.method,
                "params": request.params
            }
            
            async with self.session.post(
                f"{self.base_url}/rpc",
                json=rpc_request
            ) as response:
                
                if response.status != 200:
                    self.logger.error(f"Failed to list tools: HTTP {response.status}")
                    return []
                
                response_data = await response.json()
                
                if "error" in response_data:
                    self.logger.error(f"Failed to list tools: {response_data['error']}")
                    return []
                
                tools_data = response_data.get("result", {}).get("tools", [])
                tools = []
                
                for tool_data in tools_data:
                    tool_type = MCPToolType.FUNCTION
                    if tool_data.get("type"):
                        try:
                            tool_type = MCPToolType(tool_data["type"])
                        except ValueError:
                            tool_type = MCPToolType.FUNCTION
                    
                    tools.append(MCPTool(
                        name=tool_data["name"],
                        type=tool_type,
                        description=tool_data.get("description", ""),
                        schema=tool_data.get("inputSchema", {}),
                        metadata=tool_data.get("metadata", {})
                    ))
                
                return tools
                
        except Exception as e:
            self.logger.error(f"Failed to list tools: {e}")
            return []
    
    async def get_server_info(self) -> Optional[MCPServerInfo]:
        """
        Get information about the remote MCP server
        
        Returns:
            Server information if successful, None otherwise
        """
        if not self.session:
            # Create temporary session for server info request
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            temp_session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
            session_to_use = temp_session
            close_session = True
        else:
            session_to_use = self.session
            close_session = False
        
        try:
            request = self.create_request(
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "EFA_Backend",
                        "version": "1.0.0"
                    }
                }
            )
            
            # Send JSON-RPC request
            rpc_request = {
                "jsonrpc": "2.0",
                "id": request.id,
                "method": request.method,
                "params": request.params
            }
            
            async with session_to_use.post(
                f"{self.base_url}/rpc",
                json=rpc_request
            ) as response:
                
                if response.status != 200:
                    return None
                
                response_data = await response.json()
                
                if "error" in response_data:
                    return None
                
                result = response_data.get("result", {})
                server_info_data = result.get("serverInfo", {})
                capabilities = result.get("capabilities", {})
                
                # Get tools list
                tools = await self.list_tools() if not close_session else []
                
                return MCPServerInfo(
                    name=server_info_data.get("name", self.server_name),
                    version=server_info_data.get("version", "unknown"),
                    description=server_info_data.get("description", ""),
                    capabilities=list(capabilities.keys()),
                    tools=tools,
                    metadata={
                        "protocol_version": result.get("protocolVersion"),
                        "server_info": server_info_data,
                        "capabilities": capabilities
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get server info: {e}")
            return None
        finally:
            if close_session:
                await temp_session.close()
    
    async def health_check(self) -> bool:
        """
        Check if the remote MCP server is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        if not self.session:
            return False
        
        try:
            # Simple ping request
            request = self.create_request(
                method="ping",
                params={}
            )
            
            rpc_request = {
                "jsonrpc": "2.0",
                "id": request.id,
                "method": request.method,
                "params": request.params
            }
            
            async with self.session.post(
                f"{self.base_url}/rpc",
                json=rpc_request
            ) as response:
                return response.status == 200
                
        except Exception:
            return False
    
    async def send_notification(self, method: str, params: Dict[str, Any]) -> bool:
        """
        Send a notification to the MCP server (no response expected)
        
        Args:
            method: Notification method
            params: Notification parameters
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.session:
            return False
        
        try:
            # JSON-RPC notification (no id field)
            rpc_notification = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params
            }
            
            async with self.session.post(
                f"{self.base_url}/rpc",
                json=rpc_notification
            ) as response:
                return response.status == 200
                
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
            return False
    
    def set_timeout(self, timeout: int) -> None:
        """
        Set request timeout
        
        Args:
            timeout: Timeout in seconds
        """
        self.timeout = timeout
        
        # Update session timeout if connected
        if self.session:
            self.session._timeout = aiohttp.ClientTimeout(total=timeout)
    
    def add_header(self, key: str, value: str) -> None:
        """
        Add or update a header
        
        Args:
            key: Header key
            value: Header value
        """
        self.headers[key] = value
        
        # Update session headers if connected
        if self.session:
            self.session.headers[key] = value
    
    def remove_header(self, key: str) -> None:
        """
        Remove a header
        
        Args:
            key: Header key to remove
        """
        self.headers.pop(key, None)
        
        # Update session headers if connected
        if self.session and key in self.session.headers:
            del self.session.headers[key]
