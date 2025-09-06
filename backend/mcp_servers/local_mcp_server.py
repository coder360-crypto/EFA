"""
Local MCP Server

Implementation of MCP server that runs locally and manages tools directly.
This can be used for development, testing, or when you want to host MCP tools locally.
"""

from typing import Dict, List, Optional, Any, Callable
import asyncio
import logging
import inspect
from datetime import datetime

from .base_adapter import MCPAdapter, MCPTool, MCPResponse, MCPServerInfo, MCPToolType


class LocalMCPServer(MCPAdapter):
    """Local implementation of MCP server"""
    
    def __init__(self, server_name: str = "local_mcp_server"):
        super().__init__(server_name)
        self.tools: Dict[str, Callable] = {}
        self.tool_schemas: Dict[str, Dict[str, Any]] = {}
        self.tool_descriptions: Dict[str, str] = {}
        self.tool_metadata: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Server capabilities
        self.capabilities = [
            "tools/call",
            "tools/list",
            "server/info",
            "health/check"
        ]
    
    async def connect(self, **kwargs) -> bool:
        """
        Connect to the local MCP server (always succeeds for local server)
        
        Returns:
            True (local server is always available)
        """
        self.connected = True
        self.server_info = MCPServerInfo(
            name=self.server_name,
            version="1.0.0",
            description="Local MCP Server for direct tool execution",
            capabilities=self.capabilities,
            tools=await self.list_tools(),
            metadata={
                "type": "local",
                "started_at": datetime.now().isoformat()
            }
        )
        self.logger.info(f"Connected to local MCP server: {self.server_name}")
        return True
    
    async def disconnect(self) -> bool:
        """
        Disconnect from the local MCP server
        
        Returns:
            True (always succeeds)
        """
        self.connected = False
        self.server_info = None
        self.logger.info(f"Disconnected from local MCP server: {self.server_name}")
        return True
    
    async def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        **kwargs
    ) -> MCPResponse:
        """
        Call a tool on the local MCP server
        
        Args:
            tool_name: Name of the tool to call
            parameters: Parameters for the tool
            **kwargs: Additional call parameters
            
        Returns:
            MCP response with result or error
        """
        request_id = self.generate_request_id()
        
        try:
            if tool_name not in self.tools:
                return self.create_response(
                    request_id=request_id,
                    error={
                        "code": "TOOL_NOT_FOUND",
                        "message": f"Tool '{tool_name}' not found"
                    }
                )
            
            tool_func = self.tools[tool_name]
            
            # Execute tool function
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**parameters)
            else:
                result = tool_func(**parameters)
            
            return self.create_response(
                request_id=request_id,
                result=result,
                metadata={
                    "tool_name": tool_name,
                    "executed_at": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Tool '{tool_name}' execution failed: {e}")
            return self.create_response(
                request_id=request_id,
                error={
                    "code": "EXECUTION_ERROR",
                    "message": str(e)
                }
            )
    
    async def list_tools(self) -> List[MCPTool]:
        """
        List available tools on the local MCP server
        
        Returns:
            List of available tools
        """
        tools = []
        
        for tool_name, tool_func in self.tools.items():
            tool_type = MCPToolType.FUNCTION  # Local tools are typically functions
            description = self.tool_descriptions.get(tool_name, f"Function: {tool_name}")
            schema = self.tool_schemas.get(tool_name, self._generate_schema_from_function(tool_func))
            metadata = self.tool_metadata.get(tool_name, {})
            
            tools.append(MCPTool(
                name=tool_name,
                type=tool_type,
                description=description,
                schema=schema,
                metadata=metadata
            ))
        
        return tools
    
    async def get_server_info(self) -> MCPServerInfo:
        """
        Get information about the local MCP server
        
        Returns:
            Server information
        """
        if not self.server_info:
            await self.connect()
        
        return self.server_info
    
    async def health_check(self) -> bool:
        """
        Check if the local MCP server is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        return self.connected
    
    def register_tool(
        self,
        name: str,
        func: Callable,
        description: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a tool with the local MCP server
        
        Args:
            name: Tool name
            func: Tool function
            description: Tool description
            schema: Tool schema (auto-generated if not provided)
            metadata: Tool metadata
            
        Returns:
            True if registration successful
        """
        try:
            self.tools[name] = func
            
            if description:
                self.tool_descriptions[name] = description
            else:
                # Generate description from function docstring
                if func.__doc__:
                    self.tool_descriptions[name] = func.__doc__.strip()
                else:
                    self.tool_descriptions[name] = f"Function: {name}"
            
            if schema:
                self.tool_schemas[name] = schema
            else:
                # Auto-generate schema from function signature
                self.tool_schemas[name] = self._generate_schema_from_function(func)
            
            if metadata:
                self.tool_metadata[name] = metadata
            
            self.logger.info(f"Registered tool: {name}")
            
            # Update server info if connected
            if self.connected and self.server_info:
                self.server_info.tools = await self.list_tools()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool '{name}': {e}")
            return False
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool from the local MCP server
        
        Args:
            name: Tool name to unregister
            
        Returns:
            True if unregistration successful
        """
        try:
            if name in self.tools:
                del self.tools[name]
                self.tool_descriptions.pop(name, None)
                self.tool_schemas.pop(name, None)
                self.tool_metadata.pop(name, None)
                
                self.logger.info(f"Unregistered tool: {name}")
                
                # Update server info if connected
                if self.connected and self.server_info:
                    asyncio.create_task(self._update_server_info())
                
                return True
            else:
                self.logger.warning(f"Tool '{name}' not found for unregistration")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to unregister tool '{name}': {e}")
            return False
    
    async def _update_server_info(self):
        """Update server info with current tools"""
        if self.server_info:
            self.server_info.tools = await self.list_tools()
    
    def _generate_schema_from_function(self, func: Callable) -> Dict[str, Any]:
        """
        Generate JSON schema from function signature
        
        Args:
            func: Function to analyze
            
        Returns:
            JSON schema for the function
        """
        try:
            sig = inspect.signature(func)
            schema = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            for param_name, param in sig.parameters.items():
                # Skip self parameter
                if param_name == "self":
                    continue
                
                param_schema = {"type": "string"}  # Default type
                
                # Try to infer type from annotation
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_schema["type"] = "integer"
                    elif param.annotation == float:
                        param_schema["type"] = "number"
                    elif param.annotation == bool:
                        param_schema["type"] = "boolean"
                    elif param.annotation == list:
                        param_schema["type"] = "array"
                    elif param.annotation == dict:
                        param_schema["type"] = "object"
                
                # Add description from docstring if available
                if func.__doc__:
                    # This is a simplified docstring parser
                    # In practice, you'd want a more robust solution
                    param_schema["description"] = f"Parameter: {param_name}"
                
                schema["properties"][param_name] = param_schema
                
                # Check if parameter is required (no default value)
                if param.default == inspect.Parameter.empty:
                    schema["required"].append(param_name)
            
            return schema
            
        except Exception as e:
            self.logger.error(f"Failed to generate schema for function: {e}")
            return {
                "type": "object",
                "properties": {},
                "required": []
            }
    
    def get_tool_count(self) -> int:
        """
        Get the number of registered tools
        
        Returns:
            Number of registered tools
        """
        return len(self.tools)
    
    def get_tool_names(self) -> List[str]:
        """
        Get list of registered tool names
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    async def bulk_register_tools(
        self,
        tools: Dict[str, Dict[str, Any]]
    ) -> Dict[str, bool]:
        """
        Register multiple tools at once
        
        Args:
            tools: Dictionary mapping tool names to tool definitions
                  Format: {
                      "tool_name": {
                          "func": callable,
                          "description": str (optional),
                          "schema": dict (optional),
                          "metadata": dict (optional)
                      }
                  }
        
        Returns:
            Dictionary mapping tool names to registration success status
        """
        results = {}
        
        for tool_name, tool_config in tools.items():
            if "func" not in tool_config:
                results[tool_name] = False
                self.logger.error(f"Tool '{tool_name}' missing 'func' in configuration")
                continue
            
            success = self.register_tool(
                name=tool_name,
                func=tool_config["func"],
                description=tool_config.get("description"),
                schema=tool_config.get("schema"),
                metadata=tool_config.get("metadata")
            )
            results[tool_name] = success
        
        return results
