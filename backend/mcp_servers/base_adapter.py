"""
Base MCP Adapter

Abstract interface for all MCP server adapters.
Defines the standard interface for interacting with MCP servers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime


class MCPToolType(Enum):
    """Types of MCP tools"""
    FUNCTION = "function"
    RESOURCE = "resource"
    PROMPT = "prompt"


@dataclass
class MCPTool:
    """Represents an MCP tool"""
    name: str
    type: MCPToolType
    description: str
    schema: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MCPRequest:
    """Request to an MCP server"""
    id: str
    method: str
    params: Dict[str, Any]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MCPResponse:
    """Response from an MCP server"""
    id: str
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MCPServerInfo:
    """Information about an MCP server"""
    name: str
    version: str
    description: str
    capabilities: List[str]
    tools: List[MCPTool]
    metadata: Optional[Dict[str, Any]] = None


class MCPAdapter(ABC):
    """Abstract base class for MCP server adapters"""
    
    def __init__(self, server_name: str):
        self.server_name = server_name
        self.connected = False
        self.server_info: Optional[MCPServerInfo] = None
    
    @abstractmethod
    async def connect(self, **kwargs) -> bool:
        """
        Connect to the MCP server
        
        Args:
            **kwargs: Connection parameters specific to the adapter
            
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the MCP server
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        **kwargs
    ) -> MCPResponse:
        """
        Call a tool on the MCP server
        
        Args:
            tool_name: Name of the tool to call
            parameters: Parameters for the tool
            **kwargs: Additional call parameters
            
        Returns:
            MCP response with result or error
        """
        pass
    
    @abstractmethod
    async def list_tools(self) -> List[MCPTool]:
        """
        List available tools on the MCP server
        
        Returns:
            List of available tools
        """
        pass
    
    @abstractmethod
    async def get_server_info(self) -> MCPServerInfo:
        """
        Get information about the MCP server
        
        Returns:
            Server information
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the MCP server is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    def generate_request_id(self) -> str:
        """
        Generate a unique request ID
        
        Returns:
            Unique request ID
        """
        return str(uuid.uuid4())
    
    def create_request(
        self,
        method: str,
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> MCPRequest:
        """
        Create an MCP request
        
        Args:
            method: Method name
            params: Request parameters
            metadata: Optional metadata
            
        Returns:
            MCP request object
        """
        return MCPRequest(
            id=self.generate_request_id(),
            method=method,
            params=params,
            timestamp=datetime.now(),
            metadata=metadata
        )
    
    def create_response(
        self,
        request_id: str,
        result: Optional[Any] = None,
        error: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MCPResponse:
        """
        Create an MCP response
        
        Args:
            request_id: ID of the original request
            result: Response result
            error: Error information if any
            metadata: Optional metadata
            
        Returns:
            MCP response object
        """
        return MCPResponse(
            id=request_id,
            result=result,
            error=error,
            metadata=metadata
        )
    
    def is_connected(self) -> bool:
        """
        Check if adapter is connected to server
        
        Returns:
            True if connected, False otherwise
        """
        return self.connected
    
    def get_tool_by_name(self, tool_name: str) -> Optional[MCPTool]:
        """
        Get tool information by name
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool information if found, None otherwise
        """
        if not self.server_info:
            return None
        
        for tool in self.server_info.tools:
            if tool.name == tool_name:
                return tool
        
        return None
    
    def validate_tool_parameters(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate parameters for a tool call
        
        Args:
            tool_name: Name of the tool
            parameters: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        tool = self.get_tool_by_name(tool_name)
        if not tool:
            return False, f"Tool '{tool_name}' not found"
        
        # Basic schema validation
        # This is a simplified version - in practice you'd want more robust validation
        schema = tool.schema
        
        if "required" in schema:
            for required_param in schema["required"]:
                if required_param not in parameters:
                    return False, f"Required parameter '{required_param}' missing"
        
        if "properties" in schema:
            for param_name, param_value in parameters.items():
                if param_name not in schema["properties"]:
                    return False, f"Unknown parameter '{param_name}'"
        
        return True, None
    
    async def safe_call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        **kwargs
    ) -> MCPResponse:
        """
        Safely call a tool with parameter validation
        
        Args:
            tool_name: Name of the tool to call
            parameters: Parameters for the tool
            **kwargs: Additional call parameters
            
        Returns:
            MCP response with result or error
        """
        # Validate parameters
        is_valid, error_message = self.validate_tool_parameters(tool_name, parameters)
        if not is_valid:
            return self.create_response(
                request_id=self.generate_request_id(),
                error={
                    "code": "INVALID_PARAMETERS",
                    "message": error_message
                }
            )
        
        # Check connection
        if not self.is_connected():
            return self.create_response(
                request_id=self.generate_request_id(),
                error={
                    "code": "NOT_CONNECTED",
                    "message": "Not connected to MCP server"
                }
            )
        
        try:
            return await self.call_tool(tool_name, parameters, **kwargs)
        except Exception as e:
            return self.create_response(
                request_id=self.generate_request_id(),
                error={
                    "code": "TOOL_CALL_ERROR",
                    "message": str(e)
                }
            )
