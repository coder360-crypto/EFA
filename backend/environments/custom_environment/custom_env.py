"""
Custom Environment

Template environment implementation that can be customized for specific use cases.
This serves as an example and starting point for creating new environments.
"""

from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime

from ..base_environment.environment import Environment, EnvironmentInfo, EnvironmentStatus
from ..base_environment.tools_registry import ToolsRegistry
from .custom_tools import CustomTools


class CustomEnvironment(Environment):
    """Custom environment implementation template"""
    
    def __init__(self, config):
        super().__init__(config)
        self.custom_tools = None
        self.custom_resources = {}
    
    async def initialize(self) -> bool:
        """Initialize the custom environment"""
        try:
            # Get configuration values
            custom_setting = self.get_config_value("custom_setting", "default_value")
            debug_mode = self.get_config_value("debug_mode", False)
            
            self.logger.info(f"Initializing custom environment with setting: {custom_setting}")
            
            if debug_mode:
                self.logger.setLevel(logging.DEBUG)
            
            # Initialize tools registry
            self.tools_registry = ToolsRegistry(self.config.name)
            
            # Initialize custom tools
            self.custom_tools = CustomTools(
                config=self.config.config_data,
                logger=self.logger
            )
            
            # Register tools
            await self._register_tools()
            
            # Initialize any custom resources
            await self._initialize_resources()
            
            self.logger.info("Custom environment initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize custom environment: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the custom environment"""
        try:
            # Clean up resources
            await self._cleanup_resources()
            
            self.custom_tools = None
            self.tools_registry = None
            self.custom_resources.clear()
            
            self.logger.info("Custom environment shutdown successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to shutdown custom environment: {e}")
            return False
    
    def get_info(self) -> EnvironmentInfo:
        """Get information about the custom environment"""
        return EnvironmentInfo(
            name="custom",
            description="Custom environment template for specific use cases",
            version="1.0.0",
            capabilities=[
                "example_operations",
                "custom_tools",
                "data_processing",
                "external_integrations"
            ],
            required_config=[
                # Add required configuration keys here
            ],
            optional_config=[
                "custom_setting",
                "debug_mode",
                "timeout",
                "max_retries"
            ],
            status=self.status,
            metadata={
                "provider": "Custom",
                "type": "template",
                "customizable": True
            }
        )
    
    async def health_check(self) -> bool:
        """Check if custom environment is healthy"""
        try:
            # Perform health checks specific to your environment
            # For example, check external service connectivity,
            # verify resource availability, etc.
            
            # Example health check
            if self.custom_tools:
                # Test a simple operation
                result = await self.custom_tools.health_check()
                return result
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        if self.tools_registry:
            return self.tools_registry.get_tool_names()
        return []
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Execute a tool in the custom environment"""
        if not self.tools_registry:
            raise RuntimeError("Tools registry not initialized")
        
        return await self.tools_registry.execute_tool(tool_name, parameters, **kwargs)
    
    async def _register_tools(self) -> None:
        """Register custom tools with the tools registry"""
        if not self.tools_registry or not self.custom_tools:
            return
        
        # Example tool registration
        self.tools_registry.register_tool(
            name="echo",
            func=self.custom_tools.echo,
            description="Echo back the input message",
            schema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to echo back"
                    }
                },
                "required": ["message"]
            }
        )
        
        self.tools_registry.register_tool(
            name="process_data",
            func=self.custom_tools.process_data,
            description="Process data with custom logic",
            schema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "Data to process"
                    },
                    "operation": {
                        "type": "string",
                        "description": "Operation to perform",
                        "enum": ["transform", "validate", "analyze"]
                    }
                },
                "required": ["data", "operation"]
            }
        )
        
        self.tools_registry.register_tool(
            name="external_api_call",
            func=self.custom_tools.external_api_call,
            description="Make a call to an external API",
            schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "API endpoint URL"
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method",
                        "enum": ["GET", "POST", "PUT", "DELETE"],
                        "default": "GET"
                    },
                    "headers": {
                        "type": "object",
                        "description": "HTTP headers"
                    },
                    "data": {
                        "type": "object",
                        "description": "Request data"
                    }
                },
                "required": ["url"]
            }
        )
        
        self.tools_registry.register_tool(
            name="calculate",
            func=self.custom_tools.calculate,
            description="Perform mathematical calculations",
            schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    },
                    "variables": {
                        "type": "object",
                        "description": "Variables to use in the expression"
                    }
                },
                "required": ["expression"]
            }
        )
        
        self.tools_registry.register_tool(
            name="file_operations",
            func=self.custom_tools.file_operations,
            description="Perform file system operations",
            schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "File operation to perform",
                        "enum": ["read", "write", "delete", "list", "exists"]
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content for write operations"
                    }
                },
                "required": ["operation", "path"]
            }
        )
        
        self.logger.info("Registered custom tools successfully")
    
    async def _initialize_resources(self) -> None:
        """Initialize custom resources"""
        try:
            # Initialize any resources your environment needs
            # Examples: database connections, external service clients, caches, etc.
            
            # Example resource initialization
            self.custom_resources["cache"] = {}
            self.custom_resources["counters"] = {"api_calls": 0, "tool_executions": 0}
            self.custom_resources["start_time"] = datetime.now()
            
            self.logger.debug("Custom resources initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize resources: {e}")
            raise
    
    async def _cleanup_resources(self) -> None:
        """Clean up custom resources"""
        try:
            # Clean up any resources that need explicit cleanup
            # Examples: close database connections, cleanup temporary files, etc.
            
            # Example cleanup
            if "cache" in self.custom_resources:
                self.custom_resources["cache"].clear()
            
            self.logger.debug("Custom resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup resources: {e}")
    
    def get_resource(self, name: str) -> Any:
        """
        Get a custom resource by name
        
        Args:
            name: Resource name
            
        Returns:
            Resource value or None if not found
        """
        return self.custom_resources.get(name)
    
    def set_resource(self, name: str, value: Any) -> None:
        """
        Set a custom resource
        
        Args:
            name: Resource name
            value: Resource value
        """
        self.custom_resources[name] = value
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get environment statistics
        
        Returns:
            Dictionary with environment statistics
        """
        counters = self.custom_resources.get("counters", {})
        start_time = self.custom_resources.get("start_time")
        
        stats = {
            "environment_name": self.config.name,
            "status": self.status.value,
            "tool_count": len(self.get_available_tools()),
            "counters": counters.copy() if counters else {}
        }
        
        if start_time:
            uptime = datetime.now() - start_time
            stats["uptime_seconds"] = uptime.total_seconds()
        
        return stats
    
    def increment_counter(self, counter_name: str, amount: int = 1) -> None:
        """
        Increment a counter
        
        Args:
            counter_name: Name of the counter
            amount: Amount to increment by
        """
        counters = self.custom_resources.get("counters", {})
        counters[counter_name] = counters.get(counter_name, 0) + amount
