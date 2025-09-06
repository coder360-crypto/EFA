"""
Tools Registry

Manages registration and execution of tools within an environment.
Provides a centralized way to handle environment-specific tools.
"""

from typing import Dict, List, Optional, Any, Callable
import asyncio
import inspect
import logging
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ToolInfo:
    """Information about a registered tool"""
    name: str
    func: Callable
    description: str
    schema: Dict[str, Any]
    registered_at: datetime
    call_count: int = 0
    last_called: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class ToolsRegistry:
    """Registry for managing environment tools"""
    
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
        self.tools: Dict[str, ToolInfo] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_tool(
        self,
        name: str,
        func: Callable,
        description: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a tool
        
        Args:
            name: Tool name
            func: Tool function
            description: Tool description
            schema: Tool schema (auto-generated if not provided)
            metadata: Additional metadata
            
        Returns:
            True if registration successful
        """
        try:
            # Validate tool name
            if not name or not isinstance(name, str):
                self.logger.error("Tool name must be a non-empty string")
                return False
            
            # Check if tool already exists
            if name in self.tools:
                self.logger.warning(f"Tool '{name}' already exists, replacing it")
            
            # Generate description if not provided
            if description is None:
                if func.__doc__:
                    description = func.__doc__.strip()
                else:
                    description = f"Tool: {name}"
            
            # Generate schema if not provided
            if schema is None:
                schema = self._generate_schema_from_function(func)
            
            # Create tool info
            tool_info = ToolInfo(
                name=name,
                func=func,
                description=description,
                schema=schema,
                registered_at=datetime.now(),
                metadata=metadata or {}
            )
            
            self.tools[name] = tool_info
            self.logger.info(f"Registered tool '{name}' in environment '{self.environment_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool '{name}': {e}")
            return False
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool
        
        Args:
            name: Tool name
            
        Returns:
            True if unregistration successful
        """
        try:
            if name in self.tools:
                del self.tools[name]
                self.logger.info(f"Unregistered tool '{name}' from environment '{self.environment_name}'")
                return True
            else:
                self.logger.warning(f"Tool '{name}' not found for unregistration")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to unregister tool '{name}': {e}")
            return False
    
    def get_tool_names(self) -> List[str]:
        """
        Get list of registered tool names
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a tool
        
        Args:
            name: Tool name
            
        Returns:
            Tool information if found, None otherwise
        """
        if name not in self.tools:
            return None
        
        tool_info = self.tools[name]
        return {
            "name": tool_info.name,
            "description": tool_info.description,
            "schema": tool_info.schema,
            "registered_at": tool_info.registered_at.isoformat(),
            "call_count": tool_info.call_count,
            "last_called": tool_info.last_called.isoformat() if tool_info.last_called else None,
            "metadata": tool_info.metadata
        }
    
    def get_all_tools_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered tools
        
        Returns:
            Dictionary mapping tool names to their information
        """
        return {name: self.get_tool_info(name) for name in self.tools.keys()}
    
    async def execute_tool(
        self,
        name: str,
        parameters: Dict[str, Any],
        **kwargs
    ) -> Any:
        """
        Execute a registered tool
        
        Args:
            name: Tool name
            parameters: Tool parameters
            **kwargs: Additional execution parameters
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found
            Exception: If tool execution fails
        """
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found in environment '{self.environment_name}'")
        
        tool_info = self.tools[name]
        
        try:
            # Validate parameters
            validation_result = self._validate_parameters(name, parameters)
            if not validation_result["valid"]:
                raise ValueError(f"Parameter validation failed: {validation_result['error']}")
            
            # Update call statistics
            tool_info.call_count += 1
            tool_info.last_called = datetime.now()
            
            # Execute the tool
            if asyncio.iscoroutinefunction(tool_info.func):
                result = await tool_info.func(**parameters)
            else:
                result = tool_info.func(**parameters)
            
            self.logger.debug(f"Successfully executed tool '{name}' in environment '{self.environment_name}'")
            return result
            
        except Exception as e:
            self.logger.error(f"Tool '{name}' execution failed: {e}")
            raise
    
    def _validate_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool parameters against schema
        
        Args:
            tool_name: Tool name
            parameters: Parameters to validate
            
        Returns:
            Validation result with 'valid' and 'error' keys
        """
        if tool_name not in self.tools:
            return {"valid": False, "error": f"Tool '{tool_name}' not found"}
        
        tool_info = self.tools[tool_name]
        schema = tool_info.schema
        
        try:
            # Check required parameters
            required_params = schema.get("required", [])
            for param in required_params:
                if param not in parameters:
                    return {"valid": False, "error": f"Required parameter '{param}' is missing"}
            
            # Check parameter types (basic validation)
            properties = schema.get("properties", {})
            for param_name, param_value in parameters.items():
                if param_name in properties:
                    expected_type = properties[param_name].get("type")
                    if expected_type and not self._check_type(param_value, expected_type):
                        return {"valid": False, "error": f"Parameter '{param_name}' has incorrect type. Expected {expected_type}"}
            
            # Check for unknown parameters
            for param_name in parameters:
                if param_name not in properties:
                    return {"valid": False, "error": f"Unknown parameter '{param_name}'"}
            
            return {"valid": True, "error": None}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """
        Check if value matches expected JSON schema type
        
        Args:
            value: Value to check
            expected_type: Expected JSON schema type
            
        Returns:
            True if type matches, False otherwise
        """
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, assume valid
        
        return isinstance(value, expected_python_type)
    
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
                    param_schema = self._annotation_to_schema(param.annotation)
                
                # Add description if available from docstring
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
    
    def _annotation_to_schema(self, annotation: Any) -> Dict[str, Any]:
        """
        Convert Python type annotation to JSON schema
        
        Args:
            annotation: Python type annotation
            
        Returns:
            JSON schema for the type
        """
        if annotation == str:
            return {"type": "string"}
        elif annotation == int:
            return {"type": "integer"}
        elif annotation == float:
            return {"type": "number"}
        elif annotation == bool:
            return {"type": "boolean"}
        elif annotation == list:
            return {"type": "array"}
        elif annotation == dict:
            return {"type": "object"}
        else:
            # Handle generic types and complex annotations
            if hasattr(annotation, "__origin__"):
                if annotation.__origin__ == list:
                    return {"type": "array"}
                elif annotation.__origin__ == dict:
                    return {"type": "object"}
                elif annotation.__origin__ == Union:
                    # Handle Optional types (Union[T, None])
                    args = getattr(annotation, "__args__", ())
                    if len(args) == 2 and type(None) in args:
                        # This is Optional[T]
                        non_none_type = args[0] if args[1] is type(None) else args[1]
                        return self._annotation_to_schema(non_none_type)
            
            # Default to string for unknown types
            return {"type": "string"}
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about tool usage
        
        Returns:
            Dictionary with tool statistics
        """
        total_tools = len(self.tools)
        total_calls = sum(tool.call_count for tool in self.tools.values())
        
        # Find most used tool
        most_used_tool = None
        max_calls = 0
        for tool in self.tools.values():
            if tool.call_count > max_calls:
                max_calls = tool.call_count
                most_used_tool = tool.name
        
        # Find recently used tools
        recently_used = []
        for tool in self.tools.values():
            if tool.last_called:
                recently_used.append({
                    "name": tool.name,
                    "last_called": tool.last_called.isoformat(),
                    "call_count": tool.call_count
                })
        
        # Sort by last called time
        recently_used.sort(key=lambda x: x["last_called"], reverse=True)
        
        return {
            "environment_name": self.environment_name,
            "total_tools": total_tools,
            "total_calls": total_calls,
            "most_used_tool": most_used_tool,
            "max_calls": max_calls,
            "recently_used": recently_used[:10]  # Top 10 recently used
        }
    
    def clear_all_tools(self) -> int:
        """
        Clear all registered tools
        
        Returns:
            Number of tools that were cleared
        """
        count = len(self.tools)
        self.tools.clear()
        self.logger.info(f"Cleared {count} tools from environment '{self.environment_name}'")
        return count
    
    def bulk_register_tools(self, tools_config: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """
        Register multiple tools at once
        
        Args:
            tools_config: Dictionary mapping tool names to their configuration
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
        
        for tool_name, config in tools_config.items():
            if "func" not in config:
                self.logger.error(f"Tool '{tool_name}' missing 'func' in configuration")
                results[tool_name] = False
                continue
            
            success = self.register_tool(
                name=tool_name,
                func=config["func"],
                description=config.get("description"),
                schema=config.get("schema"),
                metadata=config.get("metadata")
            )
            results[tool_name] = success
        
        return results
