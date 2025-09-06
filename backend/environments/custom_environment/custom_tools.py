"""
Custom Tools

Implementation of custom tools for the custom environment.
These serve as examples and templates for creating your own tools.
"""

from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
import json
import os
import logging
import ast
import operator
from datetime import datetime


class CustomTools:
    """Custom tools implementation"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def health_check(self) -> bool:
        """
        Perform a health check of the tools
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Test basic functionality
            result = await self.echo("health_check")
            return result == "health_check"
        except Exception:
            return False
    
    async def echo(self, message: str) -> str:
        """
        Echo back the input message
        
        Args:
            message: Message to echo back
            
        Returns:
            The same message
        """
        self.logger.debug(f"Echo tool called with message: {message}")
        return message
    
    async def process_data(self, data: Dict[str, Any], operation: str) -> Dict[str, Any]:
        """
        Process data with custom logic
        
        Args:
            data: Data to process
            operation: Operation to perform ("transform", "validate", "analyze")
            
        Returns:
            Processed data result
        """
        try:
            self.logger.debug(f"Processing data with operation: {operation}")
            
            if operation == "transform":
                return await self._transform_data(data)
            elif operation == "validate":
                return await self._validate_data(data)
            elif operation == "analyze":
                return await self._analyze_data(data)
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            raise
    
    async def external_api_call(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a call to an external API
        
        Args:
            url: API endpoint URL
            method: HTTP method
            headers: HTTP headers
            data: Request data
            
        Returns:
            API response data
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            request_headers = headers or {}
            request_data = data
            
            if method.upper() in ["POST", "PUT", "PATCH"] and request_data:
                request_headers.setdefault("Content-Type", "application/json")
                request_data = json.dumps(request_data)
            
            self.logger.debug(f"Making {method} request to {url}")
            
            async with self.session.request(
                method, url, headers=request_headers, data=request_data
            ) as response:
                response_data = {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "url": str(response.url)
                }
                
                try:
                    response_data["data"] = await response.json()
                except Exception:
                    response_data["data"] = await response.text()
                
                return response_data
                
        except Exception as e:
            self.logger.error(f"External API call failed: {e}")
            raise
    
    async def calculate(
        self,
        expression: str,
        variables: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Perform mathematical calculations
        
        Args:
            expression: Mathematical expression to evaluate
            variables: Variables to use in the expression
            
        Returns:
            Calculation result
        """
        try:
            self.logger.debug(f"Calculating expression: {expression}")
            
            # Safe evaluation of mathematical expressions
            allowed_names = {
                k: v for k, v in vars(ast).items() if not k.startswith("_")
            }
            allowed_names.update({
                "abs": abs,
                "min": min,
                "max": max,
                "round": round,
                "pow": pow,
                "sum": sum
            })
            
            # Add variables if provided
            if variables:
                allowed_names.update(variables)
            
            # Parse and evaluate the expression safely
            node = ast.parse(expression, mode='eval')
            result = self._safe_eval(node.body, allowed_names)
            
            return {
                "expression": expression,
                "variables": variables or {},
                "result": result,
                "calculated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Calculation failed: {e}")
            raise ValueError(f"Invalid expression or calculation error: {e}")
    
    async def file_operations(
        self,
        operation: str,
        path: str,
        content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform file system operations
        
        Args:
            operation: File operation ("read", "write", "delete", "list", "exists")
            path: File or directory path
            content: Content for write operations
            
        Returns:
            Operation result
        """
        try:
            self.logger.debug(f"File operation: {operation} on {path}")
            
            if operation == "read":
                return await self._read_file(path)
            elif operation == "write":
                return await self._write_file(path, content or "")
            elif operation == "delete":
                return await self._delete_file(path)
            elif operation == "list":
                return await self._list_directory(path)
            elif operation == "exists":
                return await self._check_exists(path)
            else:
                raise ValueError(f"Unknown file operation: {operation}")
                
        except Exception as e:
            self.logger.error(f"File operation failed: {e}")
            raise
    
    async def _transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data (example implementation)"""
        transformed = {}
        
        for key, value in data.items():
            # Example transformations
            if isinstance(value, str):
                transformed[f"transformed_{key}"] = value.upper()
            elif isinstance(value, (int, float)):
                transformed[f"transformed_{key}"] = value * 2
            elif isinstance(value, list):
                transformed[f"transformed_{key}"] = len(value)
            else:
                transformed[f"transformed_{key}"] = str(value)
        
        return {
            "operation": "transform",
            "original_data": data,
            "transformed_data": transformed,
            "transformation_count": len(transformed)
        }
    
    async def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data (example implementation)"""
        validation_results = {}
        
        for key, value in data.items():
            if key == "email":
                validation_results[key] = "@" in str(value)
            elif key == "age":
                validation_results[key] = isinstance(value, int) and 0 <= value <= 150
            elif key == "name":
                validation_results[key] = isinstance(value, str) and len(value) > 0
            else:
                validation_results[key] = value is not None
        
        is_valid = all(validation_results.values())
        
        return {
            "operation": "validate",
            "data": data,
            "validation_results": validation_results,
            "is_valid": is_valid,
            "valid_fields": sum(validation_results.values()),
            "total_fields": len(validation_results)
        }
    
    async def _analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data (example implementation)"""
        analysis = {
            "field_count": len(data),
            "field_types": {},
            "string_fields": [],
            "numeric_fields": [],
            "list_fields": [],
            "dict_fields": [],
            "null_fields": []
        }
        
        for key, value in data.items():
            value_type = type(value).__name__
            analysis["field_types"][key] = value_type
            
            if isinstance(value, str):
                analysis["string_fields"].append(key)
            elif isinstance(value, (int, float)):
                analysis["numeric_fields"].append(key)
            elif isinstance(value, list):
                analysis["list_fields"].append(key)
            elif isinstance(value, dict):
                analysis["dict_fields"].append(key)
            elif value is None:
                analysis["null_fields"].append(key)
        
        return {
            "operation": "analyze",
            "data": data,
            "analysis": analysis
        }
    
    def _safe_eval(self, node, names):
        """Safely evaluate AST node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            else:
                raise NameError(f"Name '{node.id}' is not defined")
        elif isinstance(node, ast.BinOp):
            left = self._safe_eval(node.left, names)
            right = self._safe_eval(node.right, names)
            return self._get_operator(node.op)(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._safe_eval(node.operand, names)
            return self._get_unary_operator(node.op)(operand)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in names:
                func = names[node.func.id]
                args = [self._safe_eval(arg, names) for arg in node.args]
                return func(*args)
            else:
                raise ValueError("Function calls not allowed")
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")
    
    def _get_operator(self, op):
        """Get operator function from AST operator"""
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.FloorDiv: operator.floordiv
        }
        return operators[type(op)]
    
    def _get_unary_operator(self, op):
        """Get unary operator function from AST operator"""
        operators = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg
        }
        return operators[type(op)]
    
    async def _read_file(self, path: str) -> Dict[str, Any]:
        """Read file content"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        stat = os.stat(path)
        
        return {
            "operation": "read",
            "path": path,
            "content": content,
            "size": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    
    async def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        return {
            "operation": "write",
            "path": path,
            "content_length": len(content),
            "written_at": datetime.now().isoformat()
        }
    
    async def _delete_file(self, path: str) -> Dict[str, Any]:
        """Delete file or directory"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        
        if os.path.isfile(path):
            os.remove(path)
            deleted_type = "file"
        else:
            os.rmdir(path)
            deleted_type = "directory"
        
        return {
            "operation": "delete",
            "path": path,
            "type": deleted_type,
            "deleted_at": datetime.now().isoformat()
        }
    
    async def _list_directory(self, path: str) -> Dict[str, Any]:
        """List directory contents"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Path is not a directory: {path}")
        
        items = []
        for item_name in os.listdir(path):
            item_path = os.path.join(path, item_name)
            stat = os.stat(item_path)
            
            items.append({
                "name": item_name,
                "path": item_path,
                "type": "directory" if os.path.isdir(item_path) else "file",
                "size": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return {
            "operation": "list",
            "path": path,
            "items": items,
            "item_count": len(items)
        }
    
    async def _check_exists(self, path: str) -> Dict[str, Any]:
        """Check if path exists"""
        exists = os.path.exists(path)
        
        result = {
            "operation": "exists",
            "path": path,
            "exists": exists
        }
        
        if exists:
            result["type"] = "directory" if os.path.isdir(path) else "file"
            stat = os.stat(path)
            result["size"] = stat.st_size
            result["modified_time"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        return result
    
    async def close(self):
        """Close any open resources"""
        if self.session:
            await self.session.close()
            self.session = None
