"""
MCP Server

Main MCP server implementation that exposes tools and capabilities
through the Model Context Protocol.
"""

from typing import Dict, List, Optional, Any, Callable
import asyncio
import json
import logging
from datetime import datetime
from aiohttp import web, WSMsgType
import aiohttp_cors

from ..environments.environment_manager import EnvironmentManager
from ..mcp_tools.core_tools import MemoryTool, JudgmentTool, LearningTool, PlanningTool, PerceptionTool
from ..mcp_servers.performance_monitor import PerformanceMonitor


class MCPServer:
    """MCP Server implementation"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        environment_manager: Optional[EnvironmentManager] = None
    ):
        self.host = host
        self.port = port
        self.environment_manager = environment_manager or EnvironmentManager()
        self.performance_monitor = PerformanceMonitor()
        self.logger = logging.getLogger(__name__)
        
        # Core tools
        self.memory_tool = MemoryTool()
        self.judgment_tool = JudgmentTool()
        self.learning_tool = LearningTool()
        self.planning_tool = PlanningTool()
        self.perception_tool = PerceptionTool()
        
        # Server state
        self.app = None
        self.runner = None
        self.site = None
        self.websocket_connections: List[web.WebSocketResponse] = []
        
        # Request handlers
        self.request_handlers: Dict[str, Callable] = {}
        self._register_handlers()
    
    def _register_handlers(self):
        """Register request handlers"""
        # Core MCP methods
        self.request_handlers["initialize"] = self._handle_initialize
        self.request_handlers["tools/list"] = self._handle_list_tools
        self.request_handlers["tools/call"] = self._handle_call_tool
        self.request_handlers["resources/list"] = self._handle_list_resources
        self.request_handlers["resources/read"] = self._handle_read_resource
        
        # Environment methods
        self.request_handlers["environments/list"] = self._handle_list_environments
        self.request_handlers["environments/switch"] = self._handle_switch_environment
        self.request_handlers["environments/status"] = self._handle_environment_status
        
        # Core tool methods
        self.request_handlers["memory/store"] = self._handle_memory_store
        self.request_handlers["memory/search"] = self._handle_memory_search
        self.request_handlers["judgment/evaluate"] = self._handle_judgment_evaluate
        self.request_handlers["learning/record"] = self._handle_learning_record
        self.request_handlers["planning/create"] = self._handle_planning_create
        self.request_handlers["perception/process"] = self._handle_perception_process
        
        # Server management
        self.request_handlers["server/status"] = self._handle_server_status
        self.request_handlers["server/health"] = self._handle_health_check
    
    async def start(self):
        """Start the MCP server"""
        try:
            # Create aiohttp application
            self.app = web.Application()
            
            # Setup CORS
            cors = aiohttp_cors.setup(self.app, defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*"
                )
            })
            
            # Add routes
            self.app.router.add_post("/rpc", self._handle_http_request)
            self.app.router.add_get("/ws", self._handle_websocket)
            self.app.router.add_get("/health", self._handle_health_http)
            self.app.router.add_get("/status", self._handle_status_http)
            
            # Add CORS to all routes
            for route in list(self.app.router.routes()):
                cors.add(route)
            
            # Start performance monitoring
            await self.performance_monitor.start_monitoring()
            
            # Start server
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()
            
            self.logger.info(f"MCP Server started on {self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start MCP server: {e}")
            raise
    
    async def stop(self):
        """Stop the MCP server"""
        try:
            # Close WebSocket connections
            for ws in self.websocket_connections:
                if not ws.closed:
                    await ws.close()
            
            # Stop performance monitoring
            await self.performance_monitor.stop_monitoring()
            
            # Stop server
            if self.site:
                await self.site.stop()
            
            if self.runner:
                await self.runner.cleanup()
            
            self.logger.info("MCP Server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping MCP server: {e}")
    
    async def _handle_http_request(self, request: web.Request) -> web.Response:
        """Handle HTTP JSON-RPC requests"""
        try:
            data = await request.json()
            response = await self._process_request(data)
            return web.json_response(response)
            
        except Exception as e:
            self.logger.error(f"HTTP request error: {e}")
            return web.json_response({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                },
                "id": None
            }, status=500)
    
    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_connections.append(ws)
        self.logger.info("New WebSocket connection established")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        response = await self._process_request(data)
                        await ws.send_text(json.dumps(response))
                    except Exception as e:
                        error_response = {
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32603,
                                "message": "Internal error",
                                "data": str(e)
                            },
                            "id": None
                        }
                        await ws.send_text(json.dumps(error_response))
                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error: {ws.exception()}")
                    break
        
        except Exception as e:
            self.logger.error(f"WebSocket handler error: {e}")
        
        finally:
            if ws in self.websocket_connections:
                self.websocket_connections.remove(ws)
            self.logger.info("WebSocket connection closed")
        
        return ws
    
    async def _process_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process JSON-RPC request"""
        try:
            # Validate JSON-RPC format
            if "jsonrpc" not in data or data["jsonrpc"] != "2.0":
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request"
                    },
                    "id": data.get("id")
                }
            
            method = data.get("method")
            params = data.get("params", {})
            request_id = data.get("id")
            
            if not method:
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": "Method not found"
                    },
                    "id": request_id
                }
            
            # Start performance monitoring for this call
            call_id = f"{method}_{datetime.now().timestamp()}"
            self.performance_monitor.start_call(
                call_id=call_id,
                tool_name=method,
                server_name="mcp_server",
                request_data=data
            )
            
            try:
                # Handle request
                if method in self.request_handlers:
                    result = await self.request_handlers[method](params)
                    
                    # End performance monitoring (success)
                    self.performance_monitor.end_call(
                        call_id=call_id,
                        success=True,
                        response_data=result
                    )
                    
                    return {
                        "jsonrpc": "2.0",
                        "result": result,
                        "id": request_id
                    }
                else:
                    # End performance monitoring (method not found)
                    self.performance_monitor.end_call(
                        call_id=call_id,
                        success=False,
                        error_code="METHOD_NOT_FOUND",
                        error_message=f"Method '{method}' not found"
                    )
                    
                    return {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32601,
                            "message": "Method not found",
                            "data": f"Method '{method}' is not supported"
                        },
                        "id": request_id
                    }
            
            except Exception as e:
                # End performance monitoring (error)
                self.performance_monitor.end_call(
                    call_id=call_id,
                    success=False,
                    error_code="EXECUTION_ERROR",
                    error_message=str(e)
                )
                raise
                
        except Exception as e:
            self.logger.error(f"Request processing error: {e}")
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                },
                "id": data.get("id")
            }
    
    # Request handlers
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": True
                },
                "resources": {
                    "subscribe": True,
                    "listChanged": True
                },
                "prompts": {
                    "listChanged": True
                },
                "logging": {}
            },
            "serverInfo": {
                "name": "EFA MCP Server",
                "version": "1.0.0",
                "description": "Enhanced Function Agent MCP Server"
            },
            "instructions": "EFA MCP Server providing LLM tools, environment management, and core capabilities"
        }
    
    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request"""
        try:
            tools = []
            
            # Core tools
            core_tools = [
                {
                    "name": "memory_store",
                    "description": "Store information in memory",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "memory_type": {"type": "string", "enum": ["episodic", "semantic", "procedural"]},
                            "importance": {"type": "number", "minimum": 0, "maximum": 1},
                            "tags": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["content"]
                    }
                },
                {
                    "name": "memory_search",
                    "description": "Search stored memories",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "memory_type": {"type": "string"},
                            "max_results": {"type": "integer", "default": 10}
                        }
                    }
                },
                {
                    "name": "judgment_evaluate",
                    "description": "Evaluate responses and actions",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "response": {"type": "string"},
                            "context": {"type": "object"},
                            "criteria": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["response", "context"]
                    }
                },
                {
                    "name": "learning_record",
                    "description": "Record learning events",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "event_type": {"type": "string"},
                            "context": {"type": "object"},
                            "outcome": {"type": "string"},
                            "feedback": {"type": "string"}
                        },
                        "required": ["event_type", "context", "outcome"]
                    }
                },
                {
                    "name": "planning_create",
                    "description": "Create execution plans",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "metadata": {"type": "object"}
                        },
                        "required": ["name", "description"]
                    }
                },
                {
                    "name": "perception_process",
                    "description": "Process input data (text, image, audio, etc.)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "input_data": {"type": "string"},
                            "input_type": {"type": "string", "enum": ["text", "image", "audio", "video", "sensor", "document"]},
                            "analysis_options": {"type": "object"}
                        },
                        "required": ["input_data", "input_type"]
                    }
                }
            ]
            
            tools.extend(core_tools)
            
            # Environment tools
            env_tools = self.environment_manager.get_all_tools()
            for env_name, env_tool_list in env_tools.items():
                for tool in env_tool_list:
                    tools.append({
                        "name": f"{env_name}_{tool['name']}",
                        "description": f"[{env_name}] {tool.get('description', tool['name'])}",
                        "inputSchema": tool.get('schema', {})
                    })
            
            return {"tools": tools}
            
        except Exception as e:
            self.logger.error(f"Failed to list tools: {e}")
            raise
    
    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        try:
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if not tool_name:
                raise ValueError("Tool name is required")
            
            # Handle core tools
            if tool_name == "memory_store":
                result = await self.memory_tool.store_memory(**arguments)
                return {"content": [{"type": "text", "text": f"Memory stored with ID: {result}"}]}
            
            elif tool_name == "memory_search":
                result = await self.memory_tool.search_memories(**arguments)
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
            
            elif tool_name == "judgment_evaluate":
                result = await self.judgment_tool.evaluate_response(**arguments)
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
            
            elif tool_name == "learning_record":
                result = await self.learning_tool.record_event(**arguments)
                return {"content": [{"type": "text", "text": f"Learning event recorded with ID: {result}"}]}
            
            elif tool_name == "planning_create":
                result = await self.planning_tool.create_plan(**arguments)
                return {"content": [{"type": "text", "text": f"Plan created with ID: {result}"}]}
            
            elif tool_name == "perception_process":
                result = await self.perception_tool.process_input(**arguments)
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
            
            # Handle environment tools
            elif "_" in tool_name:
                env_name, env_tool_name = tool_name.split("_", 1)
                result = await self.environment_manager.execute_tool(
                    tool_name=env_tool_name,
                    parameters=arguments,
                    environment_name=env_name
                )
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
            
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
                
        except Exception as e:
            self.logger.error(f"Tool call failed: {e}")
            raise
    
    async def _handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list request"""
        resources = [
            {
                "uri": "memory://statistics",
                "name": "Memory Statistics",
                "description": "Statistics about stored memories",
                "mimeType": "application/json"
            },
            {
                "uri": "performance://metrics",
                "name": "Performance Metrics",
                "description": "Server performance metrics",
                "mimeType": "application/json"
            },
            {
                "uri": "environments://status",
                "name": "Environment Status",
                "description": "Status of all environments",
                "mimeType": "application/json"
            }
        ]
        
        return {"resources": resources}
    
    async def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request"""
        uri = params.get("uri")
        
        if uri == "memory://statistics":
            stats = await self.memory_tool.get_memory_statistics()
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(stats, indent=2)
                }]
            }
        
        elif uri == "performance://metrics":
            report = self.performance_monitor.generate_report()
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(report, indent=2)
                }]
            }
        
        elif uri == "environments://status":
            environments = self.environment_manager.list_environments()
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(environments, indent=2)
                }]
            }
        
        else:
            raise ValueError(f"Unknown resource: {uri}")
    
    async def _handle_list_environments(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle environments/list request"""
        environments = self.environment_manager.list_environments()
        return {"environments": environments}
    
    async def _handle_switch_environment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle environments/switch request"""
        env_name = params.get("name")
        if not env_name:
            raise ValueError("Environment name is required")
        
        success = await self.environment_manager.switch_environment(env_name)
        return {"success": success, "active_environment": env_name if success else None}
    
    async def _handle_environment_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle environments/status request"""
        stats = self.environment_manager.get_statistics()
        return stats
    
    # Core tool handlers (delegating to the tools directly)
    async def _handle_memory_store(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory/store request"""
        result = await self.memory_tool.store_memory(**params)
        return {"memory_id": result}
    
    async def _handle_memory_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory/search request"""
        result = await self.memory_tool.search_memories(**params)
        return {"memories": result}
    
    async def _handle_judgment_evaluate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle judgment/evaluate request"""
        result = await self.judgment_tool.evaluate_response(**params)
        return result
    
    async def _handle_learning_record(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle learning/record request"""
        result = await self.learning_tool.record_event(**params)
        return {"event_id": result}
    
    async def _handle_planning_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle planning/create request"""
        result = await self.planning_tool.create_plan(**params)
        return {"plan_id": result}
    
    async def _handle_perception_process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle perception/process request"""
        result = await self.perception_tool.process_input(**params)
        return result
    
    async def _handle_server_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle server/status request"""
        return {
            "status": "running",
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if hasattr(self, 'start_time') else 0,
            "connections": len(self.websocket_connections),
            "performance": self.performance_monitor.generate_report(),
            "environments": self.environment_manager.get_statistics()
        }
    
    async def _handle_health_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle server/health request"""
        health_checks = await self.environment_manager.health_check_all()
        
        overall_health = all(health_checks.values()) if health_checks else True
        
        return {
            "status": "healthy" if overall_health else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "environments": health_checks,
            "server": "running"
        }
    
    async def _handle_health_http(self, request: web.Request) -> web.Response:
        """Handle HTTP health check"""
        health = await self._handle_health_check({})
        status_code = 200 if health["status"] == "healthy" else 503
        return web.json_response(health, status=status_code)
    
    async def _handle_status_http(self, request: web.Request) -> web.Response:
        """Handle HTTP status check"""
        status = await self._handle_server_status({})
        return web.json_response(status)
    
    async def broadcast_notification(self, notification: Dict[str, Any]) -> None:
        """Broadcast notification to all WebSocket connections"""
        if not self.websocket_connections:
            return
        
        message = json.dumps(notification)
        disconnected = []
        
        for ws in self.websocket_connections:
            try:
                if not ws.closed:
                    await ws.send_text(message)
                else:
                    disconnected.append(ws)
            except Exception as e:
                self.logger.warning(f"Failed to send notification to WebSocket: {e}")
                disconnected.append(ws)
        
        # Remove disconnected WebSockets
        for ws in disconnected:
            if ws in self.websocket_connections:
                self.websocket_connections.remove(ws)
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.start_time = datetime.now()
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
