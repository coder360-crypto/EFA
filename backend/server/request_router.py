"""
Request Router

Routes MCP requests to appropriate handlers and manages request processing flow.
"""

from typing import Dict, List, Optional, Any, Callable
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class RequestType(Enum):
    """Types of requests"""
    TOOL_CALL = "tool_call"
    RESOURCE_READ = "resource_read"
    ENVIRONMENT = "environment"
    CORE_FUNCTION = "core_function"
    ADMIN = "admin"


@dataclass
class RouteDefinition:
    """Definition of a route"""
    pattern: str
    handler: Callable
    request_type: RequestType
    requires_auth: bool = False
    rate_limit: Optional[int] = None  # requests per minute
    description: str = ""


class RequestRouter:
    """Routes requests to appropriate handlers"""
    
    def __init__(self):
        self.routes: Dict[str, RouteDefinition] = {}
        self.middleware: List[Callable] = []
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_route(
        self,
        pattern: str,
        handler: Callable,
        request_type: RequestType,
        requires_auth: bool = False,
        rate_limit: Optional[int] = None,
        description: str = ""
    ) -> None:
        """
        Register a route
        
        Args:
            pattern: Route pattern (e.g., "tools/call", "memory/*")
            handler: Handler function
            request_type: Type of request
            requires_auth: Whether authentication is required
            rate_limit: Rate limit in requests per minute
            description: Description of the route
        """
        route = RouteDefinition(
            pattern=pattern,
            handler=handler,
            request_type=request_type,
            requires_auth=requires_auth,
            rate_limit=rate_limit,
            description=description
        )
        
        self.routes[pattern] = route
        self.logger.debug(f"Registered route: {pattern}")
    
    def add_middleware(self, middleware: Callable) -> None:
        """
        Add middleware function
        
        Args:
            middleware: Middleware function
        """
        self.middleware.append(middleware)
        self.logger.debug("Added middleware")
    
    async def route_request(
        self,
        method: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Route a request to the appropriate handler
        
        Args:
            method: Request method/path
            params: Request parameters
            context: Optional request context (auth, client info, etc.)
            
        Returns:
            Handler result
        """
        try:
            # Find matching route
            route = self._find_route(method)
            if not route:
                raise ValueError(f"No route found for method: {method}")
            
            # Apply middleware
            for middleware_func in self.middleware:
                await middleware_func(method, params, context)
            
            # Check authentication
            if route.requires_auth:
                await self._check_auth(context)
            
            # Check rate limiting
            if route.rate_limit:
                await self._check_rate_limit(method, route.rate_limit, context)
            
            # Execute handler
            self.logger.debug(f"Routing {method} to {route.handler.__name__}")
            
            # Add routing context
            routing_context = {
                "method": method,
                "route_pattern": route.pattern,
                "request_type": route.request_type.value,
                "timestamp": datetime.now().isoformat()
            }
            
            if context:
                routing_context.update(context)
            
            # Call handler with routing context
            if asyncio.iscoroutinefunction(route.handler):
                result = await route.handler(params, routing_context)
            else:
                result = route.handler(params, routing_context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Request routing failed for {method}: {e}")
            raise
    
    def _find_route(self, method: str) -> Optional[RouteDefinition]:
        """Find matching route for method"""
        # Exact match first
        if method in self.routes:
            return self.routes[method]
        
        # Pattern matching
        for pattern, route in self.routes.items():
            if self._match_pattern(pattern, method):
                return route
        
        return None
    
    def _match_pattern(self, pattern: str, method: str) -> bool:
        """Check if method matches pattern"""
        # Simple pattern matching
        if "*" in pattern:
            # Wildcard matching
            if pattern.endswith("/*"):
                prefix = pattern[:-2]
                return method.startswith(prefix + "/")
            elif pattern.startswith("*/"):
                suffix = pattern[2:]
                return method.endswith("/" + suffix)
            elif "/*/" in pattern:
                parts = pattern.split("/*/")
                return method.startswith(parts[0]) and method.endswith(parts[1])
        
        return pattern == method
    
    async def _check_auth(self, context: Optional[Dict[str, Any]]) -> None:
        """Check authentication"""
        # Placeholder authentication check
        # In practice, you'd implement proper authentication
        if not context or not context.get("authenticated"):
            raise PermissionError("Authentication required")
    
    async def _check_rate_limit(
        self,
        method: str,
        limit: int,
        context: Optional[Dict[str, Any]]
    ) -> None:
        """Check rate limiting"""
        now = datetime.now()
        
        # Get client identifier (IP, user ID, etc.)
        client_id = "default"
        if context:
            client_id = context.get("client_id", context.get("remote_addr", "default"))
        
        rate_key = f"{client_id}:{method}"
        
        # Initialize rate limit tracking
        if rate_key not in self.rate_limits:
            self.rate_limits[rate_key] = []
        
        request_times = self.rate_limits[rate_key]
        
        # Clean old requests (older than 1 minute)
        cutoff_time = now.timestamp() - 60  # 1 minute ago
        request_times[:] = [t for t in request_times if t.timestamp() > cutoff_time]
        
        # Check if limit exceeded
        if len(request_times) >= limit:
            raise Exception(f"Rate limit exceeded for {method}. Limit: {limit} requests per minute")
        
        # Add current request
        request_times.append(now)
    
    def get_routes_info(self) -> List[Dict[str, Any]]:
        """Get information about registered routes"""
        routes_info = []
        
        for pattern, route in self.routes.items():
            route_info = {
                "pattern": pattern,
                "request_type": route.request_type.value,
                "requires_auth": route.requires_auth,
                "rate_limit": route.rate_limit,
                "description": route.description,
                "handler": route.handler.__name__
            }
            routes_info.append(route_info)
        
        return routes_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics"""
        total_routes = len(self.routes)
        
        # Count by request type
        type_counts = {}
        for route in self.routes.values():
            req_type = route.request_type.value
            type_counts[req_type] = type_counts.get(req_type, 0) + 1
        
        # Rate limit statistics
        active_rate_limits = len(self.rate_limits)
        total_tracked_requests = sum(len(times) for times in self.rate_limits.values())
        
        return {
            "total_routes": total_routes,
            "routes_by_type": type_counts,
            "middleware_count": len(self.middleware),
            "active_rate_limits": active_rate_limits,
            "total_tracked_requests": total_tracked_requests
        }
    
    async def validate_request(
        self,
        method: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate a request without executing it
        
        Args:
            method: Request method
            params: Request parameters
            
        Returns:
            Validation result
        """
        try:
            route = self._find_route(method)
            
            validation_result = {
                "valid": route is not None,
                "method": method,
                "found_route": route.pattern if route else None,
                "request_type": route.request_type.value if route else None,
                "requires_auth": route.requires_auth if route else None,
                "rate_limited": route.rate_limit is not None if route else None
            }
            
            if not route:
                validation_result["error"] = f"No route found for method: {method}"
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "method": method,
                "error": str(e)
            }
    
    def remove_route(self, pattern: str) -> bool:
        """
        Remove a route
        
        Args:
            pattern: Route pattern to remove
            
        Returns:
            True if route was removed, False if not found
        """
        if pattern in self.routes:
            del self.routes[pattern]
            self.logger.debug(f"Removed route: {pattern}")
            return True
        return False
    
    def clear_routes(self) -> None:
        """Clear all routes"""
        self.routes.clear()
        self.logger.info("Cleared all routes")
    
    def clear_rate_limits(self) -> None:
        """Clear all rate limit tracking"""
        self.rate_limits.clear()
        self.logger.info("Cleared all rate limits")
    
    async def route_batch_requests(
        self,
        requests: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Route multiple requests in batch
        
        Args:
            requests: List of request dictionaries with 'method' and 'params'
            context: Optional shared context
            
        Returns:
            List of results corresponding to each request
        """
        results = []
        
        for i, request in enumerate(requests):
            try:
                method = request.get("method")
                params = request.get("params", {})
                
                if not method:
                    results.append({
                        "index": i,
                        "error": "Missing method in request",
                        "success": False
                    })
                    continue
                
                result = await self.route_request(method, params, context)
                results.append({
                    "index": i,
                    "result": result,
                    "success": True
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "error": str(e),
                    "success": False
                })
        
        return results


# Middleware functions
async def logging_middleware(method: str, params: Dict[str, Any], context: Optional[Dict[str, Any]]) -> None:
    """Logging middleware"""
    logger = logging.getLogger("request_router.middleware")
    client_id = context.get("client_id", "unknown") if context else "unknown"
    logger.info(f"Request: {method} from {client_id}")


async def timing_middleware(method: str, params: Dict[str, Any], context: Optional[Dict[str, Any]]) -> None:
    """Timing middleware"""
    if context:
        context["start_time"] = datetime.now()


async def security_middleware(method: str, params: Dict[str, Any], context: Optional[Dict[str, Any]]) -> None:
    """Security middleware"""
    # Placeholder security checks
    # In practice, you'd implement proper security validation
    
    # Check for suspicious patterns
    suspicious_patterns = ["eval", "exec", "__import__", "file://", "http://"]
    
    for pattern in suspicious_patterns:
        if pattern in str(params):
            raise SecurityError(f"Suspicious pattern detected: {pattern}")


class SecurityError(Exception):
    """Security-related error"""
    pass
