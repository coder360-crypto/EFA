"""
Performance Monitor

Monitors and logs MCP server performance, including call statistics,
response times, error rates, and resource usage.
"""

from typing import Dict, List, Optional, Any
import asyncio
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json


@dataclass
class CallMetrics:
    """Metrics for a single MCP call"""
    tool_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    success: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    request_size: int = 0  # Size of request in bytes
    response_size: int = 0  # Size of response in bytes
    server_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerMetrics:
    """Aggregated metrics for an MCP server"""
    server_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    total_request_size: int = 0
    total_response_size: int = 0
    error_rate: float = 0.0
    calls_per_minute: float = 0.0
    last_call_time: Optional[datetime] = None
    uptime_percentage: float = 100.0


@dataclass
class ToolMetrics:
    """Aggregated metrics for a specific tool"""
    tool_name: str
    server_name: str
    call_count: int = 0
    success_count: int = 0
    average_duration_ms: float = 0.0
    error_rate: float = 0.0
    last_used: Optional[datetime] = None


class PerformanceMonitor:
    """Monitors MCP server performance and collects metrics"""
    
    def __init__(
        self,
        max_history_size: int = 10000,
        metrics_window_minutes: int = 60,
        log_interval_seconds: int = 300  # 5 minutes
    ):
        self.max_history_size = max_history_size
        self.metrics_window_minutes = metrics_window_minutes
        self.log_interval_seconds = log_interval_seconds
        
        # Storage for call metrics
        self.call_history: deque[CallMetrics] = deque(maxlen=max_history_size)
        self.server_metrics: Dict[str, ServerMetrics] = {}
        self.tool_metrics: Dict[str, ToolMetrics] = {}
        
        # Active call tracking
        self.active_calls: Dict[str, Dict[str, Any]] = {}
        
        # Health check tracking
        self.server_health: Dict[str, List[bool]] = defaultdict(list)
        self.health_check_interval = 60  # seconds
        
        self.logger = logging.getLogger(__name__)
        
        # Start background tasks
        self._monitoring_task = None
        self._cleanup_task = None
        
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    def start_call(
        self,
        call_id: str,
        tool_name: str,
        server_name: str,
        request_data: Any = None
    ) -> None:
        """
        Start tracking a new MCP call
        
        Args:
            call_id: Unique identifier for the call
            tool_name: Name of the tool being called
            server_name: Name of the server
            request_data: Request data for size calculation
        """
        request_size = 0
        if request_data:
            try:
                request_size = len(json.dumps(request_data).encode('utf-8'))
            except Exception:
                request_size = 0
        
        self.active_calls[call_id] = {
            'tool_name': tool_name,
            'server_name': server_name,
            'start_time': datetime.now(),
            'request_size': request_size
        }
    
    def end_call(
        self,
        call_id: str,
        success: bool,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        response_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        End tracking an MCP call and record metrics
        
        Args:
            call_id: Unique identifier for the call
            success: Whether the call was successful
            error_code: Error code if call failed
            error_message: Error message if call failed
            response_data: Response data for size calculation
            metadata: Additional metadata
        """
        if call_id not in self.active_calls:
            self.logger.warning(f"Call ID {call_id} not found in active calls")
            return
        
        call_info = self.active_calls.pop(call_id)
        end_time = datetime.now()
        duration_ms = (end_time - call_info['start_time']).total_seconds() * 1000
        
        response_size = 0
        if response_data:
            try:
                response_size = len(json.dumps(response_data).encode('utf-8'))
            except Exception:
                response_size = 0
        
        # Create call metrics
        call_metrics = CallMetrics(
            tool_name=call_info['tool_name'],
            start_time=call_info['start_time'],
            end_time=end_time,
            duration_ms=duration_ms,
            success=success,
            error_code=error_code,
            error_message=error_message,
            request_size=call_info['request_size'],
            response_size=response_size,
            server_name=call_info['server_name'],
            metadata=metadata or {}
        )
        
        # Store in history
        self.call_history.append(call_metrics)
        
        # Update aggregated metrics
        self._update_server_metrics(call_metrics)
        self._update_tool_metrics(call_metrics)
    
    def _update_server_metrics(self, call_metrics: CallMetrics) -> None:
        """Update aggregated server metrics"""
        server_name = call_metrics.server_name
        
        if server_name not in self.server_metrics:
            self.server_metrics[server_name] = ServerMetrics(server_name=server_name)
        
        metrics = self.server_metrics[server_name]
        
        # Update counts
        metrics.total_calls += 1
        if call_metrics.success:
            metrics.successful_calls += 1
        else:
            metrics.failed_calls += 1
        
        # Update durations
        if call_metrics.duration_ms < metrics.min_duration_ms:
            metrics.min_duration_ms = call_metrics.duration_ms
        if call_metrics.duration_ms > metrics.max_duration_ms:
            metrics.max_duration_ms = call_metrics.duration_ms
        
        # Update sizes
        metrics.total_request_size += call_metrics.request_size
        metrics.total_response_size += call_metrics.response_size
        
        # Update timing
        metrics.last_call_time = call_metrics.end_time
        
        # Recalculate averages
        self._recalculate_server_metrics(server_name)
    
    def _update_tool_metrics(self, call_metrics: CallMetrics) -> None:
        """Update aggregated tool metrics"""
        tool_key = f"{call_metrics.server_name}:{call_metrics.tool_name}"
        
        if tool_key not in self.tool_metrics:
            self.tool_metrics[tool_key] = ToolMetrics(
                tool_name=call_metrics.tool_name,
                server_name=call_metrics.server_name
            )
        
        metrics = self.tool_metrics[tool_key]
        
        # Update counts
        metrics.call_count += 1
        if call_metrics.success:
            metrics.success_count += 1
        
        # Update timing
        metrics.last_used = call_metrics.end_time
        
        # Recalculate averages
        self._recalculate_tool_metrics(tool_key)
    
    def _recalculate_server_metrics(self, server_name: str) -> None:
        """Recalculate server metrics from recent call history"""
        cutoff_time = datetime.now() - timedelta(minutes=self.metrics_window_minutes)
        
        # Get recent calls for this server
        recent_calls = [
            call for call in self.call_history
            if call.server_name == server_name and call.start_time >= cutoff_time
        ]
        
        if not recent_calls:
            return
        
        metrics = self.server_metrics[server_name]
        
        # Calculate averages from recent calls
        durations = [call.duration_ms for call in recent_calls]
        metrics.average_duration_ms = statistics.mean(durations)
        
        # Calculate error rate
        total_recent = len(recent_calls)
        failed_recent = sum(1 for call in recent_calls if not call.success)
        metrics.error_rate = (failed_recent / total_recent) * 100 if total_recent > 0 else 0
        
        # Calculate calls per minute
        time_span_minutes = self.metrics_window_minutes
        metrics.calls_per_minute = total_recent / time_span_minutes
    
    def _recalculate_tool_metrics(self, tool_key: str) -> None:
        """Recalculate tool metrics from recent call history"""
        cutoff_time = datetime.now() - timedelta(minutes=self.metrics_window_minutes)
        
        metrics = self.tool_metrics[tool_key]
        
        # Get recent calls for this tool
        recent_calls = [
            call for call in self.call_history
            if (f"{call.server_name}:{call.tool_name}" == tool_key and
                call.start_time >= cutoff_time)
        ]
        
        if not recent_calls:
            return
        
        # Calculate averages from recent calls
        durations = [call.duration_ms for call in recent_calls]
        metrics.average_duration_ms = statistics.mean(durations)
        
        # Calculate error rate
        total_recent = len(recent_calls)
        failed_recent = sum(1 for call in recent_calls if not call.success)
        metrics.error_rate = (failed_recent / total_recent) * 100 if total_recent > 0 else 0
    
    def record_health_check(self, server_name: str, is_healthy: bool) -> None:
        """
        Record a health check result
        
        Args:
            server_name: Name of the server
            is_healthy: Whether the server is healthy
        """
        self.server_health[server_name].append(is_healthy)
        
        # Keep only recent health checks (last 24 hours)
        max_checks = (24 * 60 * 60) // self.health_check_interval
        if len(self.server_health[server_name]) > max_checks:
            self.server_health[server_name] = self.server_health[server_name][-max_checks:]
        
        # Update uptime percentage in server metrics
        if server_name in self.server_metrics:
            health_checks = self.server_health[server_name]
            if health_checks:
                uptime = (sum(health_checks) / len(health_checks)) * 100
                self.server_metrics[server_name].uptime_percentage = uptime
    
    def get_server_metrics(self, server_name: str) -> Optional[ServerMetrics]:
        """
        Get metrics for a specific server
        
        Args:
            server_name: Name of the server
            
        Returns:
            Server metrics if available, None otherwise
        """
        return self.server_metrics.get(server_name)
    
    def get_tool_metrics(self, server_name: str, tool_name: str) -> Optional[ToolMetrics]:
        """
        Get metrics for a specific tool
        
        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            
        Returns:
            Tool metrics if available, None otherwise
        """
        tool_key = f"{server_name}:{tool_name}"
        return self.tool_metrics.get(tool_key)
    
    def get_all_server_metrics(self) -> Dict[str, ServerMetrics]:
        """
        Get metrics for all servers
        
        Returns:
            Dictionary mapping server names to metrics
        """
        return self.server_metrics.copy()
    
    def get_slow_calls(
        self,
        threshold_ms: float = 1000,
        limit: int = 100
    ) -> List[CallMetrics]:
        """
        Get calls that took longer than threshold
        
        Args:
            threshold_ms: Duration threshold in milliseconds
            limit: Maximum number of calls to return
            
        Returns:
            List of slow calls
        """
        slow_calls = [
            call for call in self.call_history
            if call.duration_ms > threshold_ms
        ]
        
        # Sort by duration (slowest first)
        slow_calls.sort(key=lambda x: x.duration_ms, reverse=True)
        
        return slow_calls[:limit]
    
    def get_error_calls(
        self,
        server_name: Optional[str] = None,
        limit: int = 100
    ) -> List[CallMetrics]:
        """
        Get calls that resulted in errors
        
        Args:
            server_name: Filter by server name (optional)
            limit: Maximum number of calls to return
            
        Returns:
            List of error calls
        """
        error_calls = [
            call for call in self.call_history
            if not call.success and (server_name is None or call.server_name == server_name)
        ]
        
        # Sort by time (most recent first)
        error_calls.sort(key=lambda x: x.end_time, reverse=True)
        
        return error_calls[:limit]
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report
        
        Returns:
            Performance report as dictionary
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "metrics_window_minutes": self.metrics_window_minutes,
            "total_calls_in_history": len(self.call_history),
            "active_calls": len(self.active_calls),
            "servers": {},
            "tools": {},
            "summary": {}
        }
        
        # Server metrics
        for server_name, metrics in self.server_metrics.items():
            report["servers"][server_name] = {
                "total_calls": metrics.total_calls,
                "successful_calls": metrics.successful_calls,
                "failed_calls": metrics.failed_calls,
                "error_rate": metrics.error_rate,
                "average_duration_ms": metrics.average_duration_ms,
                "min_duration_ms": metrics.min_duration_ms,
                "max_duration_ms": metrics.max_duration_ms,
                "calls_per_minute": metrics.calls_per_minute,
                "uptime_percentage": metrics.uptime_percentage,
                "total_request_size": metrics.total_request_size,
                "total_response_size": metrics.total_response_size,
                "last_call_time": metrics.last_call_time.isoformat() if metrics.last_call_time else None
            }
        
        # Tool metrics
        for tool_key, metrics in self.tool_metrics.items():
            report["tools"][tool_key] = {
                "tool_name": metrics.tool_name,
                "server_name": metrics.server_name,
                "call_count": metrics.call_count,
                "success_count": metrics.success_count,
                "error_rate": metrics.error_rate,
                "average_duration_ms": metrics.average_duration_ms,
                "last_used": metrics.last_used.isoformat() if metrics.last_used else None
            }
        
        # Summary statistics
        if self.call_history:
            all_durations = [call.duration_ms for call in self.call_history]
            total_calls = len(self.call_history)
            successful_calls = sum(1 for call in self.call_history if call.success)
            
            report["summary"] = {
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "failed_calls": total_calls - successful_calls,
                "overall_error_rate": ((total_calls - successful_calls) / total_calls) * 100 if total_calls > 0 else 0,
                "average_duration_ms": statistics.mean(all_durations),
                "median_duration_ms": statistics.median(all_durations),
                "min_duration_ms": min(all_durations),
                "max_duration_ms": max(all_durations),
                "p95_duration_ms": self._percentile(all_durations, 95),
                "p99_duration_ms": self._percentile(all_durations, 99)
            }
        
        return report
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    async def _monitoring_loop(self):
        """Background task for periodic logging"""
        while True:
            try:
                await asyncio.sleep(self.log_interval_seconds)
                
                # Log current metrics
                report = self.generate_report()
                self.logger.info(f"Performance Report: {json.dumps(report['summary'], indent=2)}")
                
                # Log any concerning metrics
                for server_name, server_metrics in self.server_metrics.items():
                    if server_metrics.error_rate > 10:  # > 10% error rate
                        self.logger.warning(f"High error rate for {server_name}: {server_metrics.error_rate:.2f}%")
                    
                    if server_metrics.uptime_percentage < 95:  # < 95% uptime
                        self.logger.warning(f"Low uptime for {server_name}: {server_metrics.uptime_percentage:.2f}%")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    async def _cleanup_loop(self):
        """Background task for cleanup operations"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old health check data
                cutoff_time = datetime.now() - timedelta(hours=24)
                for server_name in self.server_health:
                    # This is handled in record_health_check, but we can do additional cleanup here
                    pass
                
                # Log cleanup statistics
                self.logger.debug(f"Cleanup completed. Call history size: {len(self.call_history)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
