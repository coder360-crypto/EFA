"""
Planning Tool

Provides DAG-based task planning capabilities for breaking down complex
tasks into manageable steps with dependencies.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Task:
    """Represents a task in the plan"""
    id: str
    name: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: Optional[timedelta] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    subtasks: List[str] = field(default_factory=list)
    assigned_tool: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class Plan:
    """Represents an execution plan"""
    id: str
    name: str
    description: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "created"
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlanningTool:
    """Tool for DAG-based task planning"""
    
    def __init__(self):
        self.plans: Dict[str, Plan] = {}
        self.logger = logging.getLogger(__name__)
    
    async def create_plan(
        self,
        name: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new execution plan
        
        Args:
            name: Plan name
            description: Plan description
            metadata: Optional metadata
            
        Returns:
            Plan ID
        """
        try:
            plan_id = f"plan_{uuid.uuid4().hex[:8]}"
            
            plan = Plan(
                id=plan_id,
                name=name,
                description=description,
                metadata=metadata or {}
            )
            
            self.plans[plan_id] = plan
            
            self.logger.info(f"Created plan: {plan_id} - {name}")
            return plan_id
            
        except Exception as e:
            self.logger.error(f"Failed to create plan: {e}")
            raise
    
    async def add_task(
        self,
        plan_id: str,
        name: str,
        description: str,
        dependencies: Optional[List[str]] = None,
        estimated_duration: Optional[int] = None,  # minutes
        priority: str = "medium",
        assigned_tool: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a task to a plan
        
        Args:
            plan_id: Plan ID
            name: Task name
            description: Task description
            dependencies: List of task IDs this task depends on
            estimated_duration: Estimated duration in minutes
            priority: Task priority ("low", "medium", "high", "urgent")
            assigned_tool: Tool to use for this task
            metadata: Optional metadata
            
        Returns:
            Task ID
        """
        try:
            if plan_id not in self.plans:
                raise ValueError(f"Plan {plan_id} not found")
            
            plan = self.plans[plan_id]
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            
            # Validate dependencies
            if dependencies:
                for dep_id in dependencies:
                    if dep_id not in plan.tasks:
                        raise ValueError(f"Dependency task {dep_id} not found in plan")
            
            # Convert priority string to enum
            try:
                priority_enum = TaskPriority[priority.upper()]
            except KeyError:
                priority_enum = TaskPriority.MEDIUM
            
            # Convert duration to timedelta
            duration = timedelta(minutes=estimated_duration) if estimated_duration else None
            
            task = Task(
                id=task_id,
                name=name,
                description=description,
                dependencies=dependencies or [],
                estimated_duration=duration,
                priority=priority_enum,
                assigned_tool=assigned_tool,
                metadata=metadata or {}
            )
            
            plan.tasks[task_id] = task
            
            # Update task status based on dependencies
            await self._update_task_status(plan_id, task_id)
            
            self.logger.debug(f"Added task {task_id} to plan {plan_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to add task: {e}")
            raise
    
    async def add_subtask(
        self,
        plan_id: str,
        parent_task_id: str,
        name: str,
        description: str,
        **kwargs
    ) -> str:
        """
        Add a subtask to an existing task
        
        Args:
            plan_id: Plan ID
            parent_task_id: Parent task ID
            name: Subtask name
            description: Subtask description
            **kwargs: Additional task parameters
            
        Returns:
            Subtask ID
        """
        try:
            if plan_id not in self.plans:
                raise ValueError(f"Plan {plan_id} not found")
            
            plan = self.plans[plan_id]
            if parent_task_id not in plan.tasks:
                raise ValueError(f"Parent task {parent_task_id} not found")
            
            # Create subtask
            subtask_id = await self.add_task(
                plan_id=plan_id,
                name=name,
                description=description,
                **kwargs
            )
            
            # Add to parent's subtasks list
            parent_task = plan.tasks[parent_task_id]
            parent_task.subtasks.append(subtask_id)
            
            self.logger.debug(f"Added subtask {subtask_id} to task {parent_task_id}")
            return subtask_id
            
        except Exception as e:
            self.logger.error(f"Failed to add subtask: {e}")
            raise
    
    async def get_ready_tasks(self, plan_id: str) -> List[Task]:
        """
        Get tasks that are ready for execution
        
        Args:
            plan_id: Plan ID
            
        Returns:
            List of ready tasks
        """
        try:
            if plan_id not in self.plans:
                raise ValueError(f"Plan {plan_id} not found")
            
            plan = self.plans[plan_id]
            ready_tasks = []
            
            for task in plan.tasks.values():
                if task.status == TaskStatus.READY:
                    ready_tasks.append(task)
            
            # Sort by priority and creation time
            ready_tasks.sort(
                key=lambda t: (t.priority.value, t.created_at),
                reverse=True
            )
            
            return ready_tasks
            
        except Exception as e:
            self.logger.error(f"Failed to get ready tasks: {e}")
            return []
    
    async def start_task(self, plan_id: str, task_id: str) -> bool:
        """
        Mark a task as started
        
        Args:
            plan_id: Plan ID
            task_id: Task ID
            
        Returns:
            True if task started successfully
        """
        try:
            if plan_id not in self.plans:
                raise ValueError(f"Plan {plan_id} not found")
            
            plan = self.plans[plan_id]
            if task_id not in plan.tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = plan.tasks[task_id]
            
            if task.status != TaskStatus.READY:
                raise ValueError(f"Task {task_id} is not ready to start (status: {task.status})")
            
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now()
            
            # Update plan status
            if plan.status == "created":
                plan.status = "in_progress"
                plan.started_at = datetime.now()
            
            self.logger.debug(f"Started task {task_id} in plan {plan_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start task: {e}")
            return False
    
    async def complete_task(
        self,
        plan_id: str,
        task_id: str,
        result: Optional[Any] = None
    ) -> bool:
        """
        Mark a task as completed
        
        Args:
            plan_id: Plan ID
            task_id: Task ID
            result: Task result
            
        Returns:
            True if task completed successfully
        """
        try:
            if plan_id not in self.plans:
                raise ValueError(f"Plan {plan_id} not found")
            
            plan = self.plans[plan_id]
            if task_id not in plan.tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = plan.tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # Update dependent tasks
            await self._update_dependent_tasks(plan_id, task_id)
            
            # Check if all tasks are completed
            await self._check_plan_completion(plan_id)
            
            self.logger.debug(f"Completed task {task_id} in plan {plan_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to complete task: {e}")
            return False
    
    async def fail_task(
        self,
        plan_id: str,
        task_id: str,
        error: str,
        retry: bool = True
    ) -> bool:
        """
        Mark a task as failed
        
        Args:
            plan_id: Plan ID
            task_id: Task ID
            error: Error message
            retry: Whether to retry the task
            
        Returns:
            True if task failure handled successfully
        """
        try:
            if plan_id not in self.plans:
                raise ValueError(f"Plan {plan_id} not found")
            
            plan = self.plans[plan_id]
            if task_id not in plan.tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = plan.tasks[task_id]
            task.error = error
            task.retry_count += 1
            
            # Determine if we should retry
            if retry and task.retry_count <= task.max_retries:
                task.status = TaskStatus.READY
                self.logger.info(f"Task {task_id} failed, retrying (attempt {task.retry_count})")
            else:
                task.status = TaskStatus.FAILED
                # Handle dependent tasks
                await self._handle_failed_task_dependencies(plan_id, task_id)
                self.logger.error(f"Task {task_id} failed permanently: {error}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to handle task failure: {e}")
            return False
    
    async def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """
        Get comprehensive status of a plan
        
        Args:
            plan_id: Plan ID
            
        Returns:
            Plan status information
        """
        try:
            if plan_id not in self.plans:
                raise ValueError(f"Plan {plan_id} not found")
            
            plan = self.plans[plan_id]
            
            # Count tasks by status
            status_counts = {}
            for status in TaskStatus:
                status_counts[status.value] = 0
            
            for task in plan.tasks.values():
                status_counts[task.status.value] += 1
            
            # Calculate progress
            total_tasks = len(plan.tasks)
            completed_tasks = status_counts[TaskStatus.COMPLETED.value]
            progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            # Estimate remaining time
            remaining_time = await self._estimate_remaining_time(plan_id)
            
            status = {
                "plan_id": plan_id,
                "name": plan.name,
                "description": plan.description,
                "status": plan.status,
                "created_at": plan.created_at.isoformat(),
                "started_at": plan.started_at.isoformat() if plan.started_at else None,
                "completed_at": plan.completed_at.isoformat() if plan.completed_at else None,
                "total_tasks": total_tasks,
                "task_status_counts": status_counts,
                "progress_percentage": progress,
                "estimated_remaining_time": remaining_time,
                "critical_path": await self._get_critical_path(plan_id)
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get plan status: {e}")
            return {}
    
    async def visualize_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Generate visualization data for the plan
        
        Args:
            plan_id: Plan ID
            
        Returns:
            Visualization data (nodes and edges)
        """
        try:
            if plan_id not in self.plans:
                raise ValueError(f"Plan {plan_id} not found")
            
            plan = self.plans[plan_id]
            
            nodes = []
            edges = []
            
            # Create nodes for tasks
            for task in plan.tasks.values():
                node = {
                    "id": task.id,
                    "label": task.name,
                    "description": task.description,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "estimated_duration": task.estimated_duration.total_seconds() / 60 if task.estimated_duration else None,
                    "assigned_tool": task.assigned_tool,
                    "subtask_count": len(task.subtasks)
                }
                nodes.append(node)
            
            # Create edges for dependencies
            for task in plan.tasks.values():
                for dep_id in task.dependencies:
                    edge = {
                        "from": dep_id,
                        "to": task.id,
                        "type": "dependency"
                    }
                    edges.append(edge)
                
                # Add subtask edges
                for subtask_id in task.subtasks:
                    edge = {
                        "from": task.id,
                        "to": subtask_id,
                        "type": "subtask"
                    }
                    edges.append(edge)
            
            return {
                "plan_id": plan_id,
                "plan_name": plan.name,
                "nodes": nodes,
                "edges": edges,
                "layout_suggestions": await self._suggest_layout(nodes, edges)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to visualize plan: {e}")
            return {}
    
    async def _update_task_status(self, plan_id: str, task_id: str) -> None:
        """Update task status based on dependencies"""
        plan = self.plans[plan_id]
        task = plan.tasks[task_id]
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.IN_PROGRESS]:
            return
        
        # Check if all dependencies are completed
        all_deps_completed = True
        for dep_id in task.dependencies:
            if dep_id in plan.tasks:
                dep_task = plan.tasks[dep_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    all_deps_completed = False
                    break
            else:
                all_deps_completed = False
                break
        
        if all_deps_completed:
            task.status = TaskStatus.READY
        else:
            task.status = TaskStatus.PENDING
    
    async def _update_dependent_tasks(self, plan_id: str, completed_task_id: str) -> None:
        """Update tasks that depend on the completed task"""
        plan = self.plans[plan_id]
        
        for task in plan.tasks.values():
            if completed_task_id in task.dependencies:
                await self._update_task_status(plan_id, task.id)
    
    async def _handle_failed_task_dependencies(self, plan_id: str, failed_task_id: str) -> None:
        """Handle tasks that depend on a failed task"""
        plan = self.plans[plan_id]
        
        def mark_dependent_as_cancelled(task_id: str, visited: Set[str]) -> None:
            if task_id in visited:
                return
            visited.add(task_id)
            
            for task in plan.tasks.values():
                if task_id in task.dependencies and task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    task.status = TaskStatus.CANCELLED
                    task.error = f"Cancelled due to failed dependency: {task_id}"
                    # Recursively cancel dependent tasks
                    mark_dependent_as_cancelled(task.id, visited)
        
        mark_dependent_as_cancelled(failed_task_id, set())
    
    async def _check_plan_completion(self, plan_id: str) -> None:
        """Check if plan is completed"""
        plan = self.plans[plan_id]
        
        all_completed = True
        has_failed = False
        
        for task in plan.tasks.values():
            if task.status in [TaskStatus.PENDING, TaskStatus.READY, TaskStatus.IN_PROGRESS]:
                all_completed = False
                break
            elif task.status == TaskStatus.FAILED:
                has_failed = True
        
        if all_completed:
            if has_failed:
                plan.status = "completed_with_failures"
            else:
                plan.status = "completed"
            plan.completed_at = datetime.now()
    
    async def _estimate_remaining_time(self, plan_id: str) -> Optional[float]:
        """Estimate remaining time for plan completion"""
        plan = self.plans[plan_id]
        
        remaining_duration = timedelta()
        has_estimates = False
        
        for task in plan.tasks.values():
            if task.status in [TaskStatus.PENDING, TaskStatus.READY, TaskStatus.IN_PROGRESS]:
                if task.estimated_duration:
                    remaining_duration += task.estimated_duration
                    has_estimates = True
        
        return remaining_duration.total_seconds() / 60 if has_estimates else None  # Return minutes
    
    async def _get_critical_path(self, plan_id: str) -> List[str]:
        """Calculate critical path through the plan"""
        plan = self.plans[plan_id]
        
        # Simplified critical path calculation
        # In a real implementation, you'd use proper critical path method
        
        # Find tasks with no dependencies (start tasks)
        start_tasks = [task.id for task in plan.tasks.values() if not task.dependencies]
        
        # Find tasks with no dependents (end tasks)
        all_dependencies = set()
        for task in plan.tasks.values():
            all_dependencies.update(task.dependencies)
        
        end_tasks = [task.id for task in plan.tasks.values() if task.id not in all_dependencies]
        
        # Simple path from start to end (this is a simplified version)
        critical_path = []
        if start_tasks and end_tasks:
            # Just return one path for now
            current = start_tasks[0]
            critical_path.append(current)
            
            # Follow dependencies forward
            visited = set()
            while current and current not in visited:
                visited.add(current)
                # Find a task that depends on current
                next_task = None
                for task in plan.tasks.values():
                    if current in task.dependencies:
                        next_task = task.id
                        break
                
                if next_task:
                    critical_path.append(next_task)
                    current = next_task
                else:
                    break
        
        return critical_path
    
    async def _suggest_layout(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
        """Suggest layout parameters for visualization"""
        return {
            "type": "hierarchical",
            "direction": "top-bottom",
            "node_spacing": 100,
            "level_spacing": 150,
            "recommendations": [
                "Use hierarchical layout for dependency visualization",
                "Color nodes by status",
                "Size nodes by priority or estimated duration"
            ]
        }
    
    async def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan"""
        try:
            if plan_id in self.plans:
                del self.plans[plan_id]
                self.logger.info(f"Deleted plan {plan_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete plan: {e}")
            return False
    
    async def list_plans(self) -> List[Dict[str, Any]]:
        """List all plans"""
        try:
            plan_list = []
            for plan in self.plans.values():
                plan_info = {
                    "id": plan.id,
                    "name": plan.name,
                    "description": plan.description,
                    "status": plan.status,
                    "task_count": len(plan.tasks),
                    "created_at": plan.created_at.isoformat(),
                    "started_at": plan.started_at.isoformat() if plan.started_at else None,
                    "completed_at": plan.completed_at.isoformat() if plan.completed_at else None
                }
                plan_list.append(plan_info)
            
            return plan_list
        except Exception as e:
            self.logger.error(f"Failed to list plans: {e}")
            return []
