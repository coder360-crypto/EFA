"""
Environment Manager

Manages loading, switching, and coordination between different environments.
Provides a centralized interface for environment operations.
"""

from typing import Dict, List, Optional, Any, Type
import asyncio
import logging
from datetime import datetime
from enum import Enum

from .base_environment.environment import Environment, EnvironmentConfig, EnvironmentStatus
from .nextcloud_environment.nextcloud_env import NextcloudEnvironment
from .custom_environment.custom_env import CustomEnvironment


class EnvironmentManager:
    """Manager for handling multiple environments"""
    
    def __init__(self):
        self.environments: Dict[str, Environment] = {}
        self.active_environment: Optional[str] = None
        self.logger = logging.getLogger(__name__)
        
        # Registry of available environment types
        self.environment_types: Dict[str, Type[Environment]] = {
            "nextcloud": NextcloudEnvironment,
            "custom": CustomEnvironment
        }
    
    def register_environment_type(self, name: str, env_class: Type[Environment]) -> None:
        """
        Register a new environment type
        
        Args:
            name: Environment type name
            env_class: Environment class
        """
        self.environment_types[name] = env_class
        self.logger.info(f"Registered environment type: {name}")
    
    async def load_environment(
        self,
        name: str,
        env_type: str,
        config_data: Dict[str, Any],
        credentials: Optional[Dict[str, Any]] = None,
        auto_initialize: bool = True
    ) -> bool:
        """
        Load and optionally initialize an environment
        
        Args:
            name: Environment instance name
            env_type: Type of environment to create
            config_data: Configuration data for the environment
            credentials: Optional credentials
            auto_initialize: Whether to automatically initialize the environment
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if name in self.environments:
                self.logger.warning(f"Environment '{name}' already loaded, replacing it")
                await self.unload_environment(name)
            
            # Check if environment type is registered
            if env_type not in self.environment_types:
                raise ValueError(f"Unknown environment type: {env_type}")
            
            # Create environment configuration
            config = EnvironmentConfig(
                name=name,
                config_data=config_data,
                credentials=credentials,
                metadata={"type": env_type, "loaded_at": datetime.now().isoformat()}
            )
            
            # Create environment instance
            env_class = self.environment_types[env_type]
            environment = env_class(config)
            
            # Store environment
            self.environments[name] = environment
            
            # Initialize if requested
            if auto_initialize:
                success = await environment.safe_initialize()
                if not success:
                    # Remove from environments if initialization failed
                    del self.environments[name]
                    return False
            
            self.logger.info(f"Loaded environment '{name}' of type '{env_type}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load environment '{name}': {e}")
            return False
    
    async def unload_environment(self, name: str) -> bool:
        """
        Unload an environment
        
        Args:
            name: Environment name to unload
            
        Returns:
            True if unloaded successfully, False otherwise
        """
        try:
            if name not in self.environments:
                self.logger.warning(f"Environment '{name}' not found")
                return False
            
            environment = self.environments[name]
            
            # Shutdown environment if it's active
            if environment.is_initialized():
                await environment.safe_shutdown()
            
            # Remove from environments
            del self.environments[name]
            
            # Update active environment if necessary
            if self.active_environment == name:
                self.active_environment = None
            
            self.logger.info(f"Unloaded environment '{name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unload environment '{name}': {e}")
            return False
    
    async def switch_environment(self, name: str) -> bool:
        """
        Switch to a different active environment
        
        Args:
            name: Environment name to switch to
            
        Returns:
            True if switched successfully, False otherwise
        """
        try:
            if name not in self.environments:
                raise ValueError(f"Environment '{name}' not found")
            
            environment = self.environments[name]
            
            # Initialize environment if not already initialized
            if not environment.is_initialized():
                success = await environment.safe_initialize()
                if not success:
                    return False
            
            # Check if environment is healthy
            is_healthy = await environment.health_check()
            if not is_healthy:
                self.logger.warning(f"Environment '{name}' failed health check")
                return False
            
            self.active_environment = name
            self.logger.info(f"Switched to environment '{name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch to environment '{name}': {e}")
            return False
    
    def get_active_environment(self) -> Optional[Environment]:
        """
        Get the currently active environment
        
        Returns:
            Active environment instance or None
        """
        if self.active_environment and self.active_environment in self.environments:
            return self.environments[self.active_environment]
        return None
    
    def get_environment(self, name: str) -> Optional[Environment]:
        """
        Get an environment by name
        
        Args:
            name: Environment name
            
        Returns:
            Environment instance or None if not found
        """
        return self.environments.get(name)
    
    def list_environments(self) -> List[Dict[str, Any]]:
        """
        List all loaded environments
        
        Returns:
            List of environment information
        """
        environments = []
        
        for name, env in self.environments.items():
            env_info = env.get_info()
            environments.append({
                "name": name,
                "type": env_info.name,
                "description": env_info.description,
                "status": env_info.status.value,
                "is_active": name == self.active_environment,
                "capabilities": env_info.capabilities,
                "tool_count": len(env.get_available_tools()) if env.is_initialized() else 0
            })
        
        return environments
    
    def get_available_environment_types(self) -> List[str]:
        """
        Get list of available environment types
        
        Returns:
            List of environment type names
        """
        return list(self.environment_types.keys())
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        environment_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute a tool in an environment
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            environment_name: Environment to use (uses active if not specified)
            **kwargs: Additional execution parameters
            
        Returns:
            Tool execution result
        """
        # Determine which environment to use
        if environment_name:
            if environment_name not in self.environments:
                raise ValueError(f"Environment '{environment_name}' not found")
            environment = self.environments[environment_name]
        else:
            environment = self.get_active_environment()
            if not environment:
                raise RuntimeError("No active environment set")
        
        # Check if environment is active
        if not environment.is_active():
            raise RuntimeError(f"Environment '{environment.config.name}' is not active")
        
        # Check if tool is available
        available_tools = environment.get_available_tools()
        if tool_name not in available_tools:
            raise ValueError(f"Tool '{tool_name}' not available in environment '{environment.config.name}'")
        
        # Execute tool
        return await environment.execute_tool(tool_name, parameters, **kwargs)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health check on all environments
        
        Returns:
            Dictionary mapping environment names to health status
        """
        health_results = {}
        
        for name, env in self.environments.items():
            try:
                if env.is_initialized():
                    health_results[name] = await env.health_check()
                else:
                    health_results[name] = False
            except Exception as e:
                self.logger.error(f"Health check failed for environment '{name}': {e}")
                health_results[name] = False
        
        return health_results
    
    async def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all loaded environments
        
        Returns:
            Dictionary mapping environment names to initialization success
        """
        init_results = {}
        
        for name, env in self.environments.items():
            try:
                if not env.is_initialized():
                    init_results[name] = await env.safe_initialize()
                else:
                    init_results[name] = True
            except Exception as e:
                self.logger.error(f"Initialization failed for environment '{name}': {e}")
                init_results[name] = False
        
        return init_results
    
    async def shutdown_all(self) -> Dict[str, bool]:
        """
        Shutdown all environments
        
        Returns:
            Dictionary mapping environment names to shutdown success
        """
        shutdown_results = {}
        
        for name, env in self.environments.items():
            try:
                if env.is_initialized():
                    shutdown_results[name] = await env.safe_shutdown()
                else:
                    shutdown_results[name] = True
            except Exception as e:
                self.logger.error(f"Shutdown failed for environment '{name}': {e}")
                shutdown_results[name] = False
        
        # Clear active environment
        self.active_environment = None
        
        return shutdown_results
    
    def get_environment_tools(self, environment_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available tools from an environment
        
        Args:
            environment_name: Environment name (uses active if not specified)
            
        Returns:
            List of tool information
        """
        # Determine which environment to use
        if environment_name:
            if environment_name not in self.environments:
                raise ValueError(f"Environment '{environment_name}' not found")
            environment = self.environments[environment_name]
        else:
            environment = self.get_active_environment()
            if not environment:
                raise RuntimeError("No active environment set")
        
        if not environment.is_initialized():
            return []
        
        tools = []
        tool_names = environment.get_available_tools()
        
        for tool_name in tool_names:
            tool_info = environment.get_tool_info(tool_name)
            if tool_info:
                tools.append(tool_info)
            else:
                # Basic tool info if detailed info not available
                tools.append({
                    "name": tool_name,
                    "description": f"Tool: {tool_name}",
                    "environment": environment.config.name
                })
        
        return tools
    
    def get_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all available tools from all environments
        
        Returns:
            Dictionary mapping environment names to their tools
        """
        all_tools = {}
        
        for name, env in self.environments.items():
            if env.is_initialized():
                all_tools[name] = self.get_environment_tools(name)
            else:
                all_tools[name] = []
        
        return all_tools
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get manager statistics
        
        Returns:
            Dictionary with manager statistics
        """
        stats = {
            "total_environments": len(self.environments),
            "active_environment": self.active_environment,
            "available_types": list(self.environment_types.keys()),
            "environments": {}
        }
        
        # Environment-specific statistics
        for name, env in self.environments.items():
            env_stats = {
                "status": env.get_status().value,
                "initialized": env.is_initialized(),
                "tool_count": len(env.get_available_tools()) if env.is_initialized() else 0
            }
            
            # Add custom statistics if environment supports it
            if hasattr(env, 'get_statistics'):
                try:
                    env_stats.update(env.get_statistics())
                except Exception:
                    pass
            
            stats["environments"][name] = env_stats
        
        return stats
    
    async def reload_environment(self, name: str) -> bool:
        """
        Reload an environment (shutdown and reinitialize)
        
        Args:
            name: Environment name to reload
            
        Returns:
            True if reloaded successfully, False otherwise
        """
        try:
            if name not in self.environments:
                self.logger.error(f"Environment '{name}' not found")
                return False
            
            environment = self.environments[name]
            
            # Shutdown if initialized
            if environment.is_initialized():
                await environment.safe_shutdown()
            
            # Reinitialize
            success = await environment.safe_initialize()
            
            if success:
                self.logger.info(f"Reloaded environment '{name}' successfully")
            else:
                self.logger.error(f"Failed to reload environment '{name}'")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Exception during environment reload '{name}': {e}")
            return False
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown_all()
