"""
Base Environment

Abstract base class for all environment implementations.
Defines the interface that all environments must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging


class EnvironmentStatus(Enum):
    """Environment status states"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class EnvironmentInfo:
    """Information about an environment"""
    name: str
    description: str
    version: str
    capabilities: List[str]
    required_config: List[str]
    optional_config: List[str]
    status: EnvironmentStatus
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EnvironmentConfig:
    """Configuration for an environment"""
    name: str
    config_data: Dict[str, Any]
    credentials: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class Environment(ABC):
    """Abstract base class for all environments"""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.status = EnvironmentStatus.INACTIVE
        self.tools_registry = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the environment
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Shutdown the environment and clean up resources
        
        Returns:
            True if shutdown successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_info(self) -> EnvironmentInfo:
        """
        Get information about this environment
        
        Returns:
            Environment information
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the environment is healthy and operational
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    @abstractmethod
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names in this environment
        
        Returns:
            List of tool names
        """
        pass
    
    @abstractmethod
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        **kwargs
    ) -> Any:
        """
        Execute a tool in this environment
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            **kwargs: Additional execution parameters
            
        Returns:
            Tool execution result
        """
        pass
    
    def validate_config(self) -> tuple[bool, Optional[str]]:
        """
        Validate the environment configuration
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            env_info = self.get_info()
            config_data = self.config.config_data
            
            # Check required configuration
            for required_key in env_info.required_config:
                if required_key not in config_data:
                    return False, f"Required configuration key '{required_key}' is missing"
                
                # Check for empty values
                if not config_data[required_key]:
                    return False, f"Required configuration key '{required_key}' is empty"
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def get_status(self) -> EnvironmentStatus:
        """
        Get current environment status
        
        Returns:
            Current status
        """
        return self.status
    
    def is_active(self) -> bool:
        """
        Check if environment is active
        
        Returns:
            True if active, False otherwise
        """
        return self.status == EnvironmentStatus.ACTIVE
    
    def is_initialized(self) -> bool:
        """
        Check if environment has been initialized
        
        Returns:
            True if initialized, False otherwise
        """
        return self._initialized
    
    async def safe_initialize(self) -> bool:
        """
        Safely initialize the environment with error handling
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            self.logger.warning("Environment already initialized")
            return True
        
        try:
            self.status = EnvironmentStatus.INITIALIZING
            self.logger.info(f"Initializing environment: {self.config.name}")
            
            # Validate configuration first
            is_valid, error_message = self.validate_config()
            if not is_valid:
                self.logger.error(f"Configuration validation failed: {error_message}")
                self.status = EnvironmentStatus.ERROR
                return False
            
            # Initialize the environment
            success = await self.initialize()
            
            if success:
                self.status = EnvironmentStatus.ACTIVE
                self._initialized = True
                self.logger.info(f"Environment initialized successfully: {self.config.name}")
            else:
                self.status = EnvironmentStatus.ERROR
                self.logger.error(f"Environment initialization failed: {self.config.name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Exception during environment initialization: {e}")
            self.status = EnvironmentStatus.ERROR
            return False
    
    async def safe_shutdown(self) -> bool:
        """
        Safely shutdown the environment with error handling
        
        Returns:
            True if shutdown successful, False otherwise
        """
        if not self._initialized:
            self.logger.warning("Environment not initialized, nothing to shutdown")
            return True
        
        try:
            self.status = EnvironmentStatus.SHUTTING_DOWN
            self.logger.info(f"Shutting down environment: {self.config.name}")
            
            success = await self.shutdown()
            
            if success:
                self.status = EnvironmentStatus.INACTIVE
                self._initialized = False
                self.logger.info(f"Environment shutdown successfully: {self.config.name}")
            else:
                self.status = EnvironmentStatus.ERROR
                self.logger.error(f"Environment shutdown failed: {self.config.name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Exception during environment shutdown: {e}")
            self.status = EnvironmentStatus.ERROR
            return False
    
    async def safe_execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        **kwargs
    ) -> tuple[bool, Any, Optional[str]]:
        """
        Safely execute a tool with error handling
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            **kwargs: Additional execution parameters
            
        Returns:
            Tuple of (success, result, error_message)
        """
        if not self.is_active():
            return False, None, "Environment is not active"
        
        if tool_name not in self.get_available_tools():
            return False, None, f"Tool '{tool_name}' not available in this environment"
        
        try:
            result = await self.execute_tool(tool_name, parameters, **kwargs)
            return True, result, None
            
        except Exception as e:
            self.logger.error(f"Tool execution failed for '{tool_name}': {e}")
            return False, None, str(e)
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.config_data.get(key, default)
    
    def get_credential(self, key: str, default: Any = None) -> Any:
        """
        Get a credential value
        
        Args:
            key: Credential key
            default: Default value if key not found
            
        Returns:
            Credential value or default
        """
        if not self.config.credentials:
            return default
        return self.config.credentials.get(key, default)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration values
        
        Args:
            updates: Dictionary of configuration updates
        """
        self.config.config_data.update(updates)
        self.logger.debug(f"Configuration updated for environment: {self.config.name}")
    
    def update_credentials(self, updates: Dict[str, Any]) -> None:
        """
        Update credential values
        
        Args:
            updates: Dictionary of credential updates
        """
        if not self.config.credentials:
            self.config.credentials = {}
        self.config.credentials.update(updates)
        self.logger.debug(f"Credentials updated for environment: {self.config.name}")
    
    def register_tool(
        self,
        name: str,
        func: Callable,
        description: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a tool with this environment
        
        Args:
            name: Tool name
            func: Tool function
            description: Tool description
            schema: Tool schema
            
        Returns:
            True if registration successful
        """
        if self.tools_registry:
            return self.tools_registry.register_tool(name, func, description, schema)
        else:
            self.logger.error("Tools registry not available")
            return False
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool from this environment
        
        Args:
            name: Tool name
            
        Returns:
            True if unregistration successful
        """
        if self.tools_registry:
            return self.tools_registry.unregister_tool(name)
        else:
            self.logger.error("Tools registry not available")
            return False
    
    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a tool
        
        Args:
            name: Tool name
            
        Returns:
            Tool information if found, None otherwise
        """
        if self.tools_registry:
            return self.tools_registry.get_tool_info(name)
        else:
            return None
    
    def __str__(self) -> str:
        """String representation of the environment"""
        return f"{self.__class__.__name__}({self.config.name}, status={self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the environment"""
        return (f"{self.__class__.__name__}("
                f"name='{self.config.name}', "
                f"status={self.status.value}, "
                f"initialized={self._initialized})")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.safe_initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.safe_shutdown()
