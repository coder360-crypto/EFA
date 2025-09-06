"""
Nextcloud Environment

Environment implementation for Nextcloud integration.
Provides access to Nextcloud services and tools.
"""

from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
import logging
from datetime import datetime

from ..base_environment.environment import Environment, EnvironmentInfo, EnvironmentStatus
from ..base_environment.tools_registry import ToolsRegistry
from .nextcloud_tools import NextcloudTools


class NextcloudEnvironment(Environment):
    """Nextcloud environment implementation"""
    
    def __init__(self, config):
        super().__init__(config)
        self.nextcloud_tools = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = ""
        self.username = ""
        self.password = ""
    
    async def initialize(self) -> bool:
        """Initialize the Nextcloud environment"""
        try:
            # Get configuration
            self.base_url = self.get_config_value("base_url", "").rstrip("/")
            self.username = self.get_credential("username", "")
            self.password = self.get_credential("password", "")
            
            if not all([self.base_url, self.username, self.password]):
                self.logger.error("Missing required Nextcloud configuration")
                return False
            
            # Create HTTP session
            auth = aiohttp.BasicAuth(self.username, self.password)
            self.session = aiohttp.ClientSession(auth=auth)
            
            # Test connection
            is_connected = await self._test_connection()
            if not is_connected:
                self.logger.error("Failed to connect to Nextcloud")
                return False
            
            # Initialize tools registry
            self.tools_registry = ToolsRegistry(self.config.name)
            
            # Initialize Nextcloud tools
            self.nextcloud_tools = NextcloudTools(
                base_url=self.base_url,
                session=self.session,
                logger=self.logger
            )
            
            # Register tools
            await self._register_tools()
            
            self.logger.info("Nextcloud environment initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Nextcloud environment: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the Nextcloud environment"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.nextcloud_tools = None
            self.tools_registry = None
            
            self.logger.info("Nextcloud environment shutdown successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to shutdown Nextcloud environment: {e}")
            return False
    
    def get_info(self) -> EnvironmentInfo:
        """Get information about the Nextcloud environment"""
        return EnvironmentInfo(
            name="nextcloud",
            description="Nextcloud cloud storage and collaboration environment",
            version="1.0.0",
            capabilities=[
                "file_management",
                "directory_operations", 
                "file_sharing",
                "calendar_access",
                "contacts_management",
                "collaborative_editing"
            ],
            required_config=[
                "base_url"
            ],
            optional_config=[
                "timeout",
                "verify_ssl",
                "app_password"
            ],
            status=self.status,
            metadata={
                "provider": "Nextcloud",
                "api_version": "WebDAV/CalDAV/CardDAV",
                "base_url": self.base_url if hasattr(self, 'base_url') else None
            }
        )
    
    async def health_check(self) -> bool:
        """Check if Nextcloud environment is healthy"""
        if not self.session:
            return False
        
        return await self._test_connection()
    
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
        """Execute a tool in the Nextcloud environment"""
        if not self.tools_registry:
            raise RuntimeError("Tools registry not initialized")
        
        return await self.tools_registry.execute_tool(tool_name, parameters, **kwargs)
    
    async def _test_connection(self) -> bool:
        """Test connection to Nextcloud"""
        try:
            if not self.session:
                return False
            
            # Test with a simple WebDAV request
            url = f"{self.base_url}/remote.php/dav/files/{self.username}/"
            
            async with self.session.request("PROPFIND", url) as response:
                return response.status in [200, 207]  # 207 is Multi-Status for WebDAV
                
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    async def _register_tools(self) -> None:
        """Register Nextcloud tools with the tools registry"""
        if not self.tools_registry or not self.nextcloud_tools:
            return
        
        # File management tools
        self.tools_registry.register_tool(
            name="list_files",
            func=self.nextcloud_tools.list_files,
            description="List files and directories in a Nextcloud path",
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to list (default: root directory)",
                        "default": "/"
                    }
                },
                "required": []
            }
        )
        
        self.tools_registry.register_tool(
            name="upload_file",
            func=self.nextcloud_tools.upload_file,
            description="Upload a file to Nextcloud",
            schema={
                "type": "object",
                "properties": {
                    "local_path": {
                        "type": "string",
                        "description": "Local file path to upload"
                    },
                    "remote_path": {
                        "type": "string",
                        "description": "Remote path where file should be uploaded"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Whether to overwrite existing file",
                        "default": False
                    }
                },
                "required": ["local_path", "remote_path"]
            }
        )
        
        self.tools_registry.register_tool(
            name="download_file",
            func=self.nextcloud_tools.download_file,
            description="Download a file from Nextcloud",
            schema={
                "type": "object",
                "properties": {
                    "remote_path": {
                        "type": "string",
                        "description": "Remote file path to download"
                    },
                    "local_path": {
                        "type": "string",
                        "description": "Local path where file should be saved"
                    }
                },
                "required": ["remote_path", "local_path"]
            }
        )
        
        self.tools_registry.register_tool(
            name="delete_file",
            func=self.nextcloud_tools.delete_file,
            description="Delete a file or directory from Nextcloud",
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to delete"
                    }
                },
                "required": ["path"]
            }
        )
        
        self.tools_registry.register_tool(
            name="create_directory",
            func=self.nextcloud_tools.create_directory,
            description="Create a directory in Nextcloud",
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to create"
                    }
                },
                "required": ["path"]
            }
        )
        
        self.tools_registry.register_tool(
            name="move_file",
            func=self.nextcloud_tools.move_file,
            description="Move or rename a file/directory in Nextcloud",
            schema={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Source path"
                    },
                    "destination_path": {
                        "type": "string",
                        "description": "Destination path"
                    }
                },
                "required": ["source_path", "destination_path"]
            }
        )
        
        self.tools_registry.register_tool(
            name="copy_file",
            func=self.nextcloud_tools.copy_file,
            description="Copy a file/directory in Nextcloud",
            schema={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Source path"
                    },
                    "destination_path": {
                        "type": "string",
                        "description": "Destination path"
                    }
                },
                "required": ["source_path", "destination_path"]
            }
        )
        
        self.tools_registry.register_tool(
            name="get_file_info",
            func=self.nextcloud_tools.get_file_info,
            description="Get information about a file or directory",
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to get information about"
                    }
                },
                "required": ["path"]
            }
        )
        
        self.tools_registry.register_tool(
            name="share_file",
            func=self.nextcloud_tools.share_file,
            description="Create a share link for a file or directory",
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to share"
                    },
                    "password": {
                        "type": "string",
                        "description": "Optional password for the share"
                    },
                    "expire_date": {
                        "type": "string",
                        "description": "Optional expiration date (YYYY-MM-DD)"
                    },
                    "permissions": {
                        "type": "integer",
                        "description": "Share permissions (1=read, 2=update, 4=create, 8=delete, 16=share)",
                        "default": 1
                    }
                },
                "required": ["path"]
            }
        )
        
        self.tools_registry.register_tool(
            name="search_files",
            func=self.nextcloud_tools.search_files,
            description="Search for files in Nextcloud",
            schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to search in (default: root)",
                        "default": "/"
                    }
                },
                "required": ["query"]
            }
        )
        
        self.logger.info("Registered Nextcloud tools successfully")
