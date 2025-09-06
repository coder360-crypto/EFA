#!/usr/bin/env python3
"""
EFA Backend Main Entry Point

Enhanced Function Agent Backend - Main application entry point.
Initializes and starts the MCP server with LLM adapters and environments.
"""

import asyncio
import logging
import os
import sys
import signal
from pathlib import Path
from typing import Optional
import yaml
import argparse

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from llm_core.llm_interface import LLMConfig
from llm_core.adapters import OpenRouterAdapter
from environments.environment_manager import EnvironmentManager
from environments.base_environment.environment import EnvironmentConfig
from server.mcp_server import MCPServer


class EFABackend:
    """Enhanced Function Agent Backend Application"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = {}
        self.credentials = {}
        self.server: Optional[MCPServer] = None
        self.environment_manager: Optional[EnvironmentManager] = None
        self.logger = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        if self.server:
            asyncio.create_task(self.shutdown())
    
    async def initialize(self):
        """Initialize the EFA Backend"""
        try:
            # Load configuration
            await self._load_config()
            
            # Setup logging
            self._setup_logging()
            
            self.logger.info("Starting EFA Backend initialization...")
            
            # Create data directories
            await self._create_directories()
            
            # Initialize environment manager
            self.environment_manager = EnvironmentManager()
            
            # Load environments
            await self._load_environments()
            
            # Initialize MCP server
            await self._initialize_server()
            
            self.logger.info("EFA Backend initialized successfully")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Initialization failed: {e}")
            else:
                print(f"Initialization failed: {e}")
            raise
    
    async def start(self):
        """Start the EFA Backend server"""
        try:
            if not self.server:
                raise RuntimeError("Server not initialized. Call initialize() first.")
            
            self.logger.info("Starting EFA Backend server...")
            await self.server.start()
            
            host = self.config.get("server", {}).get("host", "localhost")
            port = self.config.get("server", {}).get("port", 8000)
            
            self.logger.info(f"EFA Backend server started successfully")
            self.logger.info(f"Server available at:")
            self.logger.info(f"  HTTP: http://{host}:{port}")
            self.logger.info(f"  WebSocket: ws://{host}:{port}/ws")
            self.logger.info(f"  Health: http://{host}:{port}/health")
            self.logger.info(f"  Status: http://{host}:{port}/status")
            
            # Keep the server running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the EFA Backend"""
        if self.logger:
            self.logger.info("Shutting down EFA Backend...")
        
        try:
            # Stop server
            if self.server:
                await self.server.stop()
                self.server = None
            
            # Shutdown environments
            if self.environment_manager:
                await self.environment_manager.shutdown_all()
                self.environment_manager = None
            
            if self.logger:
                self.logger.info("EFA Backend shutdown complete")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during shutdown: {e}")
            else:
                print(f"Error during shutdown: {e}")
    
    async def _load_config(self):
        """Load configuration files"""
        # Load main config
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load credentials
        credentials_path = config_file.parent / "credentials.yaml"
        if credentials_path.exists():
            with open(credentials_path, 'r') as f:
                self.credentials = yaml.safe_load(f)
        else:
            print(f"Warning: Credentials file not found: {credentials_path}")
            self.credentials = {}
        
        # Resolve environment variables in credentials
        self._resolve_env_vars()
    
    def _resolve_env_vars(self):
        """Resolve environment variables in configuration"""
        def resolve_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.getenv(env_var, "")
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            return value
        
        self.credentials = resolve_value(self.credentials)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get("logging", {})
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        log_level = getattr(logging, log_config.get("level", "INFO").upper())
        log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Setup root logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[]
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(console_handler)
        
        # File handler
        if log_config.get("file_logging", False):
            log_file = log_config.get("log_file", "logs/efa_backend.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(file_handler)
        
        self.logger = logging.getLogger(__name__)
    
    async def _create_directories(self):
        """Create necessary directories"""
        data_config = self.config.get("data", {})
        directories = [
            data_config.get("base_dir", "data"),
            data_config.get("logs_dir", "logs"),
            data_config.get("cache_dir", "cache"),
            data_config.get("temp_dir", "temp"),
            data_config.get("backup_dir", "backups")
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")
    
    async def _load_environments(self):
        """Load and configure environments"""
        env_config = self.config.get("environments", {})
        env_credentials = self.credentials.get("environments", {})
        
        # Load Nextcloud environment
        if env_config.get("nextcloud", {}).get("enabled", False):
            await self._load_nextcloud_environment(env_config, env_credentials)
        
        # Load Custom environment
        if env_config.get("custom", {}).get("enabled", False):
            await self._load_custom_environment(env_config, env_credentials)
    
    async def _load_nextcloud_environment(self, env_config, env_credentials):
        """Load Nextcloud environment"""
        try:
            nextcloud_config = env_config.get("nextcloud", {})
            nextcloud_creds = env_credentials.get("nextcloud", {})
            
            if not nextcloud_creds.get("username") or not nextcloud_creds.get("password"):
                self.logger.warning("Nextcloud credentials not found, skipping Nextcloud environment")
                return
            
            config_data = {
                "base_url": nextcloud_creds.get("base_url", ""),
                "timeout": nextcloud_config.get("timeout", 30),
                "verify_ssl": nextcloud_config.get("verify_ssl", True)
            }
            
            credentials = {
                "username": nextcloud_creds.get("username"),
                "password": nextcloud_creds.get("password")
            }
            
            success = await self.environment_manager.load_environment(
                name="nextcloud",
                env_type="nextcloud",
                config_data=config_data,
                credentials=credentials,
                auto_initialize=True
            )
            
            if success:
                self.logger.info("Nextcloud environment loaded successfully")
            else:
                self.logger.error("Failed to load Nextcloud environment")
                
        except Exception as e:
            self.logger.error(f"Error loading Nextcloud environment: {e}")
    
    async def _load_custom_environment(self, env_config, env_credentials):
        """Load Custom environment"""
        try:
            custom_config = env_config.get("custom", {})
            custom_creds = env_credentials.get("custom", {})
            
            config_data = {
                "debug_mode": custom_config.get("debug_mode", False),
                "timeout": custom_config.get("timeout", 30),
                "custom_setting": "default_value"
            }
            
            credentials = {
                "api_key": custom_creds.get("api_key", ""),
                "secret": custom_creds.get("secret", "")
            }
            
            success = await self.environment_manager.load_environment(
                name="custom",
                env_type="custom",
                config_data=config_data,
                credentials=credentials,
                auto_initialize=True
            )
            
            if success:
                self.logger.info("Custom environment loaded successfully")
            else:
                self.logger.error("Failed to load Custom environment")
                
        except Exception as e:
            self.logger.error(f"Error loading Custom environment: {e}")
    
    async def _initialize_server(self):
        """Initialize the MCP server"""
        server_config = self.config.get("server", {})
        
        host = server_config.get("host", "localhost")
        port = server_config.get("port", 8000)
        
        self.server = MCPServer(
            host=host,
            port=port,
            environment_manager=self.environment_manager
        )
        
        self.logger.info(f"MCP server initialized on {host}:{port}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="EFA Backend - Enhanced Function Agent")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="EFA Backend 1.0.0"
    )
    
    args = parser.parse_args()
    
    # Create and run the application
    app = EFABackend(config_path=args.config)
    
    try:
        await app.initialize()
        await app.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication interrupted")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
