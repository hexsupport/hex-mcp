"""
Configuration and setup for the ModelManager MCP Server.

This module handles environment configuration, server initialization,
and provides the main FastMCP server instance.
"""

import os
from dotenv import load_dotenv
from fastmcp import FastMCP

# Load environment variables from .env file
load_dotenv()

class ServerConfig:
    """Configuration settings for the ModelManager MCP Server."""
    
    def __init__(self):
        self.secret_key = os.getenv("SECRET_KEY")
        self.api_base_url = os.getenv("MM_API_BASE_URL")
        self.output_dir = os.getenv("OUTPUT_DIR")
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = os.getenv("PORT", "9000")
        
        # Validate required environment variables
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required environment variables are set."""
        required_vars = ["SECRET_KEY", "MM_API_BASE_URL", "OUTPUT_DIR"]
        # Map environment variable names to attribute names
        attr_mapping = {
            "SECRET_KEY": "secret_key",
            "MM_API_BASE_URL": "api_base_url", 
            "OUTPUT_DIR": "output_dir"
        }
        missing_vars = [var for var in required_vars if not getattr(self, attr_mapping[var])]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def get_env_status(self) -> dict:
        """Get the status of environment variables (without revealing sensitive values)."""
        return {
            "SECRET_KEY": "*****" if self.secret_key else "NOT SET",
            "MM_API_BASE_URL": self.api_base_url or "NOT SET",
            "OUTPUT_DIR": self.output_dir or "NOT SET",
            "HOST": self.host,
            "PORT": self.port
        }

# Initialize server configuration
config = ServerConfig()

# Create FastMCP server instance
mcp = FastMCP("hexagonml-modelmanager")
