"""
HexagonML ModelManager MCP Server

This module provides a FastMCP server implementation for interacting with the HexagonML ModelManager API,
offering tools for model and usecase management, forecasting, and model card operations.

The server is now organized into modular components:
- config.py: Server configuration and environment management
- clients.py: ModelManager API client factory and context management
- utils.py: Common utility functions for response handling and validation
- validators.py: Payload validation functions
- base.py: Base classes and common patterns for tools
- model_tools.py: Model management tools (create, update, delete, metrics)
- usecase_tools.py: Usecase management tools (create, update, delete, list)
- modelcard_tools.py: Model card management tools
- forecasting_tools.py: Forecasting tools and operations
- main.py: Main server entry point and startup logic

Environment variables required:
- SECRET_KEY: Authentication key for the ModelManager API
- MM_API_BASE_URL: Base URL for the ModelManager API
- OUTPUT_DIR: Directory to store generated files
- HOST (optional): Host address for the MCP server (default: 0.0.0.0)
- PORT (optional): Port for the MCP server (default: 9000)

Usage:
    python main.py
"""

# Import the main configuration and server instance
from config import mcp, config
from clients import server_lifespan

# Import all tool modules to register their MCP tools
# This automatically registers all @mcp.tool decorated functions
import model_tools
import usecase_tools
import modelcard_tools
import forecasting_tools

# The server is now ready to run with all tools registered
# Use main.py to start the server with proper lifecycle management
