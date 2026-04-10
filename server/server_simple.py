"""
Simple ModelManager MCP Server entry point for fastmcp CLI.

This file can be used with: fastmcp run server/server_simple.py
"""

# Import all the modular components
from config import mcp, config
from clients import server_lifespan

# Import all tool modules to register their MCP tools
# import tools.usecase_tools
# import tools.model_tools
import tools.modelcard_tools
import tools.forecasting_tools
import tools.forecasting_governance_tools
import handlers.prompt_handler
import health

# The server is ready - fastmcp will handle the rest
