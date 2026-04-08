"""
Simple ModelManager MCP Server entry point for fastmcp CLI.

This file can be used with: fastmcp run server/server_simple.py
"""

# Import all the modular components
from config import mcp, config
from clients import server_lifespan

# Import all tool modules to register their MCP tools
# import usecase_tools
# import model_tools
# import modelcard_tools
import forecasting_tools
import forecasting_prompts  # noqa: F401 — side-effect import; registers @mcp.prompt
import health  # noqa: F401 — side-effect import; registers /health endpoint

# The server is ready - fastmcp will handle the rest
