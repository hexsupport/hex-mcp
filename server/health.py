"""
Health check endpoint for the ModelManager MCP Server.

Provides a simple HTTP health check endpoint for Docker, load balancers,
and monitoring systems.
"""

from config import mcp

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Health check endpoint returning 200 OK with status info."""
    return {
        "status": "healthy",
        "service": "hexagonml-modelmanager-mcp"
    }
