"""
Client management for the ModelManager MCP Server.

This module provides factory functions for creating ModelManager API clients
with proper authentication and configuration.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from fastmcp import Context
from mmanager import Model, Usecase, ModelCard, ModelInsights, ForecastingApi, ForecastGovernanceReport

@dataclass
class MMContext:
    """Context for the ModelManager MCP server.
    
    This class holds the configuration needed to communicate with the ModelManager API,
    including authentication credentials and API endpoint information.
    """
    secret_key: str
    api_base_url: str
    output_dir: str

@asynccontextmanager
async def server_lifespan() -> AsyncIterator[MMContext]:
    """Manage server lifespan and initialize ModelManager context.
    
    Yields:
        MMContext: Configured context for ModelManager API interactions.
    """
    from config import config
    
    context = MMContext(
        secret_key=config.secret_key,
        api_base_url=config.api_base_url,
        output_dir=config.output_dir
    )
    
    try:
        yield context
    finally:
        # Cleanup if needed
        pass

def get_mm_client(ctx: Context, client_type: str):
    """Factory function to get ModelManager API clients.
    
    Args:
        ctx: The MCP server context containing authentication and configuration.
        client_type: Type of client to create ('model', 'usecase', 'modelcard', 
                     'modelinsights', 'forecast').
    
    Returns:
        Configured ModelManager API client instance.
    
    Raises:
        ValueError: If client_type is not supported.
    """
    # Handle both FastMCP CLI and standalone usage
    try:
        # Try to get context from lifespan (standalone mode)
        mm_context = ctx.request_context.lifespan_context
        secret_key = mm_context.secret_key
        base_url = mm_context.api_base_url
    except (AttributeError, TypeError):
        # Fallback to environment variables (FastMCP CLI mode)
        from config import config
        secret_key = config.secret_key
        base_url = config.api_base_url
    
    client_map = {
        'model': Model,
        'usecase': Usecase,
        'modelcard': ModelCard,
        'modelinsights': ModelInsights,
        'forecast': ForecastingApi,
        'governance': ForecastGovernanceReport,
    }
    
    if client_type not in client_map:
        raise ValueError(f"Unsupported client type: {client_type}. "
                        f"Supported types: {', '.join(client_map.keys())}")
    
    client_class = client_map[client_type]
    return client_class(secret_key=secret_key, base_url=base_url)
