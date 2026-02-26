"""
Main entry point for the ModelManager MCP Server.

This module provides the main server setup, startup, and shutdown logic.
It imports all tool modules and configures the FastMCP server with proper
lifespan management and error handling.
"""

import asyncio
import os
from fastmcp import Context
from config import mcp, config
from clients import server_lifespan

# Import all tool modules to register their MCP tools
import model_tools
import usecase_tools
import modelcard_tools
import forecasting_tools
# Note: validators is imported by forecasting_tools, utils is imported by other modules

async def main():
    """Main entry point for the MCP server.
    
    Validates required environment variables and runs the MCP server.
    Handles graceful shutdown on keyboard interrupt.
    """
    print("-" * 60)
    print("ModelManager MCP Server Startup")
    print("-" * 60)
    
    # Print environment variable status (without revealing sensitive values)
    print("Environment configuration:")
    env_status = config.get_env_status()
    for key, value in env_status.items():
        status = "✓" if value != "NOT SET" else "✗"
        print(f"  {key}: {value} {status}")
    
    if "NOT SET" in env_status.values():
        print("\n❌ Some required environment variables are not set!")
        print("Please set the missing environment variables and restart the server.")
        return
    
    print(f"\n🚀 Starting ModelManager MCP Server...")
    print(f"   Host: {config.host}")
    print(f"   Port: {config.port}")
    print(f"   API Base URL: {config.api_base_url}")
    print("-" * 60)
    
    try:
        # Run the MCP server with proper lifespan management
        import asyncio
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            print("🔄 Running in existing event loop...")
            # Create a task for the server
            task = loop.create_task(mcp.run(
                host=config.host,
                port=int(config.port),
                lifespan=server_lifespan
            ))
            await task
        except RuntimeError:
            # No event loop running, create a new one
            print("🔄 Creating new event loop...")
            await mcp.run(
                host=config.host,
                port=int(config.port),
                lifespan=server_lifespan
            )
    except KeyboardInterrupt:
        print("\n🛑 Server shutdown requested by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {str(e)}")
        raise
    finally:
        print("👋 ModelManager MCP Server stopped")

def run_server():
    """Run the server with proper async handling."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    run_server()
