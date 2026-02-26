#!/usr/bin/env python3
"""
Test script to validate the modular server structure and imports.
"""

import sys
import os

# Add the server directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

def test_imports():
    """Test if all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        print("  ✓ Importing config...")
        from config import mcp, config
        
        print("  ✓ Importing clients...")
        from clients import server_lifespan
        
        print("  ✓ Importing utils...")
        from utils import safe_response_to_dict, create_error_response
        
        print("  ✓ Importing validators...")
        from validators import validate_forecast_payload
        
        print("  ✓ Importing tool modules...")
        import model_tools
        import usecase_tools
        import modelcard_tools
        import forecasting_tools
        
        print("  ✓ All imports successful!")
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from config import config
        
        print(f"  ✓ Config object created")
        print(f"  ✓ Host: {config.host}")
        print(f"  ✓ Port: {config.port}")
        
        # Check environment status
        env_status = config.get_env_status()
        print(f"  ✓ Environment status retrieved")
        
        for key, value in env_status.items():
            status = "✓" if value != "NOT SET" else "✗"
            print(f"    {key}: {value} {status}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_mcp_tools():
    """Test MCP tool registration."""
    print("\nTesting MCP tool registration...")
    
    try:
        from config import mcp
        
        # Get list of registered tools
        tools = mcp.get_tools()
        print(f"  ✓ {len(tools)} tools registered")
        
        for tool in tools:
            print(f"    - {tool.name}: {tool.description}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ MCP tools test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ModelManager MCP Server - Structure Test")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_imports()
    success &= test_config()
    success &= test_mcp_tools()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Server structure is valid.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)
