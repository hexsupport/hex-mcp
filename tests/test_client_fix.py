#!/usr/bin/env python3
"""
Test script to verify the client fix works correctly.
"""

import sys
import os
import asyncio

# Add the server directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

async def test_client_creation():
    """Test that clients can be created with the new fallback mechanism."""
    print("Testing client creation with fallback...")
    
    try:
        # Import required modules
        from config import config
        from clients import get_mm_client
        from fastmcp import Context
        
        print("  ✓ Imports successful")
        
        # Create a mock context (simulating FastMCP CLI mode)
        class MockRequestContext:
            def __init__(self):
                self.lifespan_context = None  # This simulates CLI mode
        
        class MockContext:
            def __init__(self):
                self.request_context = MockRequestContext()
        
        mock_ctx = MockContext()
        
        # Test client creation
        print("  ✓ Testing model client creation...")
        model_client = get_mm_client(mock_ctx, 'model')
        print(f"    ✓ Model client created: {type(model_client).__name__}")
        
        print("  ✓ Testing usecase client creation...")
        usecase_client = get_mm_client(mock_ctx, 'usecase')
        print(f"    ✓ Usecase client created: {type(usecase_client).__name__}")
        
        print("  ✓ Testing modelcard client creation...")
        modelcard_client = get_mm_client(mock_ctx, 'modelcard')
        print(f"    ✓ Modelcard client created: {type(modelcard_client).__name__}")
        
        print("  ✓ Testing forecast client creation...")
        forecast_client = get_mm_client(mock_ctx, 'forecast')
        print(f"    ✓ Forecast client created: {type(forecast_client).__name__}")
        
        print("  ✓ All client creation tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Client creation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ModelManager MCP Server - Client Fix Test")
    print("=" * 60)
    
    success = asyncio.run(test_client_creation())
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Client fix test passed! The server should now work with FastMCP CLI.")
    else:
        print("❌ Client fix test failed. Please check the errors above.")
    print("=" * 60)
