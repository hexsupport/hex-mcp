"""HexagonML ModelManager MCP Server

This module provides a FastMCP server implementation for interacting with the HexagonML ModelManager API,
offering tools for model and usecase management, causal discovery, and metrics analysis.

The server exposes tools for:
- Creating, updating, and deleting ML models and usecases
- Retrieving metrics and performance data
- Generating causal discovery and inference graphs
- Analyzing causal relationships in datasets

Environment variables required:
- SECRET_KEY: Authentication key for the ModelManager API
- MM_API_BASE_URL: Base URL for the ModelManager API
- OUTPUT_DIR: Directory to store generated graph files
- HOST (optional): Host address for the MCP server (default: 0.0.0.0)
- PORT (optional): Port for the MCP server (default: 9000)
"""

from fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
import httpx
import asyncio
import os
import sys
from mmanager.mmanager import Model, Usecase

# Load environment variables from .env file
load_dotenv()

@dataclass
class MMContext:
    """Context for the ModelManager MCP server.
    
    This class holds the configuration needed to communicate with the ModelManager API,
    including authentication credentials and API endpoint information.
    """
    secret_key: str  # Authentication key for the ModelManager API
    api_base_url: str  # Base URL of the ModelManager API service

@asynccontextmanager
async def mm_lifespan(server: FastMCP) -> AsyncIterator[MMContext]:
    """
    Manages the HexagonML ModelManager API configuration lifecycle.
    
    This context manager initializes the HexagonML ModelManager API configuration from environment
    variables and provides it to the MCP server. It handles the setup and teardown
    of resources needed for API communication.
    
    Args:
        server: The FastMCP server instance that will use this context.
        
    Yields:
        MMContext: A context object containing the API credentials and configuration.
    """
    print("Initializing MCP Server lifespan...")
    
    # Get required environment variables
    secret_key = os.getenv("SECRET_KEY")
    api_base_url = os.getenv("MM_API_BASE_URL")
    
    # Validate credentials
    if not secret_key:
        print("ERROR: Missing SECRET_KEY environment variable")
        raise ValueError("SECRET_KEY environment variable must be set")
        
    if not api_base_url:
        print("ERROR: Missing MM_API_BASE_URL environment variable")
        raise ValueError("MM_API_BASE_URL environment variable must be set")
    
    # Check if OUTPUT_DIR exists and is writable
    output_dir = os.getenv("OUTPUT_DIR")
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory confirmed: {output_dir}")
        except Exception as e:
            print(f"WARNING: Could not create output directory: {str(e)}")
    else:
        print("WARNING: OUTPUT_DIR not set, some features may not work properly")
    
    # Create context
    print(f"Connecting to ModelManager API at {api_base_url}")
    ctx = MMContext(secret_key=secret_key, api_base_url=api_base_url)
    
    # Initialize server
    print("MCP Server initialization complete! Ready to serve requests.")
    
    try:
        yield ctx
    except Exception as e:
        print(f"ERROR during MCP server operation: {str(e)}")
        raise
    finally:
        print("Shutting down MCP Server...")
        # Add cleanup if needed

# Create FastMCP server instance
# Note: Using only parameters supported by the current FastMCP version
mcp = FastMCP(
    "hex-mm-mcp",
    lifespan=mm_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "9000")
)


# === Utility Functions ===
def get_model_client(ctx: Context) -> Model:
    """Return a ModelManager Model client using credentials from context."""
    secret_key = ctx.request_context.lifespan_context.secret_key
    base_url = ctx.request_context.lifespan_context.api_base_url
    return Model(secret_key, base_url)

def get_usecase_client(ctx: Context) -> Usecase:
    """Return a ModelManager Usecase client using credentials from context."""
    secret_key = ctx.request_context.lifespan_context.secret_key
    base_url = ctx.request_context.lifespan_context.api_base_url
    return Usecase(secret_key, base_url)

def get_mm_client(ctx: Context, client_type: str):
    """Return the correct ModelManager client (Model or Usecase) based on client_type."""
    if client_type == 'model':
        return get_model_client(ctx)
    elif client_type == 'usecase':
        return get_usecase_client(ctx)
    else:
        raise ValueError(f"Unknown client_type: {client_type}")

def safe_response_to_dict(response) -> dict:
    """Convert a ModelManager response to a dictionary, handling .json() or fallback to str."""
    try:
        if hasattr(response, 'json'):
            return response.json()
        elif isinstance(response, dict):
            return response
        else:
            return {"status": "success", "message": str(response)}
    except Exception as e:
        return {"status": "error", "message": f"Failed to parse response: {str(e)}", "error_type": type(e).__name__}

def infer_forecasting_condition_count(usecase_detail: dict) -> int | None:
    """Infer the number of forecasting conditions required for a forecasting usecase.

    Returns:
        int | None: 1, 2, 3 if it can be inferred, otherwise None.
    """
    if not isinstance(usecase_detail, dict):
        return None

    template_to_count = {"one_condition": 1, "two_conditions": 2, "three_conditions": 3}

    def _get_nested(d: dict, *path: str):
        cur = d
        for p in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
        return cur

    for t in (
        usecase_detail.get("forecasting_template"),
        _get_nested(usecase_detail, "project", "forecasting_template"),
        _get_nested(usecase_detail, "config", "forecasting_template"),
        _get_nested(usecase_detail, "forecasting_config", "forecasting_template"),
        _get_nested(usecase_detail, "forecasting_fields", "forecasting_template"),
    ):
        if isinstance(t, str):
            key = t.strip().lower().replace(" ", "_")
            if key in template_to_count:
                return template_to_count[key]

    candidates = [
        usecase_detail.get("forecasting_fields"),
        usecase_detail.get("forecasting_field"),
        usecase_detail.get("forecasting_config"),
        usecase_detail.get("config"),
    ]
    for c in candidates:
        if isinstance(c, dict):
            # Most direct / explicit
            for key in ("conditions", "condition_fields", "condition_columns", "conditions_count"):
                v = c.get(key)
                if isinstance(v, int):
                    return max(1, min(3, v))
                if isinstance(v, (list, tuple)):
                    if len(v) in (1, 2, 3):
                        return len(v)
            # Heuristic based on presence of condition metadata
            has_c2 = any(k in c for k in ("condition_2", "condition2", "second_condition", "condition_2_name", "condition_2_label"))
            has_c3 = any(k in c for k in ("condition_3", "condition3", "third_condition", "condition_3_name", "condition_3_label"))
            if has_c3:
                return 3
            if has_c2:
                return 2

    # Fallback heuristic on root keys
    has_c2 = any(k in usecase_detail for k in ("condition_2", "condition2", "second_condition", "condition_2_name", "condition_2_label"))
    has_c3 = any(k in usecase_detail for k in ("condition_3", "condition3", "third_condition", "condition_3_name", "condition_3_label"))
    if has_c3:
        return 3
    if has_c2:
        return 2
    return None


async def fetch_usecase_detail(ctx: Context, usecase_id: str) -> dict:
    """Fetch usecase detail using the ModelManager Usecase client."""
    try:
        usecase_client = get_mm_client(ctx, 'usecase')
        resp = await asyncio.to_thread(usecase_client.get_detail, usecase_id)
        return safe_response_to_dict(resp)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get usecase detail: {str(e)}",
            "error_type": type(e).__name__,
        }

# === MCP Tools ===

@mcp.tool(
    name="add_usecase",
    description="Create a new usecase in the ModelManager service with optional forecasting configuration",
    tags={"usecase", "create", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def add_usecase(ctx: Context, usecase_info: dict, forecasting_fields: dict = None, forecasting_feature_tabs: dict = None) -> dict:
    """
    Create a new usecase in the ModelManager service.
    Args:
        ctx: The MCP server context.
        usecase_info: Dict of usecase metadata including name, description, and configuration.
        forecasting_fields: Optional dict for forecasting usecases with field definitions.
        forecasting_feature_tabs: Optional dict for forecasting usecases with feature tab configurations.
    Returns:
        dict: Response from the ModelManager service with the created usecase details.
    """
    forecasting_fields = forecasting_fields or {}
    forecasting_feature_tabs = forecasting_feature_tabs or {}
    
    # Validate input
    if not usecase_info:
        await ctx.error("Usecase information cannot be empty")
        return {"status": "error", "message": "Usecase information is required", "error_type": "ValidationError"}
    
    # Check for required fields
    required_fields = ['name']
    missing_fields = [field for field in required_fields if field not in usecase_info]
    if missing_fields:
        await ctx.error(f"Missing required fields: {', '.join(missing_fields)}")
        return {
            "status": "error", 
            "message": f"Missing required fields: {', '.join(missing_fields)}",
            "error_type": "ValidationError"
        }
    
    # Report progress
    await ctx.info(f"Creating new usecase: {usecase_info.get('name', 'Unnamed')}")
    await ctx.report_progress(progress=20, total=100)
    
    try:
        usecase_client = get_mm_client(ctx, 'usecase')
        await ctx.report_progress(progress=40, total=100)
        
        # Execute create operation
        response = await asyncio.to_thread(
            usecase_client.post_usecase, 
            usecase_info, 
            forecasting_fields, 
            forecasting_feature_tabs
        )
        await ctx.report_progress(progress=80, total=100)
        
        # Process response
        result = safe_response_to_dict(response)
        
        # Check for success indicator in response
        if result.get('status') == 'error':
            await ctx.error(f"Failed to create usecase: {result.get('message', 'Unknown error')}")
        else:
            usecase_id = result.get('id') or result.get('usecase_id')
            if usecase_id:
                await ctx.info(f"Usecase created successfully with ID: {usecase_id}")
            else:
                await ctx.info("Usecase created successfully")
        
        await ctx.report_progress(progress=100, total=100)
        return result
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return {
            "status": "error",
            "message": f"Validation error: {str(e)}",
            "error_type": "ValueError"
        }
    except Exception as e:
        await ctx.error(f"Failed to add usecase: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to add usecase: {str(e)}",
            "error_type": type(e).__name__
        }

@mcp.tool(
    name="update_usecase",
    description="Update an existing usecase in the ModelManager service with new configuration",
    tags={"usecase", "update", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def update_usecase(ctx: Context, usecase_id: str, usecase_data: dict) -> dict:
    """
    Update an existing usecase in the ModelManager service.
    Args:
        ctx: The MCP server context.
        usecase_id: The unique identifier of the usecase to update.
        usecase_data: Dict of updated usecase metadata including name, description, and configuration.
    Returns:
        dict: Response from the ModelManager service with the updated usecase details.
    """
    try:
        usecase_client = get_mm_client(ctx, 'usecase')
        response = await asyncio.to_thread(usecase_client.patch_usecase, usecase_data, usecase_id)
        return safe_response_to_dict(response)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to update usecase: {str(e)}",
            "error_type": type(e).__name__
        }

@mcp.tool(
    name="delete_usecase",
    description="Delete a usecase from the ModelManager service permanently",
    tags={"usecase", "delete", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def delete_usecase(ctx: Context, usecase_id: str) -> dict:
    """
    Delete a usecase from the ModelManager service.
    Args:
        ctx: The MCP server context.
        usecase_id: The unique identifier of the usecase to delete.
    Returns:
        dict: Response from the ModelManager service with status information.
    """
    try:
        usecase_client = get_mm_client(ctx, 'usecase')
        response = await asyncio.to_thread(usecase_client.delete_usecase, usecase_id)
        status_code = getattr(response, 'status_code', None)
        if status_code == 204:
            return {
                "status": "success",
                "message": f"Usecase {usecase_id} deleted successfully.",
                "code": 204
            }
        return response
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to delete usecase: {str(e)}",
            "error_type": type(e).__name__
        }


@mcp.tool(
    name="add_model",
    description="Upload a new machine learning model to the ModelManager service",
    tags={"model", "create", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def add_model(ctx: Context, model_data: dict) -> dict:
    """
    Upload a machine learning model to the ModelManager service.
    Args:
        ctx: The MCP server context containing authentication and configuration.
        model_data: Dict of model metadata/configuration including name, description, and model parameters.
    Returns:
        dict: Response from the ModelManager service containing the created model details.
    """
    # Report progress to client
    await ctx.info(f"Creating new model with name: {model_data.get('name', 'Unnamed')}")
    await ctx.report_progress(progress=10, total=100)
    
    try:
        # Validate required fields
        if not model_data:
            await ctx.error("Model data is empty or null")
            return {"status": "error", "message": "Model data cannot be empty"}
            
        required_fields = ['name', 'description']
        missing_fields = [field for field in required_fields if field not in model_data]
        
        if missing_fields:
            await ctx.error(f"Missing required fields: {', '.join(missing_fields)}")
            return {
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}",
                "error_type": "ValidationError"
            }
        
        # Proceed with model creation
        await ctx.report_progress(progress=30, total=100)
        model_client = get_mm_client(ctx, 'model')
        
        # Use asyncio.to_thread for non-blocking operation
        model_response = await asyncio.to_thread(model_client.post_model, model_data)
        await ctx.report_progress(progress=90, total=100)
        
        # Process response
        response_dict = safe_response_to_dict(model_response)
        
        if 'id' in response_dict:
            await ctx.info(f"Model created successfully with ID: {response_dict['id']}")
        else:
            await ctx.info("Model created successfully")
            
        await ctx.report_progress(progress=100, total=100)
        return response_dict
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return {
            "status": "error",
            "message": f"Validation error: {str(e)}",
            "error_type": "ValueError"
        }
    except Exception as e:
        await ctx.error(f"Failed to upload model: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to upload model: {str(e)}",
            "error_type": type(e).__name__
        }

@mcp.tool(
    name="delete_model",
    description="Delete a machine learning model from the ModelManager service permanently",
    tags={"model", "delete", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def delete_model(ctx: Context, model_id: str) -> dict:
    """
    Delete a machine learning model from the ModelManager service.
    Args:
        ctx: The MCP server context containing authentication and configuration.
        model_id: The unique identifier of the model to delete.
    Returns:
        dict: Response from the ModelManager service with status information.
    """
    # Validate input
    if not model_id:
        await ctx.error("Model ID cannot be empty")
        return {"status": "error", "message": "Model ID is required", "error_type": "ValidationError"}
    
    # Report progress
    await ctx.info(f"Deleting model with ID: {model_id}")
    await ctx.report_progress(progress=25, total=100)
    
    try:
        model_client = get_mm_client(ctx, 'model')
        await ctx.report_progress(progress=50, total=100)
        
        # Execute delete operation
        delete_response = await asyncio.to_thread(model_client.delete_model, model_id)
        await ctx.report_progress(progress=75, total=100)
        
        # Check response
        if hasattr(delete_response, 'status_code') and delete_response.status_code == 204:
            await ctx.info(f"Model {model_id} deleted successfully")
            await ctx.report_progress(progress=100, total=100)
            return {
                "status": "success",
                "message": f"Model {model_id} deleted successfully",
                "code": 204
            }
            
        # Process other responses
        response_dict = safe_response_to_dict(delete_response)
        await ctx.report_progress(progress=100, total=100)
        return response_dict
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return {
            "status": "error",
            "message": f"Validation error: {str(e)}",
            "error_type": "ValueError"
        }
    except Exception as e:
        await ctx.error(f"Failed to delete model: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to delete model: {str(e)}",
            "error_type": type(e).__name__
        }

@mcp.tool(
    name="update_model",
    description="Update a machine learning model's metadata or configuration in the ModelManager service",
    tags={"model", "update", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def update_model(ctx: Context, model_id: str, model_data: dict, create_sweetviz: bool = True) -> dict:
    """
    Update a machine learning model's metadata or configuration in the ModelManager service.
    Args:
        ctx: The MCP server context containing authentication and configuration.
        model_id: The unique identifier of the model to update.
        model_data: Dict of updated model metadata/configuration including name, description, and parameters.
        create_sweetviz: Whether to generate a Sweetviz report for data visualization (default: True).
    Returns:
        dict: Response from the ModelManager service with updated model details.
    """
    try:
        model_client = get_mm_client(ctx, 'model')
        update_response = await asyncio.to_thread(model_client.patch_model, model_data, model_id, create_sweetviz)
        return safe_response_to_dict(update_response)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to update model: {str(e)}",
            "error_type": type(e).__name__
        }

@mcp.tool(
    name="get_latest_metrics",
    description="Retrieve the latest performance metrics for a model from the ModelManager service",
    tags={"model", "metrics", "performance", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def get_latest_metrics(ctx: Context, model_id: str, metric_type: str) -> dict:
    """
    Retrieve the latest metrics for a model from the ModelManager service.
    Args:
        ctx: The MCP server context containing authentication and configuration.
        model_id: The unique identifier of the model.
        metric_type: The type of metric to retrieve (e.g., 'Scoring Metric', 'Development Metric').
    Returns:
        dict: Response from the ModelManager service containing the latest metrics and performance data.
    """
    try:
        model_client = get_mm_client(ctx, 'model')
        metrics_response = await asyncio.to_thread(model_client.get_latest_metrics, model_id, metric_type)
        return safe_response_to_dict(metrics_response)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get latest metrics: {str(e)}",
            "error_type": type(e).__name__
        }

@mcp.tool(
    name="get_usecase_data",
    description="Retrieve and summarize all usecases from the ModelManager API",
    tags={"usecase", "list", "summary", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def get_usecase_data(ctx: Context) -> dict:
    """
    Retrieve and summarize usecase data from the ModelManager API.

    Fetches all registered usecases and returns a concise summary including usecase ID, name, description, 
    insights, and metrics analyses for each usecase.

    Args:
        ctx (Context): The MCP server context containing authentication credentials and API configuration.

    Returns:
        dict: {
            'status': 'success' or 'error',
            'summary': List of summaries for each usecase (id, name, description, insights, metrics),
            'details' (optional): Full usecase data if needed for debugging
        }
    """
    api_url = f"{ctx.request_context.lifespan_context.api_base_url}/api/mcp-usecase-detail/get_usecase_data/"
    secret_key = ctx.request_context.lifespan_context.secret_key
    headers = {"Authorization": f"secret-key {secret_key}", "Accept": "application/json"}
    
    # Report progress to client
    await ctx.info("Fetching usecase data from ModelManager API...")
    await ctx.report_progress(progress=10, total=100)
    
    try:
        # Use httpx with timeout and follow_redirects for better reliability
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(api_url, headers=headers)
            response.raise_for_status()
            await ctx.report_progress(progress=50, total=100)
            
            data = response.json()
            await ctx.report_progress(progress=90, total=100)
            await ctx.info(f"Successfully retrieved {len(data) if isinstance(data, list) else 'all'} usecases")
    except httpx.HTTPStatusError as e:
        await ctx.error(f"HTTP error: {str(e)}")
        return {
            'status': 'error',
            'message': f"HTTP error: {str(e)}",
            'error_type': type(e).__name__,
            'status_code': e.response.status_code if hasattr(e, 'response') else None
        }
    except httpx.TimeoutException:
        await ctx.error("Request timed out when fetching usecase data")
        return {
            'status': 'error',
            'message': "Request timed out",
            'error_type': 'TimeoutException'
        }
    except Exception as e:
        await ctx.error(f"Failed to fetch usecase data: {str(e)}")
        return {
            'status': 'error',
            'message': f"Failed to fetch usecase data: {str(e)}",
            'error_type': type(e).__name__
        }
        
    # Report complete and return results
    await ctx.report_progress(progress=100, total=100)
    return {
        'status': 'success',
        'summary': data
    }

@mcp.tool(
    name="get_modelcard_summary",
    description="Retrieve the model card summary for a given model from the ModelManager service",
    tags={"model", "modelcard", "summary", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def get_modelcard_summary(ctx: Context, usecase_id: str, model_id:str = None, series: str = None, condition_1: str = None, condition_2: str = None, condition_3: str = None) -> dict:
    """
    Retrieve the model card summary for a model.

    Args:
        ctx: The MCP server context containing authentication and configuration.
        usecase_id: The unique identifier of the usecase.
        model_id: The unique identifier of the model, If the usecase is not Forecasting Usecase, this parameter is required (optional).
        series: The series identifier, Series (optional), If the usecase is Forecasting, this parameter is required.
        condition_1: The first condition, Region (optional), If the usecase is Forecasting, this parameter is required.
        condition_2: The second condition, Facility (optional), If the usecase is Forecasting and Two Conditions or Three Conditions, this parameter is required.
        condition_3: The third condition, Unit (optional), If the usecase is Forecasting and Three Conditions, this parameter is required.

    Returns:
        dict: Parsed JSON/dict response containing the model card summary, or an error dict.
    """
    if not usecase_id:
        await ctx.error("Usecase ID cannot be empty")
        return {"status": "error", "message": "usecase_id is required", "error_type": "ValidationError"}

    api_url = f"{ctx.request_context.lifespan_context.api_base_url}/api/mcp-usecase-detail/get_modelcard_data/"
    secret_key = ctx.request_context.lifespan_context.secret_key
    headers = {"Authorization": f"secret-key {secret_key}", "Accept": "application/json"}

    await ctx.info("Fetching usecase details...")
    await ctx.report_progress(progress=10, total=100)
    usecase_detail = await fetch_usecase_detail(ctx, usecase_id)

    if isinstance(usecase_detail, dict) and usecase_detail.get("status") == "error":
        await ctx.error(usecase_detail.get("message", "Failed to get usecase detail"))
        return usecase_detail

    usecase_type = None
    if isinstance(usecase_detail, dict):
        usecase_type = usecase_detail.get("usecase_type") or usecase_detail.get("type")
    is_forecasting = isinstance(usecase_type, str) and usecase_type.strip().lower() == "forecasting"

    if is_forecasting:
        required_conditions = infer_forecasting_condition_count(usecase_detail)

        if not series:
            await ctx.error("series is required for forecasting usecases")
            return {"status": "error", "message": "series is required for forecasting usecases", "error_type": "ValidationError"}
        if not condition_1:
            await ctx.error("condition_1 is required for forecasting usecases")
            return {"status": "error", "message": "condition_1 is required for forecasting usecases", "error_type": "ValidationError"}
        if required_conditions in (2, 3) and not condition_2:
            await ctx.error("condition_2 is required for this forecasting usecase")
            return {"status": "error", "message": "condition_2 is required for this forecasting usecase", "error_type": "ValidationError"}
        if required_conditions == 3 and not condition_3:
            await ctx.error("condition_3 is required for this forecasting usecase")
            return {"status": "error", "message": "condition_3 is required for this forecasting usecase", "error_type": "ValidationError"}
    else:
        if not model_id:
            await ctx.error("model_id is required for non-forecasting usecases")
            return {"status": "error", "message": "model_id is required for non-forecasting usecases", "error_type": "ValidationError"}

    params: dict = {"usecase_id": usecase_id}
    if is_forecasting:
        params.update({
            "series": series,
            "condition_1": condition_1,
        })
        if condition_2:
            params["condition_2"] = condition_2
        if condition_3:
            params["condition_3"] = condition_3
    else:
        params["model_id"] = model_id

    await ctx.info("Fetching model card summary from ModelManager API...")
    await ctx.report_progress(progress=40, total=100)

    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            await ctx.report_progress(progress=90, total=100)
            return {"status": "success", "summary": response.json()}
    except httpx.HTTPStatusError as e:
        await ctx.error(f"HTTP error: {str(e)}")
        return {
            "status": "error",
            "message": f"HTTP error: {str(e)}",
            "error_type": type(e).__name__,
            "status_code": e.response.status_code if hasattr(e, 'response') else None,
        }
    except httpx.TimeoutException:
        await ctx.error("Request timed out when fetching modelcard summary")
        return {
            "status": "error",
            "message": "Request timed out",
            "error_type": "TimeoutException",
        }
    except Exception as e:
        await ctx.error(f"Failed to get modelcard summary: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get modelcard summary: {str(e)}",
            "error_type": type(e).__name__,
        }

CAUSAL_DISCOVERY_GRAPH_TYPE_OPTIONS = [
    "HeatMap",
    "2D_CausalDiscovery_Comparision",
    "3D_CausalDiscovery_Comparision"
]

# === Utility Functions for Causal Graph Tools ===
def validate_graph_type(graph_type: str, allowed: list, label: str) -> dict | None:
    if graph_type not in allowed:
        return {
            "status": "error",
            "message": f"Invalid graph_type '{graph_type}'. Allowed options: {allowed}",
            "error_type": "InvalidGraphType",
            "graph_type_label": label,
        }
    return None

def extract_html_content(html_obj) -> str:
    # Handles IPython.display.HTML or plain string
    try:
        return getattr(html_obj, "data", None) or getattr(html_obj, "value", None) or str(html_obj)
    except Exception:
        return str(html_obj)

def save_html_to_file(html_content: str, prefix: str, model_id: str, graph_type: str) -> str:
    import os
    from datetime import datetime
    output_dir = os.getenv("OUTPUT_DIR")
    if not output_dir:
        raise RuntimeError("OUTPUT_DIR environment variable must be set. Please set it in your .env file.")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{prefix}_{model_id}_{graph_type}_{timestamp}.html"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return file_path

@mcp.tool(
    name="get_causal_discovery_graphs",
    description="Retrieve causal discovery visualization graphs for a model and save as HTML",
    tags={"model", "causal", "discovery", "visualization", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def get_causal_discovery_graphs(ctx: Context, model_id: str, graph_type: str) -> dict:
    """
    Retrieve causal discovery graphs for a given model and save the HTML visualization content to a file.

    Args:
        ctx: The MCP server context.
        model_id: The unique identifier of the model.
        graph_type: The type of graph to retrieve. Options: "HeatMap", "2D_CausalDiscovery_Comparision", "3D_CausalDiscovery_Comparision"
    Returns:
        dict: Contains the file path where the HTML was saved, or error info.
    """
    err = validate_graph_type(graph_type, CAUSAL_DISCOVERY_GRAPH_TYPE_OPTIONS, "causal_discovery")
    if err:
        return err
    try:
        model_client = get_mm_client(ctx, 'model')
        html_obj = await asyncio.to_thread(model_client.get_causal_discovery_graphs, model_id, graph_type)
        html_content = extract_html_content(html_obj)
        file_path = save_html_to_file(html_content, "causal_discovery", model_id, graph_type)
        return {
            "status": "success",
            "file_path": file_path,
            "message": f"Causal discovery graph saved to {file_path}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get or save causal discovery graphs: {str(e)}",
            "error_type": type(e).__name__
        }

@mcp.tool(
    name="get_causal_discovery_metrics",
    description="Retrieve causal discovery metrics and KPIs for a given model",
    tags={"model", "causal", "discovery", "metrics", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def get_causal_discovery_metrics(ctx: Context, model_id: str) -> dict:
    """
    Retrieve causal discovery metrics for a given model.

    Args:
        ctx: The MCP server context.
        model_id (str): The unique identifier of the model.

    Returns:
        dict: The parsed JSON metrics if successful, or an error dict if the request or parsing fails.

    Example structure of returned dict:
        {
            "metrics": [...],
            ...
        }
    """
    try:
        model_client = get_mm_client(ctx, 'model')
        resp = await asyncio.to_thread(model_client.get_causal_discovery_metrics, model_id)
        if hasattr(resp, 'json'):
            try:
                return resp.json()
            except Exception as json_exc:
                return {"status": "error", "message": f"JSON decode error: {json_exc}"}
        return resp
    except Exception as e:
        return {"status": "error", "message": str(e)}

CAUSAL_INFERENCE_GRAPH_TYPE_OPTIONS = [
    "coeff_graph",
    "top_effect_p_values",
    "top_effect_rsquared"
]
@mcp.tool
async def get_causal_inference_graphs(ctx: Context, model_id: str, graph_type: str, treatment: str = None, outcome: str = None) -> dict:
    """
    Retrieve causal inference graphs for a given model.

    Args:
        ctx: The MCP server context.
        model_id: The unique identifier of the model.
        graph_type: The type of graph to retrieve. Options: "coeff_graph", "top_effect_p_values", "top_effect_rsquared"
        treatment: Optional treatment variable.
        outcome: Optional outcome variable.
    Returns:
        dict: Contains the file path where the HTML was saved, or error info.
    """
    err = validate_graph_type(graph_type, CAUSAL_INFERENCE_GRAPH_TYPE_OPTIONS, "causal_inference")
    if err:
        return err
    try:
        model_client = get_mm_client(ctx, 'model')
        html_obj = await asyncio.to_thread(model_client.get_causal_inference_graphs, model_id, graph_type, treatment, outcome)
        html_content = extract_html_content(html_obj)
        file_path = save_html_to_file(html_content, "causal_inference", model_id, graph_type)
        return {
            "status": "success",
            "file_path": file_path,
            "message": f"Causal inference graph saved to {file_path}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get or save causal inference graphs: {str(e)}",
            "error_type": type(e).__name__
        }


CAUSAL_INFERENCE_CORRELATION_GRAPH_TYPE_OPTIONS = [
    "correlation_graph",
    "causal_correlation_summary"
]
@mcp.tool
async def get_causal_inference_correlation(ctx: Context, model_id: str, graph_type: str, treatment: str, outcome: str) -> dict:
    """
    Retrieve causal inference correlation for a given model.

    Args:
        ctx: The MCP server context.
        model_id: The unique identifier of the model.
        graph_type: The type of graph/correlation to retrieve. Options: "correlation_graph", "causal_correlation_summary"
        treatment: The treatment variable.
        outcome: The outcome variable.
    Returns:
        dict: Contains the file path where the HTML was saved, or error info.
    """
    err = validate_graph_type(graph_type, CAUSAL_INFERENCE_CORRELATION_GRAPH_TYPE_OPTIONS, "causal_inference_correlation")
    if err:
        return err
    try:
        model_client = get_mm_client(ctx, 'model')
        html_obj = await asyncio.to_thread(model_client.get_causal_inference_correlation, model_id, graph_type, treatment, outcome)
        html_content = extract_html_content(html_obj)
        file_path = save_html_to_file(html_content, "causal_inference_correlation", model_id, graph_type)
        return {
            "status": "success",
            "file_path": file_path,
            "message": f"Causal inference correlation saved to {file_path}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get or save causal inference correlation: {str(e)}",
            "error_type": type(e).__name__
        }

@mcp.tool
async def get_drivers_analysis(ctx: Context, file_path: str, treatment: str = None, outcome: str = None) -> dict:
    """
    Retrieve drivers (causal) analysis insights for a given dataset using the ModelManager client.

    Args:
        ctx (Context):
            The MCP server context containing authentication and configuration.
        file_path (str):
            Path to the data file for analysis. Must exist and be accessible.
        treatment (str, optional):
            The treatment variable name to analyze. If not provided, analysis is performed without a specific treatment.
        outcome (str, optional):
            The outcome variable name to analyze. If not provided, analysis is performed without a specific outcome.

    Returns:
        dict: A structured response indicating the result of the analysis request.
    """
    # Validate input parameters
    if not file_path:
        return {
            "status": "error",
            "message": "Missing required parameter: file_path",
            "error_type": "ValidationError"
        }

    # Check if file exists before proceeding
    if not os.path.exists(file_path):
        return {
            "status": "error",
            "message": f"File not found: {file_path}",
            "error_type": "FileNotFoundError"
        }

    # Log the request parameters
    print(f"Processing drivers analysis with: treatment={treatment}, outcome={outcome}, file_path={file_path}")

    # Construct input_data dictionary for ModelManager API
    input_data = {
        "file_path": file_path
    }
    
    if treatment is not None:
        input_data["treatment"] = treatment
    
    if outcome is not None:
        input_data["outcome"] = outcome

    try:
        # Get ModelManager client and process request
        model_client = get_mm_client(ctx, 'model')
        drivers_analysis_obj = await asyncio.to_thread(model_client.get_drivers_analysis, input_data)
        
        # Handle successful response
        if hasattr(drivers_analysis_obj, 'status_code') and drivers_analysis_obj.status_code >= 400:
            # Handle API error responses
            error_msg = getattr(drivers_analysis_obj, 'text', str(drivers_analysis_obj))
            return {
                "status": "error",
                "message": f"API error: {error_msg}",
                "error_type": "APIError",
                "status_code": drivers_analysis_obj.status_code
            }
        
        # Convert response to dict and add success message
        response_data = safe_response_to_dict(drivers_analysis_obj)
        response_data["status"] = "success"
        response_data["message"] = "Successfully retrieved drivers analysis"
        return response_data
        
    except FileNotFoundError as e:
        # Handle file not found errors specifically
        return {
            "status": "error",
            "message": f"File access error: {str(e)}",
            "error_type": "FileNotFoundError"
        }
    except ValueError as e:
        # Handle value errors (often from parameter validation)
        return {
            "status": "error",
            "message": f"Invalid parameter value: {str(e)}",
            "error_type": "ValueError"
        }
    except Exception as e:
        # Catch all other exceptions
        return {
            "status": "error",
            "message": f"Failed to get drivers analysis insights: {str(e)}",
            "error_type": type(e).__name__
        }

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
    env_vars = {
        "SECRET_KEY": "*****" if os.getenv("SECRET_KEY") else "NOT SET",
        "MM_API_BASE_URL": os.getenv("MM_API_BASE_URL") or "NOT SET",
        "OUTPUT_DIR": os.getenv("OUTPUT_DIR") or "NOT SET",
        "HOST": os.getenv("HOST", "0.0.0.0"),
        "PORT": os.getenv("PORT", "9000")
    }
    for key, value in env_vars.items():
        status = "✓" if value != "NOT SET" else "✗"
        print(f"  {status} {key}: {value}")
    
    # Validate required environment variables
    required_env_vars = ['SECRET_KEY', 'MM_API_BASE_URL', 'OUTPUT_DIR']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"\nERROR: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment.")
        sys.exit(1)
        
    try:
        print(f"\nStarting ModelManager MCP server on {env_vars['HOST']}:{env_vars['PORT']}")
        print("Initializing server components...")
        print("Press Ctrl+C to stop the server\n")
        
        # Run with a timeout to catch initialization issues
        try:
            await asyncio.wait_for(mcp.run_sse_async(), timeout=60.0)
        except asyncio.TimeoutError:
            print("\nERROR: Server initialization is taking too long. Check your connection to the ModelManager API.")
            print("Try verifying API credentials and connectivity.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nServer shutdown requested...")
    except Exception as e:
        print(f"\nERROR: Server failed with exception: {str(e)}")
        raise
    finally:
        print("\nServer shutdown complete.")
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(main())