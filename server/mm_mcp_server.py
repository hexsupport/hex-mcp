from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
import httpx
import asyncio
import os  
from mmanager.mmanager import Model, Usecase
from datetime import datetime
from IPython.display import HTML

import functools

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
    secret_key = os.getenv("SECRET_KEY")
    api_base_url = os.getenv("MM_API_BASE_URL")
    ctx = MMContext(secret_key=secret_key, api_base_url=api_base_url)
    try:
        yield ctx
    finally:
        pass  # Add cleanup if needed

mcp = FastMCP(
    "hex-mm-mcp",
    description="This is a HexagonML MCP server that provides a Model context protocol interface for HexagonML ModelManager tools.",
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

# === MCP Tools ===

@mcp.tool()
async def add_usecase(ctx: Context, usecase_info: dict, forecasting_fields: dict = None, forecasting_feature_tabs: dict = None) -> dict:
    """
    Create a new usecase in the ModelManager service.
    Args:
        ctx: The MCP server context.
        usecase_info: Dict of usecase metadata.
        forecasting_fields: Optional dict for forecasting usecases.
        forecasting_feature_tabs: Optional dict for forecasting usecases.
    Returns:
        dict: Response from the ModelManager service.
    """
    forecasting_fields = forecasting_fields or {}
    forecasting_feature_tabs = forecasting_feature_tabs or {}
    try:
        usecase_client = get_mm_client(ctx, 'usecase')
        response = await asyncio.to_thread(usecase_client.post_usecase, usecase_info, forecasting_fields, forecasting_feature_tabs)
        return safe_response_to_dict(response)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to add usecase: {str(e)}",
            "error_type": type(e).__name__
        }

@mcp.tool()
async def update_usecase(ctx: Context, usecase_id: str, usecase_data: dict) -> dict:
    """
    Update an existing usecase in the ModelManager service.
    Args:
        ctx: The MCP server context.
        usecase_id: The ID of the usecase to update.
        usecase_data: Dict of updated usecase metadata.
    Returns:
        dict: Response from the ModelManager service.
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

@mcp.tool()
async def delete_usecase(ctx: Context, usecase_id: str) -> dict:
    """
    Delete a usecase from the ModelManager service.
    Args:
        ctx: The MCP server context.
        usecase_id: The ID of the usecase to delete.
    Returns:
        dict: Response from the ModelManager service.
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


@mcp.tool()
async def add_model(ctx: Context, model_data: dict) -> dict:
    """
    Upload a machine learning model to the ModelManager service.
    Args:
        ctx: The MCP server context containing authentication and configuration.
        model_data: Dict of model metadata/configuration.
    Returns:
        dict: Response from the ModelManager service.
    """
    try:
        model_client = get_mm_client(ctx, 'model')
        model_response = await asyncio.to_thread(model_client.post_model, model_data)
        return safe_response_to_dict(model_response)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to upload model: {str(e)}",
            "error_type": type(e).__name__
        }

@mcp.tool()
async def delete_model(ctx: Context, model_id: str) -> dict:
    """
    Delete a machine learning model from the ModelManager service.
    Args:
        ctx: The MCP server context containing authentication and configuration.
        model_id: The unique identifier of the model to delete.
    Returns:
        dict: Response from the ModelManager service.
    """
    try:
        model_client = get_mm_client(ctx, 'model')
        delete_response = await asyncio.to_thread(model_client.delete_model, model_id)
        return safe_response_to_dict(delete_response)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to delete model: {str(e)}",
            "error_type": type(e).__name__
        }

@mcp.tool()
async def update_model(ctx: Context, model_id: str, model_data: dict, create_sweetviz: bool = True) -> dict:
    """
    Update a machine learning model's metadata or configuration in the ModelManager service.
    Args:
        ctx: The MCP server context containing authentication and configuration.
        model_id: The unique identifier of the model to update.
        model_data: Dict of updated model metadata/configuration.
        create_sweetviz: Whether to generate a Sweetviz report (default: True).
    Returns:
        dict: Response from the ModelManager service.
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

@mcp.tool()
async def get_latest_metrics(ctx: Context, model_id: str, metric_type: str) -> dict:
    """
    Retrieve the latest metrics for a model from the ModelManager service.
    Args:
        ctx: The MCP server context containing authentication and configuration.
        model_id: The unique identifier of the model.
        metric_type: The type of metric to retrieve (e.g., 'accuracy', 'loss').
    Returns:
        dict: Response from the ModelManager service containing the latest metrics.
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

@mcp.tool()
async def get_usecase_data(ctx: Context) -> dict:
    """
    Retrieve and summarize usecase data from the ModelManager API.

    Fetches all registered usecases and returns a concise summary including usecase ID, name, description, insights, and metrics analyses for each usecase.

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
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        return {
            'status': 'error',
            'message': f"HTTP error: {str(e)}",
            'error_type': type(e).__name__
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Failed to fetch usecase data: {str(e)}",
            'error_type': type(e).__name__
        }
    # Return the whole results for all usecases
    return str(data)

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

@mcp.tool()
async def get_causal_discovery_graphs(ctx: Context, model_id: str, graph_type: str) -> dict:
    """
    Retrieve causal discovery graphs for a given model and save the HTML content to a file.

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

@mcp.tool()
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
@mcp.tool()
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
@mcp.tool()
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


async def main():
    await mcp.run_sse_async()

if __name__ == "__main__":
    asyncio.run(main())