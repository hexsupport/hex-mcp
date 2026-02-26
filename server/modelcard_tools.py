"""
ModelCard management tools for the ModelManager MCP Server.

This module provides MCP tools for creating and managing model cards,
which provide standardized documentation for machine learning models.
"""

from fastmcp import Context
from config import mcp
from clients import get_mm_client
from utils import safe_response_to_dict, create_error_response
import asyncio

@mcp.tool(name="get_modelcard_data",
    description="Retrieve modelcard for a given model",
    tags={"modelcard", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"})
async def get_modelcard_data(ctx: Context, usecase_id: str = None, model_id: str = None) -> dict:
    """Retrieve modelcard data for a specific model or usecase.
    
    Args:
        ctx: The MCP server context containing authentication and configuration.
        usecase_id: The usecase ID to filter modelcards (optional).
        model_id: The model ID to filter modelcards (optional).
        
    Returns:
        dict: Response containing modelcard data or error information.
    """
    # Validate that at least one parameter is provided
    if not usecase_id and not model_id:
        await ctx.error("At least one of usecase_id or model_id must be provided")
        return create_error_response(
            message="At least one of usecase_id or model_id must be provided",
            error_type="ValidationError"
        )
    
    # Build data dict
    data = {}
    if usecase_id is not None:
        data["usecase_id"] = usecase_id
    if model_id is not None:
        data["model_id"] = model_id

    await ctx.info(f"Retrieving modelcard data")
    await ctx.report_progress(progress=20, total=100)

    try:
        modelcard_client = get_mm_client(ctx, 'modelcard')
        await ctx.report_progress(progress=40, total=100)
        
        modelcard_resp = await asyncio.to_thread(modelcard_client.get_modelcard_data, data)
        await ctx.report_progress(progress=80, total=100)

        if hasattr(modelcard_resp, 'status_code') and modelcard_resp.status_code >= 400:
            error_msg = getattr(modelcard_resp, 'text', str(modelcard_resp))
            await ctx.error(f"API error: {error_msg}")
            return create_error_response(
                message=f"API error: {error_msg}",
                error_type="APIError",
                status_code=modelcard_resp.status_code
            )

        response_data = safe_response_to_dict(modelcard_resp)
        await ctx.info("Modelcard data retrieved successfully")
        await ctx.report_progress(progress=100, total=100)
        
        response_data["status"] = "success"
        response_data["message"] = "Successfully retrieved modelcard data"
        return response_data
        
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message=f"Validation error: {str(e)}",
            error_type="ValueError"
        )
    except Exception as e:
        await ctx.error(f"Failed to retrieve modelcard data: {str(e)}")
        return create_error_response(
            message=f"Failed to retrieve modelcard data: {str(e)}",
            error_type=type(e).__name__
        )

@mcp.tool(
    name="create_modelcard",
    description="Create a modelcard",
    tags={"modelcard", "modelmanager", "create"},
    meta={"version": "1.0", "author": "HexagonML"},
)
async def create_modelcard(ctx: Context, usecase_id: str, series: str = None) -> dict:
    """Create a modelcard for a usecase.
    
    Args:
        ctx: The MCP server context containing authentication and configuration.
        usecase_id: The usecase ID to create the modelcard for (required).
        series: The series name for the modelcard (optional).
        
    Returns:
        dict: Response containing the created modelcard data or error information.
    """
    # Validate required field
    if not usecase_id or not usecase_id.strip():
        await ctx.error("Usecase ID cannot be empty")
        return create_error_response(
            message="Usecase ID is required",
            error_type="ValidationError"
        )
    
    # Build data dict
    data = {"usecase_id": usecase_id}
    if series is not None:
        data["series"] = series

    await ctx.info(f"Creating modelcard for usecase: {usecase_id}")
    await ctx.report_progress(progress=20, total=100)

    try:
        modelcard_client = get_mm_client(ctx, 'modelcard')
        await ctx.report_progress(progress=40, total=100)
        
        resp = await asyncio.to_thread(modelcard_client.create_modelcard, data)
        await ctx.report_progress(progress=80, total=100)

        if hasattr(resp, 'status_code') and resp.status_code >= 400:
            error_msg = getattr(resp, 'text', str(resp))
            await ctx.error(f"API error: {error_msg}")
            return create_error_response(
                message=f"API error: {error_msg}",
                error_type="APIError",
                status_code=resp.status_code
            )

        response_data = safe_response_to_dict(resp)
        await ctx.info("Modelcard created successfully")
        await ctx.report_progress(progress=100, total=100)
        
        response_data["status"] = "success"
        response_data["message"] = "Successfully created modelcard"
        return response_data
        
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message=f"Validation error: {str(e)}",
            error_type="ValueError"
        )
    except Exception as e:
        await ctx.error(f"Failed to create modelcard: {str(e)}")
        return create_error_response(
            message=f"Failed to create modelcard: {str(e)}",
            error_type=type(e).__name__
        )

@mcp.tool(
    name="create_modelcard_bulk",
    description="Create modelcards in bulk for a usecase",
    tags={"modelcard", "modelmanager", "create"},
    meta={"version": "1.0", "author": "HexagonML"},
)
async def create_modelcard_bulk(ctx: Context, usecase_id: str) -> dict:
    """Create modelcards in bulk for a usecase.
    
    Args:
        ctx: The MCP server context containing authentication and configuration.
        usecase_id: The usecase ID to create modelcards for (required).
        
    Returns:
        dict: Response containing the bulk creation results or error information.
    """
    # Validate required field
    if not usecase_id or not usecase_id.strip():
        await ctx.error("Usecase ID cannot be empty")
        return create_error_response(
            message="Usecase ID is required",
            error_type="ValidationError"
        )

    await ctx.info(f"Creating bulk modelcards for usecase: {usecase_id}")
    await ctx.report_progress(progress=20, total=100)

    try:
        modelcard_client = get_mm_client(ctx, 'modelcard')
        await ctx.report_progress(progress=40, total=100)
        
        resp = await asyncio.to_thread(modelcard_client.create_modelcard_bulk, usecase_id)
        await ctx.report_progress(progress=80, total=100)

        response_data = safe_response_to_dict(resp)
        await ctx.info("Bulk modelcards created successfully")
        await ctx.report_progress(progress=100, total=100)
        
        return response_data
        
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message=f"Validation error: {str(e)}",
            error_type="ValueError"
        )
    except Exception as e:
        await ctx.error(f"Failed to create bulk modelcards: {str(e)}")
        return create_error_response(
            message=f"Failed to create bulk modelcards: {str(e)}",
            error_type=type(e).__name__
        )
