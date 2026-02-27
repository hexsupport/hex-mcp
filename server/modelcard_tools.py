"""
ModelCard management tools for the ModelManager MCP Server.

This module provides MCP tools for creating and managing model cards,
which provide standardized documentation for machine learning models.
"""

from fastmcp import Context
from config import mcp
from clients import get_mm_client
from utils import safe_response_to_dict, create_error_response, normalize_tool_response
import asyncio

@mcp.tool(name="get_modelcard_data",
    description="Retrieve modelcard for a given model with optional filtering",
    tags={"modelcard", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"})
async def get_modelcard_data(ctx: Context, usecase_id: str = None, model_id: str = None, series: str = None, condition_one: str = None, condition_two: str = None, condition_three: str = None) -> dict:
    """Retrieve modelcard data for a specific model or usecase with optional filtering.
    
    Args:
        ctx: The MCP server context containing authentication and configuration.
        usecase_id: The usecase ID to filter modelcards (optional).
        model_id: The model ID to filter modelcards (optional).
        series: The series name to filter modelcards (optional).
        condition_one: First condition parameter for filtering (optional).
        condition_two: Second condition parameter for filtering (optional).
        condition_three: Third condition parameter for filtering (optional).
        
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
    
    # Build data dict with all provided parameters
    data = {}
    if usecase_id is not None and usecase_id.strip():
        data["usecase_id"] = usecase_id
    if model_id is not None and model_id.strip():
        data["model_id"] = model_id
    if series is not None and series.strip():
        data["series"] = series
    if condition_one is not None and condition_one.strip():
        data["condition_one"] = condition_one
    if condition_two is not None and condition_two.strip():
        data["condition_two"] = condition_two
    if condition_three is not None and condition_three.strip():
        data["condition_three"] = condition_three

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

        return normalize_tool_response(
            response_data,
            success_message="Successfully retrieved modelcard data",
        )
        
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
    description="Create a modelcard with required parameters",
    tags={"modelcard", "modelmanager", "create"},
    meta={"version": "1.0", "author": "HexagonML"},
)
async def create_modelcard(ctx: Context, usecase_id: str, model_id: str = None, series: str = None, condition_one: str = None, condition_two: str = None, condition_three: str = None) -> dict:
    """Create a modelcard for a usecase with required parameters.
    
    Args:
        ctx: The MCP server context containing authentication and configuration.
        usecase_id: The usecase ID to create the modelcard for (required).
        model_id: The model ID to create the modelcard for (optional, required if usecase is classification or regression).
        series: The series name for the modelcard (optional).
        condition_one: First condition parameter (optional).
        condition_two: Second condition parameter (optional).
        condition_three: Third condition parameter (optional).
        
    Returns:
        dict: Response containing the created modelcard data or error information.
    """
    # Validate required fields
    if not usecase_id or not usecase_id.strip():
        await ctx.error("Usecase ID cannot be empty")
        return create_error_response(
            message="Usecase ID is required",
            error_type="ValidationError"
        )
    
    # Build data dict with all provided parameters
    data = {}
    if usecase_id is not None and usecase_id.strip():
        data["usecase_id"] = usecase_id
    if model_id is not None and model_id.strip():
        data["model_id"] = model_id
    if series is not None and series.strip():
        data["series"] = series
    if condition_one is not None and condition_one.strip():
        data["condition_one"] = condition_one
    if condition_two is not None and condition_two.strip():
        data["condition_two"] = condition_two
    if condition_three is not None and condition_three.strip():
        data["condition_three"] = condition_three

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

        return normalize_tool_response(
            response_data,
            success_message="Successfully created modelcard",
        )
        
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

# @mcp.tool(
#     name="create_modelcard_bulk",
#     description="Create modelcards in bulk for a usecase",
#     tags={"modelcard", "modelmanager", "create"},
#     meta={"version": "1.0", "author": "HexagonML"},
# )
# async def create_modelcard_bulk(ctx: Context, usecase_id: str) -> dict:
#     """Create modelcards in bulk for a usecase.
    
#     Args:
#         ctx: The MCP server context containing authentication and configuration.
#         usecase_id: The usecase ID to create modelcards for (required).
        
#     Returns:
#         dict: Response containing the bulk creation results or error information.
#     """
#     # Validate required field
#     if not usecase_id or not usecase_id.strip():
#         await ctx.error("Usecase ID cannot be empty")
#         return create_error_response(
#             message="Usecase ID is required",
#             error_type="ValidationError"
#         )

#     await ctx.info(f"Creating bulk modelcards for usecase: {usecase_id}")
#     await ctx.report_progress(progress=20, total=100)

#     try:
#         modelcard_client = get_mm_client(ctx, 'modelcard')
#         await ctx.report_progress(progress=40, total=100)
        
#         resp = await asyncio.to_thread(modelcard_client.create_modelcard_bulk, usecase_id)
#         await ctx.report_progress(progress=80, total=100)

#         response_data = safe_response_to_dict(resp)
#         await ctx.info("Bulk modelcards created successfully")
#         await ctx.report_progress(progress=100, total=100)

#         return normalize_tool_response(
#             response_data,
#             success_message="Successfully created modelcards in bulk",
#         )
        
#     except ValueError as e:
#         await ctx.error(f"Validation error: {str(e)}")
#         return create_error_response(
#             message=f"Validation error: {str(e)}",
#             error_type="ValueError"
#         )
#     except Exception as e:
#         await ctx.error(f"Failed to create bulk modelcards: {str(e)}")
#         return create_error_response(
#             message=f"Failed to create bulk modelcards: {str(e)}",
#             error_type=type(e).__name__
#         )
