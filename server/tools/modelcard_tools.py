"""
ModelCard management tools for the ModelManager MCP Server.

This module provides MCP tools for creating and managing model cards,
which provide standardized documentation for machine learning models.
"""

from fastmcp import Context
from config import mcp
from clients import get_mm_client
from utils import safe_response_to_dict, create_error_response
from handlers import dispatch_response
import asyncio

@mcp.tool(
    name="create_modelcard",
    description="Create a forecasting modelcard with required series and condition parameters",
    tags={"modelcard", "modelmanager", "create", "forecasting"},
    meta={"version": "1.0", "author": "HexagonML"},
)
async def create_modelcard(
    ctx: Context,
    usecase_name: str,
    series: str,
    usecase_id: str = None,
    condition_one: str = None,
    condition_two: str = None,
    condition_three: str = None,
) -> dict:
    """Create a forecasting modelcard for a usecase with required parameters.

    Args:
        ctx: The MCP server context containing authentication and configuration.
        usecase_name: Name of the usecase. Used for fuzzy/semantic lookup when usecase_id is not provided. REQUIRED.
        usecase_id: Primary key of the usecase/project. Takes priority over usecase_name if both are provided (optional).
        series: The forecasting series/value_type dimension (e.g. 'value_type_A'). REQUIRED.
        condition_one: Region filter (rgn_cd). Maps to conditionOne/region in forecasting data. REQUIRED.
        condition_two: Facility filter (fac_id_cd). Maps to conditionTwo/facility (optional).
        condition_three: Unit filter (unit_id). Maps to conditionThree/unit (optional).

    Returns:
        dict: Response containing the created modelcard data or error information.
    """
    # Validate required fields
    if not series or not series.strip():
        await ctx.error("series is required")
        return create_error_response(
            message="series is required for forecasting modelcard creation",
            error_type="ValidationError",
        )

    if not condition_one or not condition_one.strip():
        await ctx.error("condition_one is required")
        return create_error_response(
            message="condition_one is required for forecasting modelcard creation",
            error_type="ValidationError",
        )

    # Validate usecase identifier
    usecase_id_clean = usecase_id.strip() if isinstance(usecase_id, str) else None
    usecase_name_clean = usecase_name.strip() if isinstance(usecase_name, str) else None

    if not usecase_id_clean and not usecase_name_clean:
        await ctx.error("At least one of usecase_id or usecase_name must be provided")
        return create_error_response(
            message="At least one of usecase_id or usecase_name must be provided",
            error_type="ValidationError",
        )

    # Build data dict with all provided parameters
    data = {
        "series": series.strip(),
        "condition_one": condition_one.strip()
    }

    if usecase_id_clean:
        data["usecase_id"] = usecase_id_clean
    if usecase_name_clean:
        data["usecase_name"] = usecase_name_clean
    if condition_two is not None and condition_two.strip():
        data["condition_two"] = condition_two.strip()
    if condition_three is not None and condition_three.strip():
        data["condition_three"] = condition_three.strip()

    await ctx.info(f"Creating modelcard for usecase: {usecase_name_clean or usecase_id_clean}")
    await ctx.report_progress(progress=20, total=100)

    try:
        modelcard_client = get_mm_client(ctx, 'modelcard')
        await ctx.report_progress(progress=40, total=100)

        resp = await asyncio.to_thread(modelcard_client.create_modelcard, data)
        await ctx.report_progress(progress=80, total=100)

        if hasattr(resp, 'status_code') and resp.status_code >= 400:
            error_msg = getattr(resp, 'text', str(resp))
            await ctx.error(f"API error (status {resp.status_code}): {error_msg}")
            return create_error_response(
                message=f"The upstream API returned an error (HTTP {resp.status_code}).",
                error_type="APIError",
                status_code=resp.status_code
            )

        response_data = safe_response_to_dict(resp)
        await ctx.info("Modelcard created successfully")
        await ctx.report_progress(progress=100, total=100)

        return dispatch_response(response_data)

    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message="A validation error occurred. Please check your input.",
            error_type="ValidationError"
        )
    except Exception as e:
        await ctx.error(f"Failed to create modelcard: {str(e)}")
        return create_error_response(
            message="An internal error occurred while creating the modelcard.",
            error_type="InternalError"
        )
