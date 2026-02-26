"""
Forecasting tools for the ModelManager MCP Server.

This module provides MCP tools for forecasting operations including
retrieving forecasts and validating forecast payloads.
"""

from fastmcp import Context
from config import mcp
from clients import get_mm_client
from validators import validate_forecast_payload
from utils import safe_response_to_dict, create_error_response
import asyncio

@mcp.tool(
    name="get_forecast",
    description="Retrieve forecast for a usecase",
    tags={"Forecast", "modelmanager", "get", "forecasting_usecase"},
    meta={"version": "1.0", "author": "HexagonML"},
)
async def get_forecast(
    ctx: Context, 
    usecase_name: str = None,
    usecase_id: int = None,
    series: str = None,
    condition_one: str = None,
    condition_two: str = None,
    condition_three: str = None,
    prediction_period: int = None
) -> dict:
    """Retrieve forecast for a usecase.
    
    Args:
        ctx: The MCP server context containing authentication and configuration.
        usecase_name: Name of the usecase to retrieve forecast for.
        usecase_id: ID of the usecase to retrieve forecast for.
        series: Series name for the forecast.
        condition_one: First condition parameter.
        condition_two: Second condition parameter.
        condition_three: Third condition parameter.
        prediction_period: Period for prediction (number of periods).
        
    Returns:
        dict: Response containing forecast data or error information.
    """
    # Build payload from individual parameters
    payload = {}
    
    if usecase_id is not None:
        payload["usecase_id"] = usecase_id
    elif usecase_name is not None:
        payload["usecase_name"] = usecase_name
    else:
        await ctx.error("Either usecase_name or usecase_id must be provided")
        return create_error_response(
            message="Either usecase_name or usecase_id must be provided",
            error_type="ValidationError"
        )
    
    # Add optional parameters
    optional_params = {
        "series": series,
        "condition_one": condition_one,
        "condition_two": condition_two,
        "condition_three": condition_three,
        "prediction_period": prediction_period
    }
    
    for param, value in optional_params.items():
        if value is not None:
            payload[param] = value
    
    # Validate payload
    validated_payload = validate_forecast_payload(payload)
    if "status" in validated_payload and validated_payload["status"] == "error":
        await ctx.error(validated_payload.get("message", "Payload validation failed"))
        return validated_payload

    await ctx.info(f"Retrieving forecast for usecase: {usecase_name or usecase_id}")
    await ctx.report_progress(progress=20, total=100)

    try:
        usecase_forecast_client = get_mm_client(ctx, 'forecast')
        await ctx.report_progress(progress=40, total=100)
        
        resp = await asyncio.to_thread(usecase_forecast_client.get_forecast, validated_payload)
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
        await ctx.info("Forecast retrieved successfully")
        await ctx.report_progress(progress=100, total=100)
        
        return response_data
        
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message=f"Validation error: {str(e)}",
            error_type="ValueError"
        )
    except Exception as e:
        await ctx.error(f"Failed to get model forecast: {str(e)}")
        return create_error_response(
            message=f"Failed to get model forecast: {str(e)}",
            error_type=type(e).__name__
        )
