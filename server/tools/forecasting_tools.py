"""
Forecasting tools for the ModelManager MCP Server.

This module provides MCP tools for forecasting operations including
retrieving forecasts and validating forecast payloads.
"""

from fastmcp import Context
from config import mcp
from clients import get_mm_client
from utils import validate_forecast_payload, safe_response_to_dict, create_error_response
from handlers import dispatch_response
import asyncio


def format_api_response(raw: dict) -> dict:
    """Adapt any forecast API response into an LLM-friendly structure.

    Delegates to the response_handlers module which classifies the response
    into one of 16 documented scenarios and applies the appropriate formatter.
    Every response includes a 'status' field and 'prompt_hint' for the LLM.

    Args:
        raw: The raw API response (may be dict, string, or other type)

    Returns:
        dict: Formatted response with status, contextual fields, and prompt_hint
    """
    return dispatch_response(raw)


@mcp.tool(
    name="get_forecast",
    description=(
        "Retrieve time-series forecast data for a ModelManager usecase. "
        "Returns predicted values with confidence intervals, the last actual "
        "data date, applied filters, and a pre-computed summary (count, date "
        "range, min/max/avg value)."
    ),
    tags={"Forecast", "modelmanager", "get", "forecasting_usecase"},
    meta={"version": "2.0", "author": "HexagonML"},
)
async def get_forecast(
    ctx: Context,
    usecase_name: str = None,
    usecase_id: str = None,
    series: str = None,
    condition_one: str = None,
    condition_two: str = None,
    condition_three: str = None,
    prediction_period: str = None,
) -> dict:
    """Retrieve time-series forecast data for a usecase.

    Args:
        ctx: MCP server context containing authentication and configuration.
        usecase_name: Name of the usecase. Either this or usecase_id is required.
        usecase_id: Numeric ID of the usecase. Either this or usecase_name is required.
        series: Time-series name to forecast (e.g. "ED Visits").
        condition_one: Primary filter/dimension (e.g. "DRV").
        condition_two: Secondary filter/dimension (e.g. "1_year").
        condition_three: Tertiary filter/dimension (e.g. "October_2025").
        prediction_period: Number of periods to forecast as a numeric string (e.g. "30").

    Returns:
        dict: Normalised forecast response containing:
            - usecase: id, name, type
            - condition_info: condition_type and required_filters
            - filters_applied: the filters the API used
            - last_actual_update: date of the most recent actual data point
            - forecast_summary: total_points, date_range, value_range (min/max/avg)
            - forecast: list of {date, value, lower_bound, upper_bound, value_type, region_code}
    """
    # --- Input cleaning & validation ---
    usecase_id_clean   = usecase_id.strip()   if isinstance(usecase_id,   str) else None
    usecase_name_clean = usecase_name.strip() if isinstance(usecase_name, str) else None

    payload: dict = {}
    if usecase_id_clean:
        payload["usecase_id"] = usecase_id_clean
    if usecase_name_clean:
        payload["usecase_name"] = usecase_name_clean

    if not payload:
        await ctx.error("At least one of usecase_id or usecase_name must be provided")
        return create_error_response(
            message="At least one of usecase_id or usecase_name must be provided",
            error_type="ValidationError",
        )

    for param, value in {
        "series":           series,
        "condition_one":    condition_one,
        "condition_two":    condition_two,
        "condition_three":  condition_three,
        "prediction_period": prediction_period,
    }.items():
        if value is not None:
            payload[param] = value

    validated_payload = validate_forecast_payload(payload)
    if isinstance(validated_payload, dict) and validated_payload.get("status") == "error":
        await ctx.error(validated_payload.get("message", "Payload validation failed"))
        return validated_payload

    await ctx.info(f"Retrieving forecast for usecase: {usecase_name_clean or usecase_id_clean}")
    await ctx.report_progress(progress=20, total=100)

    try:
        forecast_client = get_mm_client(ctx, 'forecast')
        await ctx.report_progress(progress=40, total=100)

        resp = await asyncio.to_thread(forecast_client.get_forecast, validated_payload)
        await ctx.report_progress(progress=80, total=100)

        # The mmanager client returns Exception objects instead of raising them
        if isinstance(resp, Exception):
            await ctx.error(f"Forecast API request failed: {resp}")
            return create_error_response(
                message="Failed to connect to the forecast API. Please check your configuration.",
                error_type="APIError",
            )

        # HTTP-level error (4xx / 5xx) — extract the body for formatting
        if hasattr(resp, 'status_code') and resp.status_code >= 400:
            error_msg = getattr(resp, 'text', str(resp))
            await ctx.error(f"API error (status {resp.status_code}): {error_msg}")
            return format_api_response(error_msg)

        response_data = safe_response_to_dict(resp)
        return format_api_response(response_data)

    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message="A validation error occurred. Please check your input.",
            error_type="ValidationError",
        )
    except Exception as e:
        await ctx.error(f"Failed to get model forecast: {str(e)}")
        return create_error_response(
            message="An internal error occurred while retrieving the forecast.",
            error_type="InternalError",
        )
