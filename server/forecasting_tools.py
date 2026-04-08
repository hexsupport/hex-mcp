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


def _format_forecast_response(raw: dict) -> dict:
    """Reshape the raw API response into a clean, LLM-friendly structure.

    Normalises inconsistent key casing, renames abbreviated fields, and
    adds a pre-computed summary so the model does not have to scan every
    data point to understand the shape of the result.
    """
    usecase_info = raw.get("usecase") or {}
    filters = raw.get("filters_applied") or {}
    info = raw.get("info") or {}
    raw_points = raw.get("forecast") or []

    # Normalise each forecast point to consistent snake_case keys
    points = []
    for p in raw_points:
        points.append({
            "date":        p.get("Forecast Date"),
            "value":       p.get("Forecast Value"),
            "lower_bound": p.get("value_lb"),
            "upper_bound": p.get("value_ub"),
            "value_type":  p.get("value_type"),
            "region_code": p.get("rgn_cd"),
        })

    # Pre-compute a summary to give the LLM an at-a-glance overview
    summary: dict = {"total_points": len(points)}
    if points:
        dates  = [p["date"]  for p in points if p["date"]  is not None]
        values = [p["value"] for p in points if p["value"] is not None]
        if dates:
            summary["date_range"] = {"from": dates[0], "to": dates[-1]}
        if values:
            summary["value_range"] = {
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "avg": round(sum(values) / len(values), 4),
            }

    return {
        "status":             "success",
        "message":            "Successfully retrieved forecast",
        "resolved":           raw.get("resolved"),
        "usecase": {
            "id":   usecase_info.get("id"),
            "name": usecase_info.get("name"),
            "type": usecase_info.get("usecase_type"),
        },
        "condition_info": {
            "condition_type":   info.get("condition_type"),
            "required_filters": info.get("required_filters"),
        },
        "filters_applied":    filters,
        "last_actual_update": raw.get("last_actual_update"),
        "forecast_summary":   summary,
        "forecast":           points,
        "_llm_instructions": {
            "role": "You are presenting forecast data to a non-technical business user.",
            "rules": [
                "Lead with the forecast summary: date range and value range (min/max/avg).",
                "Explain lower_bound and upper_bound as a confidence interval — the range the actual value is likely to fall within.",
                "Mention last_actual_update to give the user context on how fresh the underlying data is.",
                "If total_points is 0, tell the user no forecast data was found and suggest checking the usecase filters.",
                "Round displayed numbers to 2 decimal places.",
                "Present as a concise narrative paragraph; avoid echoing raw JSON back to the user.",
                "If resolved is false, flag that the API could not resolve the request and ask the user to verify their filters.",
            ],
        },
    }


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

        # HTTP-level error (4xx / 5xx)
        if hasattr(resp, 'status_code') and resp.status_code >= 400:
            error_msg = getattr(resp, 'text', str(resp))
            await ctx.error(f"API error (status {resp.status_code}): {error_msg}")
            return create_error_response(
                message=f"The upstream API returned an error (HTTP {resp.status_code}).",
                error_type="APIError",
                status_code=resp.status_code,
            )

        response_data = safe_response_to_dict(resp)

        # API-level failure signalled via the resolved flag
        if not response_data.get("resolved", True):
            await ctx.error("API returned resolved=false for forecast request")
            return create_error_response(
                message="The forecast API was unable to resolve the request. Check your filters and usecase configuration.",
                error_type="ForecastError",
            )

        formatted = _format_forecast_response(response_data)

        point_count = formatted["forecast_summary"]["total_points"]
        await ctx.info(f"Forecast retrieved: {point_count} data point(s)")
        await ctx.report_progress(progress=100, total=100)

        return formatted

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
