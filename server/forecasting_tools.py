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
import json


def format_api_response(raw: dict) -> dict:
    """Adapt any forecast API response into an LLM-friendly structure.

    Detects the shape of *raw* and picks the right formatting path:

    1. **success=false with available_options** → error with corrective hints
    2. **success=false (generic)** → plain error
    3. **forecast list present** → structured forecast data
    4. **anything else** → pass-through unchanged
    """
    # ── Unwrap: raw may already be a dict, or may need JSON parsing ──
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"status": "error", "error": raw}

    if not isinstance(raw, dict):
        return {"status": "error", "error": str(raw)}

    data = raw.get("data") or {}

    # ── Path 1 & 2: API signalled failure ──
    if raw.get("success") is False:
        error_msg = data.get("message") or raw.get("error", "Unknown error")
        invalid_filters = data.get("invalid_filters") or {}
        available_options = data.get("available_options") or {}

        result = {
            "status": "error",
            "status_code": raw.get("status_code"),
            "error": error_msg,
            "invalid_filters": invalid_filters,
        }

        if available_options:
            result["available_options"] = {
                "series": available_options.get("series", []),
                "condition_one": available_options.get("condition_one", []),
                "conditions": available_options.get("conditions", {}),
            }
            result["_llm_instructions"] = {
                "role": "You are helping a business user fix an invalid forecast request.",
                "rules": [
                    "The API rejected the request because the filter combination is invalid.",
                    "Show the user which filters were invalid using `invalid_filters`.",
                    "Present the `available_options` so the user knows what valid values they can choose from.",
                    "List `series` options, `condition_one` options, and the nested `conditions` mapping (condition_one -> condition_two values).",
                    "Be concise and friendly. Do not echo raw JSON — present the options as a readable list.",
                    "Suggest a corrected request based on the available options.",
                ],
            }
        else:
            result["_llm_instructions"] = {
                "role": "You are reporting a forecast API error to a business user.",
                "rules": [
                    "Tell the user the request failed and show the error message.",
                    "If invalid_filters are present, explain which filters were wrong.",
                    "Be concise and friendly. Do not echo raw JSON.",
                ],
            }

        return result

    # ── Path 3: successful response with forecast data ──
    raw_points = data.get("forecast") or raw.get("forecast")
    if raw_points:
        usecase_info = data.get("usecase") or raw.get("usecase") or {}
        filters = data.get("filters_applied") or raw.get("filters_applied") or {}
        info = data.get("info") or raw.get("info") or {}
        total_points = len(raw_points)

        return {
            "status": "success",
            "message": "Successfully retrieved forecast",
            "data_available": total_points > 0,
            "resolved": data.get("resolved") if data else raw.get("resolved"),
            "usecase": {
                "id":   usecase_info.get("id"),
                "name": usecase_info.get("name"),
                "type": usecase_info.get("usecase_type"),
            },
            "condition_info": {
                "condition_type": info.get("condition_type"),
                "required_filters": info.get("required_filters"),
            },
            "filters_applied": filters,
            "last_actual_update": data.get("last_actual_update") or raw.get("last_actual_update"),
            "forecast": raw_points,
            "_llm_instructions": {
                "role": "You are presenting forecast data to a non-technical business user.",
                "rules": [
                    "The `forecast` list contains forecast entries.",
                    "Each entry has keys: 'Forecast Date', 'Forecast Value', 'value_lb', 'value_ub', 'value_type', 'rgn_cd', 'fac_id_cd', 'model_type', 'model_id'.",
                    "Describe 'value_lb' and 'value_ub' as the confidence interval lower and upper bounds.",
                    "Mention last_actual_update to indicate how fresh the underlying data is.",
                    "Round displayed numbers to 2 decimal places.",
                    "Present as a concise narrative paragraph; do not echo raw JSON or field names to the user.",
                ],
            },
        }

    # ── Path 4: unrecognised shape → pass through unchanged ──
    return raw


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
