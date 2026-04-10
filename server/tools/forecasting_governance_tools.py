"""
Forecast governance report tools for the ModelManager MCP Server.

This module provides MCP tools for generating forecast governance reports,
which provide audit-ready documentation for forecast model performance and compliance.
"""

from fastmcp import Context
from config import mcp
from clients import get_mm_client
from utils import safe_response_to_dict, create_error_response
from handlers import dispatch_response
import asyncio


@mcp.tool(
    name="get_forecast_governance_report",
    description=(
        "Generate a forecast governance report for a ModelManager usecase. "
        "Returns an audit-ready report documenting model performance, compliance, "
        "and forecast quality metrics for the specified series and conditions."
    ),
    tags={"governance", "modelmanager", "forecast", "report"},
    meta={"version": "1.0", "author": "HexagonML"},
)
async def get_forecast_governance_report(
    ctx: Context,
    usecase_name: str = None,
    usecase_id: str = None,
    series: str = None,
    condition_one: str = None,
    condition_two: str = None,
    condition_three: str = None,
) -> dict:
    """Generate a forecast governance report for a usecase.

    Args:
        ctx: MCP server context containing authentication and configuration.
        usecase_name: Name of the usecase. Either this or usecase_id is required.
        usecase_id: Numeric ID of the usecase. Either this or usecase_name is required.
        series: Time-series name (e.g. "ED Visits"). Required for report generation.
        condition_one: Primary filter/dimension (e.g. "DRV"). Required for report generation.
        condition_two: Secondary filter/dimension (e.g. "1_year") (optional).
        condition_three: Tertiary filter/dimension (e.g. "October_2025") (optional).

    Returns:
        dict: Governance report response containing:
            - usecase: id and name
            - series: the series used for the report
            - conditions: applied condition filters
            - report data from the governance API
    """
    usecase_id_clean   = usecase_id.strip()   if isinstance(usecase_id,   str) else None
    usecase_name_clean = usecase_name.strip() if isinstance(usecase_name, str) else None

    if not usecase_id_clean and not usecase_name_clean:
        await ctx.error("At least one of usecase_id or usecase_name must be provided")
        return create_error_response(
            message="At least one of usecase_id or usecase_name must be provided",
            error_type="ValidationError",
        )

    if not series or not series.strip():
        await ctx.error("series is required")
        return create_error_response(
            message="series is required for governance report generation",
            error_type="ValidationError",
        )

    if not condition_one or not condition_one.strip():
        await ctx.error("condition_one is required")
        return create_error_response(
            message="condition_one is required for governance report generation",
            error_type="ValidationError",
        )

    data: dict = {
        "series": series.strip(),
        "condition_one": condition_one.strip(),
    }

    if usecase_id_clean:
        data["usecase_id"] = usecase_id_clean
    if usecase_name_clean:
        data["usecase_name"] = usecase_name_clean
    if condition_two is not None and condition_two.strip():
        data["condition_two"] = condition_two.strip()
    if condition_three is not None and condition_three.strip():
        data["condition_three"] = condition_three.strip()

    await ctx.info(f"Generating governance report for usecase: {usecase_name_clean or usecase_id_clean}")
    await ctx.report_progress(progress=20, total=100)

    try:
        governance_client = get_mm_client(ctx, 'governance')
        await ctx.report_progress(progress=40, total=100)

        resp = await asyncio.to_thread(governance_client.get_forecast_governance_report, data)
        await ctx.report_progress(progress=80, total=100)

        if isinstance(resp, Exception):
            await ctx.error(f"Governance report API request failed: {resp}")
            return create_error_response(
                message="Failed to connect to the governance report API. Please check your configuration.",
                error_type="APIError",
            )

        if hasattr(resp, 'status_code') and resp.status_code >= 400:
            error_msg = getattr(resp, 'text', str(resp))
            await ctx.error(f"API error (status {resp.status_code}): {error_msg}")
            return create_error_response(
                message=f"The upstream API returned an error (HTTP {resp.status_code}).",
                error_type="APIError",
                status_code=resp.status_code,
            )

        response_data = safe_response_to_dict(resp)
        await ctx.report_progress(progress=100, total=100)

        return dispatch_response(response_data)

    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message="A validation error occurred. Please check your input.",
            error_type="ValidationError",
        )
    except Exception as e:
        await ctx.error(f"Failed to generate governance report: {str(e)}")
        return create_error_response(
            message="An internal error occurred while generating the governance report.",
            error_type="InternalError",
        )
