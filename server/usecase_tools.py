"""
Usecase management tools for the ModelManager MCP Server.

This module provides MCP tools for creating, updating, deleting, and managing
usecases (projects) in the ModelManager service, including forecasting usecases.
"""

from fastmcp import Context
from config import mcp
from clients import get_mm_client
from utils import safe_response_to_dict, create_error_response, normalize_tool_response
import asyncio

@mcp.tool(
    name="add_usecase",
    description="Create a new usecase in the ModelManager service with optional forecasting configuration",
    tags={"usecase", "create", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def add_usecase(
    ctx: Context, 
    name: str,
    usecase_type: str = "General",
    author: str = None,
    description: str = None,
    source: str = None,
    contributor: str = None,
    image: str = None,
    performance_data_selection: str = None,
    applications: str = None,
    forecasting_performance_data_selection: str = None,
    notification_emails: list = None,
    forecasting_template: str = None,
    result_tab: bool = False,
    series_tab: bool = False,
    condition_tab: bool = False,
    performance_tab: bool = False,
    ab_testing_tab: bool = False,
    release_tab: bool = False
) -> dict:
    """Create a new usecase in the ModelManager service.
    
    Args:
        ctx: The MCP server context.
        name: Name of the usecase (required).
        usecase_type: Type of usecase (e.g., 'Forecasting', 'General').
        author: Author of the usecase.
        description: Description of the usecase.
        source: Source of the usecase.
        contributor: Contributor to the usecase.
        image: Image URL or path.
        performance_data_selection: Performance data selection criteria.
        applications: Applications associated with the usecase.
        forecasting_performance_data_selection: Forecasting-specific performance data selection (JSON string).
        notification_emails: List of notification emails for forecasting.
        forecasting_template: Forecasting template type.
        result_tab: Enable result tab for forecasting.
        series_tab: Enable series tab for forecasting.
        condition_tab: Enable condition tab for forecasting.
        performance_tab: Enable performance tab for forecasting.
        ab_testing_tab: Enable A/B testing tab for forecasting.
        release_tab: Enable release tab for forecasting.
        
    Returns:
        dict: Response from the ModelManager service with the created usecase details.
    """
    # Validate required field
    if not name or not name.strip():
        await ctx.error("Usecase name cannot be empty")
        return create_error_response(
            message="Usecase name is required",
            error_type="ValidationError"
        )
    
    # Build usecase_info dict
    usecase_info = {
        "name": name,
        "usecase_type": usecase_type,
        "author": author or "",
        "description": description or "",
        "source": source or "",
        "contributor": contributor or "",
        "image": image or "",
        "performance_data_selection": performance_data_selection or "",
        "applications": applications or ""
    }
    
    # Build forecasting_fields dict if any forecasting parameters are provided
    forecasting_fields = {}
    if forecasting_performance_data_selection:
        forecasting_fields["performance_data_selection"] = forecasting_performance_data_selection
    if notification_emails:
        forecasting_fields["notification_emails"] = notification_emails
    if forecasting_template:
        forecasting_fields["forecasting_template"] = forecasting_template
    
    # Build forecasting_feature_tabs dict if any tab parameters are provided
    forecasting_feature_tabs = {}
    if any([result_tab, series_tab, condition_tab, performance_tab, ab_testing_tab, release_tab]):
        forecasting_feature_tabs = {
            "result_tab": result_tab,
            "series_tab": series_tab,
            "condition_tab": condition_tab,
            "performance_tab": performance_tab,
            "ab_testing_tab": ab_testing_tab,
            "release_tab": release_tab
        }
    
    await ctx.info(f"Creating new usecase: {name}")
    await ctx.report_progress(progress=20, total=100)
    
    try:
        usecase_client = get_mm_client(ctx, 'usecase')
        await ctx.report_progress(progress=40, total=100)
        
        response = await asyncio.to_thread(
            usecase_client.post_usecase, 
            usecase_info, 
            forecasting_fields, 
            forecasting_feature_tabs
        )
        await ctx.report_progress(progress=80, total=100)
        
        result = safe_response_to_dict(response)
        
        usecase_id = result.get('id') or result.get('usecase_id')
        if usecase_id:
            await ctx.info(f"Usecase created successfully with ID: {usecase_id}")
        else:
            await ctx.info("Usecase created successfully")
        
        await ctx.report_progress(progress=100, total=100)
        return result
        
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message=f"Validation error: {str(e)}",
            error_type="ValueError"
        )
    except Exception as e:
        await ctx.error(f"Failed to add usecase: {str(e)}")
        return create_error_response(
            message=f"Failed to add usecase: {str(e)}",
            error_type=type(e).__name__
        )

@mcp.tool(
    name="update_usecase",
    description="Update an existing usecase in the ModelManager service with new configuration",
    tags={"usecase", "update", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def update_usecase(
    ctx: Context, 
    usecase_id: str,
    name: str = None,
    usecase_type: str = None,
    author: str = None,
    description: str = None,
    source: str = None,
    contributor: str = None,
    image: str = None,
    performance_data_selection: str = None,
    applications: str = None,
    forecasting_performance_data_selection: str = None,
    notification_emails: list = None,
    forecasting_template: str = None,
    result_tab: bool = None,
    series_tab: bool = None,
    condition_tab: bool = None,
    performance_tab: bool = None,
    ab_testing_tab: bool = None,
    release_tab: bool = None
) -> dict:
    """Update an existing usecase in the ModelManager service.
    
    Args:
        ctx: The MCP server context.
        usecase_id: The unique identifier of the usecase to update (required).
        name: New name of the usecase.
        usecase_type: New type of usecase (e.g., 'Forecasting', 'General').
        author: New author of the usecase.
        description: New description of the usecase.
        source: New source of the usecase.
        contributor: New contributor to the usecase.
        image: New image URL or path.
        performance_data_selection: New performance data selection criteria.
        applications: New applications associated with the usecase.
        forecasting_performance_data_selection: New forecasting-specific performance data selection (JSON string).
        notification_emails: New list of notification emails for forecasting.
        forecasting_template: New forecasting template type.
        result_tab: New result tab setting for forecasting.
        series_tab: New series tab setting for forecasting.
        condition_tab: New condition tab setting for forecasting.
        performance_tab: New performance tab setting for forecasting.
        ab_testing_tab: New A/B testing tab setting for forecasting.
        release_tab: New release tab setting for forecasting.
        
    Returns:
        dict: Response from the ModelManager service with the updated usecase details.
    """
    # Validate required field
    if not usecase_id or not usecase_id.strip():
        await ctx.error("Usecase ID cannot be empty")
        return create_error_response(
            message="Usecase ID is required",
            error_type="ValidationError"
        )
    
    # Build usecase_data dict with only provided values
    usecase_data = {}
    
    # Add usecase_info fields if provided
    usecase_info = {}
    info_fields = {
        "name": name,
        "usecase_type": usecase_type,
        "author": author,
        "description": description,
        "source": source,
        "contributor": contributor,
        "image": image,
        "performance_data_selection": performance_data_selection,
        "applications": applications
    }
    
    for field, value in info_fields.items():
        if value is not None:
            usecase_info[field] = value
    
    # Add forecasting_fields if any forecasting parameters are provided
    forecasting_fields = {}
    if forecasting_performance_data_selection is not None:
        forecasting_fields["performance_data_selection"] = forecasting_performance_data_selection
    if notification_emails is not None:
        forecasting_fields["notification_emails"] = notification_emails
    if forecasting_template is not None:
        forecasting_fields["forecasting_template"] = forecasting_template
    
    # Add forecasting_feature_tabs if any tab parameters are provided
    forecasting_feature_tabs = {}
    if any(tab is not None for tab in [result_tab, series_tab, condition_tab, performance_tab, ab_testing_tab, release_tab]):
        forecasting_feature_tabs = {
            "result_tab": result_tab,
            "series_tab": series_tab,
            "condition_tab": condition_tab,
            "performance_tab": performance_tab,
            "ab_testing_tab": ab_testing_tab,
            "release_tab": release_tab
        }
    
    # Combine all data
    if usecase_info:
        usecase_data["usecase_info"] = usecase_info
    if forecasting_fields:
        usecase_data["forecasting_fields"] = forecasting_fields
    if forecasting_feature_tabs:
        usecase_data["forecasting_feature_tabs"] = forecasting_feature_tabs
    
    # Check if there's anything to update
    if not usecase_data:
        await ctx.error("No update data provided")
        return create_error_response(
            message="At least one field must be provided for update",
            error_type="ValidationError"
        )
    
    await ctx.info(f"Updating usecase: {usecase_id}")
    await ctx.report_progress(progress=20, total=100)
    
    try:
        usecase_client = get_mm_client(ctx, 'usecase')
        await ctx.report_progress(progress=40, total=100)
        
        response = await asyncio.to_thread(usecase_client.patch_usecase, usecase_data, usecase_id)
        await ctx.report_progress(progress=80, total=100)
        
        result = safe_response_to_dict(response)
        
        if result.get('status') == 'error':
            await ctx.error(f"Failed to update usecase: {result.get('message', 'Unknown error')}")
        else:
            await ctx.info(f"Usecase updated successfully: {usecase_id}")
        
        await ctx.report_progress(progress=100, total=100)
        return result
        
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message=f"Validation error: {str(e)}",
            error_type="ValueError"
        )
    except Exception as e:
        await ctx.error(f"Failed to update usecase: {str(e)}")
        return create_error_response(
            message=f"Failed to update usecase: {str(e)}",
            error_type=type(e).__name__
        )

@mcp.tool(
    name="delete_usecase",
    description="Delete a usecase from the ModelManager service permanently",
    tags={"usecase", "delete", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def delete_usecase(ctx: Context, usecase_id: str) -> dict:
    """Delete a usecase from the ModelManager service permanently.
    
    Args:
        ctx: The MCP server context containing authentication and configuration.
        usecase_id: The unique identifier of the usecase to delete (required).
        
    Returns:
        dict: Response from the ModelManager service confirming deletion.
    """
    # Validate required field
    if not usecase_id or not usecase_id.strip():
        await ctx.error("Usecase ID cannot be empty")
        return create_error_response(
            message="Usecase ID is required",
            error_type="ValidationError"
        )
    
    await ctx.info(f"Deleting usecase: {usecase_id}")
    await ctx.report_progress(progress=20, total=100)
    
    try:
        usecase_client = get_mm_client(ctx, 'usecase')
        await ctx.report_progress(progress=40, total=100)
        
        delete_response = await asyncio.to_thread(usecase_client.delete_usecase, usecase_id)
        await ctx.report_progress(progress=80, total=100)
        
        result = safe_response_to_dict(delete_response)
        await ctx.info(f"Usecase deleted successfully: {usecase_id}")
        await ctx.report_progress(progress=100, total=100)
        
        return result
        
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message=f"Validation error: {str(e)}",
            error_type="ValueError"
        )
    except Exception as e:
        await ctx.error(f"Failed to delete usecase: {str(e)}")
        return create_error_response(
            message=f"Failed to delete usecase: {str(e)}",
            error_type=type(e).__name__
        )

@mcp.tool(
    name="get_usecase_data",
    description="Retrieve and summarize all usecases from the ModelManager API",
    tags={"usecase", "list", "summary", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def get_usecase_data(ctx: Context) -> dict:
    """Retrieve and summarize all usecases from the ModelManager API.
    
    Args:
        ctx: The MCP server context containing authentication and configuration.
        
    Returns:
        dict: Response containing summarized usecase data or error information.
    """
    await ctx.info("Retrieving all usecases")
    await ctx.report_progress(progress=20, total=100)
    
    try:
        usecase_client = get_mm_client(ctx, 'usecase')
        await ctx.report_progress(progress=40, total=100)
        
        usecases_response = await asyncio.to_thread(usecase_client.get_usecases)
        await ctx.report_progress(progress=80, total=100)
        
        data = safe_response_to_dict(usecases_response)
        await ctx.info(f"Retrieved {len(data) if isinstance(data, list) else 0} usecases")
        await ctx.report_progress(progress=100, total=100)

        return normalize_tool_response(
            {"summary": data},
            success_message="Successfully retrieved usecases",
        )
        
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message=f"Validation error: {str(e)}",
            error_type="ValueError"
        )
    except Exception as e:
        await ctx.error(f"Failed to retrieve usecases: {str(e)}")
        return create_error_response(
            message=f"Failed to retrieve usecases: {str(e)}",
            error_type=type(e).__name__
        )
