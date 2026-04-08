"""
Model management tools for the ModelManager MCP Server.

This module provides MCP tools for creating, updating, deleting, and managing
machine learning models in the ModelManager service.
"""

from fastmcp import Context
from config import mcp
from clients import get_mm_client
from utils import safe_response_to_dict, create_error_response, validate_file_path
import asyncio

@mcp.tool(
    name="add_model",
    description="Upload a new machine learning model to the ModelManager service",
    tags={"model", "create", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def add_model(
    ctx: Context,
    name: str,
    description: str,
    project: str = None,
    transformer_type: str = None,
    dataset_insertion_type: str = None,
    training_dataset: str = None,
    test_dataset: str = None,
    pred_dataset: str = None,
    actual_dataset: str = None,
    model_file_path: str = None,
    target_column: str = None
) -> dict:
    """Upload a machine learning model to the ModelManager service.
    
    Args:
        ctx: The MCP server context containing authentication and configuration.
        name: Name of the model (required).
        description: Description of the model (required).
        project: Project ID or identifier.
        transformer_type: Type of transformer (e.g., 'Classification', 'Regression').
        dataset_insertion_type: Dataset insertion method (e.g., 'Manual', 'Automatic').
        training_dataset: Path to training dataset file.
        test_dataset: Path to test dataset file.
        pred_dataset: Path to prediction dataset file.
        actual_dataset: Path to actual/truth dataset file.
        model_file_path: Path to the model file.
        target_column: Target column name in the dataset.
        
    Returns:
        dict: Response from the ModelManager service containing the created model details.
    """
    # Validate required fields
    validation_errors = []
    if not name or not name.strip():
        validation_errors.append("Model name is required")
    if not description or not description.strip():
        validation_errors.append("Model description is required")
    
    # Validate file paths against traversal sequences
    path_params = {
        "training_dataset": training_dataset,
        "test_dataset": test_dataset,
        "pred_dataset": pred_dataset,
        "actual_dataset": actual_dataset,
        "model_file_path": model_file_path,
    }
    for param_name, param_value in path_params.items():
        if param_value is not None and not validate_file_path(param_value):
            validation_errors.append(f"Invalid path for '{param_name}': path traversal sequences are not allowed")

    if validation_errors:
        await ctx.error(f"Validation failed: {'; '.join(validation_errors)}")
        return create_error_response(
            message=f"Validation failed: {'; '.join(validation_errors)}",
            error_type="ValidationError"
        )

    # Build model data dict
    model_data = {
        "name": name,
        "description": description
    }
    
    # Add optional fields if provided
    optional_fields = {
        "project": project,
        "transformerType": transformer_type,
        "datasetinsertionType": dataset_insertion_type,
        "training_dataset": training_dataset,
        "test_dataset": test_dataset,
        "pred_dataset": pred_dataset,
        "actual_dataset": actual_dataset,
        "model_file_path": model_file_path,
        "target_column": target_column
    }
    
    for field, value in optional_fields.items():
        if value is not None:
            model_data[field] = value
    
    await ctx.info(f"Creating new model: {name}")
    await ctx.report_progress(progress=10, total=100)
    
    try:
        model_client = get_mm_client(ctx, 'model')
        await ctx.report_progress(progress=30, total=100)
        
        model_response = await asyncio.to_thread(model_client.post_model, model_data)
        await ctx.report_progress(progress=90, total=100)
        
        response_dict = safe_response_to_dict(model_response)
        
        if 'id' in response_dict:
            await ctx.info(f"Model created successfully with ID: {response_dict['id']}")
        else:
            await ctx.info("Model created successfully")
        
        await ctx.report_progress(progress=100, total=100)
        return response_dict
        
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message="A validation error occurred. Please check your input.",
            error_type="ValidationError"
        )
    except Exception as e:
        await ctx.error(f"Failed to upload model: {str(e)}")
        return create_error_response(
            message="An internal error occurred while uploading the model.",
            error_type="InternalError"
        )

@mcp.tool(
    name="update_model",
    description="Update a machine learning model's metadata or configuration in the ModelManager service",
    tags={"model", "update", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def update_model(
    ctx: Context, 
    model_id: str,
    name: str = None,
    description: str = None,
    project: str = None,
    transformer_type: str = None,
    dataset_insertion_type: str = None,
    training_dataset: str = None,
    test_dataset: str = None,
    pred_dataset: str = None,
    actual_dataset: str = None,
    model_file_path: str = None,
    target_column: str = None,
    create_sweetviz: bool = True
) -> dict:
    """Update a machine learning model's metadata or configuration in the ModelManager service.
    
    Args:
        ctx: The MCP server context containing authentication and configuration.
        model_id: The unique identifier of the model to update (required).
        name: New name of the model.
        description: New description of the model.
        project: New project ID or identifier.
        transformer_type: New type of transformer (e.g., 'Classification', 'Regression').
        dataset_insertion_type: New dataset insertion method (e.g., 'Manual', 'Automatic').
        training_dataset: New path to training dataset file.
        test_dataset: New path to test dataset file.
        pred_dataset: New path to prediction dataset file.
        actual_dataset: New path to actual/truth dataset file.
        model_file_path: New path to the model file.
        target_column: New target column name in the dataset.
        create_sweetviz: Whether to generate a Sweetviz report for data visualization (default: True).
        
    Returns:
        dict: Response from the ModelManager service with updated model details.
    """
    # Validate required field
    if not model_id or not model_id.strip():
        await ctx.error("Model ID cannot be empty")
        return create_error_response(
            message="Model ID is required",
            error_type="ValidationError"
        )
    
    # Build model data dict with only provided values
    model_data = {}
    field_mapping = {
        "name": name,
        "description": description,
        "project": project,
        "transformerType": transformer_type,
        "datasetinsertionType": dataset_insertion_type,
        "training_dataset": training_dataset,
        "test_dataset": test_dataset,
        "pred_dataset": pred_dataset,
        "actual_dataset": actual_dataset,
        "model_file_path": model_file_path,
        "target_column": target_column
    }
    
    for field, value in field_mapping.items():
        if value is not None:
            model_data[field] = value
    
    # Validate file paths against traversal sequences
    path_fields = ["training_dataset", "test_dataset", "pred_dataset", "actual_dataset", "model_file_path"]
    path_errors = [
        f"Invalid path for '{f}': path traversal sequences are not allowed"
        for f in path_fields
        if model_data.get(f) is not None and not validate_file_path(model_data[f])
    ]
    if path_errors:
        await ctx.error(f"Validation failed: {'; '.join(path_errors)}")
        return create_error_response(
            message=f"Validation failed: {'; '.join(path_errors)}",
            error_type="ValidationError"
        )

    # Check if there's anything to update
    if not model_data:
        await ctx.error("No update data provided")
        return create_error_response(
            message="At least one field must be provided for update",
            error_type="ValidationError"
        )

    await ctx.info(f"Updating model: {model_id}")
    await ctx.report_progress(progress=20, total=100)
    
    try:
        model_client = get_mm_client(ctx, 'model')
        await ctx.report_progress(progress=40, total=100)
        
        update_response = await asyncio.to_thread(model_client.patch_model, model_data, model_id, create_sweetviz)
        await ctx.report_progress(progress=80, total=100)
        
        result = safe_response_to_dict(update_response)
        
        if result.get('status') == 'error':
            await ctx.error(f"Failed to update model: {result.get('message', 'Unknown error')}")
        else:
            await ctx.info(f"Model updated successfully: {model_id}")
        
        await ctx.report_progress(progress=100, total=100)
        return result
        
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message="A validation error occurred. Please check your input.",
            error_type="ValidationError"
        )
    except Exception as e:
        await ctx.error(f"Failed to update model: {str(e)}")
        return create_error_response(
            message="An internal error occurred while updating the model.",
            error_type="InternalError"
        )

@mcp.tool(
    name="delete_model",
    description="Delete a machine learning model from the ModelManager service permanently",
    tags={"model", "delete", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def delete_model(ctx: Context, model_id: str) -> dict:
    """Delete a machine learning model from the ModelManager service permanently.
    
    Args:
        ctx: The MCP server context containing authentication and configuration.
        model_id: The unique identifier of the model to delete (required).
        
    Returns:
        dict: Response from the ModelManager service confirming deletion.
    """
    # Validate required field
    if not model_id or not model_id.strip():
        await ctx.error("Model ID cannot be empty")
        return create_error_response(
            message="Model ID is required",
            error_type="ValidationError"
        )
    
    await ctx.info(f"Deleting model: {model_id}")
    await ctx.report_progress(progress=20, total=100)
    
    try:
        model_client = get_mm_client(ctx, 'model')
        await ctx.report_progress(progress=40, total=100)
        
        delete_response = await asyncio.to_thread(model_client.delete_model, model_id)
        await ctx.report_progress(progress=80, total=100)
        
        result = safe_response_to_dict(delete_response)
        await ctx.info(f"Model deleted successfully: {model_id}")
        await ctx.report_progress(progress=100, total=100)
        
        return result
        
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message="A validation error occurred. Please check your input.",
            error_type="ValidationError"
        )
    except Exception as e:
        await ctx.error(f"Failed to delete model: {str(e)}")
        return create_error_response(
            message="An internal error occurred while deleting the model.",
            error_type="InternalError"
        )

@mcp.tool(
    name="get_latest_metrics",
    description="Retrieve the latest performance metrics for a model from the ModelManager service",
    tags={"model", "metrics", "performance", "modelmanager"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def get_latest_metrics(ctx: Context, model_id: str, metric_type: str = None) -> dict:
    """Retrieve the latest performance metrics for a model from the ModelManager service.
    
    Args:
        ctx: The MCP server context containing authentication and configuration.
        model_id: The unique identifier of the model to retrieve metrics for (required).
        metric_type: Optional type of metrics to retrieve (e.g., 'accuracy', 'precision', 'recall').
        
    Returns:
        dict: Response containing the latest metrics data or error information.
    """
    # Validate required field
    if not model_id or not model_id.strip():
        await ctx.error("Model ID cannot be empty")
        return create_error_response(
            message="Model ID is required",
            error_type="ValidationError"
        )
    
    await ctx.info(f"Retrieving latest metrics for model: {model_id}")
    await ctx.report_progress(progress=20, total=100)
    
    try:
        model_client = get_mm_client(ctx, 'model')
        await ctx.report_progress(progress=40, total=100)
        
        metrics_response = await asyncio.to_thread(model_client.get_latest_metrics, model_id, metric_type)
        await ctx.report_progress(progress=80, total=100)
        
        result = safe_response_to_dict(metrics_response)
        await ctx.info(f"Metrics retrieved successfully for model: {model_id}")
        await ctx.report_progress(progress=100, total=100)
        
        return result
        
    except ValueError as e:
        await ctx.error(f"Validation error: {str(e)}")
        return create_error_response(
            message="A validation error occurred. Please check your input.",
            error_type="ValidationError"
        )
    except Exception as e:
        await ctx.error(f"Failed to retrieve metrics: {str(e)}")
        return create_error_response(
            message="An internal error occurred while retrieving metrics.",
            error_type="InternalError"
        )
