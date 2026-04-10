"""
Base classes and common patterns for ModelManager MCP tools.

This module provides base classes and decorators to reduce code duplication
and ensure consistent behavior across all MCP tools.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from fastmcp import Context
import asyncio
from clients import get_mm_client
from utils.helpers import safe_response_to_dict, create_error_response, create_success_response

class BaseMCPTool(ABC):
    """Base class for MCP tools with common functionality."""

    def __init__(self, client_type: str):
        """Initialize the base tool.

        Args:
            client_type: Type of ModelManager client to use.
        """
        self.client_type = client_type

    @abstractmethod
    async def execute(self, ctx: Context, *args, **kwargs) -> Dict:
        """Execute the tool logic. Must be implemented by subclasses."""
        pass

    async def _execute_with_progress(self, ctx: Context, operation, *args, **kwargs) -> Dict:
        """Execute an operation with progress reporting and error handling.

        Args:
            ctx: MCP server context.
            operation: The operation to execute (client method).
            *args: Arguments to pass to the operation.
            **kwargs: Keyword arguments to pass to the operation.

        Returns:
            dict: Response from the operation.
        """
        try:
            client = get_mm_client(ctx, self.client_type)
            await ctx.report_progress(progress=40, total=100)

            response = await asyncio.to_thread(operation, *args, **kwargs)
            await ctx.report_progress(progress=80, total=100)

            result = safe_response_to_dict(response)
            await ctx.report_progress(progress=100, total=100)

            return result

        except ValueError as e:
            await ctx.error(f"Validation error: {str(e)}")
            return create_error_response(
                message="A validation error occurred. Please check your input.",
                error_type="ValidationError"
            )
        except Exception as e:
            await ctx.error(f"Operation failed: {str(e)}")
            return create_error_response(
                message="An internal error occurred. Please try again.",
                error_type="InternalError"
            )

class CRUDTool(BaseMCPTool):
    """Base class for CRUD operations (Create, Read, Update, Delete)."""

    def __init__(self, client_type: str, resource_name: str):
        """Initialize CRUD tool.

        Args:
            client_type: Type of ModelManager client.
            resource_name: Name of the resource (for logging).
        """
        super().__init__(client_type)
        self.resource_name = resource_name

    async def create(self, ctx: Context, data: Dict) -> Dict:
        """Create a new resource."""
        await ctx.info(f"Creating new {self.resource_name}")
        await ctx.report_progress(progress=20, total=100)

        client = get_mm_client(ctx, self.client_type)
        return await self._execute_with_progress(
            ctx, client.post_resource, data
        )

    async def update(self, ctx: Context, data: Dict, resource_id: str) -> Dict:
        """Update an existing resource."""
        await ctx.info(f"Updating {self.resource_name}: {resource_id}")
        await ctx.report_progress(progress=20, total=100)

        client = get_mm_client(ctx, self.client_type)
        return await self._execute_with_progress(
            ctx, client.patch_resource, data, resource_id
        )

    async def delete(self, ctx: Context, resource_id: str) -> Dict:
        """Delete a resource."""
        await ctx.info(f"Deleting {self.resource_name}: {resource_id}")
        await ctx.report_progress(progress=20, total=100)

        client = get_mm_client(ctx, self.client_type)
        return await self._execute_with_progress(
            ctx, client.delete_resource, resource_id
        )

    async def get_details(self, ctx: Context, resource_id: str) -> Dict:
        """Get resource details."""
        await ctx.info(f"Retrieving {self.resource_name} details: {resource_id}")
        await ctx.report_progress(progress=20, total=100)

        client = get_mm_client(ctx, self.client_type)
        return await self._execute_with_progress(
            ctx, client.get_details, resource_id
        )

def validate_parameters(**validators):
    """Decorator to validate function parameters.

    Args:
        **validators: Mapping of parameter names to validation functions.

    Returns:
        Decorated function with validation.
    """
    def decorator(func):
        async def wrapper(ctx: Context, *args, **kwargs):
            # Run validations
            for param, validator in validators.items():
                if param in kwargs and kwargs[param] is not None:
                    validation_result = validator(kwargs[param])
                    if isinstance(validation_result, dict) and validation_result.get("status") == "error":
                        await ctx.error(validation_result.get("message", "Validation failed"))
                        return validation_result

            return await func(ctx, *args, **kwargs)
        return wrapper
    return decorator
