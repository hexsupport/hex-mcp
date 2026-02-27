"""
Utility functions for the ModelManager MCP Server.

This module provides common utility functions for response handling,
validation, and error management.
"""

from typing import Any, Dict, Union
import httpx

def safe_response_to_dict(response: Union[Dict, httpx.Response, Any]) -> Dict:
    """Safely convert various response types to dictionary format.
    
    Args:
        response: Response object that could be a dict, httpx.Response, or other type.
        
    Returns:
        dict: Response data in dictionary format.
    """
    if isinstance(response, dict):
        return response

    if hasattr(response, 'json'):
        try:
            return response.json()
        except Exception:
            return {"data": str(response)}
    elif hasattr(response, '__dict__'):
        return vars(response)
    else:
        return {"data": str(response)}

def create_error_response(message: str, error_type: str = "Error", status_code: int = None) -> Dict:
    """Create a standardized error response.
    
    Args:
        message: Error message to include.
        error_type: Type of error that occurred.
        status_code: HTTP status code if applicable.
        
    Returns:
        dict: Standardized error response.
    """
    error_response = {
        "status": "error",
        "message": message,
        "error_type": error_type
    }
    
    if status_code is not None:
        error_response["status_code"] = status_code
    
    return error_response

def create_success_response(data: Any = None, message: str = "Operation successful") -> Dict:
    """Create a standardized success response.
    
    Args:
        data: Data to include in the response.
        message: Success message to include.
        
    Returns:
        dict: Standardized success response.
    """
    response = {
        "status": "success",
        "message": message
    }
    
    if data is not None:
        response["data"] = data
    
    return response


def normalize_tool_response(data: Any, success_message: str = None) -> Dict:
    if isinstance(data, dict):
        if data.get("status") == "error":
            return data

        if data.get("error"):
            return create_error_response(
                message=str(data.get("error")),
                error_type="APIError",
            )

        if data.get("errors"):
            return create_error_response(
                message=str(data.get("errors")),
                error_type="APIError",
            )

        if success_message is not None:
            data["status"] = "success"
            data["message"] = success_message

        return data

    if success_message is None:
        success_message = "Operation successful"
    return create_success_response(data=data, message=success_message)

def validate_required_fields(data: Dict, required_fields: list) -> list:
    """Validate that required fields are present in data.
    
    Args:
        data: Dictionary to validate.
        required_fields: List of required field names.
        
    Returns:
        list: List of missing field names (empty if all present).
    """
    return [field for field in required_fields if field not in data or data[field] is None]

def validate_field_types(data: Dict, field_types: Dict[str, type]) -> Dict:
    """Validate field types in data.
    
    Args:
        data: Dictionary to validate.
        field_types: Dictionary mapping field names to expected types.
        
    Returns:
        dict: Error response if validation fails, None if valid.
    """
    for field, expected_type in field_types.items():
        if field in data and data[field] is not None:
            if not isinstance(data[field], expected_type):
                return create_error_response(
                    message=f"Field '{field}' must be of type {expected_type.__name__}",
                    error_type="ValidationError"
                )
    
    return None
