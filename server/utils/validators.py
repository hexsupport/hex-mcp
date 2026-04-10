"""
Validation functions for ModelManager MCP server payloads.
"""
from typing import Union

def validate_forecast_payload(payload: dict) -> dict:
    """Validate forecast payload parameters.
    
    Args:
        payload: Dictionary containing forecast parameters
        
    Returns:
        dict: Validated payload or error response
    """
    if not isinstance(payload, dict):
        return {
            "status": "error",
            "message": "Payload must be a dictionary",
            "error_type": "ValidationError",
        }
    
    # Check if either usecase_name or usecase_id is provided
    if "usecase_id" not in payload and "usecase_name" not in payload:
        return {
            "status": "error",
            "message": "Either usecase_name or usecase_id must be provided",
            "error_type": "ValidationError",
        }
    
    # Validate parameter types
    if "usecase_id" in payload and payload["usecase_id"] is not None:
        if not isinstance(payload["usecase_id"], Union[str, int]):
            return {
                "status": "error",
                "message": "usecase_id must be a string or integer",
                "error_type": "ValidationError",
            }
    
    if "usecase_name" in payload and payload["usecase_name"] is not None:
        if not isinstance(payload["usecase_name"], str):
            return {
                "status": "error",
                "message": "usecase_name must be a string",
                "error_type": "ValidationError",
            }
    
    if "series" in payload and payload["series"] is not None:
        if not isinstance(payload["series"], str):
            return {
                "status": "error",
                "message": "series must be a string",
                "error_type": "ValidationError",
            }
    
    if "condition_one" in payload and payload["condition_one"] is not None:
        if not isinstance(payload["condition_one"], str):
            return {
                "status": "error",
                "message": "condition_one must be a string",
                "error_type": "ValidationError",
            }
    
    if "condition_two" in payload and payload["condition_two"] is not None:
        if not isinstance(payload["condition_two"], str):
            return {
                "status": "error",
                "message": "condition_two must be a string",
                "error_type": "ValidationError",
            }
    
    if "condition_three" in payload and payload["condition_three"] is not None:
        if not isinstance(payload["condition_three"], str):
            return {
                "status": "error",
                "message": "condition_three must be a string",
                "error_type": "ValidationError",
            }
    
    if "prediction_period" in payload and payload["prediction_period"] is not None:
        if not isinstance(payload["prediction_period"], Union[str, int]):
            return {
                "status": "error",
                "message": "prediction_period must be a string or integer",
                "error_type": "ValidationError",
            }
    
    return payload
