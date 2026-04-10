"""
Response formatters for ModelManager API responses.

This module classifies and formats diverse API response shapes into a uniform
LLM-friendly structure. Each documented scenario maps to a dedicated handler,
ensuring consistent response structure across all tools: forecasting_tools,
modelcard_tools, and forecasting_governance_tools.

Each handler returns a ``prompt_hint`` field that maps to an MCP prompt
template defined in ``forecasting_prompts.py``, replacing the previous
inline ``_llm_instructions`` approach.

Scenario tags:
1.  unparseable_string        - raw input is a plain string (JSON parse failed)
2.  unparseable_non_dict      - raw input is not a dict or string
3.  validation_error          - success=false, no available_options (missing param, invalid format)
4.  invalid_filter_combination - success=false, data has available_options
5.  internal_server_error     - success=false, status_code=500
6.  embedding_service_busy    - success=false, 404 with "busy processing embeddings"
7.  usecase_not_found         - success=false, status_code=404
8.  semantic_candidates       - success=true, data has semantic_candidates
9.  multiple_candidates       - success=true, data has candidates (exact/partial matches)
10. filter_error_in_success   - success=true, data has filter_error object
11. forecast_with_data        - success=true, data has non-empty forecast list
12. empty_forecast            - success=true, data has empty forecast list
13. modelcard_created         - top-level has modelcard_id and pdf_url (full creation success)
14. modelcard_pending         - top-level has modelcard_id but no pdf_url (creation in progress)
15. governance_report         - success=true, data has governance report content
16. unknown                   - unrecognized response shape
"""

from typing import Any, Callable, Dict


def classify_response(raw: Any) -> str:
    """
    Classify API response into one of 16 scenario tags.

    Checks are ordered from most specific to least specific to avoid
    misclassification. For example, filter_error must be checked before
    empty_forecast, since both can have forecast=[] but filter_error
    has additional context.

    Args:
        raw: The raw API response (may be dict, string, or other type)

    Returns:
        Scenario tag as string: one of the 16 tags or "unknown"
    """
    # Type guards (can't call .get() safely on non-dicts)
    if isinstance(raw, str):
        return "unparseable_string"

    if not isinstance(raw, dict):
        return "unparseable_non_dict"

    # ── Modelcard responses (flat structure, no success envelope) ──
    if "modelcard_id" in raw:
        if raw.get("pdf_url") or raw.get("modelcard_pdf_id"):
            return "modelcard_created"
        return "modelcard_pending"

    # Extract success and data safely
    success = raw.get("success")
    data = raw.get("data") or {}
    status_code = raw.get("status_code")
    error_msg = raw.get("error", "")

    # ── Path: success=false (failure responses) ──
    if success is False:
        # Most specific first: available_options means filter validation error
        # Check if the key exists (even if empty), not just if it's truthy
        if isinstance(data, dict) and "available_options" in data:
            return "invalid_filter_combination"

        # Check for server error before generic 404
        if status_code == 500:
            return "internal_server_error"

        # Check for embedding service busy (uses 404 but needs special handling)
        if status_code == 404 and "busy processing embeddings" in str(error_msg):
            return "embedding_service_busy"

        # Generic 404: usecase not found
        if status_code == 404:
            return "usecase_not_found"

        # Remaining failures (validation errors, 400s, etc.)
        return "validation_error"

    # ── Path: success=true (success responses) ──
    if success is True:
        if not isinstance(data, dict):
            # success=true but data is not a dict → unusual, treat as unknown
            return "unknown"

        # Disambiguation responses (most specific for success=true paths)
        if "semantic_candidates" in data:
            return "semantic_candidates"

        if "candidates" in data:
            return "multiple_candidates"

        # Filter error inside a success response (before checking forecast)
        if "filter_error" in data:
            return "filter_error_in_success"

        # Governance report (check before forecast)
        if "report_url" in data or "governance_data" in data or "report_id" in data:
            return "governance_report"

        # Forecast data (happy path)
        forecast = data.get("forecast")
        if isinstance(forecast, list):
            if len(forecast) > 0:
                return "forecast_with_data"
            else:
                return "empty_forecast"

    # Fallback for unrecognized shapes
    return "unknown"


# ── Individual Handler Functions ──

def handle_unparseable_string(raw: str) -> Dict[str, Any]:
    """Handle raw input that is a plain string (JSON parse failed)."""
    return {
        "status": "error",
        "status_code": None,
        "message": "Failed to parse API response",
        "raw_input": raw[:200],  # Truncate for safety
        "prompt_hint": "error_recovery_guide",
    }


def handle_unparseable_non_dict(raw: Any) -> Dict[str, Any]:
    """Handle raw input that is neither dict nor string."""
    return {
        "status": "error",
        "status_code": None,
        "message": f"API response has unexpected type: {type(raw).__name__}",
        "prompt_hint": "error_recovery_guide",
    }


def handle_validation_error(raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle validation errors: missing or invalid required parameters."""
    error_msg = data.get("message") or raw.get("error", "Validation error")

    return {
        "status": "error",
        "status_code": raw.get("status_code"),
        "message": error_msg,
        "prompt_hint": "error_recovery_guide",
    }


def handle_invalid_filter_combination(raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle filter validation errors with available_options."""
    error_msg = data.get("message") or raw.get("error", "Invalid filter combination")
    invalid_filters = data.get("invalid_filters") or {}
    available_options = data.get("available_options") or {}

    result: Dict[str, Any] = {
        "status": "error",
        "status_code": raw.get("status_code"),
        "message": error_msg,
        "invalid_filters": invalid_filters,
        "prompt_hint": "filter_selection_guide",
    }

    # Include available options for user guidance (always include, even if empty)
    result["available_options"] = {
        "series": available_options.get("series", []),
        "condition_one": available_options.get("condition_one", []),
        "conditions": available_options.get("conditions", {}),
        "facilityToUnit": available_options.get("facilityToUnit", {}),
    }

    return result


def handle_internal_server_error(raw: Dict[str, Any], _data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle 500 server errors."""
    error_msg = raw.get("error", "Internal server error")
    error_type = raw.get("error_type")

    return {
        "status": "error",
        "status_code": 500,
        "message": error_msg,
        "error_type": error_type,
        "prompt_hint": "error_recovery_guide",
    }


def handle_embedding_service_busy(_raw: Dict[str, Any], _data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle embedding service busy (occurs when semantic search is overloaded)."""
    return {
        "status": "error",
        "status_code": 404,
        "message": "The embedding service is busy processing requests. Please try again in a few seconds.",
        "prompt_hint": "error_recovery_guide",
    }


def handle_usecase_not_found(raw: Dict[str, Any], _data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle usecase not found (404)."""
    error_msg = raw.get("error", "No usecase found")

    return {
        "status": "error",
        "status_code": 404,
        "message": error_msg,
        "prompt_hint": "error_recovery_guide",
    }


def handle_semantic_candidates(_raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle semantic candidates: no exact match, but similar usecases found."""
    usecase_name = data.get("usecase_name", "?")
    candidates = data.get("semantic_candidates", [])

    return {
        "status": "clarification_needed",
        "status_code": 200,
        "message": f"No exact match for '{usecase_name}'. Showing semantically similar usecases.",
        "requested_name": usecase_name,
        "candidates": candidates,
        "prompt_hint": "error_recovery_guide",
    }


def handle_multiple_candidates(_raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle multiple candidates: exact/partial matches found."""
    usecase_name = data.get("usecase_name", "?")
    candidates = data.get("candidates", [])

    return {
        "status": "clarification_needed",
        "status_code": 200,
        "message": f"Multiple usecases match '{usecase_name}'. Please select one.",
        "requested_name": usecase_name,
        "candidates": candidates,
        "prompt_hint": "error_recovery_guide",
    }


def handle_filter_error_in_success(_raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle filter error inside a success response.

    This occurs when the usecase resolves successfully, but the filter
    values are invalid. The response includes both the resolved usecase
    and details about which filters failed.
    """
    filter_error = data.get("filter_error") or {}
    error_msg = filter_error.get("message", "Invalid filter values")
    invalid_filters = filter_error.get("invalid_filters") or {}
    available_options = filter_error.get("available_options") or {}

    result: Dict[str, Any] = {
        "status": "error",
        "status_code": 200,
        "message": error_msg,
        "resolved_usecase": {
            "id": data.get("usecase", {}).get("id"),
            "name": data.get("usecase", {}).get("name"),
        },
        "invalid_filters": invalid_filters,
        "prompt_hint": "filter_selection_guide",
    }

    if available_options:
        result["available_options"] = {
            "series": available_options.get("series", []),
            "condition_one": available_options.get("condition_one", []),
            "conditions": available_options.get("conditions", {}),
            "facilityToUnit": available_options.get("facilityToUnit", {}),
        }

    return result


def handle_forecast_with_data(_raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle successful response with non-empty forecast data."""
    usecase_info = data.get("usecase") or {}
    filters = data.get("filters_applied") or {}
    info = data.get("info") or {}
    raw_points = data.get("forecast") or []
    total_points = len(raw_points)

    return {
        "status": "success",
        "status_code": 200,
        "message": "Successfully retrieved forecast",
        "data_available": total_points > 0,
        "resolved": data.get("resolved", False),
        "usecase": {
            "id": usecase_info.get("id"),
            "name": usecase_info.get("name"),
            "type": usecase_info.get("usecase_type"),
        },
        "condition_info": {
            "condition_type": info.get("condition_type"),
            "required_filters": info.get("required_filters"),
        },
        "filters_applied": filters,
        "last_actual_update": data.get("last_actual_update"),
        "forecast_count": total_points,
        "forecast": raw_points,
        "prompt_hint": "forecast_presentation_guide",
    }


def handle_empty_forecast(_raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle successful response with empty forecast list."""
    usecase_info = data.get("usecase") or {}
    filters = data.get("filters_applied") or {}
    info = data.get("info") or {}

    return {
        "status": "success",
        "status_code": 200,
        "message": "No forecast data available for the selected filters",
        "data_available": False,
        "resolved": data.get("resolved", False),
        "usecase": {
            "id": usecase_info.get("id"),
            "name": usecase_info.get("name"),
            "type": usecase_info.get("usecase_type"),
        },
        "condition_info": {
            "condition_type": info.get("condition_type"),
            "required_filters": info.get("required_filters"),
        },
        "filters_applied": filters,
        "last_actual_update": data.get("last_actual_update"),
        "forecast_count": 0,
        "forecast": [],
        "prompt_hint": "forecast_presentation_guide",
    }


def handle_modelcard_created(raw: Dict[str, Any], _data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle successful modelcard creation with PDF available."""
    return {
        "status": "success",
        "message": "Successfully created modelcard",
        "modelcard_id": raw.get("modelcard_id"),
        "modelcard_pdf_id": raw.get("modelcard_pdf_id"),
        "pdf_url": raw.get("pdf_url"),
        "prompt_hint": "modelcard_guide",
    }


def handle_modelcard_pending(raw: Dict[str, Any], _data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle modelcard creation where PDF generation is still in progress."""
    return {
        "status": "pending",
        "message": "Modelcard created but PDF is not yet available",
        "modelcard_id": raw.get("modelcard_id"),
        "prompt_hint": "modelcard_guide",
    }


def handle_governance_report(_raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle successful forecast governance report response."""
    usecase_info = data.get("usecase") or {}

    return {
        "status": "success",
        "status_code": 200,
        "message": "Successfully generated forecast governance report",
        "usecase": {
            "id": usecase_info.get("id"),
            "name": usecase_info.get("name"),
        },
        "report_id": data.get("report_id"),
        "report_url": data.get("report_url"),
        "governance_data": data.get("governance_data"),
        "prompt_hint": "governance_report_guide",
    }


def handle_unknown(raw: Dict[str, Any], _data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle unrecognized response shape."""
    return {
        "status": "unknown",
        "status_code": raw.get("status_code"),
        "message": "API response shape was not recognized by the formatter",
        "raw_keys": list(raw.keys()) if isinstance(raw, dict) else None,
        "prompt_hint": "error_recovery_guide",
    }


# ── Dispatch Table ──

HANDLER_MAP: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {
    "validation_error": handle_validation_error,
    "invalid_filter_combination": handle_invalid_filter_combination,
    "internal_server_error": handle_internal_server_error,
    "embedding_service_busy": handle_embedding_service_busy,
    "usecase_not_found": handle_usecase_not_found,
    "semantic_candidates": handle_semantic_candidates,
    "multiple_candidates": handle_multiple_candidates,
    "filter_error_in_success": handle_filter_error_in_success,
    "forecast_with_data": handle_forecast_with_data,
    "empty_forecast": handle_empty_forecast,
    "modelcard_created": handle_modelcard_created,
    "modelcard_pending": handle_modelcard_pending,
    "governance_report": handle_governance_report,
    "unknown": handle_unknown,
}


def dispatch_response(raw: Any) -> Dict[str, Any]:
    """
    Main entry point: classify and dispatch to the appropriate handler.

    This function replaces the body of format_api_response, maintaining
    the same interface while delegating to scenario-specific handlers.

    Args:
        raw: The raw API response (dict, string, or other type)

    Returns:
        A formatted response dict with status, message, data, and prompt_hint
    """
    scenario = classify_response(raw)

    # Handle special cases that need different handler signatures
    if scenario == "unparseable_string":
        return handle_unparseable_string(raw)

    if scenario == "unparseable_non_dict":
        return handle_unparseable_non_dict(raw)

    # For all other scenarios, raw is a dict
    if not isinstance(raw, dict):
        # Defensive: should not happen, but catch it
        return handle_unparseable_non_dict(raw)

    data = raw.get("data") or {}

    # Look up and invoke the handler
    handler = HANDLER_MAP.get(scenario, handle_unknown)
    return handler(raw, data)
