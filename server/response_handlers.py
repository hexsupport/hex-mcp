"""
Response formatters for forecast API responses.

This module classifies and formats diverse API response shapes into a uniform
LLM-friendly structure. Each of the 13 documented API response scenarios maps to
a dedicated handler, ensuring consistent response structure and appropriate
LLM guidance across all paths.

Scenario tags:
1. unparseable_string       - raw input is a plain string (JSON parse failed)
2. unparseable_non_dict     - raw input is not a dict or string
3. validation_error         - success=false, no available_options (missing param, invalid format)
4. invalid_filter_combination - success=false, data has available_options
5. internal_server_error    - success=false, status_code=500
6. embedding_service_busy   - success=false, 404 with "busy processing embeddings"
7. usecase_not_found        - success=false, status_code=404
8. semantic_candidates      - success=true, data has semantic_candidates
9. multiple_candidates      - success=true, data has candidates (exact/partial matches)
10. filter_error_in_success - success=true, data has filter_error object
11. forecast_with_data      - success=true, data has non-empty forecast list
12. empty_forecast          - success=true, data has empty forecast list
13. unknown                 - unrecognized response shape
"""

from typing import Any, Callable, Dict


def classify_response(raw: Any) -> str:
    """
    Classify API response into one of 13 scenario tags.

    Checks are ordered from most specific to least specific to avoid
    misclassification. For example, filter_error must be checked before
    empty_forecast, since both can have forecast=[] but filter_error
    has additional context.

    Args:
        raw: The raw API response (may be dict, string, or other type)

    Returns:
        Scenario tag as string: one of the 13 tags or "unknown"
    """
    # Type guards (can't call .get() safely on non-dicts)
    if isinstance(raw, str):
        return "unparseable_string"

    if not isinstance(raw, dict):
        return "unparseable_non_dict"

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
        "_llm_instructions": {
            "role": "You are reporting a system-level error to the user.",
            "rules": [
                "The API response could not be parsed.",
                "This is a system error, not a user input error.",
                "Suggest the user try again or contact support if the problem persists.",
            ],
        },
    }


def handle_unparseable_non_dict(raw: Any) -> Dict[str, Any]:
    """Handle raw input that is neither dict nor string."""
    return {
        "status": "error",
        "status_code": None,
        "message": f"API response has unexpected type: {type(raw).__name__}",
        "_llm_instructions": {
            "role": "You are reporting a system-level error to the user.",
            "rules": [
                "The API returned an unexpected response type.",
                "This is a system error, not a user input error.",
                "Suggest the user try again or contact support.",
            ],
        },
    }


def handle_validation_error(raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle validation errors: missing or invalid required parameters."""
    error_msg = data.get("message") or raw.get("error", "Validation error")

    return {
        "status": "error",
        "status_code": raw.get("status_code"),
        "message": error_msg,
        "_llm_instructions": {
            "role": "You are helping a user fix an invalid forecast request.",
            "rules": [
                "The API rejected the request because a required parameter is missing or invalid.",
                f"Error: {error_msg}",
                "Check that you provided:",
                "  - Either usecase_id or usecase_name",
                "  - series (the time-series to forecast)",
                "  - condition_one (primary filter/dimension)",
                "  - usecase_id must be an integer",
                "Ask the user to correct their request and try again.",
            ],
        },
    }


def handle_invalid_filter_combination(raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle filter validation errors with available_options."""
    error_msg = data.get("message") or raw.get("error", "Invalid filter combination")
    invalid_filters = data.get("invalid_filters") or {}
    available_options = data.get("available_options") or {}

    result = {
        "status": "error",
        "status_code": raw.get("status_code"),
        "message": error_msg,
        "invalid_filters": invalid_filters,
    }

    # Include available options for user guidance (always include, even if empty)
    result["available_options"] = {
        "series": available_options.get("series", []),
        "condition_one": available_options.get("condition_one", []),
        "conditions": available_options.get("conditions", {}),
        "facilityToUnit": available_options.get("facilityToUnit", {}),
    }

    result["_llm_instructions"] = {
        "role": "You are helping a business user fix invalid forecast filters.",
        "rules": [
            "The API rejected the request because the filter values are not valid for this usecase.",
            f"Invalid filters: {', '.join(k for k, v in invalid_filters.items() if v)}",
            "Present the available_options to the user as readable lists (not raw JSON):",
            "  - Available series options",
            "  - Available condition_one values",
            "  - If condition_one is selected, show the valid condition_two values in 'conditions'",
            "  - If condition_two is selected, show the valid condition_three values in 'facilityToUnit'",
            "Suggest a corrected request using only the values shown in available_options.",
            "Be concise and friendly.",
        ],
    }

    return result


def handle_internal_server_error(raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle 500 server errors."""
    error_msg = raw.get("error", "Internal server error")
    error_type = raw.get("error_type")

    return {
        "status": "error",
        "status_code": 500,
        "message": error_msg,
        "error_type": error_type,
        "_llm_instructions": {
            "role": "You are reporting a server error to the user.",
            "rules": [
                "The forecast API encountered an unexpected error (HTTP 500).",
                f"Error: {error_msg}",
                "This is not caused by the user's input — it's a server-side problem.",
                "Suggest the user should retry their request after waiting a few moments.",
                "If the problem persists after retry attempts, the user should contact support with the error message above.",
            ],
        },
    }


def handle_embedding_service_busy(raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle embedding service busy (occurs when semantic search is overloaded)."""
    return {
        "status": "error",
        "status_code": 404,
        "message": "The embedding service is busy processing requests. Please try again in a few seconds.",
        "_llm_instructions": {
            "role": "You are reporting a transient server overload to the user.",
            "rules": [
                "The semantic search service (used to find usecases by name) is temporarily busy.",
                "This is a transient issue, not a permanent failure.",
                "Suggest the user:",
                "  - Retry their request in 5–10 seconds.",
                "  - Alternatively, provide a usecase_id instead of usecase_name to bypass semantic search.",
            ],
        },
    }


def handle_usecase_not_found(raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle usecase not found (404)."""
    error_msg = raw.get("error", "No usecase found")

    return {
        "status": "error",
        "status_code": 404,
        "message": error_msg,
        "_llm_instructions": {
            "role": "You are helping a user find the correct usecase.",
            "rules": [
                "The API could not find a usecase matching the provided name or ID.",
                f"Error: {error_msg}",
                "The user should:",
                "  - Double-check the usecase name spelling.",
                "  - Try a different usecase name (e.g., shorter or with different keywords).",
                "  - If they know the usecase ID, provide that instead of the name.",
                "  - Suggest they list available usecases if that option is available.",
            ],
        },
    }


def handle_semantic_candidates(raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle semantic candidates: no exact match, but similar usecases found."""
    usecase_name = data.get("usecase_name", "?")
    candidates = data.get("semantic_candidates", [])

    return {
        "status": "clarification_needed",
        "status_code": 200,
        "message": f"No exact match for '{usecase_name}'. Showing semantically similar usecases.",
        "requested_name": usecase_name,
        "candidates": candidates,
        "_llm_instructions": {
            "role": "You are helping a user select the correct usecase from similar options.",
            "rules": [
                f"No exact usecase match for '{usecase_name}'.",
                "The API found semantically similar usecases, ranked by similarity (lower distance = better match).",
                "Present the candidate list to the user with their names and similarity scores.",
                "Ask the user to:",
                "  - Pick the usecase that best matches what they're looking for.",
                "  - Provide the usecase ID to retry the request.",
                "If none of the options match, suggest they try a different search term.",
            ],
        },
    }


def handle_multiple_candidates(raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle multiple candidates: exact/partial matches found."""
    usecase_name = data.get("usecase_name", "?")
    candidates = data.get("candidates", [])

    return {
        "status": "clarification_needed",
        "status_code": 200,
        "message": f"Multiple usecases match '{usecase_name}'. Please select one.",
        "requested_name": usecase_name,
        "candidates": candidates,
        "_llm_instructions": {
            "role": "You are helping a user select the correct usecase from multiple matches.",
            "rules": [
                f"Multiple usecases match the name '{usecase_name}'.",
                "Present the candidate list to the user with their IDs and full names.",
                "Ask the user to:",
                "  - Pick the usecase that matches their intent.",
                "  - Provide the usecase ID to retry the request.",
                "If the user is unsure, suggest they ask clarifying questions about each option.",
            ],
        },
    }


def handle_filter_error_in_success(raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle filter error inside a success response.

    This occurs when the usecase resolves successfully, but the filter
    values are invalid. The response includes both the resolved usecase
    and details about which filters failed.
    """
    filter_error = data.get("filter_error") or {}
    error_msg = filter_error.get("message", "Invalid filter values")
    invalid_filters = filter_error.get("invalid_filters") or {}
    available_options = filter_error.get("available_options") or {}

    result = {
        "status": "error",
        "status_code": 200,
        "message": error_msg,
        "resolved_usecase": {
            "id": data.get("usecase", {}).get("id"),
            "name": data.get("usecase", {}).get("name"),
        },
        "invalid_filters": invalid_filters,
    }

    if available_options:
        result["available_options"] = {
            "series": available_options.get("series", []),
            "condition_one": available_options.get("condition_one", []),
            "conditions": available_options.get("conditions", {}),
            "facilityToUnit": available_options.get("facilityToUnit", {}),
        }

    result["_llm_instructions"] = {
        "role": "You are helping a business user fix invalid forecast filters.",
        "rules": [
            "The usecase was found, but the filter values are not valid.",
            f"Invalid filters: {', '.join(k for k, v in invalid_filters.items() if v)}",
            "Present the available_options to the user as readable lists (not raw JSON):",
            "  - Available series options",
            "  - Available condition_one values",
            "  - Valid condition_two values for the selected condition_one",
            "  - Valid condition_three values for the selected condition_two",
            "Suggest a corrected request using only the values shown in available_options.",
            "Be concise and friendly.",
        ],
    }

    return result


def handle_forecast_with_data(raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
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
        "_llm_instructions": {
            "role": "You are presenting forecast data to a non-technical business user.",
            "rules": [
                "The forecast contains time-series predictions with confidence intervals.",
                "Each forecast entry has: 'Forecast Date', 'Forecast Value', 'value_type', 'rgn_cd', 'fac_id_cd'.",
                "The filters applied show which dimensions the forecast covers.",
                "The 'last_actual_update' field indicates how fresh the underlying data is.",
                "When presenting to the user:",
                "  - Round displayed values to 2 decimal places.",
                "  - Describe the date range covered by the forecast.",
                "  - Summarize the trend (increasing, decreasing, stable).",
                "  - Present as a concise narrative; do not echo raw JSON or field names.",
                "Mention filters_applied to contextualize the forecast.",
            ],
        },
    }


def handle_empty_forecast(raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
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
        "_llm_instructions": {
            "role": "You are reporting that no forecast data is available.",
            "rules": [
                "The usecase was found and filters are valid, but no forecast data exists for this combination.",
                "This is not an error — the data simply hasn't been generated yet or doesn't exist.",
                "Show the user the filters they applied and suggest:",
                "  - Broadening the filter criteria (e.g., selecting a different region or facility).",
                "  - Checking if data exists for a different prediction_period.",
                "  - Checking back later if forecasts are being generated on a schedule.",
                "Mention the 'last_actual_update' to help them understand data freshness.",
            ],
        },
    }


def handle_unknown(raw: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle unrecognized response shape."""
    return {
        "status": "unknown",
        "status_code": raw.get("status_code"),
        "message": "API response shape was not recognized by the formatter",
        "raw_keys": list(raw.keys()) if isinstance(raw, dict) else None,
        "_llm_instructions": {
            "role": "You are reporting an unexpected API response to the user.",
            "rules": [
                "The API returned a response that the formatter doesn't recognize.",
                "This may indicate an API version mismatch or an unexpected server state.",
                "Show the user the response keys or raw data cautiously.",
                "Suggest the user:",
                "  - Check if their request was valid.",
                "  - Contact support with the full error message.",
                "If safe, show the raw response to help debug.",
            ],
        },
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
        A formatted response dict with status, message, data, and _llm_instructions
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
