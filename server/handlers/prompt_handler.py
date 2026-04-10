"""
Prompt templates for the ModelManager MCP Server.

Defines reusable MCP prompt templates that guide LLMs when presenting
results from forecasting_tools, modelcard_tools, and
forecasting_governance_tools. Each tool response includes a `prompt_hint`
field that maps to one of the prompts below.
"""

from config import mcp


@mcp.prompt(name="forecast_presentation_guide")
def forecast_presentation_guide(
    usecase_name: str = "",
    date_range: str = "",
    total_points: str = "0",
) -> str:
    """Instruct the LLM how to narrate forecast results to a business user.

    Args:
        usecase_name: Name of the usecase (for contextual phrasing).
        date_range: Human-readable date range of the forecast (e.g. "Jan–Mar 2026").
        total_points: Number of forecast data points returned.
    """
    context = ""
    if usecase_name:
        context += f"The forecast is for usecase '{usecase_name}'.\n"
    if date_range:
        context += f"The forecast covers {date_range}.\n"
    if total_points and total_points != "0":
        context += f"There are {total_points} forecast data points.\n"

    return (
        "You are presenting ModelManager forecast data to a business user.\n"
        f"{context}"
        "Follow these rules:\n"
        "1. Lead with the key summary: date range and value range (min/max/avg).\n"
        "2. Describe lower_bound/upper_bound as a confidence interval — the range "
        "the actual value is likely to fall within.\n"
        "3. Reference last_actual_update to indicate how fresh the underlying data is.\n"
        "4. If total_points is 0, explain no data was returned and suggest revisiting "
        "the usecase filters.\n"
        "5. Round all numbers to 2 decimal places when presenting to the user.\n"
        "6. Write a short narrative paragraph — do not echo raw JSON or field names "
        "back to the user.\n"
        "7. Mention the filters_applied to contextualize the forecast.\n"
        "8. Summarize the trend (increasing, decreasing, stable) if data is present.\n"
        "9. If resolved is false, flag that the API could not resolve the request and "
        "ask the user to verify their filters and usecase configuration."
    )


@mcp.prompt(name="modelcard_guide")
def modelcard_guide(
    usecase_name: str = "",
    pdf_available: str = "true",
) -> str:
    """Instruct the LLM how to present modelcard creation results.

    Args:
        usecase_name: Name of the usecase the modelcard was created for.
        pdf_available: Whether the PDF report is ready ("true" or "false").
    """
    context = ""
    if usecase_name:
        context += f"The modelcard is for usecase '{usecase_name}'.\n"

    if pdf_available == "true":
        return (
            "You are confirming a successful modelcard creation to the user.\n"
            f"{context}"
            "Follow these rules:\n"
            "1. Confirm the modelcard was created successfully.\n"
            "2. Present the pdf_url as a clickable link for the user to download.\n"
            "3. Include the modelcard_id for future reference.\n"
            "4. Be concise and friendly — one short paragraph is enough."
        )

    return (
        "You are informing the user that their modelcard is being generated.\n"
        f"{context}"
        "Follow these rules:\n"
        "1. Explain that the modelcard record was created but the PDF is still being generated.\n"
        "2. This is expected — PDF generation may take a few moments.\n"
        "3. Tell the user to retry shortly using the same parameters.\n"
        "4. Include the modelcard_id so they can reference it later."
    )


@mcp.prompt(name="governance_report_guide")
def governance_report_guide(
    usecase_name: str = "",
) -> str:
    """Instruct the LLM how to present forecast governance report results.

    Args:
        usecase_name: Name of the usecase the report covers.
    """
    context = ""
    if usecase_name:
        context += f"The governance report is for usecase '{usecase_name}'.\n"

    return (
        "You are presenting a forecast governance report to the user.\n"
        f"{context}"
        "Follow these rules:\n"
        "1. The governance report documents model performance and compliance for audit purposes.\n"
        "2. If a report_url is present, present it as a downloadable link.\n"
        "3. Summarize governance_data fields in plain language — avoid echoing raw JSON.\n"
        "4. Highlight any compliance flags or performance metrics if present.\n"
        "5. Be concise and professional."
    )


@mcp.prompt(name="filter_selection_guide")
def filter_selection_guide(
    usecase_name: str = "",
    invalid_filters: str = "",
) -> str:
    """Instruct the LLM how to help a user pick valid filter values.

    This prompt applies across all three tools when filter validation fails.

    Args:
        usecase_name: Name of the usecase that was resolved.
        invalid_filters: Comma-separated list of filter names that failed validation.
    """
    context = ""
    if usecase_name:
        context += f"The usecase '{usecase_name}' was found.\n"
    if invalid_filters:
        context += f"The following filters are invalid: {invalid_filters}.\n"

    return (
        "You are helping a business user fix invalid filter values for a ModelManager request.\n"
        f"{context}"
        "Follow these rules:\n"
        "1. The API rejected the request because one or more filter values are not valid.\n"
        "2. Present the available_options to the user as readable lists (not raw JSON):\n"
        "   - Available series options\n"
        "   - Available condition_one values\n"
        "   - If condition_one is selected, show valid condition_two values from 'conditions'\n"
        "   - If condition_two is selected, show valid condition_three values from 'facilityToUnit'\n"
        "3. Suggest a corrected request using only values shown in available_options.\n"
        "4. Be concise and friendly."
    )


@mcp.prompt(name="error_recovery_guide")
def error_recovery_guide(
    error_type: str = "",
    error_message: str = "",
) -> str:
    """Instruct the LLM how to help a user recover from API errors.

    This prompt applies across all three tools for error scenarios.

    Args:
        error_type: Category of error (e.g. "ValidationError", "APIError", "500").
        error_message: The error message returned by the API.
    """
    context = ""
    if error_type:
        context += f"Error type: {error_type}.\n"
    if error_message:
        context += f"Error message: {error_message}.\n"

    return (
        "You are helping a user recover from an error in a ModelManager API request.\n"
        f"{context}"
        "Follow these rules:\n"
        "1. Classify the error for the user:\n"
        "   - Validation errors: the request is malformed — help the user fix their input.\n"
        "   - 404 / not found: the usecase or resource doesn't exist — suggest alternatives.\n"
        "   - 500 / server errors: server-side problem — suggest retrying after a moment.\n"
        "   - Embedding service busy: transient overload — suggest retrying in 5–10 seconds "
        "or using a usecase_id instead of usecase_name.\n"
        "2. Never blame the user for server-side errors.\n"
        "3. If the error includes available_options or candidates, present them.\n"
        "4. Suggest concrete next steps — do not just say 'try again'."
    )
