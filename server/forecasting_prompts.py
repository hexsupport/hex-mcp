"""
Forecasting prompts for the ModelManager MCP Server.

Defines reusable MCP prompt templates that guide LLMs when presenting
forecast data to end users.
"""

from config import mcp


@mcp.prompt(name="forecast_presentation_guide")
def forecast_presentation_guide() -> str:
    """Prompt template that instructs the LLM how to narrate forecast results."""
    return (
        "You are presenting ModelManager forecast data to a business user.\n"
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
        "7. If resolved is false, flag that the API could not resolve the request and "
        "ask the user to verify their filters and usecase configuration."
    )
