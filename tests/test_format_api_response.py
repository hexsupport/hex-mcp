"""
Comprehensive tests for format_api_response and response_handlers.

Tests all 13 documented API response scenarios plus edge cases.
Each test verifies:
1. Classification returns the correct scenario tag
2. Handler produces the expected output shape
3. Response always includes 'status' and '_llm_instructions'
4. Contextual fields are preserved correctly
"""

import json
import pytest
from pathlib import Path

# Import the formatter components
from server.response_handlers import (
    classify_response,
    dispatch_response,
    handle_validation_error,
    handle_invalid_filter_combination,
    handle_internal_server_error,
    handle_embedding_service_busy,
    handle_usecase_not_found,
    handle_semantic_candidates,
    handle_multiple_candidates,
    handle_filter_error_in_success,
    handle_forecast_with_data,
    handle_empty_forecast,
    handle_unknown,
)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures: Load response examples from resp_example directory
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def response_examples():
    """Load real-world response examples from resp_example/get_forecast_response_examples.json"""
    examples_file = Path(__file__).parent.parent / "resp_example" / "get_forecast_response_examples.json"
    with open(examples_file) as f:
        data = json.load(f)
    return data.get("examples", [])


@pytest.fixture
def response_structures():
    """Load response structure definitions from resp_example/get_forecast_response_structures.json"""
    structures_file = Path(__file__).parent.parent / "resp_example" / "get_forecast_response_structures.json"
    with open(structures_file) as f:
        data = json.load(f)
    return data.get("response_structures", {})


# ────────────────────────────────────────────────────────────────────────────
# Tests: Classification (13 scenarios)
# ────────────────────────────────────────────────────────────────────────────


class TestClassification:
    """Test classify_response against all documented scenarios."""

    def test_classify_unparseable_string(self):
        """Test classification of plain string input."""
        result = classify_response("some error text")
        assert result == "unparseable_string"

    def test_classify_unparseable_non_dict(self):
        """Test classification of non-dict, non-string inputs."""
        assert classify_response(None) == "unparseable_non_dict"
        assert classify_response(42) == "unparseable_non_dict"
        assert classify_response([1, 2, 3]) == "unparseable_non_dict"

    def test_classify_validation_error(self):
        """Test classification of validation error (missing parameter)."""
        response = {
            "success": False,
            "status_code": 400,
            "error": "series is required",
        }
        assert classify_response(response) == "validation_error"

    def test_classify_invalid_filter_combination(self):
        """Test classification of invalid filter combination."""
        response = {
            "success": False,
            "status_code": 400,
            "error": "Invalid filter combination.",
            "data": {
                "error": True,
                "message": "The conditions applied are not eligible.",
                "invalid_filters": {"series": "invalid_series"},
                "available_options": {"series": ["covid_total_census"]},
            },
        }
        assert classify_response(response) == "invalid_filter_combination"

    def test_classify_internal_server_error(self):
        """Test classification of 500 server error."""
        response = {
            "success": False,
            "status_code": 500,
            "error": "Database connection timeout",
        }
        assert classify_response(response) == "internal_server_error"

    def test_classify_embedding_service_busy(self):
        """Test classification of embedding service busy (404 with specific message)."""
        response = {
            "success": False,
            "status_code": 404,
            "error": "Server is busy processing embeddings. Please try again later.",
        }
        assert classify_response(response) == "embedding_service_busy"

    def test_classify_usecase_not_found(self):
        """Test classification of usecase not found (404 without busy message)."""
        response = {
            "success": False,
            "status_code": 404,
            "error": "No usecase found for 'NonExistentUsecase'.",
        }
        assert classify_response(response) == "usecase_not_found"

    def test_classify_semantic_candidates(self):
        """Test classification of semantic candidates response."""
        response = {
            "success": True,
            "status_code": 200,
            "data": {
                "msg": "No exact match found",
                "usecase_name": "Hospital Monitoring",
                "semantic_candidates": [
                    {"id": 5, "name": "Hospital Census Monitoring", "distance": 0.23},
                ],
            },
        }
        assert classify_response(response) == "semantic_candidates"

    def test_classify_multiple_candidates(self):
        """Test classification of multiple exact/partial matches."""
        response = {
            "success": True,
            "status_code": 200,
            "data": {
                "msg": "Multiple usecases match",
                "usecase_name": "Census",
                "candidates": [
                    {"id": 3, "name": "Covid Census Forecasting"},
                    {"id": 7, "name": "Hospital Census Monitoring"},
                ],
            },
        }
        assert classify_response(response) == "multiple_candidates"

    def test_classify_filter_error_in_success(self):
        """Test classification of filter_error inside success response."""
        response = {
            "success": True,
            "status_code": 200,
            "data": {
                "resolved": True,
                "usecase": {"id": 5, "name": "Hospital Census Forecasting"},
                "filters_applied": {"series": "invalid_series"},
                "forecast": [],
                "filter_error": {
                    "error": True,
                    "message": "Invalid filter",
                    "invalid_filters": {"series": "invalid_series"},
                    "available_options": {"series": ["covid_total_census"]},
                },
            },
        }
        assert classify_response(response) == "filter_error_in_success"

    def test_classify_forecast_with_data(self):
        """Test classification of successful forecast with data."""
        response = {
            "success": True,
            "status_code": 200,
            "data": {
                "resolved": True,
                "usecase": {"id": 5, "name": "Hospital Census Forecasting"},
                "forecast": [
                    {
                        "Forecast Date": "2026-03-08",
                        "Forecast Value": 125.5,
                    },
                ],
            },
        }
        assert classify_response(response) == "forecast_with_data"

    def test_classify_empty_forecast(self):
        """Test classification of empty forecast (no data for filter combination)."""
        response = {
            "success": True,
            "status_code": 200,
            "data": {
                "resolved": True,
                "usecase": {"id": 5, "name": "Hospital Census Forecasting"},
                "forecast": [],
            },
        }
        assert classify_response(response) == "empty_forecast"

    def test_classify_unknown(self):
        """Test classification of unrecognized response shape."""
        response = {
            "success": True,
            "unexpected_field": "value",
        }
        assert classify_response(response) == "unknown"


# ────────────────────────────────────────────────────────────────────────────
# Tests: Dispatch and Handler Contract (all responses have status + instructions)
# ────────────────────────────────────────────────────────────────────────────


class TestDispatchContract:
    """Test that all responses adhere to the uniform response contract."""

    def test_all_responses_have_status(self, response_examples):
        """Every formatted response must have a 'status' field."""
        for example in response_examples:
            raw_response = example.get("response", {})
            formatted = dispatch_response(raw_response)
            assert "status" in formatted, f"Missing status in: {example.get('scenario')}"
            assert formatted["status"] in [
                "success",
                "error",
                "clarification_needed",
                "unknown",
            ], f"Invalid status value: {formatted['status']}"

    def test_all_responses_have_llm_instructions(self, response_examples):
        """Every formatted response must include _llm_instructions."""
        for example in response_examples:
            raw_response = example.get("response", {})
            formatted = dispatch_response(raw_response)
            assert "_llm_instructions" in formatted, f"Missing _llm_instructions in: {example.get('scenario')}"
            instructions = formatted["_llm_instructions"]
            assert "role" in instructions, f"Missing role in {example.get('scenario')}"
            assert "rules" in instructions, f"Missing rules in {example.get('scenario')}"
            assert isinstance(instructions["rules"], list), f"Rules must be a list in {example.get('scenario')}"

    def test_dispatch_handles_string_input(self):
        """dispatch_response must handle plain string input."""
        result = dispatch_response("some error string")
        assert "status" in result
        assert "_llm_instructions" in result

    def test_dispatch_handles_non_dict_input(self):
        """dispatch_response must handle non-dict input types."""
        for invalid_input in [None, 42, [1, 2, 3], 3.14]:
            result = dispatch_response(invalid_input)
            assert isinstance(result, dict)
            assert "status" in result
            assert "_llm_instructions" in result


# ────────────────────────────────────────────────────────────────────────────
# Tests: Individual Scenario Handlers
# ────────────────────────────────────────────────────────────────────────────


class TestValidationErrorHandler:
    """Test handle_validation_error."""

    def test_missing_parameter_error(self):
        """Test validation error for missing required parameter."""
        raw = {
            "success": False,
            "status_code": 400,
            "error": "series is required",
        }
        result = handle_validation_error(raw, {})
        assert result["status"] == "error"
        assert result["status_code"] == 400
        assert "series is required" in result["message"]
        assert "_llm_instructions" in result

    def test_invalid_type_error(self):
        """Test validation error for invalid parameter type."""
        raw = {
            "success": False,
            "status_code": 400,
            "error": "usecase_id must be an integer",
        }
        result = handle_validation_error(raw, {})
        assert result["status"] == "error"
        assert "integer" in result["message"]


class TestInvalidFilterCombinationHandler:
    """Test handle_invalid_filter_combination."""

    def test_with_available_options(self):
        """Test filter error with available_options."""
        raw = {
            "success": False,
            "status_code": 400,
            "error": "Invalid filter combination.",
        }
        data = {
            "message": "The conditions applied are not eligible.",
            "invalid_filters": {"series": "invalid_series", "condition_two": "invalid_facility"},
            "available_options": {
                "series": ["covid_total_census", "covid_icu_census"],
                "condition_one": ["NCAL", "SCAL"],
                "conditions": {"NCAL": ["facility_1", "facility_2"]},
                "facilityToUnit": {"facility_1": ["unit_a", "unit_b"]},
            },
        }
        result = handle_invalid_filter_combination(raw, data)
        assert result["status"] == "error"
        assert "available_options" in result
        assert "facilityToUnit" in result["available_options"], "facilityToUnit must be preserved"
        assert result["available_options"]["series"] == ["covid_total_census", "covid_icu_census"]
        assert "_llm_instructions" in result


class TestInternalServerErrorHandler:
    """Test handle_internal_server_error."""

    def test_500_error_response(self):
        """Test 500 internal server error."""
        raw = {
            "success": False,
            "status_code": 500,
            "error": "Database connection timeout",
            "error_type": "OperationalError",
        }
        result = handle_internal_server_error(raw, {})
        assert result["status"] == "error"
        assert result["status_code"] == 500
        assert result["error_type"] == "OperationalError"
        # Should suggest retry somewhere in the instructions
        assert any("retry" in rule.lower() for rule in result["_llm_instructions"]["rules"])


class TestEmbeddingServiceBusyHandler:
    """Test handle_embedding_service_busy."""

    def test_embedding_busy_404(self):
        """Test embedding service busy 404 response."""
        raw = {
            "success": False,
            "status_code": 404,
            "error": "Server is busy processing embeddings. Please try again later.",
        }
        result = handle_embedding_service_busy(raw, {})
        assert result["status"] == "error"
        assert result["status_code"] == 404
        assert "embedding" in result["message"].lower()
        # Should suggest retry specifically
        assert any("retry" in rule.lower() for rule in result["_llm_instructions"]["rules"])


class TestUsecaseNotFoundHandler:
    """Test handle_usecase_not_found."""

    def test_404_not_found(self):
        """Test usecase not found 404 response."""
        raw = {
            "success": False,
            "status_code": 404,
            "error": "No usecase found for 'NonExistentUsecase'.",
        }
        result = handle_usecase_not_found(raw, {})
        assert result["status"] == "error"
        assert result["status_code"] == 404


class TestSemanticCandidatesHandler:
    """Test handle_semantic_candidates."""

    def test_semantic_candidates_response(self):
        """Test semantic candidates with distance scores."""
        raw = {"success": True, "status_code": 200}
        data = {
            "msg": "No exact match found",
            "usecase_name": "Hospital Monitoring",
            "semantic_candidates": [
                {"id": 5, "name": "Hospital Census Monitoring", "distance": 0.23},
                {"id": 12, "name": "Hospital Patient Monitoring System", "distance": 0.45},
            ],
        }
        result = handle_semantic_candidates(raw, data)
        assert result["status"] == "clarification_needed"
        assert result["requested_name"] == "Hospital Monitoring"
        assert len(result["candidates"]) == 2
        assert result["candidates"][0]["distance"] == 0.23


class TestMultipleCandidatesHandler:
    """Test handle_multiple_candidates."""

    def test_multiple_exact_matches(self):
        """Test multiple exact/partial matches."""
        raw = {"success": True, "status_code": 200}
        data = {
            "msg": "Multiple usecases match",
            "usecase_name": "Census",
            "candidates": [
                {"id": 3, "name": "Covid Census Forecasting"},
                {"id": 7, "name": "Hospital Census Monitoring"},
                {"id": 15, "name": "Regional Census Analysis"},
            ],
        }
        result = handle_multiple_candidates(raw, data)
        assert result["status"] == "clarification_needed"
        assert len(result["candidates"]) == 3
        assert all("id" in c and "name" in c for c in result["candidates"])


class TestFilterErrorInSuccessHandler:
    """Test handle_filter_error_in_success."""

    def test_filter_error_nested_in_success(self):
        """Test filter_error object inside a success=true response."""
        raw = {
            "success": True,
            "status_code": 200,
        }
        data = {
            "resolved": True,
            "usecase": {"id": 5, "name": "Hospital Census Forecasting"},
            "info": {"condition_type": "three_conditions"},
            "filters_applied": {"series": "invalid_series", "condition_one": "NCAL"},
            "forecast": [],
            "filter_error": {
                "error": True,
                "message": "The conditions applied are not eligible.",
                "invalid_filters": {"series": "invalid_series"},
                "available_options": {
                    "series": ["covid_total_census", "covid_icu_census"],
                    "facilityToUnit": {"facility_1": ["unit_a", "unit_b"]},
                },
            },
        }
        result = handle_filter_error_in_success(raw, data)
        assert result["status"] == "error"
        assert result["status_code"] == 200
        assert "resolved_usecase" in result
        assert result["resolved_usecase"]["id"] == 5
        assert "available_options" in result
        assert "facilityToUnit" in result["available_options"], "facilityToUnit must be preserved"


class TestForecastWithDataHandler:
    """Test handle_forecast_with_data."""

    def test_forecast_with_data_response(self):
        """Test successful forecast with data points."""
        raw = {"success": True, "status_code": 200}
        data = {
            "resolved": True,
            "usecase": {
                "id": 5,
                "name": "Hospital Census Forecasting",
                "usecase_type": "Multi_Model_Forecasting",
            },
            "info": {"condition_type": "three_conditions", "required_filters": ["series", "condition_one"]},
            "filters_applied": {"series": "covid_total_census", "condition_one": "NCAL"},
            "last_actual_update": "2026-04-07",
            "forecast": [
                {"Forecast Date": "2026-03-08", "Forecast Value": 125.5},
                {"Forecast Date": "2026-03-09", "Forecast Value": 128.3},
            ],
        }
        result = handle_forecast_with_data(raw, data)
        assert result["status"] == "success"
        assert result["data_available"] is True
        assert result["forecast_count"] == 2
        assert result["usecase"]["id"] == 5
        assert len(result["forecast"]) == 2


class TestEmptyForecastHandler:
    """Test handle_empty_forecast."""

    def test_empty_forecast_response(self):
        """Test successful response with no forecast data."""
        raw = {"success": True, "status_code": 200}
        data = {
            "resolved": True,
            "usecase": {
                "id": 5,
                "name": "Hospital Census Forecasting",
                "usecase_type": "Multi_Model_Forecasting",
            },
            "info": {"condition_type": "three_conditions"},
            "filters_applied": {"series": "covid_total_census", "condition_one": "NCAL"},
            "last_actual_update": "2026-04-07",
            "forecast": [],
        }
        result = handle_empty_forecast(raw, data)
        assert result["status"] == "success"
        assert result["data_available"] is False
        assert result["forecast_count"] == 0
        assert result["forecast"] == []
        # Instructions should mention something about no data or suggest alternatives
        assert "data" in result["message"].lower()  # "No forecast data available"


# ────────────────────────────────────────────────────────────────────────────
# Tests: Edge Cases
# ────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and unusual but valid inputs."""

    def test_data_is_null(self):
        """Test response where 'data' is explicitly null."""
        response = {
            "success": False,
            "status_code": 400,
            "error": "series is required",
            "data": None,
        }
        result = dispatch_response(response)
        assert result["status"] == "error"
        assert "_llm_instructions" in result

    def test_data_is_missing(self):
        """Test response where 'data' key is missing entirely."""
        response = {
            "success": False,
            "status_code": 400,
            "error": "series is required",
        }
        result = dispatch_response(response)
        assert result["status"] == "error"

    def test_forecast_is_none_instead_of_list(self):
        """Test response where 'forecast' is None instead of []."""
        response = {
            "success": True,
            "status_code": 200,
            "data": {
                "resolved": True,
                "usecase": {"id": 5, "name": "Test"},
                "forecast": None,
            },
        }
        result = dispatch_response(response)
        # Should classify as unknown since forecast is neither a list
        assert result["status"] in ["unknown", "success"]

    def test_available_options_is_empty(self):
        """Test invalid filter response where available_options is {}."""
        response = {
            "success": False,
            "status_code": 400,
            "error": "Invalid filter combination.",
            "data": {
                "message": "Invalid",
                "invalid_filters": {"series": "bad"},
                "available_options": {},
            },
        }
        result = dispatch_response(response)
        assert result["status"] == "error"
        # Should still have available_options key (even if empty)
        assert "available_options" in result

    def test_status_code_missing(self):
        """Test response where status_code is missing from input."""
        response = {
            "success": True,
            "data": {
                "resolved": True,
                "usecase": {"id": 5},
                "forecast": [{"Forecast Date": "2026-03-08", "Forecast Value": 125.5}],
            },
        }
        result = dispatch_response(response)
        assert result["status"] == "success"
        # status_code may be None if not provided in input, or set to 200 by handler
        assert result.get("status_code") is None or result.get("status_code") == 200

    def test_json_string_input(self):
        """Test that plain JSON string input is handled."""
        json_str = '{"success": false, "error": "Invalid"}'
        # Note: dispatch_response expects dict or string, but doesn't parse JSON strings
        # If the API sends JSON string instead of parsed dict, that's unusual
        result = dispatch_response(json_str)
        assert isinstance(result, dict)
        assert "status" in result


# ────────────────────────────────────────────────────────────────────────────
# Tests: Backward Compatibility (regression tests)
# ────────────────────────────────────────────────────────────────────────────


class TestBackwardCompatibility:
    """Test that refactored format_api_response maintains output consistency."""

    def test_forecast_with_data_structure(self):
        """Verify forecast_with_data output structure matches old implementation."""
        response = {
            "success": True,
            "status_code": 200,
            "data": {
                "resolved": True,
                "usecase": {
                    "id": 5,
                    "name": "Hospital Census Forecasting",
                    "usecase_type": "Multi_Model_Forecasting",
                },
                "info": {"condition_type": "three_conditions", "required_filters": ["series"]},
                "filters_applied": {"series": "covid_total_census", "condition_one": "NCAL"},
                "last_actual_update": "2026-04-07",
                "forecast": [
                    {
                        "Forecast Date": "2026-03-08",
                        "Forecast Value": 125.5,
                        "value_type": "covid_total_census",
                        "rgn_cd": "NCAL",
                        "fac_id_cd": "facility_1",
                    },
                ],
            },
        }
        result = dispatch_response(response)

        # Old implementation returned these fields for forecast_with_data
        assert result["status"] == "success"
        assert result["message"] == "Successfully retrieved forecast"
        assert result["data_available"] is True
        assert "usecase" in result
        assert result["usecase"]["id"] == 5
        assert result["usecase"]["name"] == "Hospital Census Forecasting"
        assert "forecast" in result
        assert len(result["forecast"]) == 1

    def test_invalid_filter_structure(self):
        """Verify invalid_filter output structure matches old implementation."""
        response = {
            "success": False,
            "status_code": 400,
            "error": "Invalid filter combination.",
            "data": {
                "error": True,
                "message": "The conditions applied are not eligible.",
                "invalid_filters": {
                    "series": "invalid_series_value",
                    "condition_one": None,
                    "condition_two": "invalid_facility",
                },
                "available_options": {
                    "series": ["covid_total_census", "covid_icu_census"],
                    "condition_one": ["NCAL", "SCAL"],
                    "conditions": {"NCAL": ["facility_1", "facility_2"]},
                    "facilityToUnit": {"facility_1": ["unit_a", "unit_b"]},
                },
            },
        }
        result = dispatch_response(response)

        assert result["status"] == "error"
        assert result["status_code"] == 400
        assert "invalid_filters" in result
        assert "available_options" in result
        assert "series" in result["available_options"]
        assert "condition_one" in result["available_options"]
        assert "conditions" in result["available_options"]
