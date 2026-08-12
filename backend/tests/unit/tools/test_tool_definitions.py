"""Tests for tool schema definitions."""

import pytest

from tools.tool_definitions import TOOL_NAME_TO_SCHEMA, TOOLS


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_schema_has_required_fields(tool):
    assert tool["type"] == "function"
    func = tool["function"]
    assert func["name"]
    assert func["description"]
    assert func["parameters"]["type"] == "object"
    assert "properties" in func["parameters"]


def test_all_expected_tools_present():
    names = {t["function"]["name"] for t in TOOLS}
    expected = {
        "get_weather",
        "check_reservation",
        "get_route",
        "find_restaurants",
        "find_hotels",
        "get_queue_time",
        "get_ticket_link",
        "get_local_events",
        "get_emergency_services",
        "get_poi_detail",
        "update_user_profile",
        "search_pois",
        "get_route_matrix",
        "solve_itinerary",
        "validate_itinerary",
    }
    assert names == expected


def test_tool_lookup_by_name():
    assert "get_weather" in TOOL_NAME_TO_SCHEMA
    assert TOOL_NAME_TO_SCHEMA["get_weather"]["function"]["name"] == "get_weather"
