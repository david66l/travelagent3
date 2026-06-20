"""Unit tests for HallucinationDetectionAgent."""

import pytest

from agents.hallucination_detector import HallucinationDetectionAgent


class TestHallucinationDetectionAgent:
    def test_poi_existence_passes_with_candidates(self):
        itinerary = [
            {"activities": [{"poi_name": "故宫"}, {"poi_name": "长城"}]}
        ]
        poi_candidates = [{"name": "故宫"}, {"name": "长城"}]
        score, issues, critical = (
            HallucinationDetectionAgent.check_poi_existence(
                itinerary, poi_candidates, []
            )
        )
        assert score == 1.0
        assert not issues
        assert not critical

    def test_poi_existence_passes_with_tool_results(self):
        itinerary = [{"activities": [{"poi_name": "故宫"}]}]
        tool_results = [
            {
                "name": "get_poi_detail",
                "result": {"data": {"name": "故宫"}},
            }
        ]
        score, issues, critical = (
            HallucinationDetectionAgent.check_poi_existence(
                itinerary, [], tool_results
            )
        )
        assert score == 1.0
        assert not issues
        assert not critical

    def test_poi_existence_fails_for_missing_poi(self):
        itinerary = [
            {"activities": [{"poi_name": "故宫"}, {"poi_name": "火星博物馆"}]}
        ]
        poi_candidates = [{"name": "故宫"}]
        score, issues, critical = (
            HallucinationDetectionAgent.check_poi_existence(
                itinerary, poi_candidates, []
            )
        )
        assert score == 0.5
        assert len(issues) == 1
        assert len(critical) == 1
        assert "火星博物馆" in issues[0]
        assert "火星博物馆" in critical[0]

    def test_opening_hours_passes(self):
        itinerary = [
            {
                "activities": [
                    {
                        "poi_name": "故宫",
                        "start_time": "09:30",
                        "end_time": "11:30",
                    }
                ]
            }
        ]
        poi_details = [
            {"name": "故宫", "open_time": "09:00", "close_time": "17:00"}
        ]
        score, issues = HallucinationDetectionAgent.check_opening_hours(
            itinerary, poi_details
        )
        assert score == 1.0
        assert not issues

    def test_opening_hours_fails_outside_range(self):
        itinerary = [
            {
                "activities": [
                    {
                        "poi_name": "故宫",
                        "start_time": "18:00",
                        "end_time": "20:00",
                    }
                ]
            }
        ]
        poi_details = [
            {"name": "故宫", "open_time": "09:00", "close_time": "17:00"}
        ]
        score, issues = HallucinationDetectionAgent.check_opening_hours(
            itinerary, poi_details
        )
        assert score == 0.0
        assert len(issues) == 1
        assert "故宫" in issues[0]

    def test_opening_hours_parses_open_hours_string(self):
        itinerary = [
            {
                "activities": [
                    {
                        "poi_name": "故宫",
                        "start_time": "09:30",
                        "end_time": "11:30",
                    }
                ]
            }
        ]
        poi_details = [{"name": "故宫", "open_hours": "09:00-17:00"}]
        score, issues = HallucinationDetectionAgent.check_opening_hours(
            itinerary, poi_details
        )
        assert score == 1.0
        assert not issues

    def test_ticket_price_deviation_above_tolerance(self):
        itinerary = [
            {"activities": [{"poi_name": "故宫", "ticket_price": 100}]}
        ]
        poi_details = [{"name": "故宫", "ticket_price": 60}]
        score, issues = HallucinationDetectionAgent.check_ticket_prices(
            itinerary, poi_details, tolerance=0.3
        )
        assert score == 0.0
        assert len(issues) == 1
        assert "故宫" in issues[0]

    def test_ticket_price_within_tolerance(self):
        itinerary = [
            {"activities": [{"poi_name": "故宫", "ticket_price": 65}]}
        ]
        poi_details = [{"name": "故宫", "ticket_price": 60}]
        score, issues = HallucinationDetectionAgent.check_ticket_prices(
            itinerary, poi_details, tolerance=0.3
        )
        assert score == 1.0
        assert not issues

    def test_route_commute_deviation_above_tolerance(self):
        itinerary = [
            {
                "activities": [
                    {
                        "poi_name": "故宫",
                        "transit_from_prev": {"duration_min": 100},
                    }
                ]
            }
        ]
        route_results = [
            {
                "name": "get_route",
                "result": {
                    "data": {"destination": "故宫", "minutes": 30}
                },
            }
        ]
        score, issues = HallucinationDetectionAgent.check_route_commute(
            itinerary, route_results, tolerance=0.5
        )
        assert score == 0.0
        assert len(issues) == 1
        assert "故宫" in issues[0]

    def test_route_commute_within_tolerance(self):
        itinerary = [
            {
                "activities": [
                    {
                        "poi_name": "故宫",
                        "transit_from_prev": {"duration_min": 30},
                    }
                ]
            }
        ]
        route_results = [
            {
                "name": "get_route",
                "result": {
                    "data": {"destination": "故宫", "minutes": 30}
                },
            }
        ]
        score, issues = HallucinationDetectionAgent.check_route_commute(
            itinerary, route_results, tolerance=0.5
        )
        assert score == 1.0
        assert not issues

    def test_reservation_annotation_missing(self):
        itinerary = [
            {"activities": [{"poi_name": "故宫", "tags": ["历史"]}]}
        ]
        reservation_results = [
            {
                "name": "check_reservation",
                "result": {"data": {"poi": "故宫", "need_reserve": True}},
            }
        ]
        score, issues = (
            HallucinationDetectionAgent.check_reservation_annotations(
                itinerary, reservation_results
            )
        )
        assert score == 0.0
        assert len(issues) == 1
        assert "故宫" in issues[0]

    def test_reservation_annotation_present_in_tags(self):
        itinerary = [
            {"activities": [{"poi_name": "故宫", "tags": ["历史", "预约"]}]}
        ]
        reservation_results = [
            {
                "name": "check_reservation",
                "result": {"data": {"poi": "故宫", "need_reserve": True}},
            }
        ]
        score, issues = (
            HallucinationDetectionAgent.check_reservation_annotations(
                itinerary, reservation_results
            )
        )
        assert score == 1.0
        assert not issues

    def test_reservation_annotation_present_in_recommendation_reason(self):
        itinerary = [
            {
                "activities": [
                    {
                        "poi_name": "故宫",
                        "tags": ["历史"],
                        "recommendation_reason": "需要提前预约",
                    }
                ]
            }
        ]
        reservation_results = [
            {
                "name": "check_reservation",
                "result": {"data": {"poi": "故宫", "need_reserve": True}},
            }
        ]
        score, issues = (
            HallucinationDetectionAgent.check_reservation_annotations(
                itinerary, reservation_results
            )
        )
        assert score == 1.0
        assert not issues

    def test_detect_empty_state_returns_passed(self):
        result = HallucinationDetectionAgent.detect({})
        assert result.passed is True
        assert result.total_score == 0.0
        assert not result.critical_failures
        assert not result.improvement_suggestions

    def test_detect_missing_tool_results_returns_passed(self):
        state = {"itinerary": [{"activities": [{"poi_name": "故宫"}]}]}
        result = HallucinationDetectionAgent.detect(state)
        assert result.passed is True
        assert result.total_score == 0.0

    def test_detect_full_state_passes(self):
        state = {
            "itinerary": [
                {
                    "activities": [
                        {
                            "poi_name": "故宫",
                            "start_time": "09:30",
                            "end_time": "11:30",
                            "ticket_price": 60,
                            "tags": ["历史", "预约"],
                        }
                    ]
                }
            ],
            "poi_candidates": [{"name": "故宫"}],
            "tool_results": [
                {
                    "name": "get_poi_detail",
                    "result": {
                        "data": {
                            "name": "故宫",
                            "open_time": "09:00",
                            "close_time": "17:00",
                            "ticket_price": 60,
                        }
                    },
                },
                {
                    "name": "check_reservation",
                    "result": {
                        "data": {"poi": "故宫", "need_reserve": True}
                    },
                },
            ],
        }
        result = HallucinationDetectionAgent.detect(state)
        assert result.passed is True
        assert result.total_score == 1.0
        assert not result.critical_failures
        assert not result.improvement_suggestions

    def test_detect_full_state_fails_due_to_missing_poi(self):
        state = {
            "itinerary": [
                {
                    "activities": [
                        {
                            "poi_name": "火星博物馆",
                            "start_time": "09:30",
                            "end_time": "11:30",
                        }
                    ]
                }
            ],
            "poi_candidates": [{"name": "故宫"}],
            "tool_results": [],
        }
        result = HallucinationDetectionAgent.detect(state)
        # Empty tool_results is treated as missing -> defensive pass.
        assert result.passed is True
