from collections import Counter

import pytest

from evaluation.native_react_training_corpus import (
    build_native_react_training_case,
    build_native_react_training_cases,
)


def test_first_240_cases_cover_twenty_families_and_twelve_cities() -> None:
    cases = build_native_react_training_cases(0, 240)

    assert len({case.case_id for case in cases}) == 240
    assert len({case.slice for case in cases}) == 20
    assert len({case.expected_slots["destination"] for case in cases}) == 12
    assert Counter(case.slice for case in cases).most_common(1)[0][1] == 12
    assert all(case.expected_slots["travel_days"] >= 2 for case in cases)
    assert all(case.expected_slots["total_budget"] > 0 for case in cases)


def test_current_information_families_allow_verified_draft_or_safe_tradeoff() -> None:
    opening_hours = build_native_react_training_case(9 * 12)
    event_trip = build_native_react_training_case(12 * 12)

    for case in (opening_hours, event_trip):
        assert case.expected_outcome == "draft_or_safe_termination"
        assert case.required_actions == ["search_current_info"]
        assert case.safe_required_actions == ["search_current_info", "propose_tradeoff"]


def test_invalid_case_range_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_native_react_training_cases(-1, 1)
    with pytest.raises(ValueError):
        build_native_react_training_cases(0, 0)
