from evaluation.full_agent_loop_benchmark import benchmark_hash, build_frozen_cases


def test_full_agent_loop_cases_are_unique_and_cover_required_slices() -> None:
    cases = build_frozen_cases()

    assert len(cases) == 30
    assert len({case.case_id for case in cases}) == 30
    assert {case.suite for case in cases} == {"core", "expanded"}
    assert sum(case.suite == "core" for case in cases) == 10
    assert sum(case.suite == "expanded" for case in cases) == 20
    assert {
        "ordinary",
        "family_constraints",
        "weather",
        "opening_hours",
        "restaurant",
        "seasonal_activity",
        "intercity_transport",
        "event_trip",
        "clarification",
        "user_revision",
    }.issubset({case.slice for case in cases})
    assert benchmark_hash(cases) == benchmark_hash(build_frozen_cases())


def test_every_draft_case_requires_real_loop_solver_and_verifier_actions() -> None:
    draft_cases = [case for case in build_frozen_cases() if case.expected_outcome == "draft"]

    for case in draft_cases:
        assert "finalize_research" in case.required_actions
        assert "solve_itinerary" in case.required_actions
        assert "validate_itinerary" in case.required_actions
        assert "solver_result" in case.required_artifacts
        assert "validation_report" in case.required_artifacts


def test_external_fact_cases_require_grounded_tradeoff_if_no_draft_is_safe() -> None:
    cases = [
        case
        for case in build_frozen_cases()
        if case.expected_outcome == "draft_or_safe_termination"
    ]

    assert {
        "restaurant",
        "intercity_transport",
        "event_trip",
        "temporary_closure",
        "flight_schedule",
        "current_exhibition",
        "late_restaurant",
        "tight_budget",
    } == {case.slice for case in cases}
    for case in cases:
        assert "propose_tradeoff" in case.safe_required_actions
        if case.slice != "tight_budget":
            assert any(action.startswith("search_") for action in case.safe_required_actions)


def test_revision_case_contains_a_second_user_turn() -> None:
    revisions = [case for case in build_frozen_cases() if case.expected_outcome == "revision"]

    assert len(revisions) == 4
    for revision in revisions:
        assert revision.revision_input
        assert (
            revision.expected_revision_hard
            or revision.expected_revision_soft
            or revision.expected_revision_exclusions
        )


def test_expanded_suite_checks_semantic_constraints() -> None:
    expanded = [case for case in build_frozen_cases() if case.suite == "expanded"]

    assert len(expanded) == 20
    assert all(case.expected_slots for case in expanded if not case.revision_input)
    assert {
        "wheelchair_accessibility",
        "pregnant_low_fatigue",
        "food_taboo",
        "must_not_visit",
        "max_transit",
        "tight_budget",
        "missing_destination",
        "missing_travel_days",
        "missing_transport_origin",
        "revision_add_must_visit",
        "revision_lower_budget",
        "revision_exclude_poi",
    }.issubset({case.slice for case in expanded})
