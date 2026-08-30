from collections import Counter

from evaluation.intent_revision_benchmark import (
    InitialSemanticCase,
    RevisionSemanticCase,
    benchmark_hash,
    build_frozen_cases,
    score_initial,
    score_revision,
)
from models.travel_slots import (
    RevisionOperation,
    RevisionParseOutput,
    SlotParseOutput,
    TravelSlots,
)


def test_frozen_benchmark_has_100_unique_balanced_cases() -> None:
    cases = build_frozen_cases()

    assert len(cases) == 100
    assert len({case.case_id for case in cases}) == 100
    assert Counter(case.kind for case in cases) == {"initial": 50, "revision": 50}
    assert set(Counter(case.slice for case in cases).values()) == {5}
    assert benchmark_hash(cases) == benchmark_hash(build_frozen_cases())


def test_initial_scorer_checks_structured_semantics_and_model_source() -> None:
    case = InitialSemanticCase.model_validate(
        {
            "case_id": "test-initial",
            "slice": "event",
            "text": "去看演出",
            "expected": {
                "slot_values": {"destination": "上海", "intent_kind": "event_trip"},
                "required_needs": ["event"],
                "event_query_required": True,
            },
        }
    )
    result = SlotParseOutput(
        intent="generate_itinerary",
        confidence=0.9,
        slots=TravelSlots(
            destination="上海",
            intent_kind="event_trip",
            event_query="上海演出",
            information_needs=["event"],
        ),
        parse_source="llm",
    )

    assert score_initial(case, result) == (True, [])
    result.parse_source = "deterministic_fallback"
    passed, failures = score_initial(case, result)
    assert passed is False
    assert "MODEL_FALLBACK_USED" in failures


def test_initial_scorer_accepts_an_operationally_equivalent_need() -> None:
    case = InitialSemanticCase.model_validate(
        {
            "case_id": "test-any-need",
            "slice": "restaurant",
            "text": "找一家深夜营业的餐厅",
            "expected": {"any_of_needs": ["restaurant", "opening_hours"]},
        }
    )
    result = SlotParseOutput(
        slots=TravelSlots(information_needs=["opening_hours"]),
        parse_source="llm",
    )

    assert score_initial(case, result) == (True, [])


def test_revision_scorer_accepts_field_aliases_and_rejects_forbidden_field() -> None:
    case = RevisionSemanticCase.model_validate(
        {
            "case_id": "test-revision",
            "slice": "removal",
            "text": "删掉外滩",
            "current_goal": {},
            "expected": {
                "operations": [
                    {
                        "fields": ["must_not_visit", "avoid_pois"],
                        "operation": "add",
                        "value": "外滩",
                    }
                ],
                "forbidden_fields": ["travel_days"],
            },
        }
    )
    result = RevisionParseOutput(
        intent="revise_itinerary",
        confidence=0.95,
        operations=[RevisionOperation(field="avoid_pois", operation="add", value="外滩")],
    )

    assert score_revision(case, result) == (True, [])
    result.operations.append(RevisionOperation(field="travel_days", operation="set", value=3))
    passed, failures = score_revision(case, result)
    assert passed is False
    assert "FORBIDDEN_OPERATION:travel_days" in failures


def test_revision_scorer_allows_removing_a_forbidden_positive_requirement() -> None:
    case = RevisionSemanticCase.model_validate(
        {
            "case_id": "test-negative-requirement",
            "slice": "transport",
            "text": "不用查航班",
            "current_goal": {},
            "expected": {"forbidden_fields": ["transport_modes_requested"]},
        }
    )
    result = RevisionParseOutput(
        operations=[
            RevisionOperation(field="transport_modes_requested", operation="remove", value="flight")
        ]
    )

    assert score_revision(case, result) == (True, [])


def test_revision_scorer_accepts_semantically_more_specific_text_value() -> None:
    case = RevisionSemanticCase.model_validate(
        {
            "case_id": "test-specific-value",
            "slice": "preference",
            "text": "多看自然风光",
            "current_goal": {},
            "expected": {
                "operations": [{"fields": ["interests"], "operation": "add", "value": "自然"}]
            },
        }
    )
    result = RevisionParseOutput(
        operations=[RevisionOperation(field="interests", operation="add", value="自然风光")]
    )

    assert score_revision(case, result) == (True, [])
