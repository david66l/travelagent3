"""Unit tests for conversation_turn orchestration."""

from unittest.mock import AsyncMock, patch

import pytest

from core.conversation_turn import process_user_turn, retain_agent_semantics_from_previous_turn
from core.conversation_state import default_conversation_state
from models.travel_slots import SlotParseOutput, TravelSlots


def _make_parsed(**overrides) -> SlotParseOutput:
    defaults = {
        "intent": "generate_itinerary",
        "confidence": 0.85,
        "sentiment": "neutral",
        "slots": TravelSlots(),
        "missing_slots": [],
        "clarifying_question": None,
    }
    defaults.update(overrides)
    return SlotParseOutput(**defaults)


def test_slot_filling_retains_model_derived_agent_semantics_only_when_omitted():
    previous = {
        "intent_kind": "event_trip",
        "event_query": "周杰伦上海站",
        "information_needs": ["event"],
    }
    filled = retain_agent_semantics_from_previous_turn(
        TravelSlots(travel_days=2),
        previous,
    )
    assert filled.intent_kind == "event_trip"
    assert filled.event_query == "周杰伦上海站"

    explicitly_cleared = retain_agent_semantics_from_previous_turn(
        TravelSlots(intent_kind="itinerary", information_needs=[]),
        previous,
    )
    assert explicitly_cleared.intent_kind == "itinerary"
    assert explicitly_cleared.information_needs == []


@pytest.mark.asyncio
async def test_process_user_turn_parses_and_updates_state():
    state = default_conversation_state()
    state["user_id"] = "user-1"

    parsed = _make_parsed(
        slots=TravelSlots(destination="成都", travel_days=3),
    )
    with patch(
        "core.conversation_turn.DemandParserAgent.parse", new=AsyncMock(return_value=parsed)
    ):
        with patch(
            "core.conversation_turn.ProfileRecallAgent.recall",
            new=AsyncMock(
                return_value={
                    "source": "anonymous",
                    "short_term_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "long_term_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "merged_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "recalled_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "inferred_slots": {},
                    "confidence": 0.0,
                }
            ),
        ):
            result = await process_user_turn(state, "成都3天")

    assert result.intent == "generate_itinerary"
    assert state["last_intent"] == "generate_itinerary"
    assert state["profile"]["trip"]["destination"] == "成都"
    assert state["profile"]["trip"]["travel_days"] == 3
    assert state["slots"]["destination"] == "成都"
    assert state["slots"]["travel_days"] == 3
    assert state["inferred_slots"] == {}
    assert "feasibility_report" in state
    assert state["turn"] == 1


@pytest.mark.asyncio
async def test_process_user_turn_adds_feasibility_issues_to_questions():
    state = default_conversation_state()
    state["user_id"] = "user-1"

    parsed = _make_parsed(
        slots=TravelSlots(destination="北京", travel_days=5, travelers_count=2, total_budget=500),
    )
    with patch(
        "core.conversation_turn.DemandParserAgent.parse", new=AsyncMock(return_value=parsed)
    ):
        with patch(
            "core.conversation_turn.ProfileRecallAgent.recall",
            new=AsyncMock(
                return_value={
                    "source": "anonymous",
                    "short_term_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "long_term_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "merged_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "recalled_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "inferred_slots": {},
                    "confidence": 0.0,
                }
            ),
        ):
            result = await process_user_turn(state, "北京5天预算500")

    assert result.feasibility_report is not None
    assert not result.feasibility_report["feasible"]
    assert any("预算偏低" in q for q in result.clarification_questions)


@pytest.mark.asyncio
async def test_process_user_turn_triggers_disambiguation():
    state = default_conversation_state()
    state["user_id"] = "user-1"

    parsed = _make_parsed(
        slots=TravelSlots(travel_days=3),
        disambiguation={
            "has_ambiguity": True,
            "field": "destination",
            "candidates": [{"value": "厦门", "reason": "海边休闲"}],
            "question": "您说的目的地比较宽泛，以下几个城市您更倾向哪个？",
        },
        clarifying_question="您说的目的地比较宽泛，以下几个城市您更倾向哪个？",
    )
    with patch(
        "core.conversation_turn.DemandParserAgent.parse", new=AsyncMock(return_value=parsed)
    ):
        with patch(
            "core.conversation_turn.ProfileRecallAgent.recall",
            new=AsyncMock(
                return_value={
                    "source": "anonymous",
                    "short_term_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "long_term_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "merged_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "recalled_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "inferred_slots": {},
                    "confidence": 0.0,
                }
            ),
        ):
            result = await process_user_turn(state, "想去南方玩3天")

    assert len(result.disambiguation_candidates) > 0
    assert len(result.clarification_questions) > 0


@pytest.mark.asyncio
async def test_process_user_turn_second_turn_fills_origin_dates_budget():
    """Regression: '济南出发，明天出发，预算5000' must persist origin and dates."""
    from agents import demand_parser

    state = default_conversation_state()
    state["user_id"] = "user-1"

    turn1 = _make_parsed(slots=TravelSlots(destination="上海", travel_days=4))
    turn2_llm = _make_parsed(slots=TravelSlots(total_budget=5000))

    recall_payload = {
        "source": "anonymous",
        "short_term_profile": AsyncMock(model_dump=lambda **kw: {}),
        "long_term_profile": AsyncMock(model_dump=lambda **kw: {}),
        "merged_profile": AsyncMock(model_dump=lambda **kw: {}),
        "recalled_profile": AsyncMock(model_dump=lambda **kw: {}),
        "inferred_slots": {},
        "confidence": 0.0,
    }

    with patch.object(
        demand_parser.llm,
        "structured_call",
        new=AsyncMock(side_effect=[turn1, turn2_llm]),
    ):
        with patch(
            "core.conversation_turn.ProfileRecallAgent.recall",
            new=AsyncMock(return_value=recall_payload),
        ):
            await process_user_turn(state, "我要去上海玩4天")
            result = await process_user_turn(state, "济南出发，明天出发，预算5000")

    trip = state["profile"].get("trip", state["profile"])
    assert trip.get("origin") == "济南"
    assert trip.get("travel_dates") is not None
    assert trip.get("budget_range") == 5000
    assert "origin" not in result.missing_required
    assert "travel_dates" not in result.missing_required
    assert "total_budget" not in result.missing_required


@pytest.mark.asyncio
async def test_process_user_turn_recomputes_missing_from_merged_profile():
    """Budget-only reply must not re-ask destination/days already in profile."""
    state = default_conversation_state()
    state["user_id"] = "user-1"
    state["profile"] = {
        "trip": {
            "destination": "上海",
            "travel_days": 4,
            "origin": "济南",
            "travel_dates": "2026-06-21",
        },
        "personal": {"pace": "moderate"},
    }

    parsed = _make_parsed(
        intent="update_preferences",
        slots=TravelSlots(total_budget=5000),
        missing_slots=[],
        clarifying_question=None,
    )
    with patch(
        "core.conversation_turn.DemandParserAgent.parse", new=AsyncMock(return_value=parsed)
    ):
        with patch(
            "core.conversation_turn.ProfileRecallAgent.recall",
            new=AsyncMock(
                return_value={
                    "source": "anonymous",
                    "short_term_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "long_term_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "merged_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "recalled_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "inferred_slots": {},
                    "confidence": 0.0,
                }
            ),
        ):
            result = await process_user_turn(state, "5000块")

    assert "destination" not in result.missing_required
    assert "travel_days" not in result.missing_required
    assert "has_elderly" not in result.missing_required
    assert "has_children" not in result.missing_required
    assert result.clarification_questions == []


@pytest.mark.asyncio
async def test_process_user_turn_keeps_transport_origin_as_turn_required():
    state = default_conversation_state()
    state["user_id"] = "user-1"
    parsed = _make_parsed(
        slots=TravelSlots(
            destination="成都",
            transport_modes_requested=["train"],
            information_needs=["transport"],
        ),
        missing_slots=["travel_days", "origin"],
    )
    recall_payload = {
        "source": "anonymous",
        "short_term_profile": AsyncMock(model_dump=lambda **kw: {}),
        "long_term_profile": AsyncMock(model_dump=lambda **kw: {}),
        "merged_profile": AsyncMock(model_dump=lambda **kw: {}),
        "recalled_profile": AsyncMock(model_dump=lambda **kw: {}),
        "inferred_slots": {},
        "confidence": 0.0,
    }
    with (
        patch(
            "core.conversation_turn.DemandParserAgent.parse",
            new=AsyncMock(return_value=parsed),
        ),
        patch(
            "core.conversation_turn.ProfileRecallAgent.recall",
            new=AsyncMock(return_value=recall_payload),
        ),
    ):
        result = await process_user_turn(state, "我想坐高铁去成都")

    assert result.missing_required == ["travel_days", "origin"]
    assert "出发城市" in result.clarification_questions[0]


@pytest.mark.asyncio
async def test_process_user_turn_recalls_profile_and_marks_inferred():
    state = default_conversation_state()
    state["user_id"] = "user-1"

    parsed = _make_parsed(
        slots=TravelSlots(destination="北京", travel_days=3),
    )
    with patch(
        "core.conversation_turn.DemandParserAgent.parse", new=AsyncMock(return_value=parsed)
    ):
        with patch(
            "core.conversation_turn.ProfileRecallAgent.recall",
            new=AsyncMock(
                return_value={
                    "source": "long_term",
                    "short_term_profile": AsyncMock(model_dump=lambda **kw: {}),
                    "long_term_profile": AsyncMock(model_dump=lambda **kw: {"pace": "relaxed"}),
                    "merged_profile": AsyncMock(model_dump=lambda **kw: {"pace": "relaxed"}),
                    "recalled_profile": AsyncMock(model_dump=lambda **kw: {"pace": "relaxed"}),
                    "inferred_slots": {"pace": "relaxed"},
                    "confidence": 0.75,
                }
            ),
        ):
            result = await process_user_turn(state, "北京3天")

    assert state["inferred_slots"] == {"pace": "relaxed"}
    assert result.slots["pace"] == "relaxed"


@pytest.mark.asyncio
async def test_step3_acceptance_beijing_parents_3days(mock_llm):
    """Step 3 acceptance: '我想带爸妈去北京玩3天，预算5000' produces full slots + inferred_slots, no error-level conflicts."""
    from agents import demand_parser

    state = default_conversation_state()
    state["user_id"] = "user-1"

    parsed = _make_parsed(
        slots=TravelSlots(
            destination="北京",
            travel_days=3,
            travelers_count=3,
            travel_companion="parents",
            has_elderly=True,
            total_budget=5000,
        ),
    )
    mock_llm.structured_call = AsyncMock(return_value=parsed)
    demand_parser.llm = mock_llm

    with patch(
        "core.conversation_turn.ProfileRecallAgent.recall",
        new=AsyncMock(
            return_value={
                "source": "long_term",
                "short_term_profile": AsyncMock(model_dump=lambda **kw: {}),
                "long_term_profile": AsyncMock(model_dump=lambda **kw: {"pace": "moderate"}),
                "merged_profile": AsyncMock(model_dump=lambda **kw: {"pace": "moderate"}),
                "recalled_profile": AsyncMock(model_dump=lambda **kw: {"pace": "moderate"}),
                "inferred_slots": {},
                "confidence": 0.75,
            }
        ),
    ):
        result = await process_user_turn(state, "我想带爸妈去北京玩3天，预算5000")

    assert result.intent == "generate_itinerary"
    assert result.slots["destination"] == "北京"
    assert result.slots["travel_days"] == 3
    assert result.slots["travelers_count"] == 3
    assert result.slots["travel_companion"] == "parents"
    assert result.slots["has_elderly"] is True
    assert result.slots["total_budget"] == 5000
    assert "inferred_slots" in state
    assert result.feasibility_report["feasible"] is True
    assert result.feasibility_report["budget_fit"] in ("ok", "tight")
