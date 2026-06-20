"""Unit tests for DemandParserAgent."""

from unittest.mock import AsyncMock, patch

import pytest

from agents import demand_parser
from models.travel_slots import SlotParseOutput, TravelSlots


@pytest.fixture
def parser():
    return demand_parser.DemandParserAgent()


def _make_parse_output(**overrides) -> SlotParseOutput:
    defaults = {
        "intent": "generate_itinerary",
        "confidence": 0.9,
        "sentiment": "neutral",
        "slots": TravelSlots(),
        "missing_slots": [],
    }
    defaults.update(overrides)
    return SlotParseOutput(**defaults)


@pytest.mark.asyncio
async def test_parse_extracts_slots(parser):
    slots = TravelSlots(
        destination="成都",
        travel_days=4,
        travelers_count=2,
        travel_companion="couple",
        interests=["美食", "历史"],
        food_prefs=["辣"],
    )
    fake = _make_parse_output(slots=slots)
    with patch.object(demand_parser.llm, "structured_call", new=AsyncMock(return_value=fake)):
        result = await parser.parse("我想和女朋友去成都玩4天", [], None)

    assert result.intent == "generate_itinerary"
    assert result.slots.destination == "成都"
    assert result.slots.travel_days == 4
    assert result.slots.travel_companion == "couple"
    assert "美食" in result.slots.interests


@pytest.mark.asyncio
async def test_parse_detects_positive_sentiment(parser):
    slots = TravelSlots(destination="三亚", travel_days=5)
    fake = _make_parse_output(slots=slots)
    with patch.object(demand_parser.llm, "structured_call", new=AsyncMock(return_value=fake)):
        result = await parser.parse("超期待去三亚度蜜月", [], None)
    assert result.sentiment == "positive"


@pytest.mark.asyncio
async def test_parse_detects_urgent_sentiment(parser):
    slots = TravelSlots(destination="上海", travel_days=2)
    fake = _make_parse_output(slots=slots)
    with patch.object(demand_parser.llm, "structured_call", new=AsyncMock(return_value=fake)):
        result = await parser.parse("今晚就要到上海，赶紧安排", [], None)
    assert result.sentiment == "urgent"


@pytest.mark.asyncio
async def test_parse_maps_companion_types(parser):
    slots = TravelSlots(destination="北京", travel_days=3, travel_companion="family")
    fake = _make_parse_output(slots=slots)
    with patch.object(demand_parser.llm, "structured_call", new=AsyncMock(return_value=fake)):
        result = await parser.parse("带孩子去北京", [], None)
    assert result.slots.travel_companion == "family"


@pytest.mark.asyncio
async def test_build_clarifying_question_lists_all_missing(parser):
    question = parser._build_clarifying_question(
        ["origin", "travel_dates", "total_budget", "has_elderly", "has_children"],
        "上海",
    )
    assert question.startswith("您需要把以下信息告诉我：")
    assert "出发城市" in question
    assert "出行日期" in question
    assert "预算多少" in question
    assert "有没有带老人" in question
    assert "有没有带小孩" in question


@pytest.mark.asyncio
async def test_build_clarifying_question_single_missing(parser):
    question = parser._build_clarifying_question(["has_children"], "上海")
    assert question == "您需要把这个信息告诉我：有没有带小孩。"


@pytest.mark.asyncio
async def test_parse_fills_missing_required(parser):
    slots = TravelSlots(destination="成都")
    fake = _make_parse_output(slots=slots, missing_slots=["travel_days"])
    with patch.object(demand_parser.llm, "structured_call", new=AsyncMock(return_value=fake)):
        result = await parser.parse("去成都", [], None)
    assert "travel_days" in result.missing_slots
    assert result.clarifying_question.startswith("您需要把")
    assert "玩几天" in result.clarifying_question


@pytest.mark.asyncio
async def test_parse_fallback_on_llm_failure(parser):
    with patch.object(
        demand_parser.llm, "structured_call", new=AsyncMock(side_effect=RuntimeError("llm down"))
    ):
        result = await parser.parse("我想去北京玩3天", [], None)
    assert result.intent == "generate_itinerary"
    assert result.slots.destination == "北京"
    assert result.slots.travel_days == 3


@pytest.mark.asyncio
async def test_parse_fallback_companion_couple(parser):
    with patch.object(
        demand_parser.llm, "structured_call", new=AsyncMock(side_effect=RuntimeError("llm down"))
    ):
        result = await parser.parse("我想和女朋友去成都玩4天", [], None)
    assert result.slots.destination == "成都"
    assert result.slots.travel_days == 4
    assert result.slots.travel_companion == "couple"
    assert "has_elderly" in result.missing_slots
    assert "origin" in result.missing_slots


@pytest.mark.asyncio
async def test_enrich_slots_from_multi_field_reply(parser):
    fake = _make_parse_output(slots=TravelSlots(total_budget=5000))
    known = {
        "destination": "上海",
        "travel_days": 4,
    }
    with patch.object(demand_parser.llm, "structured_call", new=AsyncMock(return_value=fake)):
        result = await parser.parse("济南出发，明天出发，预算5000", [], None, known_profile=known)

    assert result.intent == "generate_itinerary"
    assert result.slots.origin == "济南"
    assert result.slots.travel_dates is not None
    assert result.slots.total_budget == 5000
    assert "origin" not in result.missing_slots
    assert "travel_dates" not in result.missing_slots
    assert "total_budget" not in result.missing_slots


@pytest.mark.asyncio
async def test_parse_rewrites_update_preferences_when_gathering(parser):
    """Budget-only reply during gathering should stay on generate_itinerary."""
    fake = _make_parse_output(
        intent="update_preferences",
        slots=TravelSlots(total_budget=5000),
        missing_slots=[],
    )
    known = {
        "destination": "上海",
        "travel_days": 4,
        "origin": "济南",
        "travel_dates": "2026-06-21",
        "travelers_count": 1,
    }
    with patch.object(demand_parser.llm, "structured_call", new=AsyncMock(return_value=fake)):
        result = await parser.parse("5000块", [], None, known_profile=known)
    assert result.intent == "generate_itinerary"
    assert result.slots.total_budget == 5000
    assert "has_elderly" in result.missing_slots
    assert "has_children" in result.missing_slots


@pytest.mark.asyncio
async def test_parse_merges_known_profile(parser):
    slots = TravelSlots(destination="成都", travel_days=4)
    fake = _make_parse_output(slots=slots)
    known = {
        "origin": "深圳",
        "destination": "成都",
        "travel_dates": "下周",
        "travel_days": 4,
        "travelers_count": 2,
        "travelers_type": "couple",
        "has_elderly": False,
        "has_children": False,
        "budget_range": 5000,
    }
    with patch.object(demand_parser.llm, "structured_call", new=AsyncMock(return_value=fake)):
        result = await parser.parse("成都4天", [], None, known_profile=known)
    assert result.missing_slots == []
    assert result.clarifying_question is None


@pytest.mark.asyncio
async def test_enrich_slots_extracts_has_elderly(parser):
    fake = _make_parse_output(slots=TravelSlots(destination="北京", travel_days=3))
    with patch.object(demand_parser.llm, "structured_call", new=AsyncMock(return_value=fake)):
        result = await parser.parse("我想带爸妈去北京玩3天", [], None)

    assert result.slots.has_elderly is True
    assert "has_elderly" not in result.missing_slots


@pytest.mark.asyncio
async def test_parse_chitchat_no_required_missing(parser):
    fake = _make_parse_output(intent="chitchat", slots=TravelSlots())
    with patch.object(demand_parser.llm, "structured_call", new=AsyncMock(return_value=fake)):
        result = await parser.parse("你好", [], None)
    assert result.intent == "chitchat"
    assert result.missing_slots == []
    assert result.clarifying_question is None
