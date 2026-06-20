"""Unit tests for DisambiguationEngine."""

import pytest

from agents.disambiguation import DisambiguationEngine
from models.travel_slots import TravelSlots


class TestDisambiguationEngine:
    def test_no_ambiguity_for_concrete_destination(self):
        slots = TravelSlots(destination="成都", travel_days=3)
        result = DisambiguationEngine.analyze(slots, "成都3天攻略")
        assert not result["has_ambiguity"]

    def test_vague_destination_triggers_ambiguity(self):
        slots = TravelSlots(destination=None, travel_days=3)
        result = DisambiguationEngine.analyze(slots, "想去南方玩3天")
        assert result["has_ambiguity"]
        assert result["field"] == "destination"
        assert len(result["candidates"]) > 0
        assert "厦门" in {c["value"] for c in result["candidates"]}

    def test_vague_budget_triggers_question(self):
        slots = TravelSlots(destination="成都", travel_days=3)
        result = DisambiguationEngine.analyze(slots, "成都3天，便宜点")
        assert result["has_ambiguity"]
        assert result["field"] == "budget"
        assert "预算" in result["question"]

    def test_vague_days_triggers_question(self):
        slots = TravelSlots(destination="成都")
        result = DisambiguationEngine.analyze(slots, "成都玩几天合适")
        assert result["has_ambiguity"]
        assert result["field"] == "travel_days"

    def test_people_question_triggers(self):
        slots = TravelSlots(destination="成都")
        result = DisambiguationEngine.analyze(slots, "成都，几个人去合适")
        assert result["has_ambiguity"]
        assert result["field"] == "travelers"

    def test_llm_candidates_respected(self):
        slots = TravelSlots(destination=None, travel_days=3)
        candidates = [{"value": "昆明", "reason": "气候宜人"}]
        result = DisambiguationEngine.analyze(slots, "想去南边", candidates_from_llm=candidates)
        assert result["has_ambiguity"]
        assert any(c["value"] == "昆明" for c in result["candidates"])
