"""Tests for IntentRecognitionAgent."""

import pytest
from unittest.mock import AsyncMock
from schemas import IntentResult, UserProfile
from agents.intent_recognition import IntentRecognitionAgent


def make_intent_result(
    intent: str = "generate_itinerary",
    entities: dict = None,
    missing_required: list = None,
    clarification_questions: list = None,
    preference_changes: list = None,
    confidence: float = 0.95,
):
    """Helper to create IntentResult for mocks."""
    return IntentResult(
        intent=intent,
        user_entities=entities or {},
        missing_required=missing_required or [],
        clarification_questions=clarification_questions or ([] if (entities or missing_required) else ["能再说一下吗？"]),
        preference_changes=preference_changes or [],
        confidence=confidence,
    )


class TestIntentRecognition:
    """Test intent classification and entity extraction."""

    def setup_method(self):
        self.agent = IntentRecognitionAgent()

    # === Intent Classification Tests ===

    @pytest.mark.asyncio
    async def test_generate_itinerary_intent(self, mock_llm):
        import agents.intent_recognition as ir
        mock_llm.structured_call = AsyncMock(
            return_value=make_intent_result(
                "generate_itinerary",
                entities={"destination": "成都", "travel_days": 4},
            )
        )
        result = await self.agent.recognize("我想去成都玩4天", messages=[])
        assert result.intent == "generate_itinerary"
        assert result.user_entities["destination"] == "成都"

    @pytest.mark.asyncio
    async def test_modify_itinerary_intent(self, mock_llm):
        mock_llm.structured_call = AsyncMock(
            return_value=make_intent_result("modify_itinerary")
        )
        result = await self.agent.recognize("第三天换个景点", messages=[])
        assert result.intent == "modify_itinerary"

    @pytest.mark.asyncio
    async def test_update_preferences_intent(self, mock_llm):
        mock_llm.structured_call = AsyncMock(
            return_value=make_intent_result(
                "update_preferences",
                entities={"food_preferences": ["辣"]},
                preference_changes=[{"field": "food_preferences", "new_value": ["辣"]}],
            )
        )
        result = await self.agent.recognize("我喜欢吃辣", messages=[])
        assert result.intent == "update_preferences"

    @pytest.mark.asyncio
    async def test_query_info_intent(self, mock_llm):
        mock_llm.structured_call = AsyncMock(
            return_value=make_intent_result("query_info")
        )
        result = await self.agent.recognize("成都什么好吃？", messages=[])
        assert result.intent == "query_info"

    @pytest.mark.asyncio
    async def test_confirm_itinerary_intent(self, mock_llm):
        mock_llm.structured_call = AsyncMock(
            return_value=make_intent_result("confirm_itinerary")
        )
        result = await self.agent.recognize("确认行程", messages=[])
        assert result.intent == "confirm_itinerary"

    @pytest.mark.asyncio
    async def test_chitchat_intent(self, mock_llm):
        mock_llm.structured_call = AsyncMock(
            return_value=make_intent_result("chitchat")
        )
        result = await self.agent.recognize("你好", messages=[])
        assert result.intent == "chitchat"

    # === Entity Extraction Tests ===

    @pytest.mark.asyncio
    async def test_extract_destination_and_days(self, mock_llm):
        mock_llm.structured_call = AsyncMock(
            return_value=make_intent_result(
                "generate_itinerary",
                entities={"destination": "杭州", "travel_days": 3, "budget_range": 5000},
            )
        )
        result = await self.agent.recognize("我想去杭州玩3天，预算5000", messages=[])
        assert result.user_entities["destination"] == "杭州"
        assert result.user_entities["travel_days"] == 3
        assert result.user_entities["budget_range"] == 5000

    @pytest.mark.asyncio
    async def test_extract_budget_and_preferences(self, mock_llm):
        mock_llm.structured_call = AsyncMock(
            return_value=make_intent_result(
                "generate_itinerary",
                entities={
                    "destination": "西安",
                    "travel_days": 5,
                    "food_preferences": ["面食", "羊肉"],
                    "interests": ["历史", "博物馆"],
                },
            )
        )
        result = await self.agent.recognize("西安5天，爱吃面食和羊肉，喜欢历史博物馆", messages=[])
        assert "面食" in result.user_entities["food_preferences"]
        assert "历史" in result.user_entities["interests"]

    @pytest.mark.asyncio
    async def test_missing_required_fields(self, mock_llm):
        mock_llm.structured_call = AsyncMock(
            return_value=make_intent_result(
                "generate_itinerary",
                entities={"budget_range": 3000},
                missing_required=["destination", "travel_days"],
                clarification_questions=["想去哪里？", "玩几天？"],
            )
        )
        result = await self.agent.recognize("预算3000", messages=[])
        assert "destination" in result.missing_required
        assert "travel_days" in result.missing_required
        assert len(result.clarification_questions) > 0

    @pytest.mark.asyncio
    async def test_low_confidence_triggers_clarification(self, mock_llm):
        mock_llm.structured_call = AsyncMock(
            return_value=make_intent_result(
                "generate_itinerary",
                entities={},
                confidence=0.5,
            )
        )
        result = await self.agent.recognize("嗯...随便吧", messages=[])
        assert result.confidence < 0.7
        assert len(result.clarification_questions) > 0

    # === Preference Change Detection Tests ===

    @pytest.mark.asyncio
    async def test_detect_preference_changes(self, mock_llm):
        profile = UserProfile(destination="成都", food_preferences=["辣"])
        mock_llm.structured_call = AsyncMock(
            return_value=make_intent_result(
                "update_preferences",
                entities={"food_preferences": ["辣", "甜品"]},
            )
        )
        result = await self.agent.recognize("我还喜欢吃甜品", user_profile=profile, messages=[])
        assert len(result.preference_changes) > 0
        assert result.preference_changes[0]["field"] == "food_preferences"

    # === Date Resolution Tests (extended from existing) ===

    def test_resolve_date_iso_format(self):
        result = self.agent._resolve_date("2026-07-15")
        assert result == "2026-07-15"

    def test_resolve_date_month_day(self):
        result = self.agent._resolve_date("5月1日")
        assert result is not None
        assert "05-01" in result

    def test_resolve_date_relative_tomorrow(self):
        from datetime import datetime, timedelta
        result = self.agent._resolve_date("明天")
        expected = (datetime.now().date() + timedelta(days=1)).isoformat()
        assert result == expected

    def test_resolve_date_relative_day_after(self):
        from datetime import datetime, timedelta
        result = self.agent._resolve_date("后天")
        expected = (datetime.now().date() + timedelta(days=2)).isoformat()
        assert result == expected

    def test_resolve_date_range(self):
        result = self.agent._resolve_date("5月1日到5月5日")
        assert "to" in result
        parts = result.split(" to ")
        assert len(parts) == 2

    def test_resolve_date_invalid(self):
        result = self.agent._resolve_date("不知道")
        assert result == "不知道"  # preserves original if unparseable

    def test_resolve_date_empty(self):
        result = self.agent._resolve_date("")
        assert result is None

    def test_resolve_date_next_monday(self):
        result = self.agent._resolve_date("下周一")
        assert result is not None
        result_date = __import__("datetime").datetime.fromisoformat(result).date()
        assert result_date.weekday() == 0  # Monday

    def test_resolve_date_this_week(self):
        result = self.agent._resolve_date("这周五")
        assert result is not None
