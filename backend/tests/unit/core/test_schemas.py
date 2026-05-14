"""Tests for Pydantic schema validation."""

import pytest
from pydantic import ValidationError
from schemas import (
    UserProfile,
    IntentResult,
    BudgetPanel,
    Activity,
    DayPlan,
    TravelContext,
    ScoredPOI,
    Location,
)


class TestUserProfile:
    """Test UserProfile model."""

    def test_defaults(self):
        p = UserProfile()
        assert p.travelers_count == 1
        assert p.pace == "moderate"
        assert p.food_preferences == []
        assert p.interests == []

    def test_custom_values(self):
        p = UserProfile(destination="成都", travel_days=4, pace="relaxed")
        assert p.destination == "成都"
        assert p.travel_days == 4
        assert p.pace == "relaxed"

    def test_invalid_pace(self):
        # Pydantic v2 with Literal doesn't enforce at model level for str fields
        # unless explicitly constrained; pace is str not Literal
        p = UserProfile(pace="invalid_pace")
        assert p.pace == "invalid_pace"

    def test_list_fields_default_factory(self):
        p1 = UserProfile()
        p2 = UserProfile()
        p1.food_preferences.append("辣")
        # Default factory ensures separate lists
        assert "辣" not in p2.food_preferences


class TestIntentResult:
    """Test IntentResult model."""

    def test_valid_intent(self):
        r = IntentResult(intent="generate_itinerary", confidence=0.95)
        assert r.intent == "generate_itinerary"
        assert r.confidence == 0.95

    def test_invalid_intent(self):
        with pytest.raises(ValidationError):
            IntentResult(intent="invalid_intent", confidence=0.5)

    def test_confidence_too_high(self):
        with pytest.raises(ValidationError):
            IntentResult(intent="generate_itinerary", confidence=1.5)

    def test_confidence_too_low(self):
        with pytest.raises(ValidationError):
            IntentResult(intent="generate_itinerary", confidence=-0.1)

    def test_confidence_boundary(self):
        r = IntentResult(intent="generate_itinerary", confidence=0.0)
        assert r.confidence == 0.0
        r = IntentResult(intent="generate_itinerary", confidence=1.0)
        assert r.confidence == 1.0

    def test_ensure_list_validator_none(self):
        r = IntentResult(
            intent="generate_itinerary",
            confidence=0.9,
            missing_required=None,
            missing_recommended=None,
            preference_changes=None,
            clarification_questions=None,
        )
        assert r.missing_required == []
        assert r.missing_recommended == []
        assert r.preference_changes == []
        assert r.clarification_questions == []

    def test_ensure_list_validator_with_values(self):
        r = IntentResult(
            intent="generate_itinerary",
            confidence=0.9,
            missing_required=["destination"],
            clarification_questions=["想去哪里？"],
        )
        assert r.missing_required == ["destination"]
        assert r.clarification_questions == ["想去哪里？"]


class TestBudgetPanel:
    """Test BudgetPanel model."""

    def test_defaults(self):
        b = BudgetPanel()
        assert b.spent == 0
        assert b.status == "within_budget"
        assert b.breakdown == {}

    def test_custom_status(self):
        b = BudgetPanel(status="over_budget")
        assert b.status == "over_budget"


class TestActivity:
    """Test Activity model."""

    def test_defaults(self):
        a = Activity(poi_name="故宫")
        assert a.category == "attraction"
        assert a.duration_min == 120
        assert a.time_constraint == "flexible"

    def test_time_constraint_values(self):
        a = Activity(poi_name="A", time_constraint="morning_only")
        assert a.time_constraint == "morning_only"


class TestDayPlan:
    """Test DayPlan model."""

    def test_defaults(self):
        d = DayPlan(day_number=1)
        assert d.activities == []
        assert d.total_cost == 0
        assert d.total_walking_steps == 0

    def test_with_activities(self):
        d = DayPlan(
            day_number=1,
            activities=[
                Activity(poi_name="A", ticket_price=100),
                Activity(poi_name="B", meal_cost=50),
            ],
        )
        assert len(d.activities) == 2


class TestTravelContext:
    """Test TravelContext model."""

    def test_defaults(self):
        ctx = TravelContext()
        assert ctx.route_suggestions == ""
        assert ctx.upcoming_events == []
        assert ctx.pitfall_tips == []

    def test_to_prompt_text_empty(self):
        ctx = TravelContext()
        text = ctx.to_prompt_text()
        assert text == "暂无当地实用信息"

    def test_to_prompt_text_with_content(self):
        ctx = TravelContext(
            seasonal_highlights="樱花季",
            upcoming_events=[{"name": "花展", "date_range": "5月1-7日", "location": "公园"}],
            pitfall_tips=["避开高峰期"],
        )
        text = ctx.to_prompt_text()
        assert "樱花季" in text
        assert "花展" in text
        assert "避开高峰期" in text

    def test_from_dict_none(self):
        ctx = TravelContext.from_dict(None)
        assert ctx.route_suggestions == ""

    def test_from_dict_with_data(self):
        ctx = TravelContext.from_dict({"route_suggestions": "建议路线"})
        assert ctx.route_suggestions == "建议路线"


class TestScoredPOI:
    """Test ScoredPOI model."""

    def test_defaults(self):
        p = ScoredPOI(name="故宫", category="attraction", score=0.9)
        assert p.description == ""
        assert p.tags == []
        assert p.time_constraint == "flexible"

    def test_with_location(self):
        p = ScoredPOI(
            name="故宫",
            category="attraction",
            score=0.9,
            location=Location(lat=39.9, lng=116.4),
        )
        assert p.location.lat == 39.9
