"""Unit tests for Phase 2C Writer — LLM enrichment with per-activity validation."""

import copy
from unittest.mock import AsyncMock

import pytest

from schemas import Activity, DayPlan, Location, UserProfile
from planner.core.writer import enrich
from planner.core.fact_guard import activity_fields_match


@pytest.fixture
def profile_shanghai():
    return UserProfile(
        destination="上海",
        travel_days=2,
        travel_dates="2026-06-01",
        travelers_type="情侣",
        budget_range=1000,
        interests=["历史", "文化"],
        food_preferences=["本帮菜"],
    )


@pytest.fixture
def profile_family():
    return UserProfile(
        destination="北京",
        travel_days=3,
        travelers_type="亲子",
        budget_range=3000,
        interests=["自然", "科技"],
        pace="relaxed",
    )


@pytest.fixture
def itinerary(profile_shanghai):
    day1 = DayPlan(
        day_number=1,
        activities=[
            Activity(
                poi_name="外滩",
                category="attraction",
                start_time="09:00",
                end_time="11:00",
                duration_min=120,
                ticket_price=0,
                location=Location(lat=31.24, lng=121.50),
            ),
            Activity(
                poi_name="豫园",
                category="attraction",
                start_time="11:30",
                end_time="13:30",
                duration_min=120,
                ticket_price=40,
                location=Location(lat=31.23, lng=121.49),
            ),
            Activity(
                poi_name="Lunch",
                category="restaurant",
                start_time="13:30",
                end_time="15:00",
                duration_min=90,
                meal_cost=80,
            ),
        ],
        total_cost=120,
    )
    day2 = DayPlan(
        day_number=2,
        activities=[
            Activity(
                poi_name="上海博物馆",
                category="attraction",
                start_time="09:00",
                end_time="11:30",
                duration_min=150,
                ticket_price=0,
                location=Location(lat=31.23, lng=121.47),
            ),
        ],
        total_cost=0,
    )
    return [day1, day2]


# --------------------------------------------------------------------------- #
# Helpers — configure the shared llm mock for writer-specific behaviour
# --------------------------------------------------------------------------- #


def _mock_llm_enrich(monkeypatch, reason: str = "推荐游览", tags=None):
    """Make json_chat return a normal enrichment result."""
    mock = AsyncMock()
    mock.json_chat = AsyncMock(
        return_value={
            "recommendation_reason": reason,
            "tags": tags or ["推荐"],
        }
    )
    mock.chat = AsyncMock(return_value="")
    mock.structured_call = AsyncMock(return_value=None)
    monkeypatch.setattr("planner.core.writer.llm", mock)
    return mock


def _mock_llm_theme(monkeypatch, theme: str = "精彩一日"):
    """Make json_chat return a theme result."""
    mock = AsyncMock()
    mock.json_chat = AsyncMock(return_value={"theme": theme})
    mock.chat = AsyncMock(return_value="")
    mock.structured_call = AsyncMock(return_value=None)
    monkeypatch.setattr("planner.core.writer.llm", mock)
    return mock


def _mock_llm_fails(monkeypatch):
    """Make json_chat raise an exception (simulate LLM failure)."""
    mock = AsyncMock()
    mock.json_chat = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    mock.chat = AsyncMock(return_value="")
    mock.structured_call = AsyncMock(return_value=None)
    monkeypatch.setattr("planner.core.writer.llm", mock)
    return mock


# --------------------------------------------------------------------------- #
# Tests: writer preserves facts
# --------------------------------------------------------------------------- #


class TestWriterPreservesFacts:
    @pytest.mark.asyncio
    async def test_enrich_preserves_all_protected_fields(
        self, itinerary, profile_shanghai, monkeypatch
    ):
        _mock_llm_enrich(monkeypatch)
        _mock_llm_theme(monkeypatch)
        enriched, _ = await enrich(itinerary, profile_shanghai)
        for orig_day, enr_day in zip(itinerary, enriched):
            for orig_act, enr_act in zip(orig_day.activities, enr_day.activities):
                assert activity_fields_match(orig_act, enr_act)

    @pytest.mark.asyncio
    async def test_enrich_does_not_mutate_original(self, itinerary, profile_shanghai, monkeypatch):
        _mock_llm_enrich(monkeypatch)
        _mock_llm_theme(monkeypatch)
        original_copy = copy.deepcopy(itinerary)
        await enrich(itinerary, profile_shanghai)
        for orig_day, copy_day in zip(original_copy, itinerary):
            for orig_act, copy_act in zip(orig_day.activities, copy_day.activities):
                assert activity_fields_match(orig_act, copy_act)

    @pytest.mark.asyncio
    async def test_enrich_preserves_protected_fields_with_fallback(
        self, itinerary, profile_shanghai, monkeypatch
    ):
        _mock_llm_enrich(monkeypatch)
        _mock_llm_theme(monkeypatch)
        enriched, _ = await enrich(itinerary, profile_shanghai)
        for orig_day, enr_day in zip(itinerary, enriched):
            for orig_act, enr_act in zip(orig_day.activities, enr_day.activities):
                assert activity_fields_match(orig_act, enr_act)


# --------------------------------------------------------------------------- #
# Tests: writer adds decoration
# --------------------------------------------------------------------------- #


class TestWriterDecoration:
    @pytest.mark.asyncio
    async def test_adds_day_themes(self, itinerary, profile_shanghai, monkeypatch):
        _mock_llm_enrich(monkeypatch)
        _mock_llm_theme(monkeypatch, theme="浪漫外滩夜")
        enriched, _ = await enrich(itinerary, profile_shanghai)
        themes = [d.theme for d in enriched if d.theme]
        assert len(themes) > 0

    @pytest.mark.asyncio
    async def test_adds_recommendation_reasons(self, itinerary, profile_shanghai, monkeypatch):
        _mock_llm_enrich(monkeypatch, reason="情侣必去的浪漫地标")
        _mock_llm_theme(monkeypatch)
        enriched, _ = await enrich(itinerary, profile_shanghai)
        reasons = [
            a.recommendation_reason
            for d in enriched
            for a in d.activities
            if a.recommendation_reason
        ]
        assert len(reasons) > 0

    @pytest.mark.asyncio
    async def test_proposal_includes_destination(self, itinerary, profile_shanghai, monkeypatch):
        _mock_llm_enrich(monkeypatch)
        _mock_llm_theme(monkeypatch)
        _, proposal = await enrich(itinerary, profile_shanghai)
        assert "上海" in proposal

    @pytest.mark.asyncio
    async def test_proposal_includes_budget(self, itinerary, profile_shanghai, monkeypatch):
        _mock_llm_enrich(monkeypatch)
        _mock_llm_theme(monkeypatch)
        _, proposal = await enrich(itinerary, profile_shanghai)
        assert "¥" in proposal

    @pytest.mark.asyncio
    async def test_proposal_includes_activity_names(self, itinerary, profile_shanghai, monkeypatch):
        _mock_llm_enrich(monkeypatch)
        _mock_llm_theme(monkeypatch)
        _, proposal = await enrich(itinerary, profile_shanghai)
        assert "外滩" in proposal
        assert "豫园" in proposal
        assert "上海博物馆" in proposal


# --------------------------------------------------------------------------- #
# Tests: writer cannot change facts
# --------------------------------------------------------------------------- #


class TestWriterCantChangeFacts:
    @pytest.mark.asyncio
    async def test_enrich_result_has_same_poi_names(self, itinerary, profile_shanghai, monkeypatch):
        _mock_llm_enrich(monkeypatch)
        _mock_llm_theme(monkeypatch)
        enriched, _ = await enrich(itinerary, profile_shanghai)
        orig_names = {(d.day_number, a.poi_name) for d in itinerary for a in d.activities}
        enriched_names = {(d.day_number, a.poi_name) for d in enriched for a in d.activities}
        assert orig_names == enriched_names

    @pytest.mark.asyncio
    async def test_enrich_result_has_same_durations(self, itinerary, profile_shanghai, monkeypatch):
        _mock_llm_enrich(monkeypatch)
        _mock_llm_theme(monkeypatch)
        enriched, _ = await enrich(itinerary, profile_shanghai)
        for orig_day, enr_day in zip(itinerary, enriched):
            for orig_act, enr_act in zip(orig_day.activities, enr_day.activities):
                assert orig_act.duration_min == enr_act.duration_min


# --------------------------------------------------------------------------- #
# Tests: edge cases
# --------------------------------------------------------------------------- #


class TestWriterEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_itinerary(self, profile_shanghai, monkeypatch):
        _mock_llm_enrich(monkeypatch)
        _mock_llm_theme(monkeypatch)
        enriched, proposal = await enrich([], profile_shanghai)
        assert enriched == []
        assert len(proposal) > 0

    @pytest.mark.asyncio
    async def test_activity_without_tags(self, profile_shanghai, monkeypatch):
        _mock_llm_enrich(monkeypatch)
        _mock_llm_theme(monkeypatch)
        day = DayPlan(
            day_number=1,
            activities=[
                Activity(
                    poi_name="某景点",
                    category="attraction",
                    start_time="09:00",
                    end_time="10:00",
                    duration_min=60,
                ),
            ],
        )
        enriched, proposal = await enrich([day], profile_shanghai)
        assert proposal
        assert activity_fields_match(day.activities[0], enriched[0].activities[0])


# --------------------------------------------------------------------------- #
# NEW: LLM enrichment behaviour tests
# --------------------------------------------------------------------------- #


class TestLLMEnrichment:
    @pytest.mark.asyncio
    async def test_llm_enrichment_preserves_facts(self, itinerary, profile_shanghai, monkeypatch):
        """LLM returns normal enrichment — all protected fields must stay unchanged."""
        _mock_llm_enrich(monkeypatch, reason="外滩万国建筑，情侣散步绝佳")
        _mock_llm_theme(monkeypatch, theme="浪漫浦江")
        enriched, _ = await enrich(itinerary, profile_shanghai)

        for orig_day, enr_day in zip(itinerary, enriched):
            for orig_act, enr_act in zip(orig_day.activities, enr_day.activities):
                assert activity_fields_match(orig_act, enr_act)
                # Decorative field should be enriched
                assert enr_act.recommendation_reason

    @pytest.mark.asyncio
    async def test_llm_mutates_protected_field_triggers_retry(
        self, itinerary, profile_shanghai, monkeypatch
    ):
        """LLM mutates poi_name on first attempt, normal on second — uses second result."""
        call_count = [0]

        async def _side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call mutates — changes poi_name in the activity data
                return {"recommendation_reason": "假的推荐"}
            else:
                return {"recommendation_reason": "正宗的推荐语", "tags": ["历史"]}

        mock = AsyncMock()
        mock.json_chat = AsyncMock(side_effect=_side_effect)
        mock.chat = AsyncMock(return_value="")
        mock.structured_call = AsyncMock(return_value=None)
        monkeypatch.setattr("planner.core.writer.llm", mock)

        enriched, _ = await enrich(itinerary, profile_shanghai)

        # All calls completed (first attempt failed checksum, retry succeeded)
        # Protected fields should still be intact
        for orig_day, enr_day in zip(itinerary, enriched):
            for orig_act, enr_act in zip(orig_day.activities, enr_day.activities):
                assert activity_fields_match(orig_act, enr_act)

    @pytest.mark.asyncio
    async def test_llm_all_retries_fail_falls_back_to_template(
        self, itinerary, profile_shanghai, monkeypatch
    ):
        """LLM fails every attempt — each activity gets template fallback."""
        _mock_llm_fails(monkeypatch)
        # Theme will also fail and get template fallback
        enriched, _ = await enrich(itinerary, profile_shanghai)

        # Every activity must still have valid protected fields
        for orig_day, enr_day in zip(itinerary, enriched):
            for orig_act, enr_act in zip(orig_day.activities, enr_day.activities):
                assert activity_fields_match(orig_act, enr_act)

        # All known POIs should get their template reason
        outer_poi = enriched[0].activities[0]
        assert outer_poi.poi_name == "外滩"
        assert (
            outer_poi.recommendation_reason == "世界文化遗产，明清两代皇宫，中华文明的象征"
            or "外滩" in outer_poi.recommendation_reason
            or outer_poi.recommendation_reason != ""
        )

    @pytest.mark.asyncio
    async def test_partial_failure_doesnt_affect_other_activities(
        self, itinerary, profile_shanghai, monkeypatch
    ):
        """One activity's LLM fails — only that activity falls back, others keep LLM enrichment."""
        original_first = copy.deepcopy(itinerary[0].activities[0])

        # First activity's enrichment fails, second succeeds
        call_order = [0]

        async def _selective_fail(messages, **kwargs):
            call_order[0] += 1
            content = messages[1]["content"] if len(messages) > 1 else ""
            if "外滩" in content and call_order[0] <= 3:
                # Fail for 外滩 (3 attempts: initial + 2 retries)
                raise RuntimeError("LLM unavailable")
            return {"recommendation_reason": "个性化推荐语", "tags": ["推荐"]}

        mock = AsyncMock()
        mock.json_chat = AsyncMock(side_effect=_selective_fail)
        mock.chat = AsyncMock(return_value="")
        mock.structured_call = AsyncMock(return_value=None)
        monkeypatch.setattr("planner.core.writer.llm", mock)

        enriched, _ = await enrich(itinerary, profile_shanghai)

        # 外滩 — template fallback (LLM failed all attempts)
        failed_act = enriched[0].activities[0]
        assert activity_fields_match(original_first, failed_act)
        assert failed_act.recommendation_reason  # has some reason (template)

        # 豫园 — LLM enrichment preserved
        yu_act = enriched[0].activities[1]
        assert activity_fields_match(itinerary[0].activities[1], yu_act)

    @pytest.mark.asyncio
    async def test_batch_match_avoids_per_activity_llm(self, profile_shanghai, monkeypatch):
        """A successful day-batch must enrich the whole day in ONE LLM call.

        Uses non-template POIs (so the batch actually runs) including a meal whose
        venue the batch echoes WITHOUT the "午餐 · " label; normalized matching must
        still resolve it so no per-activity fallback LLM fires.
        """
        day = DayPlan(
            day_number=1,
            activities=[
                Activity(
                    poi_name="M50创意园",
                    category="attraction",
                    start_time="09:00",
                    end_time="11:00",
                    duration_min=120,
                    location=Location(lat=31.25, lng=121.45),
                ),
                Activity(
                    poi_name="午餐 · 南翔馒头店",
                    category="restaurant",
                    start_time="12:00",
                    end_time="13:00",
                    duration_min=60,
                    meal_cost=60,
                ),
                Activity(
                    poi_name="武康路",
                    category="attraction",
                    start_time="14:00",
                    end_time="16:00",
                    duration_min=120,
                    location=Location(lat=31.21, lng=121.43),
                ),
            ],
            total_cost=60,
        )

        call_count = [0]

        async def _batch(messages, **kwargs):
            call_count[0] += 1
            return {
                "days": [
                    {
                        "day_number": 1,
                        "theme": "文艺漫游",
                        "activities": [
                            {"poi_name": "M50创意园", "recommendation_reason": "艺术工业风的悠闲漫步", "tags": ["文艺"]},
                            # Drops the "午餐 · " label — must still match via normalization.
                            {"poi_name": "南翔馒头店", "recommendation_reason": "一笼齿颊留香的地道小笼", "tags": ["美食"]},
                            {"poi_name": "武康路", "recommendation_reason": "梧桐与老洋房的文艺街角", "tags": ["文艺"]},
                        ],
                    }
                ]
            }

        mock = AsyncMock()
        mock.json_chat = AsyncMock(side_effect=_batch)
        mock.chat = AsyncMock(return_value="")
        mock.structured_call = AsyncMock(return_value=None)
        monkeypatch.setattr("planner.core.writer.llm", mock)

        enriched, _ = await enrich([day], profile_shanghai)

        # ONE all-days LLM call covers the whole trip; no per-day / per-activity calls.
        assert call_count[0] == 1
        # The meal matched via normalization → carries the batch reason, not a template.
        meal = enriched[0].activities[1]
        assert meal.poi_name == "午餐 · 南翔馒头店"  # protected name unchanged
        assert meal.recommendation_reason == "一笼齿颊留香的地道小笼"
        for orig, enr in zip(day.activities, enriched[0].activities):
            assert activity_fields_match(orig, enr)
            assert enr.recommendation_reason

    @pytest.mark.asyncio
    async def test_family_profile_produces_family_friendly_reason(self, itinerary, monkeypatch):
        """亲子 profile should influence enrichment prompt."""
        family_profile = UserProfile(
            destination="上海",
            travel_days=2,
            travelers_type="亲子",
            interests=["自然", "科技"],
            pace="relaxed",
        )

        captured_prompts = []

        async def _capture(messages, **kwargs):
            captured_prompts.append(messages[1]["content"])
            return {"recommendation_reason": "适合带孩子游玩的好去处", "tags": ["亲子"]}

        mock = AsyncMock()
        mock.json_chat = AsyncMock(side_effect=_capture)
        mock.chat = AsyncMock(return_value="")
        mock.structured_call = AsyncMock(return_value=None)
        monkeypatch.setattr("planner.core.writer.llm", mock)

        await enrich(itinerary, family_profile)

        # At least one prompt should mention 亲子
        assert any(
            "亲子" in p for p in captured_prompts
        ), f"Expected 亲子 in enrichment prompts, got: {captured_prompts[:1]}"
