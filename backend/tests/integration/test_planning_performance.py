"""Performance benchmarks for Planning Core (Phase 2A + 2B).

Each scenario prints a machine-readable JSON record with stage timings.
Marked ``@pytest.mark.performance`` — not run in standard CI.
"""

import asyncio
import json
import time

import pytest
from unittest.mock import AsyncMock, patch

from models.planning_job import PlanningJob
from repositories.planning_job import PlanningJobRepository
from pipeline.planning_pipeline import PlanningPipeline
from worker.planning_worker import PlanningWorker
from schemas import IntentResult, ScoredPOI, WeatherDay


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


async def _run_scenario(
    db,
    *,
    scenario: str,
    city: str,
    days: int,
    intent_entities: dict | None = None,
    mock_pois,
    mock_weather,
    extra_patches: list | None = None,
) -> dict:
    """Run a single scenario through the full pipeline and return timing JSON."""
    repo = PlanningJobRepository(db)
    job = await repo.create(
        session_id=f"perf-{scenario}",
        user_id="user-1",
        user_input=f"{city}{days}天",
    )
    await db.commit()

    mock_intent = AsyncMock(return_value=IntentResult(
        intent="generate_itinerary",
        confidence=0.95,
        user_entities=intent_entities or {
            "destination": city,
            "travel_days": days,
            "travel_dates": "2026-06-01",
        },
    ))

    patches = [
        patch("agents.intent_recognition.IntentRecognitionAgent.recognize", mock_intent),
        patch("agents.realtime_query.RealtimeQueryAgent.query_pois", mock_pois),
        patch("agents.realtime_query.RealtimeQueryAgent.query_weather", mock_weather),
    ]
    if extra_patches:
        patches.extend(extra_patches)

    import contextlib
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)

        worker = PlanningWorker("perf-worker")
        acquired = await repo.acquire_job("perf-worker", lease_seconds=60)
        await db.commit()

        pipeline = PlanningPipeline(worker=worker)
        t0 = time.perf_counter()
        await pipeline.run(acquired)
        total_s = round(time.perf_counter() - t0, 3)

    # Re-query via a fresh session to see the pipeline's writes.
    # The db fixture session might not see cross-session updates without a
    # fresh query — expire + execute gives us the latest committed state.
    from sqlalchemy import select
    from core.database import async_session_maker as session_factory
    async with session_factory() as fresh_db:
        result = await fresh_db.execute(
            select(PlanningJob).where(PlanningJob.id == job.id)
        )
        updated = result.scalar_one()

    return {
        "scenario": scenario,
        "status": updated.status,
        "ttfi_seconds": updated.stage_timings.get("draft_ready", -1),
        "total_seconds": total_s,
        "stage_timings": updated.stage_timings or {},
        "fallback_used": (
            updated.itinerary_draft[-1].get("fallback_used", False)
            if isinstance(updated.itinerary_draft, list) and updated.itinerary_draft
            else False
        ),
        "needs_human": (
            updated.proposal_text and "needs_human" in str(updated.itinerary_final)
        ),
    }


# --------------------------------------------------------------------------- #
# scenario 1 — hot cache / built-in city
# --------------------------------------------------------------------------- #

@pytest.mark.performance
@pytest.mark.asyncio
async def test_perf_builtin_beijing_3days(db):
    """北京 3 天 — all external APIs return instantly.  Target TTFI < 5s."""

    # Use a decent set of POIs — Beijing defaults subset
    pois = [
        ScoredPOI(name="故宫", category="attraction", score=0.95,
                  area="东城区", recommended_hours="半天",
                  ticket_price=60, tags=["历史", "文化", "皇家"]),
        ScoredPOI(name="天坛", category="attraction", score=0.90,
                  area="东城区", recommended_hours="2-3小时",
                  ticket_price=34, tags=["历史", "文化"]),
        ScoredPOI(name="颐和园", category="attraction", score=0.88,
                  area="海淀区", recommended_hours="半天",
                  ticket_price=30, tags=["园林", "湖景"]),
        ScoredPOI(name="长城", category="attraction", score=0.92,
                  area="延庆区", recommended_hours="半天",
                  ticket_price=40, tags=["历史", "登山"]),
        ScoredPOI(name="南锣鼓巷", category="attraction", score=0.75,
                  area="东城区", recommended_hours="2小时",
                  ticket_price=0, tags=["胡同", "美食", "文艺"]),
        ScoredPOI(name="798艺术区", category="attraction", score=0.80,
                  area="朝阳区", recommended_hours="2-3小时",
                  ticket_price=0, tags=["艺术", "拍照"]),
        ScoredPOI(name="鸟巢", category="attraction", score=0.82,
                  area="朝阳区", recommended_hours="2小时",
                  ticket_price=50, tags=["建筑", "奥运"]),
        ScoredPOI(name="雍和宫", category="attraction", score=0.78,
                  area="东城区", recommended_hours="1-2小时",
                  ticket_price=25, tags=["宗教", "历史"]),
    ]
    mock_pois = AsyncMock(return_value=pois)
    mock_weather = AsyncMock(return_value=[
        WeatherDay(date="2026-06-01", condition="晴", temp_high=30, temp_low=20, precipitation_chance=0),
        WeatherDay(date="2026-06-02", condition="多云", temp_high=28, temp_low=21, precipitation_chance=10),
        WeatherDay(date="2026-06-03", condition="晴", temp_high=31, temp_low=22, precipitation_chance=0),
    ])

    rec = await _run_scenario(
        db,
        scenario="beijing_builtin_3d",
        city="北京",
        days=3,
        mock_pois=mock_pois,
        mock_weather=mock_weather,
    )
    print("PERF_JSON:", json.dumps(rec, ensure_ascii=False))

    assert rec["status"] == "completed"
    assert rec["ttfi_seconds"] > 0


# --------------------------------------------------------------------------- #
# scenario 2 — fallback on slow POI
# --------------------------------------------------------------------------- #

@pytest.mark.performance
@pytest.mark.asyncio
async def test_perf_fallback_slow_poi(db):
    """POI search exceeds 3s timeout → fallback draft must emit quickly."""

    async def _slow_pois(*a, **k):
        await asyncio.sleep(10)  # will trip the 3s timeout
    mock_pois = AsyncMock(side_effect=_slow_pois)
    mock_weather = AsyncMock(return_value=[])

    rec = await _run_scenario(
        db,
        scenario="fallback_slow_poi",
        city="上海",
        days=3,
        mock_pois=mock_pois,
        mock_weather=mock_weather,
    )
    print("PERF_JSON:", json.dumps(rec, ensure_ascii=False))

    assert rec["status"] == "completed"


# --------------------------------------------------------------------------- #
# scenario 3 — repair cost
# --------------------------------------------------------------------------- #

@pytest.mark.performance
@pytest.mark.asyncio
async def test_perf_repair_cost(db):
    """Inject hard-violation draft and measure repair loop time.

    Use 2 days so that move-to-next-day can resolve time overflows.
    """
    from planner.core import run_repair_loop, validate as validate_itinerary
    from planner.core.heuristic_strategy import build_strategy
    from schemas import DayPlan, Activity, UserProfile, Location

    profile = UserProfile(
        destination="上海",
        travel_days=2,
        budget_range=500,
        interests=["历史"],
    )
    pois = [
        ScoredPOI(name="外滩", category="attraction", score=0.9,
                  area="外滩", recommended_hours="2小时", ticket_price=0,
                  tags=["观光"], location=Location(lat=31.24, lng=121.49)),
        ScoredPOI(name="豫园", category="attraction", score=0.85,
                  area="城隍庙", recommended_hours="2小时", ticket_price=40,
                  tags=["历史"], location=Location(lat=31.23, lng=121.49)),
        ScoredPOI(name="上海博物馆", category="attraction", score=0.92,
                  area="人民广场", recommended_hours="2-3小时", ticket_price=0,
                  tags=["历史", "文化"], location=Location(lat=31.23, lng=121.47)),
    ]

    strategy = build_strategy(pois, profile)
    # Broken itinerary: overlap on day 1 + missing must-see + day 2 empty
    day1 = DayPlan(
        day_number=1,
        activities=[
            Activity(
                poi_name="外滩", category="attraction",
                start_time="09:00", end_time="11:00", duration_min=120,
                ticket_price=0,
            ),
            Activity(
                poi_name="豫园", category="attraction",
                start_time="10:00", end_time="12:00", duration_min=120,
                ticket_price=40,
            ),
        ],
    )
    day2 = DayPlan(day_number=2, activities=[])
    itinerary = [day1, day2]

    report = validate_itinerary(itinerary, profile, ["上海博物馆"])
    assert not report.passed, "Expected hard violations"

    t0 = time.perf_counter()
    result = run_repair_loop(
        itinerary, profile, strategy.must_see, pois, max_iterations=5,
    )
    repair_s = round(time.perf_counter() - t0, 3)

    rec = {
        "scenario": "repair_cost",
        "repair_loop_seconds": repair_s,
        "repair_success": result.success,
        "repairs_applied": len(result.applied_plans),
        "needs_human": result.needs_human,
    }
    print("PERF_JSON:", json.dumps(rec, ensure_ascii=False))

    assert repair_s < 1.0, f"Repair loop took {repair_s}s, target < 1s"
    assert result.success


# --------------------------------------------------------------------------- #
# scenario 4 — full-job event sequence
# --------------------------------------------------------------------------- #

@pytest.mark.performance
@pytest.mark.asyncio
async def test_perf_full_job_event_sequence(db):
    """Full pipeline event order and stage timing (mocked LLM/API)."""

    pois = [
        ScoredPOI(name="故宫", category="attraction", score=0.95,
                  area="东城区", recommended_hours="半天",
                  ticket_price=60, tags=["历史", "文化"]),
        ScoredPOI(name="天坛", category="attraction", score=0.90,
                  area="东城区", recommended_hours="2-3小时",
                  ticket_price=34, tags=["历史"]),
        ScoredPOI(name="全聚德", category="restaurant", score=0.80,
                  area="前门", recommended_hours="1.5小时",
                  ticket_price=0, tags=["烤鸭"]),
        ScoredPOI(name="798艺术区", category="attraction", score=0.80,
                  area="朝阳区", recommended_hours="2-3小时",
                  ticket_price=0, tags=["艺术", "拍照"]),
    ]
    mock_pois = AsyncMock(return_value=pois)
    mock_weather = AsyncMock(return_value=[
        WeatherDay(date="2026-06-01", condition="晴", temp_high=30, temp_low=20, precipitation_chance=0),
        WeatherDay(date="2026-06-02", condition="多云", temp_high=28, temp_low=21, precipitation_chance=10),
    ])

    rec = await _run_scenario(
        db,
        scenario="normal_full_job",
        city="北京",
        days=2,
        mock_pois=mock_pois,
        mock_weather=mock_weather,
    )
    print("PERF_JSON:", json.dumps(rec, ensure_ascii=False))

    assert rec["status"] == "completed"
    expected_stages = ["running", "draft_ready", "itinerary_final", "writing", "completed"]
    timings = rec["stage_timings"]
    for stage in expected_stages:
        assert stage in timings, f"Missing stage: {stage}"
    # Stages must be monotonic
    vals = [timings[s] for s in expected_stages]
    assert vals == sorted(vals), f"Non-monotonic timings: {timings}"
