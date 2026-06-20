"""Unit tests for the log analytics engine."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from monitoring.log_analytics import LogAnalyticsEngine


def _mock_execute_return(db_session, rows):
    """Configure db_session.execute to return rows from scalars().all()."""
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = rows
    db_session.execute = AsyncMock(return_value=result_mock)


@pytest.mark.asyncio
async def test_cluster_planning_failures_groups_by_error_and_city(db_session):
    job1 = MagicMock(error_message="LLM timeout", input_requirements={"city": "北京"})
    job2 = MagicMock(error_message="LLM timeout", input_requirements={"city": "北京"})
    job3 = MagicMock(error_message="Parser error", input_requirements={"city": "上海"})
    _mock_execute_return(db_session, [job1, job2, job3])

    result = await LogAnalyticsEngine.cluster_planning_failures(db_session, hours=24)

    assert len(result) == 2
    assert result[0] == {"error_message": "LLM timeout", "city": "北京", "count": 2}
    assert result[1] == {"error_message": "Parser error", "city": "上海", "count": 1}


@pytest.mark.asyncio
async def test_cluster_planning_failures_falls_back_to_unknown_city(db_session):
    job = MagicMock(error_message="Network error", input_requirements={})
    _mock_execute_return(db_session, [job])

    result = await LogAnalyticsEngine.cluster_planning_failures(db_session)

    assert result == [{"error_message": "Network error", "city": "unknown", "count": 1}]


@pytest.mark.asyncio
async def test_cluster_planning_failures_empty(db_session):
    _mock_execute_return(db_session, [])

    result = await LogAnalyticsEngine.cluster_planning_failures(db_session)

    assert result == []


@pytest.mark.asyncio
async def test_cluster_planning_failures_none_session():
    result = await LogAnalyticsEngine.cluster_planning_failures(None)

    assert result == []


@pytest.mark.asyncio
async def test_cluster_planning_failures_query_error(db_session):
    db_session.execute = AsyncMock(side_effect=RuntimeError("db down"))

    result = await LogAnalyticsEngine.cluster_planning_failures(db_session)

    assert result == []


@pytest.mark.asyncio
async def test_top_modification_intents_groups_by_payload_intent(db_session):
    log1 = MagicMock(payload={"intent": "add_poi"}, action_type="add")
    log2 = MagicMock(payload={"intent": "add_poi"}, action_type="add")
    log3 = MagicMock(payload={"intent": "remove_poi"}, action_type="remove")
    _mock_execute_return(db_session, [log1, log2, log3])

    result = await LogAnalyticsEngine.top_modification_intents(db_session, hours=24, limit=10)

    assert len(result) == 2
    assert result[0] == {"intent": "add_poi", "count": 2}
    assert result[1] == {"intent": "remove_poi", "count": 1}


@pytest.mark.asyncio
async def test_top_modification_intents_falls_back_to_action_type(db_session):
    log = MagicMock(payload={}, action_type="reorder")
    _mock_execute_return(db_session, [log])

    result = await LogAnalyticsEngine.top_modification_intents(db_session)

    assert result == [{"intent": "reorder", "count": 1}]


@pytest.mark.asyncio
async def test_top_modification_intents_empty(db_session):
    _mock_execute_return(db_session, [])

    result = await LogAnalyticsEngine.top_modification_intents(db_session)

    assert result == []


@pytest.mark.asyncio
async def test_top_modification_intents_none_session():
    result = await LogAnalyticsEngine.top_modification_intents(None)

    assert result == []


@pytest.mark.asyncio
async def test_top_modification_intents_query_error(db_session):
    db_session.execute = AsyncMock(side_effect=RuntimeError("db down"))

    result = await LogAnalyticsEngine.top_modification_intents(db_session)

    assert result == []


@pytest.mark.asyncio
async def test_destination_satisfaction_ranking_returns_scores(db_session):
    it1 = MagicMock(destination="北京")
    it2 = MagicMock(destination="北京")
    it3 = MagicMock(destination="上海")
    _mock_execute_return(db_session, [it1, it2, it3])

    result = await LogAnalyticsEngine.destination_satisfaction_ranking(
        db_session, days=30, limit=10
    )

    assert len(result) == 2
    beijing = next(item for item in result if item["destination"] == "北京")
    shanghai = next(item for item in result if item["destination"] == "上海")
    assert beijing == {"destination": "北京", "positive_count": 2, "total_count": 2, "score": 1.0}
    assert shanghai == {"destination": "上海", "positive_count": 1, "total_count": 1, "score": 1.0}


@pytest.mark.asyncio
async def test_destination_satisfaction_ranking_empty(db_session):
    _mock_execute_return(db_session, [])

    result = await LogAnalyticsEngine.destination_satisfaction_ranking(db_session)

    assert result == []


@pytest.mark.asyncio
async def test_destination_satisfaction_ranking_none_session():
    result = await LogAnalyticsEngine.destination_satisfaction_ranking(None)

    assert result == []


@pytest.mark.asyncio
async def test_destination_satisfaction_ranking_query_error(db_session):
    db_session.execute = AsyncMock(side_effect=RuntimeError("db down"))

    result = await LogAnalyticsEngine.destination_satisfaction_ranking(db_session)

    assert result == []


def test_generate_iteration_suggestions_for_top_failures():
    failures = [
        {"error_message": "LLM timeout", "city": "北京", "count": 5},
        {"error_message": "Parser error", "city": "上海", "count": 3},
        {"error_message": "Network error", "city": "广州", "count": 2},
        {"error_message": "Ignored", "city": "深圳", "count": 1},
    ]

    suggestions = LogAnalyticsEngine.generate_iteration_suggestions(failures)

    assert len(suggestions) == 3
    assert "北京" in suggestions[0] and "LLM timeout" in suggestions[0]
    assert "上海" in suggestions[1] and "Parser error" in suggestions[1]
    assert "广州" in suggestions[2] and "Network error" in suggestions[2]


def test_generate_iteration_suggestions_empty():
    assert LogAnalyticsEngine.generate_iteration_suggestions([]) == []


@pytest.mark.asyncio
async def test_analyze_returns_all_sections(db_session):
    _mock_execute_return(db_session, [])

    result = await LogAnalyticsEngine.analyze(db_session)

    assert set(result.keys()) == {
        "planning_failures",
        "modification_intents",
        "destination_ranking",
        "iteration_suggestions",
    }
    assert result["planning_failures"] == []
    assert result["modification_intents"] == []
    assert result["destination_ranking"] == []
    assert result["iteration_suggestions"] == []


@pytest.mark.asyncio
async def test_analyze_links_suggestions_to_failures(db_session):
    job = MagicMock(error_message="LLM timeout", input_requirements={"city": "北京"})
    _mock_execute_return(db_session, [job])

    result = await LogAnalyticsEngine.analyze(db_session)

    assert result["planning_failures"] == [
        {"error_message": "LLM timeout", "city": "北京", "count": 1}
    ]
    assert len(result["iteration_suggestions"]) == 1
    assert "北京" in result["iteration_suggestions"][0]
