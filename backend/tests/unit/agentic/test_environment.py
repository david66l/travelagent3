"""Tests for deterministic, isolated Agentic RL rollout environments."""

import json

import pytest

from agentic.environment import (
    EnvironmentSnapshot,
    EnvironmentTask,
    SnapshotToolResponse,
    SnapshotToolExecutor,
    create_rollout_group,
)
from agentic.loop import PolicyAction, PolicyContext
from agentic.trl_environment import (
    TRLClarificationEnvironment,
    TRLTradeoffEnvironment,
)
from agentic.legacy.environments import (  # archived pre-react modes
    TRLSearchEnvironment,
    TRLTravelEnvironment,
)


class FirstAllowedPolicy:
    async def propose(self, context: PolicyContext) -> PolicyAction:
        return PolicyAction(action=context.allowed_actions[0])


def _task() -> EnvironmentTask:
    return EnvironmentTask(
        task_id="shanghai-one-day",
        template_family="normal-city-trip",
        difficulty="L1",
        seed=42,
        user_request="Plan one day in Shanghai",
        slots={"destination": "Shanghai", "travel_days": 1},
    )


def _snapshot() -> EnvironmentSnapshot:
    poi = {
        "name": "Museum",
        "category": "attraction",
        "score": 0.9,
        "location": {"lat": 31.23, "lng": 121.47},
        "ticket_price": 0,
        "open_time": "08:00",
        "close_time": "18:00",
    }
    return EnvironmentSnapshot(
        environment_version="travel-env-test-v1",
        snapshot_version="snapshot-2026-08-12-v1",
        state_id="state-shanghai-1",
        tool_responses={
            "get_weather": [
                {
                    "data": [{"date": "2026-08-12", "condition": "sunny"}],
                    "expected_arguments": {"city": "Shanghai"},
                }
            ],
            "search_pois": [
                {
                    "data": [poi],
                    "expected_arguments": {"city": "Shanghai"},
                }
            ],
            "get_poi_detail": [{"data": poi}],
            "get_route_matrix": [
                {
                    "data": {
                        "poi_ids": ["__hotel", "Museum"],
                        "time_minutes": [[0, 10], [10, 0]],
                        "transport_cost": [[0.0, 3.0], [3.0, 0.0]],
                    }
                }
            ],
            "solve_itinerary": [
                {
                    "data": {
                        "status": "optimal",
                        "days": [
                            {
                                "day_number": 1,
                                "activities": [
                                    {
                                        "poi_id": "Museum",
                                        "poi_name": "Museum",
                                        "category": "attraction",
                                        "start_time": "09:00",
                                        "end_time": "10:00",
                                        "duration_min": 60,
                                    }
                                ],
                                "total_cost": 0,
                                "transport_cost": 0,
                            }
                        ],
                        "solve_time_ms": 4,
                    }
                }
            ],
            "validate_itinerary": [
                {
                    "data": {
                        "validator_version": "travel-validator.v1",
                        "hard_pass": True,
                        "hard_violations": [],
                        "soft_scores": {"route_efficiency": 1.0},
                        "metrics": {"budget_error_rate": 0},
                    }
                }
            ],
        },
        hidden_test_facts={"closed_pois": []},
    )



async def test_group_members_share_fingerprint_but_not_tool_counters():
    environments = create_rollout_group(_task(), _snapshot(), 2)

    first = await environments[0].rollout(FirstAllowedPolicy())
    second = await environments[1].rollout(FirstAllowedPolicy())

    assert first.initial_state_fingerprint == second.initial_state_fingerprint
    assert first.episode.trajectory_id != second.episode.trajectory_id
    assert first.tool_call_counts == second.tool_call_counts
    assert first.reward.episode_reward == second.reward.episode_reward


async def test_snapshot_fault_sequence_is_local_to_executor():
    snapshot = _snapshot()
    snapshot.tool_responses["get_weather"][0].data = None
    snapshot.tool_responses["get_weather"][0].data_source = "unavailable"
    snapshot.tool_responses["get_weather"][0].error_code = "UPSTREAM_TIMEOUT"
    snapshot.tool_responses["get_weather"][0].fallback_reason = "timeout"
    snapshot.tool_responses["get_weather"][0].retryable = True
    first = SnapshotToolExecutor(snapshot)
    second = SnapshotToolExecutor(snapshot)
    call = {
        "id": "weather-1",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"city":"Shanghai"}',
        },
    }

    first_record = (await first.execute([call], {"allowed_tools": {"get_weather"}}))[0]
    second_record = (await second.execute([call], {"allowed_tools": {"get_weather"}}))[0]

    assert first_record["observation"]["error"]["code"] == "UPSTREAM_TIMEOUT"
    assert second_record["observation"]["error"]["code"] == "UPSTREAM_TIMEOUT"
    assert first.call_counts == second.call_counts == {"get_weather": 1}


async def test_snapshot_contract_selects_response_by_arguments_not_position():
    snapshot = _snapshot()
    snapshot.tool_responses["get_poi_detail"] = [
        SnapshotToolResponse(
            data={"name": "Restaurant"},
            expected_arguments={"poi_name": "Restaurant", "city": "Shanghai"},
        ),
        SnapshotToolResponse(
            data={"name": "Museum"},
            expected_arguments={"poi_name": "Museum", "city": "Shanghai"},
        ),
    ]
    executor = SnapshotToolExecutor(snapshot)
    call = {
        "id": "detail-1",
        "type": "function",
        "function": {
            "name": "get_poi_detail",
            "arguments": json.dumps({"poi_name": "Museum", "city": "Shanghai"}),
        },
    }

    record = (await executor.execute([call], {"allowed_tools": {"get_poi_detail"}}))[0]

    assert record["observation"]["ok"] is True
    assert record["observation"]["data"]["name"] == "Museum"


async def test_context_tolerant_keyword_contract_ignores_only_out_of_contract_context():
    snapshot = _snapshot()
    snapshot.tool_responses["search_pois"] = [
        SnapshotToolResponse(
            data=None,
            data_source="unavailable",
            expected_arguments={"keywords": ["历史", "博物馆"]},
            argument_match_mode="context_tolerant_keywords",
            ignored_keyword_values=["上海", "2026-08-12"],
            error_code="QUERY_TOO_BROAD",
            retryable=True,
        ),
        SnapshotToolResponse(
            data=[{"name": "Museum"}],
            expected_arguments={"keywords": ["历史"]},
            argument_match_mode="context_tolerant_keywords",
            ignored_keyword_values=["上海", "2026-08-12"],
        ),
    ]
    executor = SnapshotToolExecutor(snapshot)
    unexpected_executor = SnapshotToolExecutor(snapshot)

    unexpected = await unexpected_executor.execute(
        [
            {
                "id": "search-unexpected",
                "type": "function",
                "function": {
                    "name": "search_pois",
                    "arguments": json.dumps(
                        {
                            "city": "Shanghai",
                            "keywords": ["历史", "博物馆", "购物"],
                        }
                    ),
                },
            }
        ],
        {"allowed_tools": {"search_pois"}},
    )

    broad = await executor.execute(
        [
            {
                "id": "search-1",
                "type": "function",
                "function": {
                    "name": "search_pois",
                    "arguments": json.dumps(
                        {
                            "city": "Shanghai",
                            "keywords": ["历史", "博物馆", "上海", "2026-08-12"],
                        }
                    ),
                },
            }
        ],
        {"allowed_tools": {"search_pois"}},
    )
    narrowed = await executor.execute(
        [
            {
                "id": "search-2",
                "type": "function",
                "function": {
                    "name": "search_pois",
                    "arguments": json.dumps({"city": "Shanghai", "keywords": ["历史", "上海"]}),
                },
            }
        ],
        {"allowed_tools": {"search_pois"}},
    )

    assert broad[0]["observation"]["error"]["code"] == "QUERY_TOO_BROAD"
    assert narrowed[0]["observation"]["ok"] is True
    assert unexpected[0]["observation"]["error"]["code"] == ("SNAPSHOT_ARGUMENT_MISMATCH")




def test_trl_policy_driven_environment_rejects_teacher_trajectory_prefix():
    environment = TRLTravelEnvironment()

    with pytest.raises(ValueError, match="trajectory prefixes are forbidden"):
        environment.reset(
            task=_task().model_dump(mode="json"),
            snapshot=_snapshot().model_dump(mode="json"),
            prompt=[
                {"role": "system", "content": "policy"},
                {"role": "user", "content": _task().user_request},
                {"role": "assistant", "content": "teacher action"},
            ],
        )



def test_trl_environments_expose_only_state_specific_policy_tools():
    import inspect

    def tools(environment):
        return {
            name
            for name, member in inspect.getmembers(environment, predicate=inspect.ismethod)
            if name not in {"reset", "get_reward"} and not name.startswith("_")
        }

    assert tools(TRLSearchEnvironment()) == {"search_pois"}
    assert tools(TRLClarificationEnvironment()) == {"ask_user"}
    assert tools(TRLTradeoffEnvironment()) == {"abort", "propose_tradeoff"}
    assert tools(TRLTravelEnvironment()) == {
        "accept_candidates",
        "accept_itinerary",
        "abort",
        "ask_user",
        "capability_check",
        "compose_draft",
        "finish",
        "finalize_research",
        "get_poi_detail",
        "get_route_matrix",
        "get_weather",
        "propose_tradeoff",
        "retrieve_city_knowledge",
        "retry_solve",
        "search_current_info",
        "search_pois",
        "search_transport",
        "solve_itinerary",
        "validate_itinerary",
    }
