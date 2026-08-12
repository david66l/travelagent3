"""Tests for deterministic, isolated Agentic RL rollout environments."""

import json

from agentic.environment import (
    EnvironmentSnapshot,
    EnvironmentTask,
    SnapshotToolExecutor,
    TravelAgentEnvironment,
    create_rollout_group,
)
from agentic.loop import PolicyAction, PolicyContext
from agentic.trl_environment import TRLTravelEnvironment


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


async def test_snapshot_rollout_produces_replayable_rewarded_episode():
    rollout = await TravelAgentEnvironment(_task(), _snapshot()).rollout(FirstAllowedPolicy())

    assert rollout.episode.status == "interrupted"
    assert rollout.episode.termination_reason == "awaiting_user"
    assert rollout.reward.gate_status == "passed"
    assert rollout.reward.episode_reward > 0
    assert rollout.tool_call_counts == {
        "get_weather": 1,
        "search_pois": 1,
        "get_poi_detail": 1,
        "get_route_matrix": 1,
        "solve_itinerary": 1,
        "validate_itinerary": 1,
    }
    observations = [item for step in rollout.episode.steps for item in step.observations]
    assert all(item.snapshot_version == "snapshot-2026-08-12-v1" for item in observations)


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


async def test_trl_environment_runs_production_loop_and_six_component_reward():
    environment = TRLTravelEnvironment()

    initial = json.loads(
        environment.reset(
            task=_task().model_dump(mode="json"),
            snapshot=_snapshot().model_dump(mode="json"),
        )
    )
    assert initial["policy_state"]["allowed_actions"] == [
        "capability_check",
        "ask_user",
        "propose_tradeoff",
        "abort",
    ]

    await environment.capability_check()
    await environment.get_weather()
    await environment.search_pois()
    await environment.get_poi_detail()
    await environment.get_route_matrix()
    await environment.solve_itinerary()
    await environment.validate_itinerary()
    await environment.compose_draft()
    terminal = json.loads(await environment.finish())
    reward = await environment.get_reward()

    assert terminal["done"] is True
    assert terminal["termination_reason"] == "awaiting_user"
    assert reward > 0
    assert environment.reward_record is not None
    assert environment.reward_record.gate_status == "passed"
    assert set(environment.reward_record.components.model_dump()) == {
        "task",
        "constraint",
        "format",
        "tool",
        "grounding",
        "efficiency",
        "quality",
    }


async def test_trl_environment_rejects_out_of_state_action_without_state_write():
    environment = TRLTravelEnvironment()
    environment.reset(
        task=_task().model_dump(mode="json"),
        snapshot=_snapshot().model_dump(mode="json"),
    )

    transition = json.loads(await environment.finish())

    assert transition["done"] is False
    assert transition["last_transition"]["verification"]["error_code"] == "ACTION_NOT_ALLOWED"
    assert "capability_check" in transition["policy_state"]["allowed_actions"]
