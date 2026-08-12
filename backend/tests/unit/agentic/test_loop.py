"""End-to-end tests for the bounded Agent Loop kernel."""

from typing import Any

import pytest

from agentic.loop import ActionOutcome, BoundedAgentLoop, PolicyAction, PolicyContext
from agentic.state import (
    AgentLedgerState,
    ArtifactRecord,
    BudgetLedger,
    GoalLedger,
    TaskGraph,
    TaskNode,
)
from agentic.trajectory import EpisodeRecorder, EpisodeReplayVerifier


class ScriptedPolicy:
    def __init__(self, actions: dict[str, str]) -> None:
        self.actions = actions

    async def propose(self, context: PolicyContext) -> PolicyAction:
        task_id = context.current_subtask["task_id"]
        return PolicyAction(action=self.actions[task_id])


class ArtifactExecutor:
    def __init__(self, payloads: dict[str, tuple[str, dict[str, Any]]]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []

    async def execute(self, *, task, action, ledger) -> ActionOutcome:
        self.calls.append(action.action)
        artifact_type, payload = self.payloads[task.task_id]
        return ActionOutcome(
            artifacts=[
                ArtifactRecord(
                    artifact_id=f"artifact-{task.task_id}",
                    artifact_type=artifact_type,
                    payload=payload,
                    evidence_refs=[action.action_id],
                    goal_version=ledger.goal.goal_version,
                    plan_version=ledger.task_graph.plan_version,
                )
            ]
        )


def _ledger(*, max_steps: int = 4) -> AgentLedgerState:
    return AgentLedgerState(
        goal=GoalLedger(original_request="Plan a trip"),
        task_graph=TaskGraph(
            goal_version=1,
            tasks=(
                TaskNode(
                    task_id="solve",
                    goal="solve",
                    allowed_actions=("solve_itinerary",),
                    success_criteria={"required_artifact_types": ["solver_result"]},
                ),
                TaskNode(
                    task_id="validate",
                    goal="validate",
                    depends_on=("solve",),
                    allowed_actions=("validate_itinerary",),
                    success_criteria={
                        "required_artifact_types": ["validation_report"],
                        "require_hard_pass": True,
                    },
                ),
            ),
        ),
        budget=BudgetLedger(max_episode_steps=max_steps),
    )


@pytest.mark.asyncio
async def test_loop_finishes_only_after_verified_task_closure():
    ledger = _ledger()
    policy = ScriptedPolicy({"solve": "solve_itinerary", "validate": "validate_itinerary"})
    executor = ArtifactExecutor(
        {
            "solve": ("solver_result", {"itinerary": []}),
            "validate": (
                "validation_report",
                {
                    "hard_pass": True,
                    "hard_violations": [],
                    "validator_version": "travel-validator.v1",
                },
            ),
        }
    )

    result = await BoundedAgentLoop().run(ledger, policy=policy, executor=executor)
    assert result.status == "finished"
    assert result.termination_reason == "validated_finish"
    assert all(task.status == "succeeded" for task in result.ledger.task_graph.tasks)
    assert [event.event_type for event in result.events].count("task_verified") == 2


@pytest.mark.asyncio
async def test_loop_rejects_unverified_executor_success():
    ledger = AgentLedgerState(
        goal=GoalLedger(original_request="Plan a trip"),
        task_graph=TaskGraph(
            goal_version=1,
            tasks=(
                TaskNode(
                    task_id="solve",
                    goal="solve",
                    allowed_actions=("solve_itinerary",),
                    success_criteria={"required_artifact_types": ["solver_result"]},
                    max_attempts=1,
                ),
            ),
        ),
    )
    policy = ScriptedPolicy({"solve": "solve_itinerary"})
    executor = ArtifactExecutor({"solve": ("wrong_artifact", {})})

    result = await BoundedAgentLoop().run(ledger, policy=policy, executor=executor)
    assert result.status == "failed"
    assert result.ledger.task_graph.get("solve").status == "failed"
    assert result.ledger.failures[0].code == "SUBTASK_VERIFICATION_FAILED"


@pytest.mark.asyncio
async def test_loop_rejects_policy_action_outside_task_allowlist():
    ledger = _ledger()
    ledger.task_graph = ledger.task_graph.model_copy(
        update={"tasks": (ledger.task_graph.get("solve").model_copy(update={"max_attempts": 1}),)}
    )
    policy = ScriptedPolicy({"solve": "get_weather"})
    executor = ArtifactExecutor({})

    result = await BoundedAgentLoop().run(ledger, policy=policy, executor=executor)
    assert result.status == "failed"
    assert executor.calls == []
    assert result.ledger.failures[0].code == "ACTION_NOT_ALLOWED"


@pytest.mark.asyncio
async def test_loop_stops_before_exceeding_durable_budget():
    ledger = _ledger(max_steps=1)
    policy = ScriptedPolicy({"solve": "solve_itinerary", "validate": "validate_itinerary"})
    executor = ArtifactExecutor(
        {
            "solve": ("solver_result", {}),
            "validate": ("validation_report", {"hard_pass": True}),
        }
    )

    result = await BoundedAgentLoop().run(ledger, policy=policy, executor=executor)
    assert result.status == "failed"
    assert result.termination_reason == "budget_exhausted_fallback"
    assert executor.calls == ["solve_itinerary"]


@pytest.mark.asyncio
async def test_loop_accounts_policy_tokens_and_actual_nested_tool_calls():
    ledger = AgentLedgerState(
        goal=GoalLedger(original_request="Plan a trip"),
        task_graph=TaskGraph(
            goal_version=1,
            tasks=(
                TaskNode(
                    task_id="details",
                    goal="details",
                    allowed_actions=("get_poi_detail",),
                    success_criteria={"required_artifact_types": ["poi_detail_set"]},
                ),
            ),
        ),
    )

    class MeteredPolicy:
        async def propose(self, context: PolicyContext) -> PolicyAction:
            return PolicyAction(action="get_poi_detail", token_usage=321)

    class MeteredExecutor:
        async def execute(self, *, task, action, ledger) -> ActionOutcome:
            return ActionOutcome(
                tool_calls_used=3,
                artifacts=[
                    ArtifactRecord(
                        artifact_id="details",
                        artifact_type="poi_detail_set",
                        payload={"details": [{}, {}, {}]},
                        goal_version=1,
                        plan_version=1,
                    )
                ],
            )

    result = await BoundedAgentLoop().run(
        ledger,
        policy=MeteredPolicy(),
        executor=MeteredExecutor(),
    )

    assert result.ledger.budget.used_tool_calls == 3
    assert result.ledger.budget.used_tokens == 321
    assert result.ledger.budget.used_latency_ms >= 0


@pytest.mark.asyncio
async def test_loop_interrupts_for_user_without_marking_task_success():
    ledger = AgentLedgerState(
        goal=GoalLedger(original_request="Plan a trip"),
        task_graph=TaskGraph(
            goal_version=1,
            tasks=(
                TaskNode(
                    task_id="ask",
                    goal="get destination",
                    allowed_actions=("ask_user",),
                    success_criteria={"required_fact_keys": ["destination"]},
                ),
            ),
        ),
    )
    policy = ScriptedPolicy({"ask": "ask_user"})

    class InterruptExecutor:
        async def execute(self, *, task, action, ledger) -> ActionOutcome:
            return ActionOutcome(status="awaiting_user")

    result = await BoundedAgentLoop().run(ledger, policy=policy, executor=InterruptExecutor())
    assert result.status == "interrupted"
    assert result.termination_reason == "awaiting_user"
    assert result.ledger.task_graph.get("ask").status == "blocked"


@pytest.mark.asyncio
async def test_loop_automatically_records_replayable_episode():
    ledger = _ledger()
    recorder = EpisodeRecorder(
        ledger,
        environment_version="travel-env.v1",
        validator_version="travel-validator.v1",
        policy_name="scripted",
        policy_version="v1",
    )
    policy = ScriptedPolicy({"solve": "solve_itinerary", "validate": "validate_itinerary"})
    executor = ArtifactExecutor(
        {
            "solve": ("solver_result", {}),
            "validate": (
                "validation_report",
                {"hard_pass": True, "hard_violations": []},
            ),
        }
    )

    result = await BoundedAgentLoop().run(
        ledger, policy=policy, executor=executor, recorder=recorder
    )

    assert result.status == "finished"
    assert len(recorder.episode.steps) == 2
    assert recorder.episode.status == "finished"
    assert EpisodeReplayVerifier().verify(recorder.episode) == []


def test_policy_context_contains_bounded_artifact_summaries():
    ledger = _ledger()
    ledger.artifacts["pois"] = ArtifactRecord(
        artifact_id="pois",
        artifact_type="poi_candidate_set",
        payload={
            "pois": [{"name": f"POI-{index}", "description": "x" * 1000} for index in range(20)]
        },
        goal_version=1,
        plan_version=1,
    )

    context = BoundedAgentLoop._policy_context(ledger, ledger.task_graph.get("solve"))

    assert context.relevant_artifact_refs == ["pois"]
    assert context.relevant_artifacts[0]["poi_count"] == 20
    assert len(context.relevant_artifacts[0]["poi_names"]) == 10
    assert "description" not in context.relevant_artifacts[0]
