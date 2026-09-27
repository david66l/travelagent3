"""Behavioral contract for model-owned planning and the shared execution harness."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from agentic.action_executor import TravelActionExecutor
from agentic.harness import assert_session_contract, invalidate_derived_evidence, preflight_action
from agentic.interactive import InteractiveAgentSession
from agentic.loop import ActionOutcome, BoundedAgentLoop, PolicyAction
from agentic.policy_actions import POLICY_ACTION_MODELS
from agentic.policy_controller import constrain_policy_context
from agentic.runtime import initialize_agent_ledger
from agentic.state import AgentLedgerState, ArtifactRecord, BudgetLedger
from agentic.trajectory import EpisodeRecorder
from agentic.trl_environment import TRLAgentEnvironment, build_trl_environment_factories


def ledger(**slots):
    state = initialize_agent_ledger(
        {
            "user_input": "Plan a trip",
            "slots": {"destination": "Hangzhou", "travel_days": 2, **slots},
        },
        mode="agent",
    )
    return AgentLedgerState(**state["agent_ledger"])


def artifact(state, kind, payload, **kwargs):
    item = ArtifactRecord(
        artifact_id=str(uuid4()),
        artifact_type=kind,
        payload=payload,
        goal_version=1,
        plan_version=1,
        **kwargs,
    )
    state.artifacts[item.artifact_id] = item
    return item


class SequencePolicy:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.contexts = []

    async def propose(self, context):
        self.contexts.append(context)
        return next(self.actions)


def test_one_session_and_complete_shared_action_surface():
    state = ledger()
    assert_session_contract(state)
    assert len(state.task_graph.tasks) == 1
    assert set(state.task_graph.tasks[0].allowed_actions) == set(POLICY_ACTION_MODELS)
    assert len(POLICY_ACTION_MODELS) == 13
    assert set(build_trl_environment_factories()) == {"travel"}
    assert all(callable(getattr(TRLAgentEnvironment, name)) for name in POLICY_ACTION_MODELS)
    with pytest.raises(ValueError):
        build_trl_environment_factories("controller_first")


def test_missing_information_does_not_force_an_action_or_hide_other_missing_fields():
    state = ledger()
    state.goal.missing_information = ["origin", "start_date"]
    state.goal.capability.status = "needs_user"
    context = constrain_policy_context(
        BoundedAgentLoop._policy_context(state, state.task_graph.tasks[0])
    )
    assert context.missing_information == ["origin", "start_date"]
    assert {"ask_user", "search_pois", "abort", "solve_itinerary"} <= set(context.allowed_actions)


@pytest.mark.asyncio
async def test_early_finish_is_feedback_and_model_can_choose_to_ask():
    state = ledger()
    policy = SequencePolicy(
        [
            PolicyAction(action="finish"),
            PolicyAction(action="ask_user", arguments={"question": "Which dates?"}),
        ]
    )
    recorder = EpisodeRecorder(
        state,
        environment_version="agent-harness-v1",
        validator_version="test",
        policy_name="test",
        policy_version="1",
    )
    result = await BoundedAgentLoop().run(
        state, policy=policy, executor=TravelActionExecutor(AsyncMock()), recorder=recorder
    )
    assert result.status == "interrupted"
    assert len(policy.contexts) == 2
    assert policy.contexts[1].failure_summary[-1]["code"] == "VALIDATION_NOT_PASSED"
    assert [s.action.action for s in recorder.episode.steps] == ["finish", "ask_user"]
    assert all(s.action.decision_source == "policy" for s in recorder.episode.steps)


@pytest.mark.asyncio
async def test_successful_tool_returns_to_model_without_automatic_progression():
    state = ledger()
    policy = SequencePolicy(
        [
            PolicyAction(action="retrieve_city_knowledge", arguments={"topic": "accessibility"}),
            PolicyAction(action="ask_user", arguments={"question": "Any mobility constraints?"}),
        ]
    )
    executor = AsyncMock()
    executor.execute.side_effect = [ActionOutcome(), ActionOutcome(status="awaiting_user")]
    result = await BoundedAgentLoop().run(state, policy=policy, executor=executor)
    assert result.status == "interrupted"
    assert executor.execute.await_count == len(policy.contexts) == 2
    assert [call.kwargs["action"].action for call in executor.execute.call_args_list] == [
        "retrieve_city_knowledge",
        "ask_user",
    ]


@pytest.mark.asyncio
async def test_online_and_interactive_use_identical_decision_contexts(monkeypatch):
    moment = datetime.now(UTC)
    monkeypatch.setattr("agentic.loop.reference_now", lambda: moment)
    initial = ledger()
    action = PolicyAction(action="ask_user", arguments={"question": "Which dates?"})
    policy = SequencePolicy([action.model_copy(deep=True)])
    await BoundedAgentLoop().run(
        initial.model_copy(deep=True), policy=policy, executor=TravelActionExecutor(AsyncMock())
    )
    session = InteractiveAgentSession(
        initial,
        executor=TravelActionExecutor(AsyncMock()),
        environment_version="test",
        validator_version="test",
        policy_name="test",
        policy_version="1",
    )
    try:
        transition = await session.start()
        from agentic.policy_prompts import policy_prompt_payload

        assert policy_prompt_payload(transition.next_context) == policy_prompt_payload(
            policy.contexts[0]
        )
        result = await session.submit(action)
        assert result.done and result.status == "interrupted"
        assert len(result.episode.steps) == 1
    finally:
        await session.aclose()


def test_semantic_query_and_solver_strategy_survive_and_execution_is_audited():
    state = ledger(
        current_info_queries=["old query"],
        start_date="2030-10-01",
        transport_modes_requested=["train"],
        origin="Shanghai",
    )
    executor = TravelActionExecutor(AsyncMock())
    args = {"query": "new evidence query", "info_type": "closure"}
    action = PolicyAction(action="search_current_info", arguments=args, model_arguments=args)
    payload = executor._hydrate_arguments(state, action)
    assert payload["query"] == "new evidence query" and payload["info_type"] == "closure"
    assert payload["date"] == "2030-10-01"
    assert action.executed_arguments == payload and action.model_arguments == args
    assert (
        action.argument_sources["query"] == "model" and action.argument_sources["date"] == "harness"
    )
    solve = PolicyAction(action="solve_itinerary", arguments={"strategy": "greedy"})
    assert executor._hydrate_arguments(state, solve)["strategy"] == "greedy"
    search = PolicyAction(action="search_pois", arguments={"keywords": []})
    assert executor._hydrate_arguments(state, search)["keywords"] == []


def test_conflicting_model_parameters_are_rejected_not_repaired():
    state = ledger(origin="Shanghai", start_date="2030-10-01", transport_modes_requested=["train"])
    for action, args, code in [
        ("search_transport", {"mode": "flight"}, "TRANSPORT_MODE_CONSTRAINT_MISMATCH"),
        (
            "search_current_info",
            {"query": "hours", "date": "2030-11-01"},
            "DATE_CONSTRAINT_MISMATCH",
        ),
    ]:
        proposal = PolicyAction(action=action, arguments=args)
        rejection = preflight_action(state, proposal)
        assert rejection.error_code == code
        assert proposal.arguments == args


def test_new_evidence_invalidates_old_solver_and_verification():
    state = ledger()
    for kind in [
        "solver_result",
        "validation_report",
        "itinerary_draft",
        "poi_detail_set",
        "route_matrix",
    ]:
        artifact(state, kind, {"hard_pass": True})
    invalidate_derived_evidence(state, "search_pois")
    assert not state.artifacts
    assert (
        preflight_action(state, PolicyAction(action="finish")).error_code == "VALIDATION_NOT_PASSED"
    )


def test_finish_rejects_stale_or_expired_validation():
    state = ledger()
    artifact(state, "solver_result", {}, created_at=datetime.now(UTC))
    report = artifact(
        state,
        "validation_report",
        {"hard_pass": True},
        created_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    assert preflight_action(state, PolicyAction(action="finish")).error_code == "STALE_VALIDATION"
    state.artifacts[report.artifact_id] = report.model_copy(
        update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    assert (
        preflight_action(state, PolicyAction(action="finish")).error_code == "VALIDATION_NOT_PASSED"
    )


@pytest.mark.asyncio
async def test_bad_repeated_actions_hit_budget_without_controller_rescue():
    state = ledger()
    state.budget = BudgetLedger(max_episode_steps=2)
    policy = SequencePolicy([PolicyAction(action="finish") for _ in range(3)])
    executor = AsyncMock()
    result = await BoundedAgentLoop().run(state, policy=policy, executor=executor)
    assert result.status == "failed" and result.termination_reason == "budget_exhausted_fallback"
    executor.execute.assert_not_called()


def seed_evidence(state):
    artifact(state, "city_knowledge", {"summary": "Grounded destination knowledge"})
    artifact(state, "poi_candidate_set", {"pois": [{"name": f"POI {i}"} for i in range(4)]})
    artifact(state, "poi_detail_set", {"details": [{"name": f"POI {i}"} for i in range(4)]})
    artifact(state, "route_matrix", {"time_minutes": [[0] * 5 for _ in range(5)]})


@pytest.mark.asyncio
async def test_model_selects_solve_validate_submit_and_user_alone_confirms():
    from agentic.runtime import confirm_agent_ledger
    from agentic.trajectory import EpisodeReplayVerifier

    state = ledger()
    seed_evidence(state)
    policy = SequencePolicy(
        [PolicyAction(action=n) for n in ["solve_itinerary", "validate_itinerary", "finish"]]
    )
    tools = AsyncMock()
    tools.execute.side_effect = [
        [
            {
                "observation": {
                    "ok": True,
                    "tool": "solve_itinerary",
                    "data": {"days": [], "status": "optimal"},
                    "source": "solver",
                    "confidence": 1,
                }
            }
        ],
        [
            {
                "observation": {
                    "ok": True,
                    "tool": "validate_itinerary",
                    "data": {
                        "hard_pass": True,
                        "hard_violations": [],
                        "soft_score": 1,
                        "validator_version": "test",
                    },
                    "source": "validator",
                    "confidence": 1,
                }
            }
        ],
    ]
    recorder = EpisodeRecorder(
        state,
        environment_version="agent-harness-v1",
        validator_version="test",
        policy_name="test",
        policy_version="1",
    )
    result = await BoundedAgentLoop().run(
        state, policy=policy, executor=TravelActionExecutor(tools), recorder=recorder
    )
    assert result.status == "interrupted"
    assert len(policy.contexts) == 3 and tools.execute.await_count == 2
    assert not any(f.key == "user_confirmation" for f in result.ledger.facts.values())
    from agentic.reward import HierarchicalRewardEngine

    reward = HierarchicalRewardEngine().score(recorder.episode)
    assert reward.gate_status == "passed" and reward.components.task == 1.0
    confirmed, decision = confirm_agent_ledger(result.ledger)
    assert decision.allowed and confirmed.termination_reason == "validated_finish"
    assert confirmed.task_graph.tasks[0].status == "succeeded"
    assert result.ledger.task_graph.tasks[0].status == "blocked"  # confirmation is transactional
    EpisodeReplayVerifier().verify(recorder.episode)


@pytest.mark.asyncio
async def test_solver_failure_returns_to_model_for_research_instead_of_forced_retry():
    state = ledger()
    seed_evidence(state)
    policy = SequencePolicy(
        [
            PolicyAction(action="solve_itinerary"),
            PolicyAction(action="retrieve_city_knowledge", arguments={"topic": "alternatives"}),
            PolicyAction(action="ask_user", arguments={"question": "Would you like another area?"}),
        ]
    )
    executor = AsyncMock()
    executor.execute.side_effect = [
        ActionOutcome(
            status="failed",
            error_code="SOLVER_INFEASIBLE",
            error_message="No feasible route",
            retryable=True,
        ),
        ActionOutcome(),
        ActionOutcome(status="awaiting_user"),
    ]
    result = await BoundedAgentLoop().run(state, policy=policy, executor=executor)
    assert result.status == "interrupted" and len(policy.contexts) == 3
    assert {"solve_itinerary", "retrieve_city_knowledge", "ask_user"} <= set(
        policy.contexts[1].allowed_actions
    )
    assert policy.contexts[1].failure_summary[-1]["code"] == "SOLVER_INFEASIBLE"


@pytest.mark.asyncio
async def test_harness_enforces_deadline_and_records_model_failure():
    import asyncio

    state = ledger()
    state.budget = BudgetLedger(timeout_ms=10)

    class SlowPolicy:
        async def propose(self, context):
            await asyncio.sleep(1)

    result = await BoundedAgentLoop().run(state, policy=SlowPolicy(), executor=AsyncMock())
    assert result.termination_reason == "agent_deadline_exceeded"


@pytest.mark.asyncio
async def test_schema_error_repairs_with_same_model_not_another_planner():
    from agentic.integration import run_agent_branch
    from agentic.policy import PolicyOutputError

    class RepairPolicy:
        def __init__(self):
            self.calls = 0

        async def propose(self, context):
            self.calls += 1
            if self.calls == 1:
                raise PolicyOutputError("malformed tool call")
            assert context.policy_feedback
            return PolicyAction(action="ask_user", arguments={"question": "Which dates?"})

    state = ledger()
    policy = RepairPolicy()
    result = await run_agent_branch(
        {"agent_ledger": state.model_dump(mode="json")},
        policy=policy,
        executor=TravelActionExecutor(AsyncMock()),
    )
    assert policy.calls == 2
    assert result["agent_status"] == "awaiting_information"
    assert len(result["agent_episode"]["steps"]) == 1


def test_grpo_uses_one_full_episode_route_and_keeps_hidden_data_out_of_prompt():
    import json
    from agentic.environment import EnvironmentTask, EnvironmentSnapshot
    from agentic.grpo_training import GRPOCorpusRow, to_trl_environment_rows

    row = GRPOCorpusRow(
        task=EnvironmentTask(
            template_family="travel",
            difficulty="L0",
            seed=1,
            task_id="train-1",
            user_request="Plan Hangzhou",
        ),
        snapshot=EnvironmentSnapshot(
            environment_version="agent-harness-v1",
            snapshot_version="test-v1",
            state_id="test-state",
            hidden_test_facts={"secret_expected_answer": "hidden-value"},
        ),
    )
    result = to_trl_environment_rows([row])[0]
    assert result["environment"] == "travel"
    assert "hidden-value" not in json.dumps(result["prompt"])
    row.snapshot.hidden_test_facts["grpo_decision_state"] = {"target_action": "abort"}
    with pytest.raises(ValueError, match="retired"):
        to_trl_environment_rows([row])


def test_graph_contains_no_alternate_planning_chain():
    from graph.graph import build_graph

    graph = build_graph().get_graph()
    assert set(graph.nodes) == {
        "__start__",
        "__end__",
        "gathering",
        "profile_recall",
        "agent_loop",
        "output",
        "confirm_gate",
        "booking",
    }


@pytest.mark.asyncio
async def test_training_adapter_rejects_early_finish_then_accepts_model_question():
    import json
    from agentic.environment import EnvironmentTask, EnvironmentSnapshot

    env = TRLAgentEnvironment(audit_enabled=False)
    try:
        task = EnvironmentTask(
            template_family="travel",
            difficulty="L0",
            seed=1,
            task_id="dev-1",
            user_request="Plan Hangzhou",
            slots={"destination": "Hangzhou", "travel_days": 2},
        )
        initial = json.loads(
            env.reset(
                task=task.model_dump(mode="json"),
                snapshot=EnvironmentSnapshot(
                    environment_version="agent-harness-v1",
                    snapshot_version="test-v1",
                    state_id="test-state",
                ).model_dump(mode="json"),
            )
        )
        assert {"finish", "search_current_info", "search_transport"} <= set(
            initial["policy_state"]["allowed_actions"]
        )
        env.finish()
        assert not env._transition.done
        env.ask_user("Which dates?")
        assert env._transition.done
        assert len(env._session.recorder.episode.steps) == 2
    finally:
        if env._runner:
            env._runner.run(env._session.aclose())
            env._runner.close()


@pytest.mark.asyncio
async def test_clarification_updates_typed_constraint_using_question_context_without_budget_reset():
    from agentic.runtime import resume_agent_conversation

    state = ledger()
    state.goal.hard_constraints.pop("origin", None)
    state.goal.missing_information = ["origin", "budget_range"]
    policy = SequencePolicy(
        [PolicyAction(action="ask_user", arguments={"question": "你从哪个城市出发？"})]
    )
    result = await BoundedAgentLoop().run(
        state, policy=policy, executor=TravelActionExecutor(AsyncMock())
    )
    budget = result.ledger.budget.model_dump()
    interpretation = {
        "confidence": 0.98,
        "operations": [{"field": "origin", "operation": "set", "value": "济南"}],
    }
    revised = await resume_agent_conversation(
        result.ledger, user_input="济南", interpretation=interpretation
    )
    assert revised.goal.hard_constraints["origin"] == "济南"
    assert revised.goal.missing_information == ["budget_range"]
    assert revised.budget.model_dump() == budget
    assert revised.task_graph.tasks[0].status == "ready"
    assert revised.goal.soft_preferences["last_user_answer"]["answer"] == "济南"
    assert result.ledger.task_graph.tasks[0].status == "blocked"


@pytest.mark.asyncio
async def test_uncertain_answer_never_guesses_first_missing_field():
    from agentic.runtime import resume_agent_conversation

    state = ledger()
    state.goal.hard_constraints.pop("origin", None)
    state.goal.missing_information = ["origin"]
    result = await BoundedAgentLoop().run(
        state,
        policy=SequencePolicy(
            [PolicyAction(action="ask_user", arguments={"question": "从哪里出发？"})]
        ),
        executor=TravelActionExecutor(AsyncMock()),
    )
    revised = await resume_agent_conversation(
        result.ledger,
        user_input="还没想好",
        interpretation={"confidence": 0.1, "operations": [], "needs_clarification": True},
    )
    assert not revised.goal.hard_constraints.get("origin")
    assert "origin" in revised.goal.missing_information
    assert revised.goal.soft_preferences["last_user_answer"]["answer"] == "还没想好"


def test_research_does_not_require_rag_or_fixed_candidate_quota():
    from agentic.react import infer_research_requirements, ResearchSufficiencyVerifier

    state = ledger()
    seed_evidence(state)
    state.artifacts = {
        k: v for k, v in state.artifacts.items() if v.artifact_type != "city_knowledge"
    }
    assert ResearchSufficiencyVerifier().evaluate(state).sufficient
    requirements = infer_research_requirements(state.goal)
    assert requirements.min_candidate_count == 1 and requirements.min_detail_count == 1


@pytest.mark.asyncio
async def test_preflight_rejection_uses_turn_and_tokens_but_not_execution_budget():
    state = ledger()
    state.budget = BudgetLedger(max_tool_calls=0, max_solver_calls=0)
    policy = SequencePolicy(
        [
            PolicyAction(action="solve_itinerary", token_usage=7),
            PolicyAction(action="ask_user", arguments={"question": "Which dates?"}, token_usage=5),
        ]
    )
    tools = AsyncMock()
    result = await BoundedAgentLoop().run(
        state, policy=policy, executor=TravelActionExecutor(tools)
    )
    assert result.status == "interrupted"
    assert state.budget.used_episode_steps == 2 and state.budget.used_tokens == 12
    assert state.budget.used_solver_calls == state.budget.used_tool_calls == 0
    assert policy.contexts[1].failure_summary[-1]["code"] == "RESEARCH_EVIDENCE_INSUFFICIENT"
    assert policy.contexts[1].remaining_budget["solver_calls"] == 0
    tools.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_execution_failure_is_charged_and_limit_blocks_next_execution():
    state = ledger()
    seed_evidence(state)
    state.budget = BudgetLedger(max_solver_calls=1)
    policy = SequencePolicy([PolicyAction(action="solve_itinerary") for _ in range(2)])
    executor = AsyncMock()
    executor.execute.side_effect = RuntimeError("provider unavailable")
    result = await BoundedAgentLoop().run(state, policy=policy, executor=executor)
    assert result.termination_reason == "budget_exhausted_fallback"
    assert state.budget.used_solver_calls == state.budget.used_tool_calls == 1
    assert state.budget.used_episode_steps == 2
    executor.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_identical_research_keeps_validation_and_still_returns_to_model():
    state = ledger()
    seed_evidence(state)
    old = next(a for a in state.artifacts.values() if a.artifact_type == "poi_detail_set")
    solver = artifact(
        state,
        "solver_result",
        {"status": "fallback", "days": [{"activities": [{"poi_name": "A"}]}]},
    )
    validation = artifact(state, "validation_report", {"hard_pass": True})
    repeat = old.model_copy(update={"artifact_id": "new-detail", "created_at": datetime.now(UTC)})
    executor = AsyncMock()
    executor.execute.side_effect = [
        ActionOutcome(artifacts=[repeat]),
        ActionOutcome(status="awaiting_user"),
    ]
    policy = SequencePolicy([PolicyAction(action="get_poi_detail"), PolicyAction(action="finish")])
    result = await BoundedAgentLoop().run(state, policy=policy, executor=executor)
    assert result.status == "interrupted" and len(policy.contexts) == 2
    assert solver.artifact_id in state.artifacts and validation.artifact_id in state.artifacts
    assert preflight_action(state, PolicyAction(action="finish")) is None
    assert (
        sum(a["artifact_type"] == "poi_detail_set" for a in policy.contexts[1].relevant_artifacts)
        == 1
    )


@pytest.mark.parametrize(
    "kind,change",
    [
        ("poi_detail_set", {"details": [{"name": "POI 0", "ticket_price": 2000}]}),
        ("poi_candidate_set", {"pois": [{"name": "Different POI"}]}),
        ("current_info_search", {"query": "hours", "results": [{"snippet": "Closed today"}]}),
    ],
)
def test_changed_content_invalidates_dependent_results(kind, change):
    state = ledger()
    seed_evidence(state)
    artifact(state, "solver_result", {"days": []})
    artifact(state, "validation_report", {"hard_pass": True})
    incoming = ArtifactRecord(
        artifact_id="new", artifact_type=kind, payload=change, goal_version=1, plan_version=1
    )
    action = {
        "poi_detail_set": "get_poi_detail",
        "poi_candidate_set": "search_pois",
        "current_info_search": "search_current_info",
    }[kind]
    invalidate_derived_evidence(state, action, artifacts=[incoming])
    kinds = {a.artifact_type for a in state.artifacts.values()}
    assert not {"solver_result", "validation_report"} & kinds
    if kind in {"poi_detail_set", "poi_candidate_set"}:
        assert "route_matrix" not in kinds
    if kind == "poi_candidate_set":
        assert "poi_detail_set" not in kinds


def test_expired_predecessor_cannot_preserve_validation_on_identical_refresh():
    state = ledger()
    old = artifact(
        state,
        "current_info_search",
        {"query": "hours"},
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    artifact(state, "solver_result", {})
    incoming = old.model_copy(
        update={"artifact_id": "fresh", "expires_at": datetime.now(UTC) + timedelta(hours=1)}
    )
    invalidate_derived_evidence(state, "search_current_info", artifacts=[incoming])
    assert not any(a.artifact_type == "solver_result" for a in state.artifacts.values())


def test_model_sees_bounded_search_and_detail_facts_without_claiming_validation():
    from agentic.policy_prompts import policy_prompt_payload

    state = ledger()
    artifact(
        state,
        "current_info_search",
        {
            "query": "Museum opening hours",
            "queried_at": "2026-09-05T00:00:00+00:00",
            "results": [
                {
                    "title": "Museum",
                    "snippet": "Closed on September 18. " + "x" * 10000,
                    "url": "https://museum.example/" + "x" * 10000,
                }
                for _ in range(40)
            ],
        },
    )
    artifact(
        state,
        "poi_detail_set",
        {
            "details": [
                {
                    "name": "Museum",
                    "ticket_price": 2000,
                    "closed_dates": ["2026-09-18"],
                    "open_time": "09:00",
                    "close_time": "17:00",
                }
            ]
        },
    )
    artifact(
        state,
        "solver_result",
        {"status": "fallback", "days": [{"activities": [{"poi_name": "Museum"}]}]},
    )
    payload = policy_prompt_payload(
        BoundedAgentLoop._policy_context(state, state.task_graph.tasks[0])
    )
    items = {a["artifact_type"]: a for a in payload["relevant_artifacts"]}
    search = items["current_info_search"]
    assert len(search["results"]) == 5 and search["omitted_results"] == 35
    assert search["results"][0]["snippet"].startswith("Closed on September 18.")
    assert len(search["results"][0]["snippet"]) <= 240
    assert search["results"][0]["source_domain"] == "museum.example"
    assert items["poi_detail_set"]["details"][0]["closed_dates"] == ["2026-09-18"]
    assert items["poi_detail_set"]["details"][0]["ticket_price"] == 2000
    assert items["solver_result"]["candidate_available"] is True
    assert items["solver_result"]["validation_is_separate"] is True
    assert "hard_pass" not in items["solver_result"]
    assert payload["observation_time"] and payload["remaining_budget"]["solver_calls"] == 3


def test_repeated_details_do_not_hide_current_solver_from_context():
    state = ledger()
    artifact(state, "solver_result", {"status": "fallback", "days": []})
    for _ in range(12):
        artifact(state, "poi_detail_set", {"details": [{"name": "A"}]})
    kinds = [
        a["artifact_type"]
        for a in BoundedAgentLoop._policy_context(
            state, state.task_graph.tasks[0]
        ).relevant_artifacts
    ]
    assert kinds.count("solver_result") == kinds.count("poi_detail_set") == 1


@pytest.mark.asyncio
async def test_diagnostic_does_not_reward_generic_question_after_rejected_solve():
    from scripts.evaluate_harness_base import judge

    state = ledger()
    policy = SequencePolicy(
        [
            PolicyAction(action="solve_itinerary"),
            PolicyAction(action="ask_user", arguments={"question": "Should I adjust the trip?"}),
        ]
    )
    recorder = EpisodeRecorder(
        state,
        environment_version="agent-harness-v1",
        validator_version="test",
        policy_name="test",
        policy_version="1",
    )
    await BoundedAgentLoop().run(
        state, policy=policy, executor=TravelActionExecutor(AsyncMock()), recorder=recorder
    )
    assert recorder.episode.steps[0].verification["error_code"] == "RESEARCH_EVIDENCE_INSUFFICIENT"
    case = {
        "id": "negative",
        "family": "constraint",
        "expected": "grounded_stop",
        "missing_field": None,
    }
    result = judge(case, recorder.episode, [])
    assert result["grounded_stop"] is False and result["criterion_passed"] is False
