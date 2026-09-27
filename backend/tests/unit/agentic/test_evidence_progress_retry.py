"""A repaired prerequisite unlocks retries without disabling the failure bound."""

import pytest

from agentic.action_executor import TravelActionExecutor
from agentic.clock import frozen_reference_time
from agentic.loop import BoundedAgentLoop, PolicyAction
from agentic.policy_repair import (
    SelfRepairingAgentPolicy,
    _repeats_failed_action_without_progress,
)
from scripts.build_harness_training_pilot import build
from scripts.evaluate_harness_base import FrozenResearchExecutor, MOMENT, initial
from backend.tests.unit.agentic.test_policy import _context


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh_missing_entity", [True, False])
async def test_current_evidence_recheck_unblocks_only_satisfied_prerequisites(refresh_missing_entity):
    case = build()[20]
    actions = [PolicyAction(action="search_current_info", arguments={"query": q})
               for q in case["slots"]["current_info_queries"]]
    actions += [PolicyAction(action=name) for name in [
        "search_pois", "get_poi_detail", "get_route_matrix", "solve_itinerary", "solve_itinerary"]]
    entity = case["slots"]["must_visit"][0 if refresh_missing_entity else 1]
    actions.append(PolicyAction(action="search_current_info", arguments={
        "query": entity + " 最新开放公告", "date": case["slots"]["start_date"]}))
    actions += [PolicyAction(action=name) for name in [
        "solve_itinerary", "solve_itinerary", "validate_itinerary", "finish"]]

    class Policy:
        def __init__(self):
            self.actions = iter(actions)
            self.contexts = []

        async def propose(self, context):
            self.contexts.append(context)
            return next(self.actions)

    policy = Policy()
    with frozen_reference_time(MOMENT):
        state = initial(case)
        result = await BoundedAgentLoop().run(state, policy=SelfRepairingAgentPolicy(policy),
            executor=TravelActionExecutor(FrozenResearchExecutor(case)))
    if refresh_missing_entity:
        assert result.termination_reason == "awaiting_user"
        assert any(c.resolved_precondition_failures for c in policy.contexts)
    else:
        assert result.termination_reason == "policy_error_fallback"
        assert not any(c.resolved_precondition_failures for c in policy.contexts)
    assert all("resolved_precondition_failures" not in c.model_dump() for c in policy.contexts)


@pytest.mark.asyncio
@pytest.mark.parametrize("early_solves", [False, True])
async def test_candidate_search_and_route_evidence_unblock_failed_prerequisites(early_solves):
    case = build()[1]
    actions = [
        PolicyAction(action=name)
        for name in ["get_poi_detail", "get_poi_detail", "search_pois", "get_poi_detail"]
    ]
    actions.extend(
        PolicyAction(action="search_current_info", arguments={"query": query})
        for query in case["slots"]["current_info_queries"]
    )
    if early_solves:
        actions.extend([PolicyAction(action="solve_itinerary") for _ in range(2)])
    actions.extend(
        PolicyAction(action=name)
        for name in ["get_route_matrix", "solve_itinerary", "validate_itinerary", "finish"]
    )

    class Policy:
        def __init__(self):
            self.actions = iter(actions)
            self.feedback = []

        async def propose(self, context):
            self.feedback.extend(context.policy_feedback)
            return next(self.actions)

    policy = Policy()
    with frozen_reference_time(MOMENT):
        state = initial(case)
        result = await BoundedAgentLoop().run(
            state,
            policy=SelfRepairingAgentPolicy(policy),
            executor=TravelActionExecutor(FrozenResearchExecutor(case)),
        )
    assert result.termination_reason == "awaiting_user"
    assert state.decision_history[-1].action == "finish"
    assert state.decision_history[2].outcome_status == "completed"
    assert state.decision_history[3].action == "get_poi_detail"
    assert state.decision_history[3].outcome_status == "completed"
    expected = ["CANDIDATES_REQUIRED"] * 2
    if early_solves:
        expected += ["RESEARCH_EVIDENCE_INSUFFICIENT"] * 2
        assert all(f.message == "MISSING_ARTIFACT:route_matrix" for f in state.failures[2:])
    assert [f.code for f in state.failures] == expected
    assert policy.feedback == []


def retry_context():
    context = _context()
    context.failure_summary = [
        {
            "code": "CANDIDATES_REQUIRED",
            "attempted_strategy": "get_poi_detail",
            "attempted_arguments": {},
        }
    ] * 2
    context.decision_history = [
        {
            "task_id": "weather",
            "action": "search_pois",
            "arguments": {},
            "outcome_status": "completed",
            "progress_made": True,
        }
    ]
    return context


def test_new_candidate_evidence_resets_only_failures_before_progress():
    context = retry_context()
    action = PolicyAction(action="get_poi_detail")
    assert not _repeats_failed_action_without_progress(context, action)
    failure = {
        "task_id": "weather",
        "action": "get_poi_detail",
        "arguments": {},
        "outcome_status": "failed",
        "progress_made": False,
    }
    context.decision_history.append(failure)
    assert not _repeats_failed_action_without_progress(context, action)
    context.decision_history.append(failure)
    assert _repeats_failed_action_without_progress(context, action)


@pytest.mark.parametrize(
    "update",
    [
        {"progress_made": False},
        {"outcome_status": "failed"},
        {"outcome_status": "succeeded"},  # Task status is not an action outcome.
        {"action": "get_weather"},
        {"task_id": "another-task"},
    ],
)
def test_unchanged_unrelated_or_unsuccessful_observations_do_not_unlock(update):
    context = retry_context()
    context.decision_history[0].update(update)
    assert _repeats_failed_action_without_progress(context, PolicyAction(action="get_poi_detail"))


@pytest.mark.parametrize("code", ["INVALID_ARGUMENTS", "EXECUTOR_ERROR"])
def test_candidate_progress_does_not_erase_other_failure_contracts(code):
    context = retry_context()
    context.failure_summary = [dict(failure, code=code) for failure in context.failure_summary]
    assert _repeats_failed_action_without_progress(context, PolicyAction(action="get_poi_detail"))


@pytest.mark.parametrize(
    "producer,progress,blocked",
    [
        ("get_route_matrix", True, False),
        ("get_route_matrix", False, True),
        ("get_poi_detail", True, True),
        ("get_weather", True, True),
    ],
)
def test_missing_route_retry_requires_new_route_evidence(producer, progress, blocked):
    context = retry_context()
    context.failure_summary = [
        {
            "code": "RESEARCH_EVIDENCE_INSUFFICIENT",
            "attempted_strategy": "solve_itinerary",
            "attempted_arguments": {"strategy": "cpsat"},
            "message": "MISSING_ARTIFACT:route_matrix",
        }
    ] * 2
    context.decision_history[0].update(action=producer, progress_made=progress)
    action = PolicyAction(action="solve_itinerary", arguments={"strategy": "cpsat"})
    assert _repeats_failed_action_without_progress(context, action) is blocked


def test_new_failure_after_resolved_prerequisite_starts_a_new_retry_window():
    context = retry_context()
    original = {
        "code": "RESEARCH_EVIDENCE_INSUFFICIENT",
        "attempted_strategy": "solve_itinerary",
        "attempted_arguments": {},
        "message": "MISSING_ARTIFACT:route_matrix",
    }
    context.failure_summary = [original] * 2 + [
        dict(original, message="MISSING_ARTIFACT:weather_snapshot")
    ]
    context.decision_history[0].update(action="get_route_matrix")
    failure = {
        "task_id": "weather",
        "action": "solve_itinerary",
        "arguments": {},
        "outcome_status": "failed",
        "progress_made": False,
    }
    context.decision_history.append(failure)
    action = PolicyAction(action="solve_itinerary")
    assert not _repeats_failed_action_without_progress(context, action)
    context.decision_history.append(failure)
    assert _repeats_failed_action_without_progress(context, action)
