"""Tests for episode recording, privacy and replay integrity."""

import hashlib
import json

from agentic.loop import AgentLoopResult, PolicyAction, PolicyContext
from agentic.state import AgentLedgerState, GoalLedger, TaskGraph, TaskNode
from agentic.trajectory import AgentEpisode, EpisodeRecorder, EpisodeReplayVerifier, redact_pii


def _state() -> AgentLedgerState:
    return AgentLedgerState(
        goal=GoalLedger(original_request="Call 13812345678 and plan Shanghai"),
        task_graph=TaskGraph(
            goal_version=1,
            tasks=(
                TaskNode(
                    task_id="search",
                    goal="search",
                    status="running",
                    allowed_actions=("search_pois",),
                ),
            ),
        ),
    )


def _context(state: AgentLedgerState) -> PolicyContext:
    return PolicyContext(
        trajectory_id=state.trajectory_id,
        goal_version=1,
        plan_version=1,
        original_request=state.goal.original_request,
        current_subtask=state.task_graph.get("search").model_dump(mode="json"),
        hard_constraints={},
        soft_preferences={},
        relevant_fact_refs=[],
        relevant_artifact_refs=[],
        failure_summary=[],
        remaining_tasks=1,
        remaining_steps=5,
        allowed_actions=["search_pois"],
    )


def test_recursive_redaction_removes_fields_and_inline_identifiers():
    value = redact_pii(
        {
            "user_id": "user-123",
            "nested": {"email": "me@example.com"},
            "message": "phone 13812345678, ID 11010519491231002X",
        }
    )

    assert value["user_id"] == "[REDACTED]"
    assert value["nested"]["email"] == "[REDACTED]"
    assert "13812345678" not in value["message"]
    assert "11010519491231002X" not in value["message"]


def test_redaction_does_not_corrupt_alphanumeric_trajectory_ids():
    trajectory_id = "d0435c1e-b67a-4ed6-bc1b-17505859080b"
    content_hash = "f483188732e109646663136847491b44223345efdac815d434d4317103728912"

    assert redact_pii(trajectory_id) == trajectory_id
    assert redact_pii(content_hash) == content_hash


def test_recorder_builds_versioned_hash_verified_episode():
    state = _state()
    recorder = EpisodeRecorder(
        state,
        environment_version="travel-env.v1",
        validator_version="travel-validator.v1",
        policy_name="api-teacher",
        policy_version="v1",
    )
    context = _context(state)
    action = PolicyAction(action="search_pois", arguments={"city": "Shanghai"})
    recorder.record_step(
        task_id="search",
        context=context,
        action=action,
        observations=[],
        verification={"passed": True},
        state_before=state,
        state_after=state,
    )
    result = AgentLoopResult(
        ledger=state,
        status="failed",
        termination_reason="partial_finish",
        events=[],
    )
    episode = recorder.finalize(result)

    assert episode.schema_version == "agent-episode.v2"
    assert "13812345678" not in episode.model_dump_json()
    assert EpisodeReplayVerifier().verify(episode) == []


def test_replay_verifier_preserves_v1_hash_without_inference_metrics_field():
    state = _state()
    recorder = EpisodeRecorder(
        state,
        environment_version="travel-env.v1",
        validator_version="travel-validator.v1",
        policy_name="legacy-policy",
        policy_version="v1",
    )
    recorder.record_step(
        task_id="search",
        context=_context(state),
        action=PolicyAction(action="search_pois", arguments={"city": "Shanghai"}),
        observations=[],
        verification={"passed": True},
        state_before=state,
        state_after=state,
    )
    episode = recorder.finalize(
        AgentLoopResult(
            ledger=state,
            status="failed",
            termination_reason="partial_finish",
            events=[],
        )
    )
    legacy_payload = episode.model_dump(mode="json", exclude={"content_hash"})
    legacy_payload["schema_version"] = "agent-episode.v1"
    for step in legacy_payload["steps"]:
        step["action"].pop("inference_metrics", None)
    serialized = json.dumps(
        legacy_payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode()
    legacy_payload["content_hash"] = hashlib.sha256(serialized).hexdigest()

    parsed = AgentEpisode(**legacy_payload)

    assert parsed.steps[0].action.inference_metrics is None
    assert EpisodeReplayVerifier().verify(parsed) == []


def test_replay_verifier_detects_tampering_and_invalid_action():
    state = _state()
    recorder = EpisodeRecorder(
        state,
        environment_version="travel-env.v1",
        validator_version="travel-validator.v1",
        policy_name="test",
        policy_version="v1",
    )
    recorder.record_step(
        task_id="search",
        context=_context(state),
        action=PolicyAction(action="get_weather"),
        observations=[],
        verification={},
        state_before=state,
        state_after=state,
    )
    episode = recorder.finalize(
        AgentLoopResult(
            ledger=state,
            status="failed",
            termination_reason="partial_finish",
            events=[],
        )
    )
    episode.policy_version = "tampered"

    errors = EpisodeReplayVerifier().verify(episode)
    assert "ACTION_NOT_ALLOWED:0" in errors
    assert "CONTENT_HASH_MISMATCH" in errors
