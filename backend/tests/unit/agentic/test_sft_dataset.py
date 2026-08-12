"""Tests for replay-gated policy SFT dataset construction."""

import json
from datetime import UTC, datetime

from agentic.loop import PolicyAction, PolicyContext
from agentic.observations import ObservationEnvelope
from agentic.sft_dataset import EpisodeCandidate, SFTDatasetBuilder
from agentic.trajectory import AgentEpisode, TrajectoryStep, episode_content_hash


def _context(trajectory_id: str, task_id: str, action: str) -> PolicyContext:
    return PolicyContext(
        trajectory_id=trajectory_id,
        goal_version=1,
        plan_version=1,
        original_request="Plan one day in Shanghai",
        current_subtask={"task_id": task_id},
        hard_constraints={"destination": "Shanghai", "travel_days": 1},
        soft_preferences={"interests": ["museum"]},
        relevant_fact_refs=[],
        relevant_artifact_refs=[],
        relevant_facts=[],
        relevant_artifacts=[],
        failure_summary=[],
        remaining_tasks=2,
        remaining_steps=5,
        allowed_actions=[action],
    )


def _episode(trajectory_id: str = "trajectory-1") -> AgentEpisode:
    steps = []
    for index, (task_id, action) in enumerate(
        (("solve_itinerary", "solve_itinerary"), ("validate_itinerary", "validate_itinerary"))
    ):
        steps.append(
            TrajectoryStep(
                step_index=index,
                task_id=task_id,
                context=_context(trajectory_id, task_id, action),
                action=PolicyAction(action_id=f"call-{index}", action=action, arguments={}),
                observations=[
                    ObservationEnvelope(
                        ok=True,
                        tool=action,
                        data={"status": "optimal"},
                        source="built_in",
                        confidence=1.0,
                        tool_call_id=f"call-{index}",
                    )
                ],
                verification={"passed": True},
                state_before_hash=f"before-{index}",
                state_after_hash=f"after-{index}",
            )
        )
    episode = AgentEpisode(
        trajectory_id=trajectory_id,
        environment_version="travel-env-test-v1",
        validator_version="travel-validator.v1",
        policy_name="teacher",
        policy_version="v1",
        initial_state={
            "goal": {
                "original_request": "Plan one day in Shanghai",
                "missing_information": [],
            }
        },
        steps=steps,
        final_state={
            "goal": {
                "original_request": "Plan one day in Shanghai",
                "missing_information": [],
            },
            "artifacts": {
                "solver": {"artifact_type": "solver_result", "payload": {"days": [{}]}},
                "validation": {
                    "artifact_type": "validation_report",
                    "payload": {"hard_pass": True, "hard_violations": []},
                },
            },
        },
        status="interrupted",
        termination_reason="awaiting_user",
        completed_at=datetime.now(UTC),
    )
    episode.content_hash = episode_content_hash(episode)
    return episode


def _candidate(
    scenario_id: str = "scenario-1",
    trajectory_id: str = "trajectory-1",
    **updates,
) -> EpisodeCandidate:
    values = {
        "scenario_id": scenario_id,
        "source": "teacher",
        "template_family": "normal-city-trip",
        "city": "Shanghai",
        "episode": _episode(trajectory_id),
    }
    values.update(updates)
    return EpisodeCandidate(**values)


def test_valid_episode_exports_one_policy_example_per_real_decision():
    result = SFTDatasetBuilder().build([_candidate()])

    assert result.manifest.accepted_episodes == 1
    assert result.manifest.exported_examples == 2
    assert result.manifest.rejected_episodes == 0
    example = result.examples[0]
    assert example.quality_label == "validated_plan"
    assert [message.role for message in example.messages] == ["system", "user", "assistant"]
    assert json.loads(example.messages[-1].content) == {
        "action": "solve_itinerary",
        "arguments": {},
    }


def test_pii_and_policy_supplied_trusted_payload_are_rejected():
    pii = _candidate("pii", "trajectory-pii")
    pii.episode.initial_state["note"] = "call 13812345678"
    forged = _candidate("forged", "trajectory-forged")
    forged.episode.steps[0].action.arguments = {"constraints": {"travel_days": 999}}

    result = SFTDatasetBuilder().build([pii, forged])

    assert result.manifest.accepted_episodes == 0
    codes = {code for review in result.reviews for code in review.rejection_codes}
    assert "L1_PII_DETECTED" in codes
    assert "L2_POLICY_SUPPLIED_PROTECTED_ARGUMENT" in codes


def test_unfinalized_or_ungrounded_episode_is_rejected():
    unfinalized = _candidate("unfinalized", "trajectory-unfinalized")
    unfinalized.episode.content_hash = None
    ungrounded = _candidate("ungrounded", "trajectory-ungrounded")
    ungrounded.episode.steps[0].action.arguments = {"strategy": "invented-strategy"}

    result = SFTDatasetBuilder().build([unfinalized, ungrounded])

    codes = {code for review in result.reviews for code in review.rejection_codes}
    assert "L1_UNFINALIZED_EPISODE" in codes
    assert "L2_ARGUMENT_NOT_GROUNDED" in codes


def test_missing_tool_observation_and_successful_duplicate_are_rejected():
    missing = _candidate("missing", "trajectory-missing")
    missing.episode.steps[0].observations = []
    duplicate = _candidate("duplicate", "trajectory-duplicate")
    extra = duplicate.episode.steps[0].model_copy(deep=True)
    extra.step_index = 1
    duplicate.episode.steps.insert(1, extra)
    duplicate.episode.steps[2].step_index = 2

    result = SFTDatasetBuilder().build([missing, duplicate])

    codes = {code for review in result.reviews for code in review.rejection_codes}
    assert "L2_TOOL_OBSERVATION_MISSING" in codes
    assert "L2_DUPLICATE_SUCCESSFUL_CALL" in codes


def test_group_split_keeps_template_city_and_production_user_isolated():
    same_family = [
        _candidate("one", "trajectory-one"),
        _candidate("two", "trajectory-two"),
    ]
    same_user = [
        _candidate(
            "prod-one",
            "trajectory-prod-one",
            source="production",
            template_family="family",
            city="Beijing",
            user_partition_key="hashed-user-1",
            contains_production_data=True,
        ),
        _candidate(
            "prod-two",
            "trajectory-prod-two",
            source="production",
            template_family="business",
            city="Shenzhen",
            user_partition_key="hashed-user-1",
            contains_production_data=True,
        ),
    ]

    result = SFTDatasetBuilder().build([*same_family, *same_user])

    splits = {review.scenario_id: review.split for review in result.reviews}
    assert splits["one"] == splits["two"]
    assert splits["prod-one"] == splits["prod-two"]
    assert result.manifest.split_group_overlap is False


def test_export_writes_manifest_reviews_and_all_splits(tmp_path):
    builder = SFTDatasetBuilder()
    result = builder.build([_candidate()])

    builder.export(result, tmp_path)

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "reviews.jsonl").exists()
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "validation.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
