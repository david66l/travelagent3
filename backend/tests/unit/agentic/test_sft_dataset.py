"""Tests for replay-gated policy SFT dataset construction."""

from datetime import UTC, datetime

import pytest

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
    tool_call = example.messages[-1].tool_calls[0]
    assert tool_call.function.name == "solve_itinerary"
    assert tool_call.function.arguments == {}
    assert example.messages[-1].content is None
    assert [tool["function"]["name"] for tool in example.tools] == ["solve_itinerary"]
    parameters = example.tools[0]["function"]["parameters"]
    assert "pois" not in parameters["properties"]
    assert "constraints" not in parameters["properties"]


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


def test_unicode_replacement_character_is_rejected():
    candidate = _candidate("broken-text", "trajectory-broken-text")
    candidate.episode.initial_state["note"] = "broken � city"

    result = SFTDatasetBuilder().build([candidate])

    assert result.manifest.accepted_episodes == 0
    assert "L1_TEXT_ENCODING_CORRUPT" in result.reviews[0].rejection_codes


def test_unfinalized_or_schema_invalid_episode_is_rejected():
    unfinalized = _candidate("unfinalized", "trajectory-unfinalized")
    unfinalized.episode.content_hash = None
    ungrounded = _candidate("ungrounded", "trajectory-ungrounded")
    ungrounded.episode.steps[0].action.arguments = {"strategy": "invented-strategy"}

    result = SFTDatasetBuilder().build([unfinalized, ungrounded])

    codes = {code for review in result.reviews for code in review.rejection_codes}
    assert "L1_UNFINALIZED_EPISODE" in codes
    assert "L2_POLICY_ARGUMENT_INVALID" in codes


def test_composed_search_text_is_allowed_but_structured_date_must_be_grounded():
    composed = _candidate("composed", "trajectory-composed")
    composed.episode.steps[0].task_id = "retrieve_city_knowledge"
    composed.episode.steps[0].context = _context(
        composed.episode.trajectory_id,
        "retrieve_city_knowledge",
        "retrieve_city_knowledge",
    )
    composed.episode.steps[0].action = PolicyAction(
        action_id="composed-search",
        action="retrieve_city_knowledge",
        arguments={"topic": "Shanghai history museum"},
    )
    composed.episode.steps[0].observations[0].tool = "retrieve_city_knowledge"
    composed.episode.steps[0].observations[0].tool_call_id = "composed-search"
    composed.episode.content_hash = episode_content_hash(composed.episode)

    hallucinated_date = _candidate("date", "trajectory-date")
    hallucinated_date.episode.steps[0].task_id = "get_weather"
    hallucinated_date.episode.steps[0].context = _context(
        hallucinated_date.episode.trajectory_id,
        "get_weather",
        "get_weather",
    )
    hallucinated_date.episode.steps[0].action = PolicyAction(
        action_id="weather-date",
        action="get_weather",
        arguments={"date": "2099-01-01"},
    )
    hallucinated_date.episode.steps[0].observations[0].tool = "get_weather"
    hallucinated_date.episode.steps[0].observations[0].tool_call_id = "weather-date"
    hallucinated_date.episode.content_hash = episode_content_hash(hallucinated_date.episode)

    result = SFTDatasetBuilder().build([composed, hallucinated_date])

    reviews = {review.scenario_id: review for review in result.reviews}
    assert reviews["composed"].accepted is True
    assert "L2_ARGUMENT_NOT_GROUNDED" in reviews["date"].rejection_codes


def test_missing_observation_is_rejected_and_duplicate_target_is_skipped():
    missing = _candidate("missing", "trajectory-missing")
    missing.episode.steps[0].observations = []
    duplicate = _candidate("duplicate", "trajectory-duplicate")
    extra = duplicate.episode.steps[0].model_copy(deep=True)
    extra.step_index = 1
    duplicate.episode.steps.insert(1, extra)
    duplicate.episode.steps[2].step_index = 2
    duplicate.episode.content_hash = episode_content_hash(duplicate.episode)

    result = SFTDatasetBuilder().build([missing, duplicate])

    reviews = {review.scenario_id: review for review in result.reviews}
    assert "L2_TOOL_OBSERVATION_MISSING" in reviews["missing"].rejection_codes
    assert reviews["duplicate"].accepted is True
    assert reviews["duplicate"].example_count == 2
    assert result.manifest.excluded_duplicate_policy_steps == 1


def test_repeated_successful_call_after_goal_revision_is_not_a_duplicate():
    candidate = _candidate()
    revised = candidate.episode.steps[0].model_copy(deep=True)
    revised.step_index = 2
    revised.context.goal_version = 2
    revised.context.plan_version = 2
    revised.action.action_id = "call-revised"
    revised.observations[0].tool_call_id = "call-revised"
    candidate.episode.steps.append(revised)
    candidate.episode.content_hash = episode_content_hash(candidate.episode)

    result = SFTDatasetBuilder().build([candidate])

    assert result.manifest.accepted_episodes == 1
    assert result.manifest.exported_examples == 3


def test_revision_snapshots_can_share_a_conversation_trajectory_id():
    initial = _candidate("initial", "trajectory-revised")
    revised = initial.model_copy(deep=True)
    revised.scenario_id = "revised"
    extra = revised.episode.steps[0].model_copy(deep=True)
    extra.step_index = 2
    extra.context.goal_version = 2
    extra.context.plan_version = 2
    extra.action.action_id = "call-revised"
    extra.observations[0].tool_call_id = "call-revised"
    revised.episode.steps.append(extra)
    revised.episode.content_hash = episode_content_hash(revised.episode)

    result = SFTDatasetBuilder().build([initial, revised])

    assert result.manifest.candidate_episodes == 2
    assert result.manifest.shared_trajectory_snapshots == 1
    assert result.manifest.exported_examples == 5
    assert len({item.example_id for item in result.examples}) == 5


def test_exact_duplicate_episode_snapshots_are_rejected():
    first = _candidate("first", "trajectory-conflict")
    second = first.model_copy(deep=True)
    second.scenario_id = "second"

    with pytest.raises(ValueError, match="duplicate episode snapshot"):
        SFTDatasetBuilder().build([first, second])


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


def test_dataset_version_changes_when_model_visible_projection_changes():
    candidate = _candidate()
    builder = SFTDatasetBuilder()
    result = builder.build([candidate])
    changed_examples = [item.model_copy(deep=True) for item in result.examples]
    changed_examples[0].messages[0].content += "\nA new policy rule."

    changed_manifest = builder._manifest(
        [candidate],
        result.reviews,
        changed_examples,
        overlap=False,
    )

    assert changed_manifest.dataset_version != result.manifest.dataset_version


def test_failed_policy_decision_is_not_imitation_target():
    candidate = _candidate()
    candidate.episode.steps[0].verification = {"task_status": "ready", "error_code": "TIMEOUT"}
    candidate.episode.content_hash = episode_content_hash(candidate.episode)

    result = SFTDatasetBuilder().build([candidate])

    assert result.manifest.accepted_episodes == 1
    assert result.manifest.exported_examples == 1
    assert result.manifest.excluded_policy_steps == 1
    assert result.examples[0].step_index == 1


def test_successful_intermediate_react_action_is_an_imitation_target():
    candidate = _candidate()
    candidate.episode.steps[0].verification = {
        "task_status": "ready",
        "error_code": None,
    }
    candidate.episode.content_hash = episode_content_hash(candidate.episode)

    result = SFTDatasetBuilder().build([candidate])

    assert result.manifest.accepted_episodes == 1
    assert result.manifest.exported_examples == 2
    assert result.examples[0].step_index == 0


def test_verified_finish_waiting_for_confirmation_is_an_imitation_target():
    candidate = _candidate()
    finish = candidate.episode.steps[-1].model_copy(deep=True)
    finish.step_index = 2
    finish.task_id = "await_confirmation"
    finish.context = _context(
        candidate.episode.trajectory_id,
        "await_confirmation",
        "finish",
    )
    finish.action = PolicyAction(action_id="finish", action="finish", arguments={})
    finish.observations = []
    finish.verification = {"task_status": "blocked", "error_code": None}
    finish.state_before_hash = "before-finish"
    finish.state_after_hash = "after-finish"
    candidate.episode.steps.append(finish)
    candidate.episode.content_hash = episode_content_hash(candidate.episode)

    result = SFTDatasetBuilder().build([candidate])

    assert result.manifest.accepted_episodes == 1
    assert result.manifest.exported_examples == 3
    assert result.examples[-1].messages[-1].tool_calls[0].function.name == "finish"


def test_evidence_exhausted_tradeoff_is_a_safe_termination_target():
    candidate = _candidate()
    tradeoff = candidate.episode.steps[-1].model_copy(deep=True)
    tradeoff.step_index = 2
    tradeoff.task_id = "research_evidence"
    tradeoff.context = _context(
        candidate.episode.trajectory_id,
        "research_evidence",
        "propose_tradeoff",
    )
    tradeoff.context.current_subtask["action_attempt_counts"] = {
        "search_current_info": 2,
        "finalize_research": 1,
    }
    tradeoff.action = PolicyAction(
        action_id="tradeoff",
        action="propose_tradeoff",
        arguments={"reason": "Evidence is insufficient", "options": ["Change date"]},
    )
    tradeoff.observations = []
    tradeoff.verification = {"task_status": "blocked", "error_code": None}
    candidate.episode.steps.append(tradeoff)
    candidate.episode.final_state["artifacts"] = {}
    candidate.episode.content_hash = episode_content_hash(candidate.episode)

    result = SFTDatasetBuilder().build([candidate])

    assert result.manifest.accepted_episodes == 1
    assert result.reviews[0].quality_label == "safe_termination"
    assert result.examples[-1].messages[-1].tool_calls[0].function.name == "propose_tradeoff"


def test_episode_with_only_controller_or_failed_policy_steps_is_rejected():
    candidate = _candidate()
    for step in candidate.episode.steps:
        step.verification = {"task_status": "failed"}
    candidate.episode.content_hash = episode_content_hash(candidate.episode)

    result = SFTDatasetBuilder().build([candidate])

    assert result.manifest.accepted_episodes == 0
    assert result.reviews[0].rejection_codes == ["L3_NO_VERIFIED_POLICY_DECISION"]
