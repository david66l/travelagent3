"""Verified priority preferences must come from real failed rollouts."""

import importlib.util
from pathlib import Path

import pytest

from agentic.corpus_generation import AdaptiveRecoveryTeacherPolicy, build_curriculum_case
from agentic.distillation import build_teacher_candidate, select_teacher_group
from agentic.environment import TravelAgentEnvironment
from agentic.grpo_training import GRPOCorpusRow
from agentic.loop import PolicyAction
from agentic.policy import ControllerFirstPolicy


SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRIORITY = _load_script("build_priority_search_corpus")
RECOVERY = _load_script("build_adaptive_recovery_corpus")
PREFERENCES = _load_script("build_verified_priority_preferences")


class _PrioritySuccessPolicy:
    async def propose(self, context):
        if "accept_candidates" in context.allowed_actions and any(
            item.get("artifact_type") == "poi_candidate_set" for item in context.relevant_artifacts
        ):
            return PolicyAction(action="accept_candidates")
        if "search_pois" in context.allowed_actions:
            target = context.soft_preferences["interests"][-1]
            return PolicyAction(
                action="search_pois",
                arguments={"keywords": [target]},
            )
        return PolicyAction(action=context.allowed_actions[0])


@pytest.mark.asyncio
async def test_priority_negative_is_executed_and_verifier_attributed():
    task, snapshot = build_curriculum_case(0)
    row = PRIORITY.derive_priority_search(GRPOCorpusRow(task=task, snapshot=snapshot))
    environment = TravelAgentEnvironment(row.task, row.snapshot)
    successful = await environment.rollout(ControllerFirstPolicy(_PrioritySuccessPolicy()))
    negative = await environment.rollout(
        ControllerFirstPolicy(PREFERENCES.PrioritySearchPerturbationPolicy())
    )

    chosen = build_teacher_candidate(successful, family="search", sample_index=0)
    rejected = build_teacher_candidate(negative, family="search", sample_index=1)
    selection = select_teacher_group([chosen, rejected])

    assert chosen.score.successful is True
    assert rejected.score.successful is False
    assert rejected.score.invalid_model_steps >= 1
    assert "SNAPSHOT_ARGUMENT_MISMATCH" in PREFERENCES._failure_codes(rejected)
    assert len(selection.preference_pairs) == 1
    pair = selection.preference_pairs[0]
    assert "VERIFIER_SUCCESS_OVER_FAILURE" in pair.reason_codes
    assert pair.chosen["tool_calls"][0]["function"]["arguments"] == {
        "keywords": [row.task.profile["interests"][-1]]
    }
    assert pair.rejected["tool_calls"][0]["function"]["arguments"] == {
        "keywords": row.task.profile["interests"][:2]
    }


@pytest.mark.asyncio
async def test_recovery_negative_repeats_broad_query_after_visible_feedback():
    task, snapshot = build_curriculum_case(7)
    row = RECOVERY.derive_adaptive_recovery(GRPOCorpusRow(task=task, snapshot=snapshot))
    environment = TravelAgentEnvironment(row.task, row.snapshot)
    successful = await environment.rollout(ControllerFirstPolicy(AdaptiveRecoveryTeacherPolicy()))
    negative = await environment.rollout(
        ControllerFirstPolicy(PREFERENCES.PrioritySearchPerturbationPolicy())
    )

    chosen = build_teacher_candidate(successful, family="recovery", sample_index=0)
    rejected = build_teacher_candidate(negative, family="recovery", sample_index=1)
    selection = select_teacher_group([chosen, rejected])

    assert PREFERENCES._target_contract(row, "recovery_repeat") == [
        row.task.profile["interests"][-1]
    ]
    assert rejected.score.successful is False
    assert "SNAPSHOT_ARGUMENT_MISMATCH" in PREFERENCES._failure_codes(rejected)
    assert len(selection.preference_pairs) == 1
    pair = selection.preference_pairs[0]
    assert pair.family == "recovery"
    assert pair.chosen["tool_calls"][0]["function"]["arguments"] == {
        "keywords": [row.task.profile["interests"][-1]]
    }
    assert pair.rejected["tool_calls"][0]["function"]["arguments"] == {
        "keywords": row.task.profile["interests"][:2]
    }
