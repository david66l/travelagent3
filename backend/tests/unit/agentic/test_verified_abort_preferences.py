"""Premature-abort negatives must be executed and verifier-attributed."""

import importlib.util
from pathlib import Path

import pytest

from agentic.corpus_generation import CurriculumTeacherPolicy, build_curriculum_case
from agentic.distillation import build_teacher_candidate, select_teacher_group
from agentic.environment import TravelAgentEnvironment
from agentic.policy import ControllerFirstPolicy


SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "build_verified_abort_preferences.py"
SPEC = importlib.util.spec_from_file_location("build_verified_abort_preferences", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.asyncio
async def test_premature_abort_becomes_real_failed_preference():
    task, snapshot = build_curriculum_case(6)
    environment = TravelAgentEnvironment(task, snapshot)
    successful = await environment.rollout(ControllerFirstPolicy(CurriculumTeacherPolicy()))
    failed = await environment.rollout(ControllerFirstPolicy(MODULE.PrematureAbortPolicy()))
    chosen = build_teacher_candidate(successful, family="clarification", sample_index=0)
    rejected = build_teacher_candidate(failed, family="clarification", sample_index=1)

    selection = select_teacher_group([chosen, rejected])

    assert rejected.score.successful is False
    assert "POLICY_ABORT" in MODULE._failure_codes(rejected)
    assert len(selection.preference_pairs) == 1
    pair = selection.preference_pairs[0]
    assert pair.rejected["tool_calls"][0]["function"]["name"] == "abort"
    assert "VERIFIER_SUCCESS_OVER_FAILURE" in pair.reason_codes
