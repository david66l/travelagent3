import pytest

from agentic.corpus_generation import build_curriculum_case
from agentic.distillation import build_teacher_candidate
from agentic.environment import TravelAgentEnvironment
from agentic.policy import ControllerFirstPolicy
from scripts.build_verified_tradeoff_preferences import FakeContinuationPolicy
from scripts.merge_reverified_teacher_sft import _scenario_split, load_and_reverify


@pytest.mark.asyncio
async def test_merge_reverification_rejects_stale_success_score(tmp_path):
    task, snapshot = build_curriculum_case(8)
    rollout = await TravelAgentEnvironment(task, snapshot).rollout(
        ControllerFirstPolicy(FakeContinuationPolicy())
    )
    candidate = build_teacher_candidate(rollout, family="tradeoff", sample_index=0)
    stale = candidate.model_copy(deep=True)
    stale.score.successful = True
    stale.score.gate_status = "passed"
    stale.rollout.reward.gate_status = "passed"
    stale.rollout.reward.components.task = 1
    stale.rollout.reward.components.constraint = 1
    stale.rollout.reward.episode_reward = 1
    source = tmp_path / "teacher.jsonl"
    source.write_text(stale.model_dump_json() + "\n", encoding="utf-8")

    groups, sources = load_and_reverify([source])

    reverified = groups[task.task_id][0]
    assert reverified.score.successful is False
    assert reverified.score.gate_status == "task_failed"
    assert sources[0]["currently_failed"] == 1


def test_reverified_sft_split_is_scenario_stable_and_has_all_partitions():
    splits = {_scenario_split(f"scenario-{index}") for index in range(100)}

    assert splits == {"train", "validation", "test"}
    assert _scenario_split("scenario-42") == _scenario_split("scenario-42")
