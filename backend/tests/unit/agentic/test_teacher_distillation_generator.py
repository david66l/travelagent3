"""Dependency-light checks for the teacher distillation generator."""

import importlib.util
import json
from pathlib import Path

import pytest

from agentic.corpus_generation import CurriculumTeacherPolicy, build_curriculum_case
from agentic.distillation import build_teacher_candidate
from agentic.environment import TravelAgentEnvironment
from agentic.grpo_training import GRPOCorpusRow
from agentic.policy import ControllerFirstPolicy
from agentic.sft_dataset import EpisodeCandidate, SFTDatasetBuilder


SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "generate_teacher_distillation.py"
SPEC = importlib.util.spec_from_file_location("generate_teacher_distillation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stratified_selection_and_holdout_contract(tmp_path: Path):
    rows = []
    for index in range(40):
        task, snapshot = build_curriculum_case(index)
        rows.append(GRPOCorpusRow(task=task, snapshot=snapshot))

    selected = MODULE.select_stratified(
        rows,
        tasks_per_family=1,
        offset_per_family=0,
    )

    assert {MODULE.task_family(row) for row in selected} == {
        "clarification",
        "recovery",
        "search",
        "tradeoff",
    }

    payload = {
        "case_id": "holdout-1",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [],
    }
    for split in ("regular", "hard", "adversarial"):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps(payload) + "\n",
            encoding="utf-8",
        )
    task_ids, hashes = MODULE.load_holdout_contract(tmp_path)

    assert task_ids == {"holdout-1"}
    assert len(hashes) == 1
    assert (
        MODULE.model_payload_hash(
            [
                {
                    "role": "user",
                    "content": "test",
                    "name": None,
                    "tool_calls": [],
                }
            ],
            [],
        )
        in hashes
    )


@pytest.mark.asyncio
async def test_verified_single_candidate_is_accepted_for_formal_scale():
    task, snapshot = build_curriculum_case(0)
    rollout = await TravelAgentEnvironment(task, snapshot).rollout(
        ControllerFirstPolicy(CurriculumTeacherPolicy())
    )
    candidate = build_teacher_candidate(rollout, family="search", sample_index=0)

    selection = MODULE.select_candidate_group([candidate])

    assert selection.chosen.score.successful is True
    assert selection.rejected == []
    assert selection.preference_pairs == []


@pytest.mark.asyncio
async def test_sft_projection_deduplicates_identical_model_payloads():
    task, snapshot = build_curriculum_case(0)
    environment = TravelAgentEnvironment(task, snapshot)
    first = await environment.rollout(ControllerFirstPolicy(CurriculumTeacherPolicy()))
    second = await environment.rollout(ControllerFirstPolicy(CurriculumTeacherPolicy()))
    raw = SFTDatasetBuilder().build(
        [
            EpisodeCandidate(
                scenario_id="scenario-first",
                source="teacher",
                template_family=task.template_family,
                city=str(task.slots["destination"]),
                episode=first.episode,
            ),
            EpisodeCandidate(
                scenario_id="scenario-second",
                source="teacher",
                template_family=task.template_family,
                city=str(task.slots["destination"]),
                episode=second.episode,
            ),
        ]
    )

    deduplicated, dropped, conflicts = MODULE.deduplicate_sft_result(raw)

    assert len(raw.examples) >= 2
    assert len(raw.examples) % 2 == 0
    assert len(deduplicated.examples) == len(raw.examples) // 2
    assert dropped == len(raw.examples) - len(deduplicated.examples)
    assert conflicts == []
    assert deduplicated.manifest.exported_examples == len(deduplicated.examples)

    conflicting = raw.model_copy(deep=True)
    prompt_groups = {}
    for example in conflicting.examples:
        prompt_hash = MODULE.model_payload_hash(
            [message.model_dump(mode="json") for message in example.messages[:-1]],
            example.tools,
        )
        prompt_groups.setdefault(prompt_hash, []).append(example)
    duplicate_group = next(group for group in prompt_groups.values() if len(group) > 1)
    duplicate_group[1].messages[-1].tool_calls[0].function.arguments = {
        "keywords": ["different-grounded-choice"]
    }
    quarantined, conflict_dropped, conflict_rows = MODULE.deduplicate_sft_result(conflicting)

    assert len(quarantined.examples) == len(deduplicated.examples) - 1
    assert conflict_dropped == dropped - (len(duplicate_group) - 1)
    assert len(conflict_rows) == 1
    assert len(conflict_rows[0]["responses"]) == 2
