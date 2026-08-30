import json
from collections import Counter

import pytest

from agentic.corpus_generation import (
    AdaptiveRecoveryTeacherPolicy,
    build_curriculum_case,
)
from agentic.environment import TravelAgentEnvironment
from agentic.grpo_training import GRPOCorpusRow
from agentic.policy import ControllerFirstPolicy
from agentic.sft_dataset import EpisodeCandidate
from scripts.build_adaptive_recovery_corpus import derive_adaptive_recovery
from scripts.build_adaptive_recovery_sft_dataset import _audit_adaptive_example
from scripts.build_adaptive_recovery_sft_dataset import _assert_unique_model_visible_payloads
from scripts.build_adaptive_recovery_sft_dataset import _balanced_replay_partition
from scripts.build_adaptive_recovery_sft_dataset import _partition
from scripts.build_adaptive_recovery_sft_dataset import _replay_family
from scripts.build_multiturn_recovery_dataset import build_multiturn_example


async def test_adaptive_teacher_uses_visible_failure_to_change_strategy():
    task, snapshot = build_curriculum_case(7)
    derived = derive_adaptive_recovery(GRPOCorpusRow(task=task, snapshot=snapshot))
    rollout = await TravelAgentEnvironment(derived.task, derived.snapshot).rollout(
        ControllerFirstPolicy(AdaptiveRecoveryTeacherPolicy())
    )
    candidate = EpisodeCandidate(
        scenario_id=derived.task.task_id,
        source="synthetic",
        template_family=derived.task.template_family,
        city=str(derived.task.slots["destination"]),
        episode=rollout.episode,
    )

    example = build_multiturn_example(
        candidate,
        split="train",
        expected_error="QUERY_TOO_BROAD",
        require_adaptation=True,
        example_prefix="adaptive-recovery",
    )
    audit = _audit_adaptive_example(example)

    first = example.messages[2].tool_calls[0].function.arguments
    second = example.messages[4].tool_calls[0].function.arguments
    transition = json.loads(example.messages[3].content)
    assert first != second
    assert second == {"keywords": [task.slots["interests"][-1]]}
    assert transition["policy_state"]["failure_summary"][-1]["code"] == "QUERY_TOO_BROAD"
    assert audit["evidence_source"] == "policy_state.failure_summary.message"
    assert "hidden_test_facts" not in example.model_dump_json()

    duplicate = example.model_copy(update={"example_id": "different-id"}, deep=True)
    with pytest.raises(ValueError, match="MODEL_VISIBLE_PAYLOAD_DUPLICATE:1"):
        _assert_unique_model_visible_payloads(
            {"train": [example], "validation": [duplicate], "test": []}
        )


def test_partition_can_allocate_extra_replay_without_source_overlap():
    rows = []
    for index in range(20):
        task, snapshot = build_curriculum_case(index * 10 + 7)
        rows.append(GRPOCorpusRow(task=task, snapshot=snapshot))

    splits = _partition(rows, {"train": 9, "validation": 3, "test": 3})
    ids = {split: {row.task.task_id for row in items} for split, items in splits.items()}

    assert {split: len(items) for split, items in splits.items()} == {
        "train": 9,
        "validation": 3,
        "test": 3,
    }
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]


def test_balanced_replay_partition_preserves_all_decision_families():
    rows = []
    for family_index, family in enumerate(("clarification", "tradeoff", "search")):
        for index in range(12):
            task, snapshot = build_curriculum_case((family_index * 20 + index) * 10 + 7)
            task.task_id = f"{family}-{index}"
            task.missing_slots = ["budget"] if family == "clarification" else []
            task.feasibility_report = {"feasible": family != "tradeoff"}
            rows.append(GRPOCorpusRow(task=task, snapshot=snapshot))

    splits = _balanced_replay_partition(
        rows,
        {"train": 3, "validation": 3, "test": 3},
        replay_ratio=2,
    )
    ids = {split: {row.task.task_id for row in items} for split, items in splits.items()}

    assert all(len(items) == 6 for items in splits.values())
    assert all(
        Counter(_replay_family(row) for row in items)
        == {"clarification": 2, "tradeoff": 2, "search": 2}
        for items in splits.values()
    )
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]
