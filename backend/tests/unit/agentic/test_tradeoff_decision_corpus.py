from agentic.corpus_generation import build_curriculum_case
from agentic.grpo_training import GRPOCorpusRow
from scripts.build_tradeoff_decision_corpus import derive_tradeoff_decision, is_eligible


def _row(index: int) -> GRPOCorpusRow:
    task, snapshot = build_curriculum_case(index)
    return GRPOCorpusRow(task=task, snapshot=snapshot)


def test_tradeoff_decision_is_explicit_and_preserves_grounded_reason():
    source = _row(8)

    derived = derive_tradeoff_decision(source)

    assert derived.task.task_id.endswith("-tradeoff-decision")
    assert "不要继续生成" in derived.task.user_request
    assert derived.snapshot.hidden_test_facts["tradeoff_decision"] == {
        "expected_action": "propose_tradeoff",
        "source_task_id": source.task.task_id,
        "reasons": ["预算不足以覆盖指定天数"],
    }


def test_tradeoff_decision_rejects_feasible_task():
    assert is_eligible(_row(0)) is False
