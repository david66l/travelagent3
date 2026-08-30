"""Priority-search derivation must remain grounded and split-safe."""

import importlib.util
from pathlib import Path

from agentic.corpus_generation import build_curriculum_case
from agentic.grpo_training import GRPOCorpusRow


SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "build_priority_search_corpus.py"
SPEC = importlib.util.spec_from_file_location("build_priority_search_corpus", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(index: int) -> GRPOCorpusRow:
    task, snapshot = build_curriculum_case(index)
    return GRPOCorpusRow(task=task, snapshot=snapshot)


def test_priority_search_derivation_uses_visible_grounded_interest():
    source = _row(0)
    target = source.task.profile["interests"][-1]

    derived = MODULE.derive_priority_search(source)

    assert derived.task.task_id.endswith("-priority-search")
    assert target in derived.task.user_request
    assert derived.snapshot.tool_responses["search_pois"][0].expected_arguments == {
        "keywords": [target]
    }
    assert derived.snapshot.hidden_test_facts["priority_search"]["target_keywords"] == [target]
    assert source.snapshot.tool_responses["search_pois"][0].expected_arguments == {}


def test_priority_search_rejects_missing_information_case():
    with_missing = _row(6)

    assert MODULE.is_eligible(with_missing) is False


def test_priority_search_can_target_first_interest_without_colliding_with_last():
    source = _row(0)
    target = source.task.profile["interests"][0]

    derived = MODULE.derive_priority_search(source, target_position="first")

    assert derived.task.task_id.endswith("-priority-search-first")
    assert target in derived.task.user_request
    assert derived.snapshot.tool_responses["search_pois"][0].expected_arguments == {
        "keywords": [target]
    }
    assert derived.snapshot.hidden_test_facts["priority_search"] == {
        "target_keywords": [target],
        "target_position": "first",
        "source_task_id": source.task.task_id,
    }
