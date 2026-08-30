import importlib.util
import json
from pathlib import Path

from agentic.grpo_training import load_grpo_corpus, preflight_grpo_corpus
from agentic.corpus_generation import build_curriculum_case
from agentic.grpo_training import GRPOCorpusRow
from agentic.trl_environment import TRLSearchEnvironment


SCRIPT = (
    Path(__file__).resolve().parents[4] / "scripts" / "build_stage3_decision_loop_curriculum.py"
)
SPEC = importlib.util.spec_from_file_location("stage3_decision_loop_curriculum", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_builds_balanced_semantic_recovery_scenarios(tmp_path: Path):
    manifest = MODULE.build(
        tmp_path,
        start_index=71000,
        train_count=8,
        validation_count=4,
        test_count=4,
    )

    assert manifest["counts"] == {"train": 8, "validation": 4, "test": 4}
    assert manifest["scenario_counts"]["train"] == {
        "change_arguments": 4,
        "retry_same_arguments": 4,
    }
    assert manifest["evidence_style_counts"]["train"] == {
        "explicit_instruction": 4,
        "diagnostic_evidence": 4,
    }
    rows = load_grpo_corpus(tmp_path / "train.jsonl")
    for row in rows:
        metadata = row.snapshot.hidden_test_facts["decision_loop_curriculum"]
        first, second = row.snapshot.tool_responses["search_pois"]
        assert first.retryable is True
        if metadata["scenario"] == "change_arguments":
            assert first.error_code == "QUERY_TOO_BROAD"
            assert first.expected_arguments != second.expected_arguments
            assert metadata["requires_argument_change"] is True
        else:
            assert first.error_code == "UPSTREAM_TIMEOUT"
            assert first.expected_arguments == second.expected_arguments
            assert metadata["requires_argument_change"] is False


def test_decision_loop_curriculum_is_grpo_preflight_compatible(tmp_path: Path):
    MODULE.build(
        tmp_path,
        start_index=72000,
        train_count=8,
        validation_count=4,
        test_count=4,
    )

    report = preflight_grpo_corpus(
        tmp_path,
        minimum_train_tasks=8,
        require_dependencies=False,
    )

    assert report.errors == []
    assert report.train_tasks == 8
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_test_message_overlap"] is False


def test_snapshot_contract_requires_different_recovery_for_each_error():
    task, snapshot = build_curriculum_case(73007)
    source = GRPOCorpusRow(task=task, snapshot=snapshot)

    for ordinal, scenario in enumerate(("change_arguments", "retry_same_arguments")):
        row = MODULE.derive_decision_loop_case(
            source,
            ordinal=ordinal,
            scenario=scenario,
            evidence_style="diagnostic_evidence",
            change_message_template="删除“{drop}”，仅保留“{target}”。",
            timeout_message="服务暂时超时，请保持相同关键词重试。",
        )
        metadata = row.snapshot.hidden_test_facts["decision_loop_curriculum"]
        environment = TRLSearchEnvironment()
        environment.reset(
            task=row.task.model_dump(mode="json"),
            snapshot=row.snapshot.model_dump(mode="json"),
        )

        first = json.loads(environment.search_pois(metadata["initial_keywords"]))
        second = json.loads(environment.search_pois(metadata["expected_recovery_keywords"]))

        assert (
            first["last_transition"]["verification"]["error_code"] == metadata["first_error_code"]
        )
        assert second["last_transition"]["verification"]["error_code"] is None
        environment.get_reward()
