import asyncio
import importlib.util
from pathlib import Path

from agentic.grpo_training import load_grpo_corpus


ROOT = Path(__file__).resolve().parents[4]
CURRICULUM_SCRIPT = ROOT / "scripts" / "build_stage3_decision_loop_curriculum_v3.py"
SFT_SCRIPT = ROOT / "scripts" / "build_stage3_decision_loop_sft_v3.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CURRICULUM = _load("stage3_decision_loop_curriculum_v3_for_sft", CURRICULUM_SCRIPT)
SFT = _load("stage3_decision_loop_sft_v3", SFT_SCRIPT)


def test_verified_sft_teacher_handles_change_and_same_argument_retry(tmp_path: Path):
    CURRICULUM.build(
        tmp_path,
        start_index=103000,
        train_count=32,
        validation_count=32,
        test_count=32,
    )
    rows = load_grpo_corpus(tmp_path / "train.jsonl")
    by_scenario = {}
    for row in rows:
        metadata = row.snapshot.hidden_test_facts["decision_loop_curriculum"]
        by_scenario.setdefault(metadata["scenario"], row)

    for scenario, row in by_scenario.items():
        example = asyncio.run(SFT._recovery_example(row, split="train"))
        first_call = example.messages[2].tool_calls[0].model_dump(mode="json")
        second_call = example.messages[4].tool_calls[0].model_dump(mode="json")
        first = first_call["function"]["arguments"]
        second = second_call["function"]["arguments"]
        if scenario == "change_arguments":
            assert first != second
        else:
            assert first == second
