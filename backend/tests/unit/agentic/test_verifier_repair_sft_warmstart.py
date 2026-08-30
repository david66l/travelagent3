import json
from pathlib import Path

from agentic.sft_dataset import SFTExample
from scripts.build_verifier_repair_sft_warmstart import build


SOURCE = Path("ml/agentic/datasets/native-react-verifier-repair-grpo-v1")


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_warmstart_builder_is_split_safe_and_balances_repair_with_replay(tmp_path):
    report = build(SOURCE, tmp_path)

    assert report["status"] == "passed"
    assert report["split_counts"] == {"train": 540, "validation": 96, "test": 90}
    assert report["source_state_counts"] == {"train": 60, "validation": 16, "test": 15}
    assert report["frozen_test_in_training"] is False
    assert report["unique_model_visible_payloads"] == 726
    assert report["action_counts"] == {
        "abort": 151,
        "get_poi_detail": 91,
        "propose_tradeoff": 151,
        "retrieve_city_knowledge": 91,
        "retry_solve": 151,
        "search_pois": 91,
    }


def test_warmstart_decisions_use_visible_verifier_evidence_and_full_schema(tmp_path):
    build(SOURCE, tmp_path)
    rows = [SFTExample(**row) for row in _read(tmp_path / "train.jsonl")]
    decisions = [row for row in rows if row.example_id.startswith("verifier-repair-decision:")]

    assert len(decisions) == 360
    for example in decisions:
        visible = json.loads(example.messages[-2].content or "{}")["policy_state"]
        report = next(
            item
            for item in reversed(visible["relevant_artifacts"])
            if item["artifact_type"] == "validation_report"
        )
        reason = example.messages[-1].tool_calls[0].function.arguments["reason"]
        assert reason == report["violations"][0]["message"]
        assert [tool["function"]["name"] for tool in example.tools] == visible[
            "allowed_actions"
        ]
