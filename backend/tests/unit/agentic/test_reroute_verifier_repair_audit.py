import json
from pathlib import Path

from agentic.grpo_training import load_grpo_corpus
from scripts.build_verifier_repair_grpo_corpus import _TEMPLATES, _prepare_variant
from scripts.reroute_verifier_repair_audit import reroute


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_reward_replay_preserves_exact_gate_and_routes_partial_variance(tmp_path):
    source = load_grpo_corpus(
        Path("ml/agentic/datasets/native-react-grpo-v1/train.jsonl")
    )[0]
    row = _prepare_variant(
        source,
        split="validation",
        template=_TEMPLATES["validation"][0],
        ordinal=0,
    )
    contract = row.snapshot.hidden_test_facts["grpo_decision_state"]
    grounded = contract["grounding_phrases"][0]
    corpus_file = tmp_path / "corpus.jsonl"
    rollouts_file = tmp_path / "rollouts.jsonl"
    _write_jsonl(corpus_file, [row.model_dump(mode="json")])
    _write_jsonl(
        rollouts_file,
        [
            {
                "task_id": row.task.task_id,
                "sample_index": 0,
                "reward": 1.0,
                "actions": [
                    {
                        "action": "retry_solve",
                        "arguments": {"strategy": "greedy", "reason": grounded},
                    }
                ],
            },
            {
                "task_id": row.task.task_id,
                "sample_index": 1,
                "reward": -1.0,
                "actions": [
                    {
                        "action": "retry_solve",
                        "arguments": {"strategy": "greedy", "reason": "未引用证据"},
                    }
                ],
            },
            {
                "task_id": row.task.task_id,
                "sample_index": 2,
                "reward": -1.0,
                "actions": [{"action": "get_route_matrix", "arguments": {}}],
            },
            {
                "task_id": row.task.task_id,
                "sample_index": 3,
                "reward": -1.0,
                "actions": [],
            },
        ],
    )

    report = reroute(corpus_file, rollouts_file, tmp_path / "output")

    assert report["exact_successes_preserved"] == 1
    assert report["routes"] == {"grpo_update": 1}
    assert report["route_target_counts"]["grpo_update"] == {"retry_solve": 1}
    replayed = [
        json.loads(line)
        for line in (tmp_path / "output" / "replayed_rollouts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["reward"] for item in replayed] == [1.0, 0.5, -1.0, -1.0]
