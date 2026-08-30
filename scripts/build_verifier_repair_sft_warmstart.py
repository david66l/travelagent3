"""Build a split-safe SFT warm-start for verifier-repair GRPO.

The decision labels come only from the verifier-repair training corpus.  Each
split also receives one replay anchor per source state for the three preceding
production ReAct actions, which reduces catastrophic forgetting without
mixing in a second data distribution or touching the frozen hard benchmark.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402
from agentic.policy_actions import policy_action_schemas  # noqa: E402
from agentic.sft_dataset import DatasetManifest, SFTExample  # noqa: E402

SPLITS = ("train", "validation", "test")
SCHEMA_VERSION = "react-verifier-repair-sft-warmstart.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _transition(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("role") not in {"user", "tool"}:
        raise ValueError("policy transition must be a user or tool message")
    payload = json.loads(str(message.get("content") or "{}"))
    state = payload.get("policy_state")
    if not isinstance(state, dict):
        raise ValueError("policy transition is missing policy_state")
    return state


def _review_violation(messages: list[dict[str, Any]]) -> str:
    state = _transition(messages[-1])
    reports = [
        item
        for item in state.get("relevant_artifacts") or []
        if isinstance(item, dict) and item.get("artifact_type") == "validation_report"
    ]
    violations = (reports[-1].get("violations") or []) if reports else []
    message = str((violations[0] if violations else {}).get("message") or "").strip()
    if not message:
        raise ValueError("review state does not expose verifier grounding evidence")
    return message


def _assistant_call(action: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": action, "arguments": arguments},
            }
        ],
    }


def _decision_arguments(row: GRPOCorpusRow, messages: list[dict[str, Any]]) -> dict[str, Any]:
    contract = row.snapshot.hidden_test_facts.get("grpo_decision_state")
    if not isinstance(contract, dict):
        raise ValueError("verifier-repair decision contract is missing")
    action = str(contract.get("target_action") or "")
    reason = _review_violation(messages)
    if not any(str(phrase) in reason for phrase in contract.get("grounding_phrases") or []):
        raise ValueError("supervised reason is not grounded in visible verifier evidence")
    if action == "retry_solve":
        strategy = str((contract.get("expected_arguments") or {}).get("strategy") or "")
        if strategy not in {"cpsat", "greedy"}:
            raise ValueError("retry decision has no valid expected strategy")
        return {"strategy": strategy, "reason": reason}
    if action == "propose_tradeoff":
        options = [
            str(item).strip()
            for item in contract.get("supervised_options") or []
            if str(item).strip()
        ]
        if not options:
            raise ValueError("trade-off decision has no supervised options")
        return {"reason": reason, "options": options[:3]}
    if action == "abort":
        return {"reason": reason}
    raise ValueError(f"unsupported verifier-repair target: {action}")


def _decision_example(row: GRPOCorpusRow, split: str) -> SFTExample:
    contract = row.snapshot.hidden_test_facts["grpo_decision_state"]
    messages = copy.deepcopy(list(contract.get("prompt_messages") or []))
    state = _transition(messages[-1])
    allowed = list(state.get("allowed_actions") or [])
    action = str(contract.get("target_action") or "")
    if action not in allowed:
        raise ValueError("target action is not allowed by the visible production state")
    messages.append(_assistant_call(action, _decision_arguments(row, messages)))
    return SFTExample(
        example_id=f"verifier-repair-decision:{row.task.task_id}",
        scenario_id=row.task.task_id,
        trajectory_id=str(contract.get("source_task_id") or row.task.task_id),
        step_index=len(contract.get("prefix_actions") or []),
        split=split,
        quality_label=("validated_plan" if action == "retry_solve" else "safe_termination"),
        source="synthetic",
        environment_version=row.snapshot.environment_version,
        policy_name="verified-verifier-repair-teacher",
        policy_version=SCHEMA_VERSION,
        messages=messages,
        tools=policy_action_schemas(allowed),
    )


def _replay_examples(row: GRPOCorpusRow, split: str) -> list[SFTExample]:
    contract = row.snapshot.hidden_test_facts["grpo_decision_state"]
    history = copy.deepcopy(list(contract.get("prompt_messages") or []))
    prefix = list(contract.get("prefix_actions") or [])
    source_id = str(contract.get("source_task_id") or row.task.task_id)
    examples: list[SFTExample] = []
    for index, expected in enumerate(prefix):
        assistant_index = 2 + index * 2
        if assistant_index >= len(history):
            raise ValueError("verified prefix history is incomplete")
        assistant = history[assistant_index]
        calls = assistant.get("tool_calls") or []
        function = (calls[0] if calls else {}).get("function") or {}
        action = str(function.get("name") or "")
        if action != expected.get("action"):
            raise ValueError("verified prefix history and action contract disagree")
        prompt_messages = history[:assistant_index]
        state = _transition(prompt_messages[-1])
        allowed = list(state.get("allowed_actions") or [])
        if action not in allowed:
            raise ValueError("replay anchor action is not visible in its policy state")
        examples.append(
            SFTExample(
                example_id=f"verifier-repair-replay:{split}:{source_id}:{index}:{action}",
                scenario_id=f"verifier-repair-replay:{source_id}",
                trajectory_id=source_id,
                step_index=index,
                split=split,
                quality_label="validated_plan",
                source="synthetic",
                environment_version=row.snapshot.environment_version,
                policy_name="verified-prefix-replay",
                policy_version=SCHEMA_VERSION,
                messages=[*prompt_messages, assistant],
                tools=policy_action_schemas(allowed),
            )
        )
    return examples


def _split_examples(rows: list[GRPOCorpusRow], split: str) -> list[SFTExample]:
    decisions = [_decision_example(row, split) for row in rows]
    by_source: dict[str, GRPOCorpusRow] = {}
    for row in rows:
        source_id = str(
            row.snapshot.hidden_test_facts["grpo_decision_state"].get("source_task_id")
            or row.task.task_id
        )
        current = by_source.get(source_id)
        if current is None or row.task.task_id < current.task.task_id:
            by_source[source_id] = row
    anchors = [
        example
        for source_id in sorted(by_source)
        for example in _replay_examples(by_source[source_id], split)
    ]
    return sorted([*decisions, *anchors], key=lambda item: item.example_id)


def _write_jsonl(path: Path, rows: list[SFTExample]) -> None:
    path.write_text(
        "".join(item.model_dump_json() + "\n" for item in rows),
        encoding="utf-8",
    )


def build(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    corpora = {
        split: load_grpo_corpus(source_dir / f"{split}.jsonl") for split in SPLITS
    }
    examples = {
        split: _split_examples(corpora[split], split) for split in SPLITS
    }
    source_ids = {
        split: {
            str(row.snapshot.hidden_test_facts["grpo_decision_state"]["source_task_id"])
            for row in rows
        }
        for split, rows in corpora.items()
    }
    errors: list[str] = []
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if source_ids[left] & source_ids[right]:
            errors.append(f"SOURCE_STATE_OVERLAP:{left}:{right}")

    visible_hashes: set[str] = set()
    duplicate_payloads = 0
    action_counts: Counter[str] = Counter()
    for split in SPLITS:
        for example in examples[split]:
            payload = {
                "messages": [item.model_dump(mode="json") for item in example.messages],
                "tools": example.tools,
            }
            digest = _digest(payload)
            if digest in visible_hashes:
                duplicate_payloads += 1
            visible_hashes.add(digest)
            action_counts[example.messages[-1].tool_calls[0].function.name] += 1
    if duplicate_payloads:
        errors.append(f"MODEL_VISIBLE_PAYLOAD_DUPLICATES:{duplicate_payloads}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        _write_jsonl(output_dir / f"{split}.jsonl", examples[split])
    source_hashes = {
        split: _sha256(source_dir / f"{split}.jsonl") for split in SPLITS
    }
    dataset_version = "verifier-repair-sft-" + _digest(
        {
            "source_hashes": source_hashes,
            "example_ids": [
                item.example_id for split in SPLITS for item in examples[split]
            ],
        }
    )[:16]
    quality_counts = Counter(
        item.quality_label for split in SPLITS for item in examples[split]
    )
    manifest = DatasetManifest(
        dataset_version=dataset_version,
        created_at=datetime.now(UTC),
        candidate_episodes=sum(len(rows) for rows in examples.values()),
        accepted_episodes=sum(len(rows) for rows in examples.values()),
        rejected_episodes=0,
        exported_examples=sum(len(rows) for rows in examples.values()),
        split_examples={split: len(examples[split]) for split in SPLITS},
        source_episodes={"synthetic": sum(len(rows) for rows in examples.values())},
        quality_episodes=dict(quality_counts),
        rejection_codes={},
        environment_versions=sorted(
            {item.environment_version for split in SPLITS for item in examples[split]}
        ),
        policy_versions=[SCHEMA_VERSION],
        split_group_overlap=bool(errors),
    )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not errors else "rejected",
        "dataset_version": dataset_version,
        "source_dir": source_dir.as_posix(),
        "source_split_sha256": source_hashes,
        "split_counts": manifest.split_examples,
        "source_state_counts": {split: len(source_ids[split]) for split in SPLITS},
        "action_counts": dict(sorted(action_counts.items())),
        "unique_model_visible_payloads": len(visible_hashes),
        "frozen_test_in_training": bool(source_ids["train"] & source_ids["test"]),
        "data_scope": (
            "verifier-repair corpus only; train labels use train split only; "
            "validation and frozen test are evaluation-only"
        ),
        "errors": errors,
    }
    (output_dir / "derivation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.source_dir, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
