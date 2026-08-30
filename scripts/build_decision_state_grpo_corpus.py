"""Build held-out, state-scoped GRPO rows from audited policy failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus, preflight_grpo_corpus  # noqa: E402
from agentic.grpo_training import to_trl_environment_rows  # noqa: E402
from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT  # noqa: E402
from agentic.trl_environment import build_trl_environment_factories  # noqa: E402

_TARGET_PATTERN = re.compile(r"arguments for ([a-z_]+):")
_SUPPORTED_TARGETS = {"get_poi_detail"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[GRPOCorpusRow]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def _decision_rows(
    source_file: Path,
    audit_rollouts: Path,
    *,
    split: str,
) -> list[GRPOCorpusRow]:
    source = {row.task.task_id: row for row in load_grpo_corpus(source_file)}
    selected: dict[str, GRPOCorpusRow] = {}
    for rollout in _read_jsonl(audit_rollouts):
        task_id = str(rollout.get("task_id") or "")
        source_row = source.get(task_id)
        if source_row is None:
            continue
        target = ""
        for error in rollout.get("policy_errors") or []:
            match = _TARGET_PATTERN.search(str(error.get("message") or ""))
            if match and match.group(1) in _SUPPORTED_TARGETS:
                target = match.group(1)
                break
        if not target:
            continue
        prefix_actions = [
            {
                "action": str(action.get("action") or ""),
                "arguments": dict(action.get("arguments") or {}),
            }
            for action in rollout.get("actions") or []
            if action.get("action")
        ]
        if not prefix_actions:
            continue
        signature_payload = json.dumps(
            {"task_id": task_id, "target": target, "prefix_actions": prefix_actions},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        signature = hashlib.sha256(signature_payload.encode()).hexdigest()[:16]
        if signature in selected:
            continue
        route = to_trl_environment_rows([source_row])[0]["environment"]
        environment = build_trl_environment_factories("react")[route](audit_enabled=False)
        prompt_messages: list[dict[str, Any]] = []
        try:
            initial = environment.reset(
                task=source_row.task.model_dump(mode="json"),
                snapshot=source_row.snapshot.model_dump(mode="json"),
            )
            prompt_messages = [
                {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": initial},
            ]
            for action in prefix_actions:
                prompt_messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": action["action"],
                                    "arguments": action["arguments"],
                                },
                            }
                        ],
                    }
                )
                result = environment._act(action["action"], action["arguments"])
                prompt_messages.append(
                    {"role": "tool", "name": action["action"], "content": result}
                )
        finally:
            environment.get_reward()
        hidden = dict(source_row.snapshot.hidden_test_facts)
        hidden["grpo_decision_state"] = {
            "schema_version": "react-decision-state.v1",
            "target_action": target,
            "prefix_actions": prefix_actions,
            "prompt_messages": prompt_messages,
            "source_task_id": task_id,
            "source_trajectory_id": rollout.get("trajectory_id"),
            "source_rollout_seed": rollout.get("rollout_seed"),
        }
        selected[signature] = GRPOCorpusRow(
            task=source_row.task.model_copy(
                update={"task_id": f"decision-{split}-{target}-{signature}"}
            ),
            snapshot=source_row.snapshot.model_copy(
                deep=True,
                update={
                    "snapshot_version": f"{source_row.snapshot.snapshot_version}-ds-{signature[:8]}",
                    "state_id": f"{source_row.snapshot.state_id}-ds-{signature[:8]}",
                    "hidden_test_facts": hidden,
                },
            ),
        )
    return [selected[key] for key in sorted(selected)]


def build(
    *,
    train_source: Path,
    train_audit: Path,
    validation_source: Path,
    validation_audit: Path,
    output_dir: Path,
) -> dict[str, Any]:
    train = _decision_rows(train_source, train_audit, split="train")
    validation = _decision_rows(validation_source, validation_audit, split="validation")
    if len(train) < 4 or len(validation) < 4:
        raise ValueError(
            f"decision-state corpus requires at least four rows per split: "
            f"train={len(train)}, validation={len(validation)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "validation.jsonl", validation)
    preflight = preflight_grpo_corpus(
        output_dir,
        minimum_train_tasks=4,
        require_dependencies=False,
    )
    if not preflight.ready:
        raise ValueError("decision-state corpus failed preflight: " + ",".join(preflight.errors))
    manifest = {
        "schema_version": "react-decision-state-corpus.v1",
        "objective": "state-scoped schema-valid tool decision",
        "target_actions": sorted(_SUPPORTED_TARGETS),
        "counts": {"train": len(train), "validation": len(validation)},
        "sources": {
            "train_corpus": str(train_source),
            "train_audit": str(train_audit),
            "validation_corpus": str(validation_source),
            "validation_audit": str(validation_audit),
        },
        "preflight": preflight.model_dump(mode="json"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--validation-source", type=Path, required=True)
    parser.add_argument("--validation-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
