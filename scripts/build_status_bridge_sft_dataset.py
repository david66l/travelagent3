"""Build verifier-labelled SFT bridge data before status-balanced GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402
from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT  # noqa: E402
from agentic.policy_actions import policy_action_schemas  # noqa: E402
from agentic.sft_dataset import (  # noqa: E402
    DatasetManifest,
    SFTExample,
    SFTMessage,
    SFTToolCall,
    SFTToolFunction,
)
from agentic.training import load_jsonl, select_sft_smoke_rows  # noqa: E402
from agentic.trl_environment import build_trl_environment_factories  # noqa: E402


DEFAULT_TRAIN_PAIR_QUOTAS = {
    "infeasible": 32,
    "missing_tool": 32,
    "unsafe": 120,
}
DEFAULT_EVAL_PAIR_QUOTAS = {
    "infeasible": 6,
    "missing_tool": 6,
    "unsafe": 12,
}


def _stable_key(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()


def _boundary_metadata(row: GRPOCorpusRow) -> dict[str, Any] | None:
    metadata = row.snapshot.hidden_test_facts.get("decision_boundary_training")
    return metadata if isinstance(metadata, dict) else None


def _boundary_visible_signature(row: GRPOCorpusRow) -> str:
    payload = {
        "user_request": row.task.user_request,
        "slots": row.task.slots,
        "profile": row.task.profile,
        "missing_slots": row.task.missing_slots,
        "feasibility_report": row.task.feasibility_report,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _select_pairs(
    rows: list[GRPOCorpusRow],
    quotas: dict[str, int],
    *,
    salt: str,
    forbidden_visible_signatures: set[str] | None = None,
) -> list[GRPOCorpusRow]:
    pairs: dict[tuple[str, str], list[GRPOCorpusRow]] = defaultdict(list)
    for row in rows:
        metadata = _boundary_metadata(row)
        if metadata is None:
            continue
        kind = str(metadata.get("boundary_kind") or "")
        pair_id = str(metadata.get("pair_id") or "")
        if kind in quotas and pair_id:
            pairs[(kind, pair_id)].append(row)

    selected: list[GRPOCorpusRow] = []
    for kind, quota in quotas.items():
        candidates = [
            (pair_id, siblings)
            for (candidate_kind, pair_id), siblings in pairs.items()
            if candidate_kind == kind
            and not (
                forbidden_visible_signatures
                and any(
                    _boundary_visible_signature(row) in forbidden_visible_signatures
                    for row in siblings
                )
            )
            and {
                str((_boundary_metadata(row) or {}).get("expected_action"))
                for row in siblings
            }
            == {"abort", "propose_tradeoff"}
        ]
        candidates.sort(key=lambda item: _stable_key(item[0], f"{salt}:{kind}"))
        if len(candidates) < quota:
            raise ValueError(f"insufficient {kind} boundary pairs: {len(candidates)}<{quota}")
        for _, siblings in candidates[:quota]:
            selected.extend(siblings)
    return sorted(selected, key=lambda row: _stable_key(row.task.task_id, salt))


def _split_validation_pairs(
    rows: list[GRPOCorpusRow],
    quotas: dict[str, int],
) -> tuple[list[GRPOCorpusRow], list[GRPOCorpusRow]]:
    validation: list[GRPOCorpusRow] = []
    test: list[GRPOCorpusRow] = []
    for kind, quota in quotas.items():
        double_quota = {kind: quota * 2}
        selected = _select_pairs(rows, double_quota, salt=f"eval:{kind}")
        pair_ids = sorted(
            {
                str((_boundary_metadata(row) or {}).get("pair_id"))
                for row in selected
            },
            key=lambda pair_id: _stable_key(pair_id, f"eval-split:{kind}"),
        )
        validation_ids = set(pair_ids[:quota])
        for row in selected:
            pair_id = str((_boundary_metadata(row) or {}).get("pair_id"))
            (validation if pair_id in validation_ids else test).append(row)
    return validation, test


def _target(row: GRPOCorpusRow) -> tuple[str, dict[str, Any]]:
    metadata = _boundary_metadata(row)
    if metadata is None:
        raise ValueError(f"missing boundary metadata: {row.task.task_id}")
    action = str(metadata.get("expected_action") or "")
    report = row.task.feasibility_report
    reasons = [str(item) for item in (report.get("reasons") or [])]
    if not reasons:
        raise ValueError(f"missing grounded reason: {row.task.task_id}")
    if action == "abort":
        return action, {"reason": reasons[0]}
    if action == "propose_tradeoff":
        options = [str(item) for item in (report.get("alternatives") or [])][:3]
        if not options:
            raise ValueError(f"missing grounded alternatives: {row.task.task_id}")
        return action, {"reason": reasons[0], "options": options}
    raise ValueError(f"unsupported boundary action {action}: {row.task.task_id}")


def _verified_example(row: GRPOCorpusRow, split: str) -> SFTExample:
    action, arguments = _target(row)
    environment = build_trl_environment_factories("policy_driven")["tradeoff"]()
    initial = environment.reset(
        task=row.task.model_dump(mode="json"),
        snapshot=row.snapshot.model_dump(mode="json"),
        prompt=[
            {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": row.task.user_request},
        ],
    )
    policy_state = json.loads(initial)["policy_state"]
    result = getattr(environment, action)(**arguments)
    transition = json.loads(result)
    scalar_reward = environment.get_reward()
    rollout = environment.rollout_record
    if transition.get("done") is not True:
        raise ValueError(f"boundary target did not terminate: {row.task.task_id}")
    if rollout is None:
        raise ValueError(f"boundary target produced no rollout: {row.task.task_id}")
    reward = rollout.reward
    if (
        reward.gate_status != "passed"
        or reward.episode_reward <= 0
        or scalar_reward <= 0
    ):
        raise ValueError(
            f"boundary target failed verification: {row.task.task_id}:{reward.gate_status}"
        )

    allowed_actions = list(policy_state.get("allowed_actions") or [])
    return SFTExample(
        example_id=f"status-bridge:{row.task.task_id}:0",
        scenario_id=row.task.task_id,
        trajectory_id=f"status-bridge:{row.task.task_id}",
        step_index=0,
        split=split,
        quality_label="safe_termination",
        source="synthetic",
        environment_version=row.snapshot.environment_version,
        policy_name="VerifierLabelPolicy",
        policy_version="status-bridge.v1",
        messages=[
            SFTMessage(role="system", content=AGENT_TOOL_POLICY_SYSTEM_PROMPT),
            SFTMessage(role="user", content=initial),
            SFTMessage(
                role="assistant",
                tool_calls=[
                    SFTToolCall(
                        function=SFTToolFunction(name=action, arguments=arguments)
                    )
                ],
            ),
        ],
        tools=policy_action_schemas(allowed_actions),
    )


def _action(example: SFTExample) -> str:
    return example.messages[-1].tool_calls[0].function.name


def _select_replay(path: Path, limit: int) -> list[SFTExample]:
    rows = [
        row
        for row in load_jsonl(path)
        if _action(SFTExample(**row)) in {"ask_user", "search_pois"}
    ]
    return [SFTExample(**row) for row in select_sft_smoke_rows(rows, limit)]


def _write_jsonl(path: Path, rows: list[SFTExample]) -> None:
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows),
        encoding="utf-8",
    )


def _model_visible_hash(example: SFTExample) -> str:
    payload = {
        "messages": [
            message.model_dump(mode="json") for message in example.messages[:-1]
        ],
        "tools": example.tools,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def build(
    grpo_dir: Path,
    replay_dir: Path,
    output_dir: Path,
    *,
    train_pair_quotas: dict[str, int] | None = None,
    eval_pair_quotas: dict[str, int] | None = None,
    train_replay: int = 256,
    eval_replay: int = 24,
) -> dict[str, Any]:
    train_quotas = train_pair_quotas or DEFAULT_TRAIN_PAIR_QUOTAS
    eval_quotas = eval_pair_quotas or DEFAULT_EVAL_PAIR_QUOTAS
    grpo_train = load_grpo_corpus(grpo_dir / "train.jsonl")
    grpo_validation = load_grpo_corpus(grpo_dir / "validation.jsonl")
    validation_signatures = {
        _boundary_visible_signature(row)
        for row in grpo_validation
        if _boundary_metadata(row) is not None
    }
    selected_train = _select_pairs(
        grpo_train,
        train_quotas,
        salt="train",
        forbidden_visible_signatures=validation_signatures,
    )
    selected_validation, selected_test = _split_validation_pairs(
        grpo_validation, eval_quotas
    )

    boundary_rows = {
        "train": selected_train,
        "validation": selected_validation,
        "test": selected_test,
    }
    replay_limits = {
        "train": train_replay,
        "validation": eval_replay,
        "test": eval_replay,
    }
    output: dict[str, list[SFTExample]] = {}
    for split, rows in boundary_rows.items():
        examples = [_verified_example(row, split) for row in rows]
        examples.extend(_select_replay(replay_dir / f"{split}.jsonl", replay_limits[split]))
        output[split] = sorted(
            examples,
            key=lambda example: _stable_key(example.example_id, f"output:{split}"),
        )

    all_rows = [row for rows in output.values() for row in rows]
    ids = [row.example_id for row in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate SFT example IDs")
    visible_hashes = [_model_visible_hash(row) for row in all_rows]
    if len(visible_hashes) != len(set(visible_hashes)):
        raise ValueError(
            "duplicate model-visible SFT prompts: "
            f"{len(visible_hashes) - len(set(visible_hashes))}"
        )
    scenario_splits: dict[str, set[str]] = defaultdict(set)
    for split, rows in output.items():
        for row in rows:
            scenario_splits[row.scenario_id].add(split)
    overlap = {key: value for key, value in scenario_splits.items() if len(value) > 1}
    if overlap:
        raise ValueError(f"scenario split overlap: {len(overlap)}")

    digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]
    manifest = DatasetManifest(
        dataset_version=f"status-bridge-sft-{digest}",
        candidate_episodes=len(all_rows),
        accepted_episodes=len(all_rows),
        rejected_episodes=0,
        exported_examples=len(all_rows),
        split_examples={split: len(rows) for split, rows in output.items()},
        source_episodes=dict(Counter(row.source for row in all_rows)),
        quality_episodes=dict(Counter(row.quality_label for row in all_rows)),
        rejection_codes={},
        environment_versions=sorted({row.environment_version for row in all_rows}),
        policy_versions=sorted(
            {f"{row.policy_name}:{row.policy_version}" for row in all_rows}
        ),
        split_group_overlap=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in output.items():
        _write_jsonl(output_dir / f"{split}.jsonl", rows)
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": "status-bridge-sft-derivation.v1",
        "dataset_version": manifest.dataset_version,
        "objective": "move unsafe necessary-abort into GRPO sampling support",
        "grpo_dir": str(grpo_dir),
        "replay_dir": str(replay_dir),
        "train_pair_quotas": train_quotas,
        "eval_pair_quotas_per_split": eval_quotas,
        "replay_limits": replay_limits,
        "split_counts": manifest.split_examples,
        "action_counts": dict(Counter(_action(row) for row in all_rows)),
        "boundary_cells": dict(
            Counter(
                f"{metadata['boundary_kind']}/{metadata['variant']}"
                for rows in boundary_rows.values()
                for row in rows
                if (metadata := _boundary_metadata(row)) is not None
            )
        ),
        "verified_boundary_examples": sum(len(rows) for rows in boundary_rows.values()),
        "unique_model_visible_prompts": len(set(visible_hashes)),
        "scenario_split_overlap": 0,
    }
    (output_dir / "derivation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grpo-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.grpo_dir, args.replay_dir, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
