"""Build tradeoff repair SFT data from the exact production GRPO state contract."""

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
from agentic.trl_environment import TRL_ENVIRONMENT_FACTORIES  # noqa: E402


def task_family(row: GRPOCorpusRow) -> str:
    """Classify a snapshot task using the same routing precedence as GRPO."""
    if row.task.missing_slots:
        return "clarification"
    if row.task.feasibility_report.get("feasible", True) is False:
        return "tradeoff"
    search = row.snapshot.tool_responses.get("search_pois") or []
    if search and (search[0].error_code or search[0].data_source == "unavailable"):
        return "recovery"
    return "search"


def _stable(rows: list[GRPOCorpusRow]) -> list[GRPOCorpusRow]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(row.task.task_id.encode()).hexdigest(),
    )


def _split(rows: list[GRPOCorpusRow]) -> dict[str, list[GRPOCorpusRow]]:
    result: dict[str, list[GRPOCorpusRow]] = defaultdict(list)
    for index, row in enumerate(_stable(rows)):
        bucket = index % 20
        split = "validation" if bucket < 3 else ("test" if bucket < 6 else "train")
        result[split].append(row)
    return result


def _teacher_action(row: GRPOCorpusRow) -> tuple[str, list[str], dict[str, Any]]:
    family = task_family(row)
    if family == "tradeoff":
        reasons = list(row.task.feasibility_report.get("reasons") or [])
        reason = str(reasons[0] if reasons else "当前约束不可同时满足")
        options = (
            ["提高预算", "缩短行程天数", "降低住宿或活动消费标准"]
            if "预算" in reason
            else ["放宽当前约束", "调整行程要求"]
        )
        return (
            "tradeoff",
            ["propose_tradeoff", "abort"],
            {"reason": reason, "options": options},
        )
    if family == "clarification":
        missing = str(row.task.missing_slots[0])
        questions = {
            "destination": "请问你想去哪个城市？",
            "travel_days": "请问计划游玩几天？",
            "start_date": "请问计划哪天出发？",
            "end_date": "请问计划哪天返程？",
            "budget_range": "请问这次旅行的预算大约是多少？",
        }
        return (
            "clarification",
            ["ask_user"],
            {"question": questions.get(missing, f"请补充{missing}。")},
        )
    return (
        "search",
        ["search_pois"],
        {"keywords": list(row.task.profile.get("interests") or [])},
    )


def aligned_example(row: GRPOCorpusRow, *, split: str, kind: str) -> SFTExample:
    """Render one example through the actual TRL environment reset contract."""
    route, allowed_actions, arguments = _teacher_action(row)
    action = "propose_tradeoff" if route == "tradeoff" else (
        "ask_user" if route == "clarification" else "search_pois"
    )
    environment = TRL_ENVIRONMENT_FACTORIES[route](audit_enabled=False)
    reset_succeeded = False
    try:
        initial = environment.reset(
            task=row.task.model_dump(mode="json"),
            snapshot=row.snapshot.model_dump(mode="json"),
        )
        reset_succeeded = True
    finally:
        if reset_succeeded:
            environment.get_reward()
    digest = hashlib.sha256(
        f"{row.task.task_id}:{route}:{action}".encode()
    ).hexdigest()[:16]
    return SFTExample(
        example_id=f"aligned-tradeoff-repair:{kind}:{row.task.task_id}:0",
        scenario_id=row.task.task_id,
        trajectory_id=f"aligned-{digest}",
        step_index=0,
        split=split,  # type: ignore[arg-type]
        quality_label="safe_termination" if route == "tradeoff" else (
            "clarification" if route == "clarification" else "validated_plan"
        ),
        source="synthetic",
        environment_version=row.snapshot.environment_version,
        policy_name="ProductionAlignedTeacher",
        policy_version="trl-reset-v1",
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


def build(source_file: Path, output_dir: Path) -> DatasetManifest:
    rows = load_grpo_corpus(source_file)
    target = [row for row in rows if task_family(row) == "tradeoff"]
    if not target:
        raise ValueError("official GRPO train split contains no tradeoff tasks")
    target_splits = _split(target)

    protected = {row.task.task_id for row in target}
    replay_candidates = [
        row
        for row in rows
        if row.task.task_id not in protected
        and task_family(row) in {"clarification", "search", "recovery"}
    ]
    replay_splits_all = _split(replay_candidates)

    output: dict[str, list[SFTExample]] = {}
    for split in ("train", "validation", "test"):
        targets = target_splits[split]
        # A 1:1 replay mix protects the already-qualified clarification/search
        # policy while keeping the repair action at half of the full dataset.
        replay = _stable(replay_splits_all[split])[: len(targets)]
        if len(replay) != len(targets):
            raise ValueError(f"insufficient aligned replay examples for {split}")
        examples = [
            aligned_example(row, split=split, kind="tradeoff") for row in targets
        ] + [aligned_example(row, split=split, kind="replay") for row in replay]
        output[split] = sorted(examples, key=lambda item: item.example_id)

    scenario_sets = {split: {item.scenario_id for item in part} for split, part in output.items()}
    if any(
        scenario_sets[left] & scenario_sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("aligned repair split leakage detected")

    version = "sft-aligned-tradeoff-repair-" + hashlib.sha256(
        json.dumps(
            {split: [item.example_id for item in part] for split, part in output.items()},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    all_examples = [item for part in output.values() for item in part]
    manifest = DatasetManifest(
        dataset_version=version,
        candidate_episodes=len(all_examples),
        accepted_episodes=len(all_examples),
        rejected_episodes=0,
        exported_examples=len(all_examples),
        split_examples={split: len(part) for split, part in output.items()},
        source_episodes=dict(Counter(item.source for item in all_examples)),
        quality_episodes=dict(Counter(item.quality_label for item in all_examples)),
        rejection_codes={},
        environment_versions=sorted({item.environment_version for item in all_examples}),
        policy_versions=sorted(
            {f"{item.policy_name}:{item.policy_version}" for item in all_examples}
        ),
        split_group_overlap=False,
        excluded_policy_steps=0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, examples in output.items():
        (output_dir / f"{split}.jsonl").write_text(
            "\n".join(item.model_dump_json() for item in examples) + "\n",
            encoding="utf-8",
        )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "derivation.json").write_text(
        json.dumps(
            {
                "source_split": "official GRPO train only",
                "official_validation_used": False,
                "target_family": "tradeoff",
                "target_examples": len(target),
                "replay_ratio": "1:1",
                "prompt_contract": "TRL_ENVIRONMENT_FACTORIES.reset",
                "tool_contract": "route-specific production allowed actions",
                "target_failure": "out-of-contract capability_check generation",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source_file, args.output_dir).model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
