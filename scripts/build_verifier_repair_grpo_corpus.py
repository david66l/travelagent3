"""Build a split-safe GRPO corpus for ReAct verifier-repair decisions.

The corpus is derived only from previously accepted Native ReAct train and
validation snapshots. It never reads the independent hard benchmark. Each row
replays a verified research/solver prefix and exposes one production
``review_itinerary`` decision: retry a repairable solve, propose an actionable
trade-off, or abort when the user forbids every safe alternative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import (  # noqa: E402
    GRPOCorpusRow,
    load_grpo_corpus,
    preflight_grpo_corpus,
)
from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT  # noqa: E402
from agentic.trl_environment import TRLReactEnvironment  # noqa: E402


SCHEMA_VERSION = "react-verifier-repair-corpus.v1"
DECISION_SCHEMA_VERSION = "react-verifier-repair-decision.v1"
PREFIX_ACTIONS = (
    ("retrieve_city_knowledge", {}),
    ("search_pois", None),
    ("get_poi_detail", {}),
)

_TEMPLATES: dict[str, tuple[dict[str, Any], ...]] = {
    "train": (
        {
            "family": "repairable_solver_failure",
            "target_action": "retry_solve",
            "request_suffix": " 若校验只发现求解策略问题，请先更换策略重算一次，不要直接放弃。",
            "violation": "求解策略未满足晚间固定时窗；允许使用贪心策略进行一次有界重算",
            "grounding_phrases": ["固定时窗", "有界重算"],
            "expected_arguments": {"strategy": "greedy"},
        },
        {
            "family": "repairable_solver_failure",
            "target_action": "retry_solve",
            "request_suffix": " 如果只是排程器卡在当前策略，可切换一次算法重新排程。",
            "violation": "当前CP-SAT排程与预约时间冲突；替换为greedy策略仍可能修复",
            "grounding_phrases": ["预约时间冲突", "greedy"],
            "expected_arguments": {"strategy": "greedy"},
        },
        {
            "family": "actionable_tradeoff",
            "target_action": "propose_tradeoff",
            "request_suffix": " 若预算与景点数量冲突，我接受提高预算或减少一个景点，请先让我选择。",
            "violation": "当前行程超出总预算300元，不能直接通过硬约束校验",
            "grounding_phrases": ["超出总预算300元", "硬约束"],
            "require_options": True,
            "supervised_options": ["提高总预算", "减少一个非必去景点"],
        },
        {
            "family": "actionable_tradeoff",
            "target_action": "propose_tradeoff",
            "request_suffix": " 若固定活动和交通时间无法同时满足，可以给我改时间或删活动的选项。",
            "violation": "固定活动与末班交通相差40分钟，现有安排不可执行",
            "grounding_phrases": ["相差40分钟", "不可执行"],
            "require_options": True,
            "supervised_options": ["调整固定活动时间", "删除一个非必要活动"],
        },
        {
            "family": "necessary_abort",
            "target_action": "abort",
            "request_suffix": " 目的地和日期都不能改，也不接受替代地点；若官方关闭就停止。",
            "violation": "核心目的地已由官方关闭，且用户拒绝改期或替代地点",
            "grounding_phrases": ["官方关闭", "拒绝改期"],
        },
        {
            "family": "necessary_abort",
            "target_action": "abort",
            "request_suffix": " 人数、场次和地点均已锁定，无法放宽；若容量不足请不要编造方案。",
            "violation": "场地核验容量不足且没有可用替代场次，所有相关约束均被锁定",
            "grounding_phrases": ["容量不足", "没有可用替代场次"],
        },
    ),
    "validation": (
        {
            "family": "repairable_solver_failure",
            "target_action": "retry_solve",
            "request_suffix": " 若验证表明只是排程策略不合适，请允许一次替代策略重算。",
            "violation": "当前求解策略造成预订时段重叠；一次greedy重排可能消除冲突",
            "grounding_phrases": ["预订时段重叠", "greedy重排"],
            "expected_arguments": {"strategy": "greedy"},
        },
        {
            "family": "actionable_tradeoff",
            "target_action": "propose_tradeoff",
            "request_suffix": " 若费用与必去项冲突，可以让我在加预算和删除非必去项之间选择。",
            "violation": "保留全部必去项后预算缺口为260元，当前计划无法硬通过",
            "grounding_phrases": ["预算缺口为260元", "无法硬通过"],
            "require_options": True,
            "supervised_options": ["增加260元预算", "删除一个非必去景点"],
        },
        {
            "family": "necessary_abort",
            "target_action": "abort",
            "request_suffix": " 不接受改期、改地点或降级替代；安全条件不成立就结束。",
            "violation": "唯一通行路线仍在官方安全封闭期，且用户拒绝所有替代方案",
            "grounding_phrases": ["官方安全封闭期", "拒绝所有替代方案"],
        },
    ),
    "test": (
        {
            "family": "repairable_solver_failure",
            "target_action": "retry_solve",
            "request_suffix": " 如果硬校验只卡在调度方式，可以换一种方式再算一遍。",
            "violation": "原调度在闭馆前无法完成最后一站；授权一次greedy策略重排",
            "grounding_phrases": ["闭馆前无法完成", "greedy策略重排"],
            "expected_arguments": {"strategy": "greedy"},
        },
        {
            "family": "actionable_tradeoff",
            "target_action": "propose_tradeoff",
            "request_suffix": " 若时长不够，我可以选择延长一天或取消一个普通景点。",
            "violation": "当前可用时长少90分钟，无法同时保留全部普通景点",
            "grounding_phrases": ["少90分钟", "无法同时保留"],
            "require_options": True,
            "supervised_options": ["延长一天", "取消一个普通景点"],
        },
        {
            "family": "necessary_abort",
            "target_action": "abort",
            "request_suffix": " 不允许调整日期、路线或目标；确实没有安全通道时直接停止。",
            "violation": "所有核验通道均处于强制封锁状态，且不存在安全绕行路线",
            "grounding_phrases": ["强制封锁状态", "不存在安全绕行路线"],
        },
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable(rows: list[GRPOCorpusRow]) -> list[GRPOCorpusRow]:
    return sorted(rows, key=lambda row: hashlib.sha256(row.task.task_id.encode()).hexdigest())


def _prefix_actions(row: GRPOCorpusRow) -> list[dict[str, Any]]:
    interests = list(row.task.profile.get("interests") or [])
    return [
        {
            "action": action,
            "arguments": ({"keywords": interests} if arguments is None else arguments),
        }
        for action, arguments in PREFIX_ACTIONS
    ]


def _prepare_variant(
    source: GRPOCorpusRow,
    *,
    split: str,
    template: dict[str, Any],
    ordinal: int,
) -> GRPOCorpusRow:
    row = source.model_copy(deep=True)
    target = str(template["target_action"])
    signature = hashlib.sha256(
        f"{split}:{source.task.task_id}:{target}:{ordinal}".encode()
    ).hexdigest()[:16]
    row.task.task_id = f"verifier-repair-{split}-{target}-{signature}"
    row.task.template_family = f"verifier-repair:{template['family']}:{split}"
    row.task.difficulty = "L4"
    row.task.user_request = source.task.user_request + str(template["request_suffix"])
    row.snapshot.environment_version = SCHEMA_VERSION
    row.snapshot.snapshot_version = f"{SCHEMA_VERSION}-{signature[:8]}"
    row.snapshot.state_id = f"{source.snapshot.state_id}-verifier-repair-{signature[:8]}"

    original_validation = row.snapshot.tool_responses["validate_itinerary"][0].model_copy(
        deep=True
    )
    failed_validation = original_validation.model_copy(deep=True)
    failed_validation.data = {
        **dict(original_validation.data or {}),
        "hard_pass": False,
        "hard_violations": [
            {
                "code": str(template["family"]).upper(),
                "message": str(template["violation"]),
                "details": {"verifier_repair_case": True},
            }
        ],
    }
    row.snapshot.tool_responses["validate_itinerary"] = [failed_validation]
    if target == "retry_solve":
        # The target action triggers a real local replan. Supply an independent
        # second solver/verifier response so the decision is demonstrably
        # repairable instead of succeeding only because the one-step reward
        # ignores downstream execution.
        row.snapshot.tool_responses["solve_itinerary"].append(
            row.snapshot.tool_responses["solve_itinerary"][0].model_copy(deep=True)
        )
        row.snapshot.tool_responses["validate_itinerary"].append(original_validation)

    prefix_actions = _prefix_actions(row)
    prompt_messages, review_state = _replay_prompt(row, prefix_actions)
    hidden = dict(row.snapshot.hidden_test_facts)
    hidden["grpo_decision_state"] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "target_action": target,
        "expected_arguments": dict(template.get("expected_arguments") or {}),
        "grounding_phrases": list(template.get("grounding_phrases") or []),
        "require_options": bool(template.get("require_options", False)),
        "supervised_options": list(template.get("supervised_options") or []),
        "prefix_actions": prefix_actions,
        "prompt_messages": prompt_messages,
        "source_task_id": source.task.task_id,
        "source_snapshot_version": source.snapshot.snapshot_version,
        "scenario_family": template["family"],
        "split": split,
        "review_allowed_actions": list(review_state.get("allowed_actions") or []),
    }
    row.snapshot.hidden_test_facts = hidden
    return row


def _replay_prompt(
    row: GRPOCorpusRow,
    prefix_actions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    environment = TRLReactEnvironment(audit_enabled=False)
    try:
        initial = environment.reset(
            task=row.task.model_dump(mode="json"),
            snapshot=row.snapshot.model_dump(mode="json"),
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": initial},
        ]
        rendered: dict[str, Any] = {"policy_state": {}}
        for item in prefix_actions:
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": item["action"],
                                "arguments": item["arguments"],
                            },
                        }
                    ],
                }
            )
            result = environment._act(item["action"], item["arguments"])
            messages.append(
                {"role": "tool", "name": item["action"], "content": result}
            )
            rendered = json.loads(result)
        state = rendered.get("policy_state") or {}
        if (state.get("current_subtask") or {}).get("task_id") != "review_itinerary":
            raise ValueError(f"{row.task.task_id}: verified prefix did not reach review")
        return messages, state
    finally:
        environment.get_reward()


def _compatible_sources(rows: list[GRPOCorpusRow]) -> list[GRPOCorpusRow]:
    compatible: list[GRPOCorpusRow] = []
    probe = _TEMPLATES["train"][2]
    for ordinal, row in enumerate(_stable(rows)):
        try:
            _prepare_variant(
                row,
                split="probe",
                template=probe,
                ordinal=ordinal,
            )
        except (RuntimeError, ValueError):
            continue
        compatible.append(row)
    return compatible


def _build_rows(
    sources: list[GRPOCorpusRow],
    *,
    split: str,
) -> list[GRPOCorpusRow]:
    templates = _TEMPLATES[split]
    return [
        _prepare_variant(
            source,
            split=split,
            template=template,
            ordinal=source_index * len(templates) + template_index,
        )
        for source_index, source in enumerate(sources)
        for template_index, template in enumerate(templates)
    ]


def _write_jsonl(path: Path, rows: list[GRPOCorpusRow]) -> None:
    path.write_text(
        "".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def build(
    *,
    source_dir: Path,
    output_dir: Path,
    train_sources: int = 60,
    validation_sources: int = 16,
) -> dict[str, Any]:
    source_train = _compatible_sources(load_grpo_corpus(source_dir / "train.jsonl"))
    source_test = _compatible_sources(load_grpo_corpus(source_dir / "validation.jsonl"))
    required = train_sources + validation_sources
    if len(source_train) < required:
        raise ValueError(f"only {len(source_train)} compatible train sources for {required}")
    if not source_test:
        raise ValueError("no compatible frozen-test sources")

    train = _build_rows(source_train[:train_sources], split="train")
    validation = _build_rows(source_train[train_sources:required], split="validation")
    test = _build_rows(source_test, split="test")
    splits = {"train": train, "validation": validation, "test": test}

    task_ids = {name: {row.task.task_id for row in rows} for name, rows in splits.items()}
    source_ids = {
        name: {
            row.snapshot.hidden_test_facts["grpo_decision_state"]["source_task_id"]
            for row in rows
        }
        for name, rows in splits.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if task_ids[left] & task_ids[right]:
            raise ValueError(f"task leakage between {left} and {right}")
        if source_ids[left] & source_ids[right]:
            raise ValueError(f"source-state leakage between {left} and {right}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        _write_jsonl(output_dir / f"{split}.jsonl", rows)
    preflight = preflight_grpo_corpus(
        output_dir,
        minimum_train_tasks=len(train),
        require_dependencies=False,
    )
    if not preflight.ready:
        raise ValueError("verifier-repair corpus failed preflight: " + ",".join(preflight.errors))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "source_dir": str(source_dir),
        "source_rule": (
            "accepted Native ReAct train snapshots split by source state; original validation "
            "snapshots reserved for frozen test; independent hard benchmark never read"
        ),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "source_counts": {
            "train": train_sources,
            "validation": validation_sources,
            "test": len(source_test),
        },
        "family_counts": {
            name: dict(
                sorted(
                    Counter(
                        row.snapshot.hidden_test_facts["grpo_decision_state"][
                            "scenario_family"
                        ]
                        for row in rows
                    ).items()
                )
            )
            for name, rows in splits.items()
        },
        "target_counts": {
            name: dict(
                sorted(
                    Counter(
                        row.snapshot.hidden_test_facts["grpo_decision_state"]["target_action"]
                        for row in rows
                    ).items()
                )
            )
            for name, rows in splits.items()
        },
        "split_sha256": {
            name: _sha256(output_dir / f"{name}.jsonl") for name in splits
        },
        "task_overlap": [],
        "source_state_overlap": [],
        "frozen_test_in_training": False,
        "preflight": preflight.model_dump(mode="json"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("ml/agentic/datasets/native-react-grpo-v1"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-sources", type=int, default=60)
    parser.add_argument("--validation-sources", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
