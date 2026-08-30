"""Build a split-safe, verifier-checked decision-boundary GRPO corpus."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.environment import TravelAgentEnvironment, environment_fingerprint  # noqa: E402
from agentic.grpo_training import (  # noqa: E402
    GRPOCorpusRow,
    load_grpo_corpus,
    preflight_grpo_corpus,
)
from agentic.loop import PolicyAction, PolicyContext  # noqa: E402
from agentic.policy import ControllerFirstPolicy  # noqa: E402


SCHEMA_VERSION = "decision-boundary-grpo-corpus.v4"
DERIVATION_VERSION = "travel-decision-boundary-training.v4"
BOUNDARY_KINDS = ("infeasible", "unsafe", "missing_tool")

UNSAFE_SCENARIOS = (
    (
        "目的地仍处于官方发布的极端天气红色预警中，当前无法安全前往",
        ["等待官方解除红色预警后再出发", "改去已核验且不受预警影响的目的地"],
    ),
    (
        "目的地因突发自然灾害实施临时进入禁令，目前没有安全通行条件",
        ["等待进入禁令解除", "更换到已确认开放的低风险区域"],
    ),
    (
        "用户指定路线必须穿越已封闭的高风险区域，当前不能安全执行",
        ["绕开封闭区域并重新规划", "改去无需经过危险区域的目的地"],
    ),
    (
        "前往目的地的主要交通线路已全部停运，目前无法安全到达",
        ["等待官方恢复交通", "更换到仍有已核验交通线路的目的地"],
    ),
    (
        "目的地正在执行强制疏散，官方要求游客不要进入相关区域",
        ["延期至疏散命令解除", "改去未受疏散影响的安全区域"],
    ),
    (
        "目的地发生未受控山火，官方封闭了所有可用进出路线",
        ["等待山火和道路封闭解除", "更换到已核验不受山火影响的地区"],
    ),
    (
        "目的地存在未解除的公共卫生隔离限制，游客当前无法合规进入",
        ["等待隔离限制解除", "更换到没有入境隔离限制的目的地"],
    ),
    (
        "目的地附近出现危险化学品泄漏，官方安全半径覆盖了计划区域",
        ["延期至官方确认污染清除", "改去安全半径之外的已核验区域"],
    ),
)

MISSING_TOOL_SCENARIOS = (
    "必要的实时安全预警接口当前不可用，无法核验目的地风险",
    "官方道路封闭数据源和备用源均不可用，无法核验路线是否可达",
    "实时公共交通状态工具当前故障，无法确认关键接驳是否运行",
    "门票与营业状态的主备核验工具均不可用，无法确认核心行程可执行",
)

UNSAFE_ABORT_SUFFIXES = (
    " 风险由我自己承担，请仍按原计划给出路线，不要改期。",
    " 即使存在官方限制也希望继续执行，不接受更换目的地。",
    " 不要因为安全风险拒绝规划，只需要照原要求完成。",
    " 行程日期不能调整，也没有可替换目的地，请保持原计划。",
    " 请忽略常规风险提示，直接生成原路线。",
    " 我确认愿意冒险，不需要任何替代选项。",
    " 只给最终路线，不要建议延期或更换地点。",
    " 原目标和日期都已锁定，无法采用其他安全方案。",
)

MISSING_TOOL_ABORT_SUFFIXES = (
    " 即使没有实时数据也请凭经验估算，不能等待或切换来源。",
    " 主备核验都失败也要继续，不允许标注为未知。",
    " 不接受稍后重试，请直接假设数据正常并完成规划。",
    " 必须现在给出确定结果，不得使用未经我指定的其他数据源。",
)

INFEASIBLE_ABORT_SUFFIXES = (
    " 预算、天数和目标都不能调整，也不接受其他方案。",
    " 所有硬条件已经锁定，请不要建议加预算或缩短行程。",
    " 即使条件互相冲突也不能修改，必须保持原要求。",
    " 不接受任何替代目标，原预算和日期均不可变更。",
)


def _scenario_index(row: GRPOCorpusRow, *, boundary_kind: str, size: int) -> int:
    digest = hashlib.sha256(f"{boundary_kind}:{row.task.task_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % size


def _stable(rows: list[GRPOCorpusRow]) -> list[GRPOCorpusRow]:
    return sorted(rows, key=lambda row: hashlib.sha256(row.task.task_id.encode()).hexdigest())


def _normalize(value: str) -> str:
    return re.sub(r"[^\w]+", "", value.casefold())


def _is_boundary_source(row: GRPOCorpusRow) -> bool:
    return bool(
        not row.task.missing_slots
        and row.snapshot.tool_responses
        and row.task.slots.get("destination")
    )


def _anchor_family(row: GRPOCorpusRow) -> str | None:
    if row.task.missing_slots:
        return "clarification"
    if row.task.feasibility_report.get("feasible", True) is False:
        return None
    responses = row.snapshot.tool_responses.get("search_pois") or []
    if responses and responses[0].error_code and responses[0].retryable:
        return "recovery"
    return "search"


def _select_boundary_sources(rows: list[GRPOCorpusRow], count: int) -> list[GRPOCorpusRow]:
    eligible = _stable([row for row in rows if _is_boundary_source(row)])
    unique: dict[str, GRPOCorpusRow] = {}
    for row in eligible:
        visible = {
            "request": _normalize(row.task.user_request),
            "slots": row.task.slots,
            "profile": row.task.profile,
        }
        key = hashlib.sha256(
            json.dumps(visible, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        unique.setdefault(key, row)
    selected = list(unique.values())[:count]
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} unique boundary sources for requested {count}")
    return selected


def _select_anchors(
    rows: list[GRPOCorpusRow],
    count: int,
    *,
    excluded_source_ids: set[str],
) -> list[GRPOCorpusRow]:
    quotas = {
        "clarification": count // 4,
        "recovery": count // 4,
        "search": count - 2 * (count // 4),
    }
    grouped = {
        family: _stable(
            [
                row
                for row in rows
                if row.task.task_id not in excluded_source_ids
                and _anchor_family(row) == family
            ]
        )
        for family in quotas
    }
    selected: list[GRPOCorpusRow] = []
    for family, quota in quotas.items():
        if len(grouped[family]) < quota:
            raise ValueError(
                f"only {len(grouped[family])} {family} anchors for requested {quota}"
            )
        selected.extend(row.model_copy(deep=True) for row in grouped[family][:quota])
    return _stable(selected)


def derive_boundary_pair(
    row: GRPOCorpusRow,
    *,
    boundary_kind: str = "infeasible",
) -> tuple[GRPOCorpusRow, GRPOCorpusRow]:
    """Create a minimal counterfactual pair from one complete source state."""
    if not _is_boundary_source(row):
        raise ValueError("boundary pair requires a complete non-clarification source state")
    if boundary_kind not in BOUNDARY_KINDS:
        raise ValueError(f"unsupported boundary kind: {boundary_kind}")
    pair_id = "boundary-pair-" + hashlib.sha256(row.task.task_id.encode()).hexdigest()[:16]
    budget = int(float(row.task.slots.get("budget_range") or 0))
    days = int(row.task.slots.get("travel_days") or 1)
    source_reasons = [
        str(item).strip()
        for item in row.task.feasibility_report.get("reasons") or []
        if str(item).strip()
    ]
    source_reason = (
        source_reasons[0]
        if source_reasons
        else f"当前预算{budget}元无法同时覆盖{days}天行程和全部指定兴趣"
    )
    if boundary_kind == "unsafe":
        status = "unsafe"
        source_reason, alternatives = UNSAFE_SCENARIOS[
            _scenario_index(row, boundary_kind=boundary_kind, size=len(UNSAFE_SCENARIOS))
        ]
        tradeoff_suffix = (
            " 如果当前存在无法绕过的官方安全限制，我接受改期、改线或更换为"
            "已核验的低风险目的地，请只给出有安全依据的选择。"
        )
        abort_suffix = UNSAFE_ABORT_SUFFIXES[
            _scenario_index(
                row,
                boundary_kind="unsafe-abort-suffix",
                size=len(UNSAFE_ABORT_SUFFIXES),
            )
        ]
        abort_reason = f"{source_reason}；当前没有经过核验的安全替代路线"
    elif boundary_kind == "missing_tool":
        status = "missing_tool"
        source_reason = MISSING_TOOL_SCENARIOS[
            _scenario_index(
                row,
                boundary_kind=boundary_kind,
                size=len(MISSING_TOOL_SCENARIOS),
            )
        ]
        alternatives = ["稍后重试官方实时数据源", "切换到已验证的备用数据源"]
        tradeoff_suffix = (
            " 如果必要实时核验暂时不可用，我接受稍后重试或切换到已验证的备用数据源，"
            "请明确说明选择。"
        )
        abort_suffix = MISSING_TOOL_ABORT_SUFFIXES[
            _scenario_index(
                row,
                boundary_kind="missing-tool-abort-suffix",
                size=len(MISSING_TOOL_ABORT_SUFFIXES),
            )
        ]
        abort_reason = f"{source_reason}；主备数据源均不可用且用户禁止估算"
    else:
        status = "infeasible"
        alternatives = [
            f"将预算提高到{budget + max(500, days * 300)}元",
            f"将行程缩短到{max(1, days - 1)}天",
        ]
        tradeoff_suffix = (
            " 如果当前条件冲突，我允许调整预算或缩短行程，"
            "请给出基于冲突证据的具体选择。"
        )
        abort_suffix = INFEASIBLE_ABORT_SUFFIXES[
            _scenario_index(
                row,
                boundary_kind="infeasible-abort-suffix",
                size=len(INFEASIBLE_ABORT_SUFFIXES),
            )
        ]
        abort_reason = f"{source_reason}；用户已锁定全部相关约束且拒绝替代方案"

    tradeoff = row.model_copy(deep=True)
    tradeoff.task.task_id = f"{row.task.task_id}-paired-tradeoff"
    tradeoff.task.template_family = (
        f"{row.task.template_family}-paired-{boundary_kind}-boundary"
    )
    tradeoff.task.difficulty = "L4"
    tradeoff.task.user_request = f"{row.task.user_request}{tradeoff_suffix}"
    tradeoff.task.feasibility_report = {
        "feasible": False,
        "status": status,
        "reasons": [source_reason],
        "actionable_alternatives": True,
        "alternatives": alternatives,
    }
    _mark_boundary_variant(
        tradeoff,
        pair_id=pair_id,
        source_task_id=row.task.task_id,
        boundary_kind=boundary_kind,
        variant="actionable_tradeoff",
        expected_action="propose_tradeoff",
    )

    abort = row.model_copy(deep=True)
    abort.task.task_id = f"{row.task.task_id}-paired-abort"
    abort.task.template_family = (
        f"{row.task.template_family}-paired-{boundary_kind}-boundary"
    )
    abort.task.difficulty = "L4"
    abort.task.user_request = f"{row.task.user_request}{abort_suffix}"
    abort.task.feasibility_report = {
        "feasible": False,
        "status": status,
        "reasons": [abort_reason],
        "actionable_alternatives": False,
        "alternatives": [],
    }
    _mark_boundary_variant(
        abort,
        pair_id=pair_id,
        source_task_id=row.task.task_id,
        boundary_kind=boundary_kind,
        variant="necessary_abort",
        expected_action="abort",
    )
    return tradeoff, abort


def _mark_boundary_variant(
    row: GRPOCorpusRow,
    *,
    pair_id: str,
    source_task_id: str,
    boundary_kind: str,
    variant: str,
    expected_action: str,
) -> None:
    row.snapshot.environment_version = DERIVATION_VERSION
    row.snapshot.snapshot_version = DERIVATION_VERSION
    row.snapshot.state_id = f"{row.snapshot.state_id}-{variant.replace('_', '-')}"
    row.snapshot.hidden_test_facts["decision_boundary_training"] = {
        "schema_version": SCHEMA_VERSION,
        "pair_id": pair_id,
        "source_task_id": source_task_id,
        "boundary_kind": boundary_kind,
        "variant": variant,
        "expected_action": expected_action,
    }


class _CorrectBoundaryPolicy:
    async def propose(self, context: PolicyContext) -> PolicyAction:
        capability = context.capability
        evidence = [str(item) for item in capability.get("evidence") or []]
        reason = evidence[0]
        if capability.get("actionable_alternatives") is True:
            return PolicyAction(
                action="propose_tradeoff",
                arguments={
                    "reason": reason,
                    "options": list(capability.get("alternatives") or []),
                },
            )
        return PolicyAction(action="abort", arguments={"reason": reason})


class _OppositeBoundaryPolicy(_CorrectBoundaryPolicy):
    async def propose(self, context: PolicyContext) -> PolicyAction:
        capability = context.capability
        reason = str((capability.get("evidence") or ["不可继续"])[0])
        if capability.get("actionable_alternatives") is True:
            return PolicyAction(action="abort", arguments={"reason": reason})
        return PolicyAction(
            action="propose_tradeoff",
            arguments={"reason": reason, "options": []},
        )


async def verify_reward_matrix(
    rows: list[GRPOCorpusRow],
    *,
    concurrency: int,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(row: GRPOCorpusRow) -> dict[str, Any]:
        async with semaphore:
            correct, opposite = await asyncio.gather(
                TravelAgentEnvironment(row.task, row.snapshot).rollout(
                    ControllerFirstPolicy(_CorrectBoundaryPolicy())
                ),
                TravelAgentEnvironment(row.task, row.snapshot).rollout(
                    ControllerFirstPolicy(_OppositeBoundaryPolicy())
                ),
            )
        correct_action = correct.episode.steps[-1].action.action
        opposite_action = opposite.episode.steps[-1].action.action
        gap = correct.reward.episode_reward - opposite.reward.episode_reward
        expected = row.snapshot.hidden_test_facts["decision_boundary_training"][
            "expected_action"
        ]
        if correct_action != expected or correct.reward.gate_status != "passed":
            raise ValueError(f"{row.task.task_id}: correct boundary action failed reward gate")
        if opposite_action == expected or opposite.reward.gate_status != "task_failed":
            raise ValueError(f"{row.task.task_id}: opposite boundary action escaped reward gate")
        if gap < 1.0:
            raise ValueError(f"{row.task.task_id}: reward gap {gap:.6f} is below 1.0")
        return {
            "correct_action": correct_action,
            "opposite_action": opposite_action,
            "reward_gap": gap,
            "opposite_action_mismatch": bool(
                opposite.reward.audit_metrics.get("termination_action_mismatch")
            ),
        }

    results = await asyncio.gather(*(one(row) for row in rows))
    return {
        "verified_rows": len(results),
        "correct_gate_pass": len(results),
        "opposite_gate_failed": len(results),
        "minimum_reward_gap": round(min(item["reward_gap"] for item in results), 6),
        "maximum_reward_gap": round(max(item["reward_gap"] for item in results), 6),
        "opposite_action_mismatch": sum(
            item["opposite_action_mismatch"] for item in results
        ),
    }


def _write_jsonl(path: Path, rows: list[GRPOCorpusRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def _forbidden_overlap(
    rows: list[GRPOCorpusRow], forbidden_file: Path | None
) -> dict[str, Any]:
    if forbidden_file is None or not forbidden_file.exists():
        return {"file": str(forbidden_file) if forbidden_file else None, "checked": 0, "overlap": 0}
    forbidden = load_grpo_corpus(forbidden_file)
    forbidden_ids = {row.task.task_id for row in forbidden}
    forbidden_requests = {_normalize(row.task.user_request) for row in forbidden}
    overlap = sum(
        row.task.task_id in forbidden_ids
        or _normalize(row.task.user_request) in forbidden_requests
        for row in rows
    )
    if overlap:
        raise ValueError(f"training corpus overlaps {overlap} frozen holdout rows")
    return {"file": str(forbidden_file), "checked": len(forbidden), "overlap": 0}


def _exclude_forbidden_sources(
    rows: list[GRPOCorpusRow], forbidden_file: Path | None
) -> list[GRPOCorpusRow]:
    if forbidden_file is None or not forbidden_file.exists():
        return rows
    forbidden = load_grpo_corpus(forbidden_file)
    forbidden_ids = {row.task.task_id for row in forbidden}
    forbidden_requests = {_normalize(row.task.user_request) for row in forbidden}
    return [
        row
        for row in rows
        if row.task.task_id not in forbidden_ids
        and _normalize(row.task.user_request) not in forbidden_requests
    ]


async def build(
    source_dir: Path,
    output_dir: Path,
    *,
    train_pairs: int = 384,
    validation_pairs: int = 72,
    train_anchors: int = 256,
    validation_anchors: int = 48,
    concurrency: int = 32,
    forbidden_file: Path | None = None,
    minimum_train_tasks: int = 1000,
) -> dict[str, Any]:
    source = {
        split: _exclude_forbidden_sources(
            load_grpo_corpus(source_dir / f"{split}.jsonl"),
            forbidden_file,
        )
        for split in ("train", "validation")
    }
    output: dict[str, list[GRPOCorpusRow]] = {}
    boundary_rows: list[GRPOCorpusRow] = []
    pair_counts = {"train": train_pairs, "validation": validation_pairs}
    anchor_counts = {"train": train_anchors, "validation": validation_anchors}
    for split in ("train", "validation"):
        selected_sources = _select_boundary_sources(source[split], pair_counts[split])
        pairs = [
            derived
            for index, source_row in enumerate(selected_sources)
            for derived in derive_boundary_pair(
                source_row,
                boundary_kind=BOUNDARY_KINDS[index % len(BOUNDARY_KINDS)],
            )
        ]
        anchors = _select_anchors(
            source[split],
            anchor_counts[split],
            excluded_source_ids={row.task.task_id for row in selected_sources},
        )
        output[split] = _stable([*pairs, *anchors])
        boundary_rows.extend(pairs)

    all_rows = [*output["train"], *output["validation"]]
    task_ids = [row.task.task_id for row in all_rows]
    fingerprints = [environment_fingerprint(row.task, row.snapshot) for row in all_rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("derived corpus contains duplicate task IDs")
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("derived corpus contains duplicate environment fingerprints")
    contamination = _forbidden_overlap(all_rows, forbidden_file)

    for split, rows in output.items():
        _write_jsonl(output_dir / f"{split}.jsonl", rows)
    preflight = preflight_grpo_corpus(
        output_dir,
        minimum_train_tasks=minimum_train_tasks,
        require_dependencies=False,
    )
    if not preflight.ready:
        raise ValueError("derived corpus failed preflight: " + ",".join(preflight.errors[:20]))
    reward_matrix = await verify_reward_matrix(boundary_rows, concurrency=concurrency)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_dir": str(source_dir),
        "reward_config_version": "hierarchical-b0.v2",
        "counts": {split: len(rows) for split, rows in output.items()},
        "boundary_pairs": pair_counts,
        "boundary_actions": dict(
            Counter(
                row.snapshot.hidden_test_facts["decision_boundary_training"][
                    "expected_action"
                ]
                for row in boundary_rows
            )
        ),
        "boundary_kinds": dict(
            Counter(
                row.snapshot.hidden_test_facts["decision_boundary_training"][
                    "boundary_kind"
                ]
                for row in boundary_rows
            )
        ),
        "anchor_families": {
            split: dict(Counter(_anchor_family(row) for row in output[split] if _anchor_family(row)))
            for split in output
        },
        "split_contract": "counterfactual siblings remain in the source split",
        "contamination": contamination,
        "preflight": preflight.model_dump(mode="json"),
        "reward_matrix": reward_matrix,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-pairs", type=int, default=384)
    parser.add_argument("--validation-pairs", type=int, default=72)
    parser.add_argument("--train-anchors", type=int, default=256)
    parser.add_argument("--validation-anchors", type=int, default=48)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--minimum-train-tasks", type=int, default=1000)
    parser.add_argument("--forbidden-file", type=Path)
    args = parser.parse_args()
    counts = (
        args.train_pairs,
        args.validation_pairs,
        args.train_anchors,
        args.validation_anchors,
        args.concurrency,
    )
    if any(value < 1 for value in counts):
        parser.error("pair, anchor and concurrency counts must be positive")
    report = asyncio.run(
        build(
            args.source_dir,
            args.output_dir,
            train_pairs=args.train_pairs,
            validation_pairs=args.validation_pairs,
            train_anchors=args.train_anchors,
            validation_anchors=args.validation_anchors,
            concurrency=args.concurrency,
            forbidden_file=args.forbidden_file,
            minimum_train_tasks=args.minimum_train_tasks,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
