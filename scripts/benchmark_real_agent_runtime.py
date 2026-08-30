"""Replay the real bounded TravelAgent runtime against the frozen pure-ReAct arm.

The current-architecture arm calls ``run_agent_branch`` without replacing its
policy, controller, tools, solver, verifier, budget or episode recorder.  The
paired pure-agent rows come from the Stage38 direct-itinerary ablation and are
matched by case ID.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend" / "src"))

from agentic.integration import run_agent_branch  # noqa: E402
from agentic.runtime import initialize_agent_ledger  # noqa: E402
from core.redis_client import redis_client  # noqa: E402
from core.settings import settings  # noqa: E402
from scripts.benchmark_pure_agent_vs_verified import (  # noqa: E402
    DEFAULT_CASES,
    _bootstrap_delta,
    _hash,
    _percentile,
    load_cases,
)


SCHEMA_VERSION = "real-agent-runtime-vs-pure-react.v1"
DEFAULT_PURE_RUNS = (
    ROOT
    / "ml"
    / "agentic"
    / "reports"
    / "stage38-pure-agent-vs-verified-final-v1"
    / "runs.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "ml"
    / "agentic"
    / "reports"
    / "stage39-real-agent-runtime-vs-pure-react-v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--pure-runs", type=Path, default=DEFAULT_PURE_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=("controller_first", "policy_driven"),
        default="controller_first",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _case_state(case: dict[str, Any]) -> dict[str, Any]:
    interests = list(case.get("interests") or [])
    state = {
        "user_input": (
            f"规划{case['destination']}{case['travel_days']}日游，"
            f"{case['start_date']}至{case['end_date']}，预算{case['budget']:.0f}元，"
            f"兴趣为{'、'.join(interests)}"
        ),
        "slots": {
            "destination": case["destination"],
            "travel_days": int(case["travel_days"]),
            "start_date": case["start_date"],
            "end_date": case["end_date"],
            "budget_range": float(case["budget"]),
            "interests": interests,
        },
        "profile": {
            "destination": case["destination"],
            "travel_days": int(case["travel_days"]),
            "travel_dates": f"{case['start_date']}|{case['end_date']}",
            "budget_range": float(case["budget"]),
            "travelers_count": 2,
            "interests": interests,
        },
    }
    return {**state, **initialize_agent_ledger(state, mode="agent")}


def _actual_row(
    case: dict[str, Any],
    result: dict[str, Any],
    elapsed_ms: float,
    *,
    execution_mode: str,
) -> dict[str, Any]:
    ledger = result.get("agent_ledger") or {}
    budget = ledger.get("budget") or {}
    episode = result.get("agent_episode") or {}
    steps = list(episode.get("steps") or [])
    policy_steps = [step for step in steps if (step.get("action") or {}).get("decision_source") == "policy"]
    validation = result.get("validation_report") or {}
    itinerary = list(result.get("itinerary") or [])
    hard_pass = bool(itinerary) and bool(validation.get("hard_pass"))
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case["case_id"],
        "input_hash": _hash(case),
        "destination": case["destination"],
        "travel_days": int(case["travel_days"]),
        "mode": "real_agent_runtime",
        "execution_mode": result.get("agent_execution_mode") or execution_mode,
        "model": episode.get("policy_version") or settings.llm_model,
        "policy_name": episode.get("policy_name") or "unknown",
        "policy_version": episode.get("policy_version") or settings.llm_model,
        "status": result.get("agent_status"),
        "termination_reason": result.get("termination_reason"),
        "hard_pass": hard_pass,
        "validation_hard_pass": bool(validation.get("hard_pass")),
        "total_tokens": int(budget.get("used_tokens") or 0),
        "model_calls": len(policy_steps),
        "episode_steps": int(budget.get("used_episode_steps") or 0),
        "tool_calls": int(budget.get("used_tool_calls") or 0),
        "solver_calls": int(budget.get("used_solver_calls") or 0),
        "latency_ms": round(elapsed_ms, 3),
        "solve_status": result.get("solve_status"),
        "itinerary_days": len(itinerary),
        "activity_count": sum(len(day.get("activities") or []) for day in itinerary),
        "violation_codes": [
            str(item.get("code"))
            for item in validation.get("hard_violations") or []
            if item.get("code")
        ],
        "failures": list(ledger.get("failures") or []),
        "actions": [
            {
                "task": step.get("task_id"),
                "action": (step.get("action") or {}).get("action"),
                "source": (step.get("action") or {}).get("decision_source"),
                "token_usage": int((step.get("action") or {}).get("token_usage") or 0),
                "policy_latency_ms": int(step.get("policy_latency_ms") or 0),
                "action_latency_ms": int(step.get("action_latency_ms") or 0),
            }
            for step in steps
        ],
        "itinerary": itinerary,
        "episode_content_hash": episode.get("content_hash"),
    }


async def run_actual(case: dict[str, Any], *, execution_mode: str) -> dict[str, Any]:
    started = time.perf_counter()
    result = await run_agent_branch(_case_state(case), execution_mode=execution_mode)
    return _actual_row(
        case,
        result,
        (time.perf_counter() - started) * 1000,
        execution_mode=execution_mode,
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [float(row["total_tokens"]) for row in rows]
    latency = [float(row["latency_ms"]) for row in rows]
    successful = [row for row in rows if row["hard_pass"]]
    passes = sum(bool(row["hard_pass"]) for row in rows)
    return {
        "tasks": len(rows),
        "hard_pass_count": passes,
        "hard_pass_rate": passes / len(rows) if rows else 0.0,
        "total_tokens": sum(tokens),
        "mean_total_tokens": statistics.fmean(tokens) if rows else 0.0,
        "median_total_tokens": statistics.median(tokens) if rows else 0.0,
        "p95_total_tokens": _percentile(tokens, 0.95),
        "mean_model_calls": statistics.fmean(float(row["model_calls"]) for row in rows) if rows else 0.0,
        "mean_tool_calls": statistics.fmean(float(row["tool_calls"]) for row in rows) if rows else 0.0,
        "mean_episode_steps": statistics.fmean(float(row.get("episode_steps") or 0) for row in rows) if rows else 0.0,
        "mean_latency_ms": statistics.fmean(latency) if rows else 0.0,
        "p95_latency_ms": _percentile(latency, 0.95),
        "max_latency_ms": max(latency, default=0.0),
        "successful_mean_total_tokens": (
            statistics.fmean(float(row["total_tokens"]) for row in successful)
            if successful
            else 0.0
        ),
        "successful_mean_model_calls": (
            statistics.fmean(float(row["model_calls"]) for row in successful)
            if successful
            else 0.0
        ),
        "solver_status_counts": dict(Counter(str(row.get("solve_status")) for row in rows)),
        "termination_reason_counts": dict(
            Counter(str(row.get("termination_reason")) for row in rows)
        ),
        "violation_counts": dict(Counter(code for row in rows for code in row.get("violation_codes") or [])),
    }


def build_report(
    actual_rows: list[dict[str, Any]],
    pure_rows: list[dict[str, Any]],
    cases_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    actual_by_id = {row["case_id"]: row for row in actual_rows}
    pure_by_id = {row["case_id"]: row for row in pure_rows}
    paired_ids = sorted(set(actual_by_id) & set(pure_by_id))
    actual = [actual_by_id[item] for item in paired_ids]
    pure: list[dict[str, Any]] = []
    for item in paired_ids:
        row = dict(pure_by_id[item])
        row["episode_steps"] = int(row.get("model_calls") or 0)
        pure.append(row)
    actual_summary = _summary(actual)
    pure_summary = _summary(pure)
    execution_modes = {
        str(row["execution_mode"])
        for row in actual
        if row.get("execution_mode")
    }
    policy_versions = sorted(
        {
            str(row.get("policy_version") or row.get("model"))
            for row in actual
            if row.get("policy_version") or row.get("model")
        }
    )
    token_pairs = [(float(pure_by_id[item]["total_tokens"]), float(actual_by_id[item]["total_tokens"])) for item in paired_ids]
    pure_mean = pure_summary["mean_total_tokens"]
    actual_mean = actual_summary["mean_total_tokens"]
    day_buckets: dict[str, dict[str, Any]] = {}
    for days in sorted({int(cases_by_id[item]["travel_days"]) for item in paired_ids}):
        ids = [item for item in paired_ids if int(cases_by_id[item]["travel_days"]) == days]
        day_buckets[str(days)] = {
            "tasks": len(ids),
            "pure_mean_tokens": statistics.fmean(float(pure_by_id[item]["total_tokens"]) for item in ids),
            "actual_mean_tokens": statistics.fmean(float(actual_by_id[item]["total_tokens"]) for item in ids),
            "actual_hard_pass_rate": sum(bool(actual_by_id[item]["hard_pass"]) for item in ids) / len(ids),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "frozen pure-ReAct reference vs repository run_agent_branch runtime",
        "model": policy_versions[0] if len(policy_versions) == 1 else "mixed",
        "policy_versions": policy_versions,
        "execution_mode": (
            next(iter(execution_modes)) if len(execution_modes) == 1 else "mixed"
        ),
        "paired_tasks": len(paired_ids),
        "covered_cities": sorted({actual_by_id[item]["destination"] for item in paired_ids}),
        "pure_agent_reference": pure_summary,
        "real_agent_runtime": actual_summary,
        "paired_delta": {
            "pure_minus_actual_mean_tokens": pure_mean - actual_mean,
            "mean_token_delta_95pct_bootstrap_ci": _bootstrap_delta(token_pairs, 20260824),
            "pure_to_actual_token_ratio": pure_mean / actual_mean if actual_mean else None,
            "actual_token_reduction_vs_pure_percent": (
                (pure_mean - actual_mean) / pure_mean * 100 if pure_mean else None
            ),
            "hard_pass_rate_delta_actual_minus_pure": (
                actual_summary["hard_pass_rate"] - pure_summary["hard_pass_rate"]
            ),
            "actual_latency_reduction_vs_pure_percent": (
                (pure_summary["mean_latency_ms"] - actual_summary["mean_latency_ms"])
                / pure_summary["mean_latency_ms"]
                * 100
                if pure_summary["mean_latency_ms"]
                else None
            ),
        },
        "by_travel_days": day_buckets,
        "token_accounting": {
            "pure_agent": "provider-reported prompt plus completion tokens accumulated by Stage38",
            "real_agent_runtime": "Agent Ledger used_tokens from actual policy actions",
            "runtime_prompt_completion_split": "unavailable for the current JSON policy adapter",
        },
        "limitations": [
            "两组使用相同 case_id 和模型，但纯 ReAct 使用冻结工具夹具，真实运行时使用仓库生产工具栈与 PostgreSQL，因此不是逐观察字节相同。",
            "本报告只覆盖信息完整的 solvable_plan；澄清、用户修改和故障恢复需单独场景集。",
            "顺序云端回放，不是真实并发线上流量。",
            "真实架构按生产 auto 策略选择 CP-SAT 或 Greedy，并由同一 Verifier 验收，不强制所有任务使用 CP-SAT。",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    pure = report["pure_agent_reference"]
    actual = report["real_agent_runtime"]
    delta = report["paired_delta"]
    title = (
        "真正逐步工具决策 Agent vs 纯 ReAct"
        if report["execution_mode"] == "policy_driven"
        else "Controller-first 运行时 vs 纯 ReAct"
    )
    return "\n".join(
        [
            f"# {title}",
            "",
            f"- 协议：`{report['schema_version']}`",
            f"- 模型：`{report['model']}`",
            f"- 执行模式：`{report['execution_mode']}`",
            f"- 配对任务：{report['paired_tasks']}",
            "- 新架构入口：仓库实际 `run_agent_branch`",
            "",
            "## 结果",
            "",
            "| 指标 | 纯 ReAct 参考 | 真实新架构 |",
            "|---|---:|---:|",
            f"| Verifier 硬通过率 | {pure['hard_pass_rate'] * 100:.2f}% | {actual['hard_pass_rate'] * 100:.2f}% |",
            f"| 平均总 Token/任务 | {pure['mean_total_tokens']:.2f} | {actual['mean_total_tokens']:.2f} |",
            f"| 硬通过任务平均 Token | {pure['successful_mean_total_tokens']:.2f} | {actual['successful_mean_total_tokens']:.2f} |",
            f"| 中位总 Token/任务 | {pure['median_total_tokens']:.2f} | {actual['median_total_tokens']:.2f} |",
            f"| P95 总 Token/任务 | {pure['p95_total_tokens']:.2f} | {actual['p95_total_tokens']:.2f} |",
            f"| 平均模型调用 | {pure['mean_model_calls']:.2f} | {actual['mean_model_calls']:.2f} |",
            f"| 平均工具调用 | {pure['mean_tool_calls']:.2f} | {actual['mean_tool_calls']:.2f} |",
            f"| 平均端到端延迟 | {pure['mean_latency_ms']:.2f} ms | {actual['mean_latency_ms']:.2f} ms |",
            f"| P95 端到端延迟 | {pure['p95_latency_ms']:.2f} ms | {actual['p95_latency_ms']:.2f} ms |",
            f"| 最大端到端延迟 | {pure['max_latency_ms']:.2f} ms | {actual['max_latency_ms']:.2f} ms |",
            "",
            "## 结论",
            "",
            f"- 真实新架构相对纯 ReAct 的 LLM Token 降低 **{delta['actual_token_reduction_vs_pure_percent']:.2f}%**。",
            f"- 纯 ReAct Token 是真实新架构的 **{delta['pure_to_actual_token_ratio']:.2f}x**。",
            f"- Verifier 硬通过率由 **{pure['hard_pass_rate'] * 100:.2f}%** 提升至 **{actual['hard_pass_rate'] * 100:.2f}%**（+{delta['hard_pass_rate_delta_actual_minus_pure'] * 100:.2f}pp）。",
            f"- 平均端到端延迟降低 **{delta['actual_latency_reduction_vs_pure_percent']:.2f}%**。",
            f"- 新架构硬通过任务平均调用模型 **{actual['successful_mean_model_calls']:.2f} 次**、使用 **{actual['successful_mean_total_tokens']:.2f} Token**。",
            f"- 新架构终止原因：`{actual['termination_reason_counts']}`。",
            f"- 平均每任务减少 **{delta['pure_minus_actual_mean_tokens']:.2f} Token**，配对 Bootstrap 95% CI 为 {delta['mean_token_delta_95pct_bootstrap_ci']}。",
            "",
            "## 天数分桶",
            "",
            "| 天数 | 任务数 | 纯 ReAct 平均 Token | 新架构平均 Token | 新架构硬通过率 |",
            "|---:|---:|---:|---:|---:|",
            *[
                f"| {days} | {row['tasks']} | {row['pure_mean_tokens']:.2f} | {row['actual_mean_tokens']:.2f} | {row['actual_hard_pass_rate'] * 100:.2f}% |"
                for days, row in report["by_travel_days"].items()
            ],
            "",
            "## 边界",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    pure_all = [row for row in _read_jsonl(args.pure_runs) if row.get("mode") == "pure_agent"]
    selected_pure = pure_all[: args.limit]
    selected_ids = {row["case_id"] for row in selected_pure}
    cases_by_id = {case["case_id"]: case for case in load_cases(args.cases) if case["case_id"] in selected_ids}
    if len(cases_by_id) != len(selected_ids):
        raise ValueError("pure reference contains case IDs missing from the case registry")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.output_dir / "runs.jsonl"
    existing = _read_jsonl(runs_path) if args.resume and runs_path.exists() else []
    if runs_path.exists() and not args.resume:
        raise FileExistsError(f"{runs_path} exists; pass --resume or choose another output dir")
    done = {row["case_id"] for row in existing}
    rows = list(existing)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    await redis_client.connect()
    try:
        with runs_path.open("a", encoding="utf-8") as handle:
            for index, pure in enumerate(selected_pure, 1):
                case_id = pure["case_id"]
                if case_id in done:
                    continue
                row = await run_actual(
                    cases_by_id[case_id], execution_mode=args.execution_mode
                )
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                rows.append(row)
                done.add(case_id)
                print(
                    json.dumps(
                        {
                            "progress": f"{len(done)}/{len(selected_pure)}",
                            "case_id": case_id,
                            "tokens": row["total_tokens"],
                            "hard_pass": row["hard_pass"],
                            "latency_ms": row["latency_ms"],
                            "solve_status": row["solve_status"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        await redis_client.disconnect()
    for row in rows:
        row.setdefault("execution_mode", args.execution_mode)
    report = build_report(rows, selected_pure, cases_by_id)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run(parse_args())), ensure_ascii=False, indent=2))
