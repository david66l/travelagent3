"""Run the frozen Native ReAct hard benchmark through the production Agent Loop."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend" / "src"))

from agentic.policy import NativeToolAgentPolicy  # noqa: E402
from core.inference_metrics import percentile  # noqa: E402
from core.llm_client import LLMClient  # noqa: E402
from core.redis_client import redis_client  # noqa: E402
from core.settings import settings  # noqa: E402
from scripts.evaluate_full_agent_loop import evaluate_case  # noqa: E402
from evaluation.native_react_hard_benchmark import NativeReactHardCase  # noqa: E402
from evaluation.native_react_hard_runtime import (  # noqa: E402
    HardBenchmarkFaultExecutor,
    RUNTIME_SCHEMA_VERSION,
    cluster_bootstrap_rate_ci,
    load_frozen_split,
    score_hard_case,
    summarize_strata,
)


DEFAULT_BENCHMARK_DIR = ROOT / "evals" / "native-react-hard-v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--allow-frozen-test",
        action="store_true",
        help="Required to open the one-time final test split.",
    )
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--family", action="append", dest="families")
    parser.add_argument("--difficulty", action="append", dest="difficulties")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--policy-model")
    parser.add_argument("--policy-base-url")
    parser.add_argument("--policy-api-key", default="not-needed")
    parser.add_argument("--policy-temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.split == "test" and not args.allow_frozen_test:
        raise ValueError(
            "the frozen test split is sealed; pass --allow-frozen-test only for the final claim"
        )
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")


def rollout_seed(base_seed: int, case_id: str) -> int:
    payload = f"{base_seed}:{case_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def select_cases(
    cases: list[NativeReactHardCase],
    args: argparse.Namespace,
) -> list[NativeReactHardCase]:
    selected = list(cases)
    if args.case_ids:
        requested = set(args.case_ids)
        unknown = requested - {row.case_id for row in selected}
        if unknown:
            raise ValueError(f"unknown cases: {sorted(unknown)}")
        selected = [row for row in selected if row.case_id in requested]
    if args.families:
        families = set(args.families)
        selected = [row for row in selected if row.metadata.family in families]
    if args.difficulties:
        difficulties = set(args.difficulties)
        selected = [row for row in selected if row.metadata.difficulty in difficulties]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("case selection is empty")
    return selected


def _mean(records: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key) or 0) for row in records]
    return round(statistics.fmean(values), 3) if values else 0.0


def _stage_pass_rate(records: list[dict[str, Any]], predicate: Any) -> float:
    if not records:
        return 0.0
    return round(sum(bool(predicate(row)) for row in records) / len(records), 4)


def build_report(
    selected: list[NativeReactHardCase],
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    frozen_file_sha256: str,
    dataset_sha256: str,
    dataset_git_commit: str,
    policy_model: str,
    policy_backend: str,
) -> dict[str, Any]:
    passed = sum(bool(row.get("passed")) for row in records)
    latencies = [float(row.get("latency_ms") or 0) for row in records]
    recovery_rows = [row for row in records if row.get("recovery") is not None]
    planned_rows = [
        row
        for row in records
        if row.get("expected_outcome") in {"draft", "draft_or_safe_termination", "revision"}
    ]
    verifier_rows = [
        row
        for row in records
        if any(
            action.get("action") == "validate_itinerary"
            for action in row.get("actions") or []
        )
    ]
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "split": args.split,
            "frozen_file_sha256": frozen_file_sha256,
            "dataset_sha256": dataset_sha256,
            "dataset_git_commit": dataset_git_commit,
            "selected_case_ids": [row.case_id for row in selected],
            "test_split_explicitly_unsealed": bool(args.allow_frozen_test),
        },
        "runtime": {
            "execution_mode": "react",
            "policy_protocol": "native_tool",
            "policy_model": policy_model,
            "policy_backend": policy_backend,
            "intent_model": settings.llm_model,
            "fault_protocol": "declared-one-shot-at-production-action-boundary",
            "seed_protocol": "sha256-base-seed-case-id-v1",
            "base_seed": args.seed,
            "temperature": args.policy_temperature,
        },
        "summary": {
            "total": len(records),
            "passed": passed,
            "failed": len(records) - passed,
            "pass_rate": round(passed / len(records), 4) if records else 0.0,
            "cluster_bootstrap_95ci": cluster_bootstrap_rate_ci(
                records,
                samples=args.bootstrap_samples,
                seed=args.seed,
            ),
            "intent_contract_pass_rate": _stage_pass_rate(
                records,
                lambda row: not any(
                    "INTENT_SLOT" in str(code)
                    or "MISSING_CLARIFICATION_FIELD" in str(code)
                    for code in row.get("failures") or []
                ),
            ),
            "planned_terminal_success_rate": _stage_pass_rate(
                planned_rows,
                lambda row: row.get("passed") is True,
            ),
            "verifier_attempted_cases": len(verifier_rows),
            "verifier_hard_pass_cases": sum(
                row.get("validation_hard_pass") is True for row in verifier_rows
            ),
            "verifier_final_hard_pass_rate": _stage_pass_rate(
                verifier_rows,
                lambda row: row.get("validation_hard_pass") is True,
            ),
            "verifier_resolved_rate": _stage_pass_rate(
                verifier_rows,
                lambda row: row.get("passed") is True,
            ),
            "fault_injection_rate": _stage_pass_rate(
                recovery_rows,
                lambda row: bool((row.get("recovery") or {}).get("fault_injected")),
            ),
            "bounded_recovery_rate": _stage_pass_rate(
                recovery_rows,
                lambda row: bool((row.get("recovery") or {}).get("bounded_recovery")),
            ),
            "mean_tokens": _mean(records, "total_tokens"),
            "mean_policy_calls": _mean(records, "policy_calls"),
            "mean_tool_calls": _mean(records, "tool_calls"),
            "mean_latency_ms": _mean(records, "latency_ms"),
            "p95_latency_ms": round(percentile(latencies, 0.95), 3),
            "failure_counts": dict(
                Counter(code for row in records for code in row.get("failures") or [])
            ),
            "by_family": summarize_strata(records, "family"),
            "by_difficulty": summarize_strata(records, "difficulty"),
            "by_cluster": summarize_strata(records, "cluster_id"),
        },
        "records": records,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def load_resume_records(
    path: Path,
    *,
    args: argparse.Namespace,
    frozen_file_sha256: str,
    dataset_sha256: str,
    policy_model: str,
    policy_backend: str,
) -> list[dict[str, Any]]:
    """Reject partial reports produced by another dataset or inference protocol."""
    if not args.resume or not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid resume report: {exc}") from exc
    benchmark = payload.get("benchmark") or {}
    runtime = payload.get("runtime") or {}
    expected = {
        "benchmark.split": args.split,
        "benchmark.frozen_file_sha256": frozen_file_sha256,
        "benchmark.dataset_sha256": dataset_sha256,
        "runtime.policy_model": policy_model,
        "runtime.policy_backend": policy_backend,
        "runtime.base_seed": args.seed,
        "runtime.temperature": args.policy_temperature,
    }
    actual = {
        "benchmark.split": benchmark.get("split"),
        "benchmark.frozen_file_sha256": benchmark.get("frozen_file_sha256"),
        "benchmark.dataset_sha256": benchmark.get("dataset_sha256"),
        "runtime.policy_model": runtime.get("policy_model"),
        "runtime.policy_backend": runtime.get("policy_backend"),
        "runtime.base_seed": runtime.get("base_seed"),
        "runtime.temperature": runtime.get("temperature"),
    }
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError(
            "resume report protocol mismatch: "
            + ", ".join(f"{key}={actual.get(key)!r} expected {expected[key]!r}" for key in mismatches)
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("resume report records must be a list")
    return [dict(row) for row in records]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    loaded = load_frozen_split(args.benchmark_dir, args.split)
    selected = select_cases(loaded.cases, args)
    policy_model = args.policy_model or settings.agentic_policy_model or settings.llm_model
    if args.policy_base_url:
        client = LLMClient(
            base_url=args.policy_base_url,
            api_key=args.policy_api_key,
            using_vllm=True,
        )
        policy_backend = args.policy_base_url
    else:
        client = LLMClient()
        policy_backend = "configured_default"
    existing = load_resume_records(
        args.output,
        args=args,
        frozen_file_sha256=loaded.file_sha256,
        dataset_sha256=loaded.dataset_sha256,
        policy_model=policy_model,
        policy_backend=policy_backend,
    )
    by_id = {str(row["case_id"]): row for row in existing}
    policy = NativeToolAgentPolicy(
        client,
        model=policy_model,
        temperature=args.policy_temperature,
        max_tokens=256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    await redis_client.connect()
    try:
        for benchmark_case in selected:
            case_id = benchmark_case.case_id
            if case_id in by_id and not (
                args.rerun_failed and not bool(by_id[case_id].get("passed"))
            ):
                continue
            seed = rollout_seed(args.seed, case_id)
            policy.set_rollout_seed(seed)
            fault_executor = (
                HardBenchmarkFaultExecutor(benchmark_case.metadata.fault_spec)
                if benchmark_case.metadata.fault_spec is not None
                else None
            )
            base_record = await evaluate_case(
                benchmark_case.case,
                policy=policy,
                executor=fault_executor,
                rollout_id=f"hard-v2:{args.seed}",
            )
            record = score_hard_case(
                benchmark_case,
                base_record,
                fault_executor.trace if fault_executor is not None else None,
            )
            record["rollout_seed"] = seed
            by_id[case_id] = record
            records = [by_id[row.case_id] for row in selected if row.case_id in by_id]
            report = build_report(
                selected,
                records,
                args=args,
                frozen_file_sha256=loaded.file_sha256,
                dataset_sha256=loaded.dataset_sha256,
                dataset_git_commit=loaded.git_commit,
                policy_model=policy_model,
                policy_backend=policy_backend,
            )
            _write_report(args.output, report)
            print(
                json.dumps(
                    {
                        "progress": f"{len(records)}/{len(selected)}",
                        "case_id": case_id,
                        "family": benchmark_case.metadata.family,
                        "passed": record["passed"],
                        "failures": record["failures"],
                        "tokens": record.get("total_tokens"),
                        "policy_calls": record.get("policy_calls"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        await redis_client.disconnect()

    records = [by_id[row.case_id] for row in selected if row.case_id in by_id]
    report = build_report(
        selected,
        records,
        args=args,
        frozen_file_sha256=loaded.file_sha256,
        dataset_sha256=loaded.dataset_sha256,
        dataset_git_commit=loaded.git_commit,
        policy_model=policy_model,
        policy_backend=policy_backend,
    )
    _write_report(args.output, report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("sqlalchemy.engine").disabled = True
    final = asyncio.run(run(parse_args()))
    print(json.dumps(final["summary"], ensure_ascii=False, indent=2))
