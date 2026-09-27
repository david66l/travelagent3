"""Evaluate one policy on paired, seeded full-loop fault-recovery rollouts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from agentic.policy import NativeToolAgentPolicy  # noqa: E402
from agentic.sft_dataset import EpisodeCandidate  # noqa: E402
from core.inference_metrics import percentile  # noqa: E402
from core.llm_client import LLMClient  # noqa: E402
from core.redis_client import redis_client  # noqa: E402
from core.settings import settings  # noqa: E402
from evaluate_full_agent_loop import _write_episode_candidates, evaluate_case  # noqa: E402
from evaluation.full_agent_loop_recovery import (  # noqa: E402
    SCHEMA_VERSION,
    OneShotFaultExecutor,
    benchmark_hash,
    build_recovery_cases,
    score_recovery,
)


SCORING_CONTRACT_VERSION = "recovery-injection-accounting.v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--episode-output",
        type=Path,
        required=True,
        help="EpisodeCandidate JSONL sidecar with complete recovery observations.",
    )
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument(
        "--rollout-id",
        required=True,
        help="Unique run label included in benchmark user identities.",
    )
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--policy-base-url", required=True)
    parser.add_argument("--policy-api-key", default="not-needed")
    parser.add_argument("--policy-temperature", type=float, default=0.8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    return parser.parse_args()


def paired_rollout_seed(base_seed: int, *, case_id: str, sample_index: int) -> int:
    payload = f"{base_seed}:{case_id}:{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _rollout_key(case_id: str, sample_index: int) -> str:
    return f"{case_id}::sample-{sample_index}"


def _mean(records: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key) or 0) for row in records]
    return round(statistics.fmean(values), 3) if values else 0.0


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _stratum_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    injected = [
        row
        for row in rows
        if (row.get("recovery") or {}).get("injection_status") == "injected_once"
    ]
    first_try = sum(
        bool((row.get("recovery") or {}).get("first_try_recovery")) for row in rows
    )
    return {
        "total": len(rows),
        "passed": sum(bool(row.get("passed")) for row in rows),
        "pass_rate": _rate(sum(bool(row.get("passed")) for row in rows), len(rows)),
        "injection_reach_rate": _rate(len(injected), len(rows)),
        "first_try_recovery_rate": _rate(first_try, len(rows)),
        "conditional_first_try_recovery_rate": _rate(first_try, len(injected)),
        "full_chain_pass_rate": _rate(
            sum(bool(row.get("base_passed")) for row in rows), len(rows)
        ),
    }


def build_report(
    selected_cases: list[Any],
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    passed = sum(bool(row.get("passed")) for row in records)
    recovery_passed = sum(
        bool((row.get("recovery") or {}).get("first_try_recovery")) for row in records
    )
    full_chain_passed = sum(bool(row.get("base_passed")) for row in records)
    injected_records = [
        row
        for row in records
        if (row.get("recovery") or {}).get("injection_status") == "injected_once"
    ]
    injection_status_counts = Counter(
        str((row.get("recovery") or {}).get("injection_status") or "missing")
        for row in records
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        recovery = row.get("recovery") or {}
        key = f"{recovery.get('scenario')}/{recovery.get('evidence_style')}"
        grouped[key].append(row)
    strata = {key: _stratum_summary(rows) for key, rows in sorted(grouped.items())}
    latencies = [float(row.get("latency_ms") or 0) for row in records]
    return {
        "schema_version": SCHEMA_VERSION,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_hash": benchmark_hash(),
        "selected_case_ids": [row.case_id for row in selected_cases],
        "policy_model": args.policy_model,
        "policy_backend": args.policy_base_url,
        "intent_model": settings.llm_model,
        "execution_mode": "react",
        "fault_protocol": "one-shot-search-failure-then-real-tools",
        "seed_protocol": "sha256-case-sample-v1",
        "base_seed": args.seed,
        "rollout_id": args.rollout_id,
        "group_size": args.group_size,
        "temperature": args.policy_temperature,
        "summary": {
            "total": len(records),
            "passed": passed,
            "failed": len(records) - passed,
            "pass_rate": round(passed / len(records), 4) if records else 0.0,
            "end_to_end_strict_pass_rate": _rate(passed, len(records)),
            "injection_reach_rate": _rate(len(injected_records), len(records)),
            "injection_status_counts": dict(injection_status_counts),
            "first_try_recovery_rate": round(recovery_passed / len(records), 4)
            if records
            else 0.0,
            "conditional_first_try_recovery_rate": _rate(
                recovery_passed, len(injected_records)
            ),
            "conditional_full_chain_pass_rate": _rate(
                sum(bool(row.get("base_passed")) for row in injected_records),
                len(injected_records),
            ),
            "full_chain_pass_rate": round(full_chain_passed / len(records), 4)
            if records
            else 0.0,
            "mean_tokens": _mean(records, "total_tokens"),
            "mean_latency_ms": _mean(records, "latency_ms"),
            "p95_latency_ms": round(percentile(latencies, 0.95), 3),
            "mean_policy_calls": _mean(records, "policy_calls"),
            "mean_tool_calls": _mean(records, "tool_calls"),
            "failure_counts": dict(
                Counter(code for row in records for code in row.get("failures") or [])
            ),
            "strata": strata,
        },
        "records": records,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.group_size < 1:
        raise ValueError("--group-size must be positive")
    cases = build_recovery_cases()
    if args.case_ids:
        requested = set(args.case_ids)
        unknown = requested - {case.case_id for case in cases}
        if unknown:
            raise ValueError(f"unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case.case_id in requested]
    if args.limit is not None:
        cases = cases[: args.limit]

    existing: list[dict[str, Any]] = []
    if args.resume and args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8")).get("records", [])
    by_key = {str(row["rollout_key"]): row for row in existing}
    episode_candidates_by_id: dict[str, EpisodeCandidate] = {}
    if args.resume and args.episode_output.is_file():
        for line_number, line in enumerate(
            args.episode_output.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                candidate = EpisodeCandidate(**json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{args.episode_output}:{line_number}: {exc}") from exc
            episode_candidates_by_id[candidate.scenario_id] = candidate
    client = LLMClient(
        base_url=args.policy_base_url,
        api_key=args.policy_api_key,
        using_vllm=True,
    )
    policy = NativeToolAgentPolicy(
        client,
        model=args.policy_model,
        temperature=args.policy_temperature,
        max_tokens=256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.episode_output.parent.mkdir(parents=True, exist_ok=True)
    await redis_client.connect()
    try:
        total = len(cases) * args.group_size
        for case in cases:
            for sample_index in range(args.group_size):
                key = _rollout_key(case.case_id, sample_index)
                if key in by_key and not (
                    args.rerun_failed and not bool(by_key[key].get("passed"))
                ):
                    continue
                rollout_seed = paired_rollout_seed(
                    args.seed,
                    case_id=case.case_id,
                    sample_index=sample_index,
                )
                policy.set_rollout_seed(rollout_seed)
                executor = OneShotFaultExecutor(case.fault)
                episode_collector: list[EpisodeCandidate] = []
                effective_rollout_id = f"{args.rollout_id}:{args.seed}:{sample_index}"
                prefix = f"{case.case_id}:{effective_rollout_id}:"
                for scenario_id in [
                    item for item in episode_candidates_by_id if item.startswith(prefix)
                ]:
                    episode_candidates_by_id.pop(scenario_id)
                base_record = await evaluate_case(
                    case.case,
                    policy=policy,
                    executor=executor,
                    rollout_id=effective_rollout_id,
                    episode_collector=episode_collector,
                )
                for candidate in episode_collector:
                    episode_candidates_by_id[candidate.scenario_id] = candidate
                _write_episode_candidates(
                    args.episode_output,
                    list(episode_candidates_by_id.values()),
                )
                record = score_recovery(case, base_record, executor.trace)
                record.update(
                    {
                        "rollout_key": key,
                        "sample_index": sample_index,
                        "rollout_seed": rollout_seed,
                    }
                )
                by_key[key] = record
                records = [
                    by_key[_rollout_key(item.case_id, index)]
                    for item in cases
                    for index in range(args.group_size)
                    if _rollout_key(item.case_id, index) in by_key
                ]
                args.output.write_text(
                    json.dumps(
                        build_report(cases, records, args=args),
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(
                    json.dumps(
                        {
                            "progress": f"{len(records)}/{total}",
                            "rollout_key": key,
                            "passed": record["passed"],
                            "first_try_recovery": record["recovery"][
                                "first_try_recovery"
                            ],
                            "full_chain_passed": record["base_passed"],
                            "failures": record["failures"],
                            "tokens": record.get("total_tokens"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        await redis_client.disconnect()

    records = [
        by_key[_rollout_key(item.case_id, index)]
        for item in cases
        for index in range(args.group_size)
        if _rollout_key(item.case_id, index) in by_key
    ]
    report = build_report(cases, records, args=args)
    _write_episode_candidates(
        args.episode_output,
        list(episode_candidates_by_id.values()),
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("sqlalchemy.engine").disabled = True
    final = asyncio.run(run(parse_args()))
    print(json.dumps(final["summary"], ensure_ascii=False, indent=2))
