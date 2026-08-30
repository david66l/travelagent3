"""Evaluate demand and revision understanding on the frozen semantic set."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from agents.demand_parser import DemandParserAgent
from core.inference_metrics import InferenceMetrics, percentile, summarize_inference_metrics
from core.llm_client import llm
from evaluation.intent_revision_benchmark import (
    SCHEMA_VERSION,
    InitialSemanticCase,
    RevisionSemanticCase,
    benchmark_hash,
    build_frozen_cases,
    score_initial,
    score_revision,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--replay-report",
        type=Path,
        help="Rescore stored raw outputs with the current frozen contract without API calls.",
    )
    parser.add_argument(
        "--slice",
        action="append",
        dest="slices",
        help="Run only a named slice; may be repeated.",
    )
    return parser.parse_args()


def _metric_payload() -> dict[str, Any] | None:
    metric = llm.last_request_metrics
    if metric is None:
        return None
    return metric.model_dump(mode="json")


async def _evaluate_case(
    case: InitialSemanticCase | RevisionSemanticCase,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        started = time.perf_counter()
        parser = DemandParserAgent()
        try:
            if isinstance(case, InitialSemanticCase):
                result = await parser.parse(case.text, [], None)
                passed, failures = score_initial(case, result)
            else:
                result = await parser.parse_revision(
                    case.text,
                    current_goal=case.current_goal,
                )
                passed, failures = score_revision(case, result)
            output: dict[str, Any] | None = result.model_dump(mode="json")
            error = None
        except Exception as exc:  # The report must retain provider/schema failures.
            passed = False
            failures = [f"EXCEPTION:{type(exc).__name__}"]
            output = None
            error = str(exc)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        return {
            "case_id": case.case_id,
            "kind": case.kind,
            "slice": case.slice,
            "text": case.text,
            "passed": passed,
            "failures": failures,
            "error": error,
            "elapsed_ms": elapsed_ms,
            "inference": _metric_payload(),
            "token_usage": llm.last_token_usage,
            "output": output,
        }


def _group_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {
        name: {
            "total": len(items),
            "passed": sum(bool(item["passed"]) for item in items),
            "pass_rate": round(sum(bool(item["passed"]) for item in items) / len(items), 4),
        }
        for name, items in sorted(groups.items())
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    cases = build_frozen_cases()
    if args.slices:
        requested = set(args.slices)
        known = {case.slice for case in cases}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown slices: {', '.join(unknown)}")
        cases = [case for case in cases if case.slice in requested]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        cases = cases[: args.limit]

    if args.replay_report:
        source = json.loads(args.replay_report.read_text(encoding="utf-8"))
        stored = {record["case_id"]: record for record in source.get("records", [])}
        records = []
        for case in cases:
            previous = stored.get(case.case_id)
            if previous is None or previous.get("output") is None:
                records.append(
                    {
                        "case_id": case.case_id,
                        "kind": case.kind,
                        "slice": case.slice,
                        "text": case.text,
                        "passed": False,
                        "failures": ["MISSING_REPLAY_OUTPUT"],
                        "error": "stored output is unavailable",
                        "elapsed_ms": 0.0,
                        "inference": None,
                        "token_usage": 0,
                        "output": None,
                    }
                )
                continue
            if isinstance(case, InitialSemanticCase):
                from models.travel_slots import SlotParseOutput

                parsed = SlotParseOutput.model_validate(previous["output"])
                passed, failures = score_initial(case, parsed)
            else:
                from models.travel_slots import RevisionParseOutput

                parsed = RevisionParseOutput.model_validate(previous["output"])
                passed, failures = score_revision(case, parsed)
            records.append(
                {
                    **previous,
                    "passed": passed,
                    "failures": failures,
                    "error": None,
                    "token_usage": int(previous.get("token_usage") or 0),
                }
            )
    else:
        semaphore = asyncio.Semaphore(args.concurrency)
        records = await asyncio.gather(*(_evaluate_case(case, semaphore) for case in cases))
    inference = [
        InferenceMetrics(**record["inference"])
        for record in records
        if record["inference"] is not None
    ]
    passed = sum(bool(record["passed"]) for record in records)
    elapsed = [float(record["elapsed_ms"]) for record in records]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_hash": benchmark_hash(),
        "selection": {
            "slices": args.slices or [],
            "limit": args.limit,
            "concurrency": args.concurrency,
            "replay_report": str(args.replay_report) if args.replay_report else None,
        },
        "summary": {
            "total": len(records),
            "passed": passed,
            "failed": len(records) - passed,
            "pass_rate": round(passed / len(records), 4) if records else 0.0,
            "exception_count": sum(record["error"] is not None for record in records),
            "fallback_count": sum(
                "MODEL_FALLBACK_USED" in record["failures"] for record in records
            ),
            "case_latency_ms": {
                "mean": round(fmean(elapsed), 3) if elapsed else 0.0,
                "p50": round(percentile(elapsed, 0.50), 3),
                "p95": round(percentile(elapsed, 0.95), 3),
                "max": round(max(elapsed), 3) if elapsed else 0.0,
            },
            "by_kind": _group_summary(records, "kind"),
            "by_slice": _group_summary(records, "slice"),
        },
        "token_usage": {
            "total": sum(int(record["token_usage"]) for record in records),
            "mean_per_case": round(
                sum(int(record["token_usage"]) for record in records) / len(records), 3
            )
            if records
            else 0.0,
        },
        "last_request_inference_summary": summarize_inference_metrics(inference),
        "records": records,
    }


def main() -> None:
    args = _arguments()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("core.cost_circuit_breaker").setLevel(logging.ERROR)
    report = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
