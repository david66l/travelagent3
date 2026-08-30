"""Evaluate one OpenAI-compatible policy checkpoint on an audited SFT split."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from agentic.policy_actions import validate_policy_arguments  # noqa: E402
from agentic.sft_dataset import SFTExample  # noqa: E402
from core.llm_client import LLMClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args()


def _load_examples(path: Path, limit: int | None) -> list[SFTExample]:
    examples = [
        SFTExample(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return examples[:limit] if limit is not None else examples


async def _evaluate_one(
    example: SFTExample,
    *,
    clients: asyncio.Queue[LLMClient],
    model: str,
    temperature: float,
) -> dict[str, Any]:
    expected_call = example.messages[-1].tool_calls[0].function
    messages = [message.model_dump(exclude_none=True) for message in example.messages[:-1]]
    started = time.perf_counter()
    client = await clients.get()
    try:
        actual = await client.tool_call(
            messages,
            example.tools,
            temperature=temperature,
            max_tokens=256,
            model_override=model,
        )
        error = None
        token_usage = int(client.last_token_usage or 0)
    except Exception as exc:
        actual = {"action": None, "arguments": {}}
        error = f"{type(exc).__name__}:{exc}"
        token_usage = int(client.last_token_usage or 0)
    finally:
        clients.put_nowait(client)
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    action = actual.get("action")
    arguments = actual.get("arguments") or {}
    schema_valid = False
    if isinstance(action, str) and isinstance(arguments, dict):
        try:
            validate_policy_arguments(action, arguments)
            schema_valid = True
        except ValueError:
            pass
    action_correct = action == expected_call.name
    arguments_exact = error is None and arguments == expected_call.arguments
    return {
        "example_id": example.example_id,
        "scenario_id": example.scenario_id,
        "expected_action": expected_call.name,
        "expected_arguments": expected_call.arguments,
        "actual_action": action,
        "actual_arguments": arguments,
        "call_success": error is None,
        "schema_valid": schema_valid,
        "action_correct": action_correct,
        "arguments_exact": arguments_exact,
        "full_exact": action_correct and arguments_exact,
        "latency_ms": latency_ms,
        "tokens": token_usage,
        "error": error,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    examples = _load_examples(
        args.dataset_dir / f"{args.split}.jsonl",
        args.limit,
    )
    clients: asyncio.Queue[LLMClient] = asyncio.Queue()
    for _ in range(max(1, args.concurrency)):
        clients.put_nowait(
            LLMClient(base_url=args.base_url, api_key=args.api_key, using_vllm=True)
        )
    records = await asyncio.gather(
        *(
            _evaluate_one(
                example,
                clients=clients,
                model=args.model,
                temperature=args.temperature,
            )
            for example in examples
        )
    )
    total = len(records)
    latencies = [float(record["latency_ms"]) for record in records]
    report = {
        "schema_version": "agent-policy-holdout-eval.v1",
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "summary": {
            "total": total,
            "call_success_rate": round(
                sum(record["call_success"] for record in records) / total, 4
            )
            if total
            else 0.0,
            "schema_valid_rate": round(
                sum(record["schema_valid"] for record in records) / total, 4
            )
            if total
            else 0.0,
            "action_accuracy": round(
                sum(record["action_correct"] for record in records) / total, 4
            )
            if total
            else 0.0,
            "argument_exact_rate": round(
                sum(record["arguments_exact"] for record in records) / total, 4
            )
            if total
            else 0.0,
            "full_exact_rate": round(
                sum(record["full_exact"] for record in records) / total, 4
            )
            if total
            else 0.0,
            "mean_latency_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "total_tokens": sum(int(record["tokens"]) for record in records),
            "expected_actions": dict(Counter(record["expected_action"] for record in records)),
            "actual_actions": dict(Counter(str(record["actual_action"]) for record in records)),
            "errors": dict(
                Counter(record["error"] for record in records if record["error"])
            ),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = parse_args()
    report = asyncio.run(run(args))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
