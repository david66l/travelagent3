"""Benchmark TTFT, TPOT, tokens and tool-call correctness over vLLM HTTP."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from evaluation.inference_benchmark import (  # noqa: E402
    StreamAccumulator,
    VLLMBenchmarkCase,
    build_chat_payload,
    summarize_benchmark_runs,
)
from agentic.loop import PolicyContext  # noqa: E402
from agentic.policy import (  # noqa: E402
    AGENT_TOOL_POLICY_SYSTEM_PROMPT,
    constrain_policy_context,
    policy_prompt_payload,
)
from agentic.policy_actions import policy_action_schemas  # noqa: E402


def project_runtime_policy_case(case: VLLMBenchmarkCase) -> VLLMBenchmarkCase:
    """Rebuild a frozen case exactly as the current native-tool policy sends it."""
    user_messages = [message for message in case.messages if message.get("role") == "user"]
    if not user_messages:
        raise ValueError(f"case {case.case_id} has no user policy context")
    raw_content = user_messages[-1].get("content")
    if not isinstance(raw_content, str):
        raise ValueError(f"case {case.case_id} user policy context is not JSON text")
    try:
        context = PolicyContext(**json.loads(raw_content))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"case {case.case_id} has invalid policy context: {exc}") from exc
    constrained = constrain_policy_context(context)
    return case.model_copy(
        deep=True,
        update={
            "messages": [
                {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        policy_prompt_payload(constrained),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "tools": policy_action_schemas(constrained.allowed_actions),
            "allowed_actions": constrained.allowed_actions,
        },
    )


def load_cases(
    path: Path, *, runtime_policy_projection: bool = False
) -> list[VLLMBenchmarkCase]:
    cases = [
        VLLMBenchmarkCase(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if runtime_policy_projection:
        cases = [project_runtime_policy_case(case) for case in cases]
    if not cases:
        raise ValueError("benchmark case file is empty")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("benchmark case_id values must be unique")
    return cases


async def run_one(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    case: VLLMBenchmarkCase,
    model: str,
    repetition: int,
    max_tokens: int,
    temperature: float,
    thinking_enabled: bool,
) -> dict[str, Any]:
    payload = build_chat_payload(
        case,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking_enabled=thinking_enabled,
    )
    accumulator = StreamAccumulator()
    started = time.perf_counter()
    http_error: str | None = None
    try:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode(errors="replace")[:2000]
                http_error = f"HTTP {response.status_code}: {body}"
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    accumulator.consume(
                        json.loads(data),
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                    )
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        http_error = f"{type(exc).__name__}: {exc}"[:2000]
    total_latency_ms = (time.perf_counter() - started) * 1000
    observed_actions = [accumulator.tool_names[index] for index in sorted(accumulator.tool_names)]
    action_mismatch = bool(
        (case.tools and len(observed_actions) != 1)
        or (
            case.allowed_actions
            and any(action not in case.allowed_actions for action in observed_actions)
        )
        or (case.expected_action and case.expected_action not in observed_actions)
    )
    observed_arguments: dict[str, Any] | None = None
    if len(accumulator.tool_arguments) == 1:
        try:
            observed_arguments = json.loads(next(iter(accumulator.tool_arguments.values())))
        except (json.JSONDecodeError, TypeError):
            observed_arguments = None
    argument_mismatch = bool(
        case.expected_arguments is not None
        and observed_arguments != case.expected_arguments
    )
    label_contract_conflict = bool(
        case.expected_action
        and case.allowed_actions
        and case.expected_action not in case.allowed_actions
    )
    policy_contract_success = bool(
        http_error is None
        and len(observed_actions) == 1
        and (
            not case.allowed_actions
            or observed_actions[0] in case.allowed_actions
        )
        and (
            label_contract_conflict
            or not case.expected_action
            or case.expected_action in observed_actions
        )
        and (label_contract_conflict or not argument_mismatch)
    )
    metrics = accumulator.inference_metrics(
        model=model,
        total_latency_ms=total_latency_ms,
        thinking_enabled=thinking_enabled,
    )
    return {
        "case_id": case.case_id,
        "family": case.family,
        "repetition": repetition,
        "expected_action": case.expected_action,
        "expected_arguments": case.expected_arguments,
        "allowed_actions": case.allowed_actions,
        "observed_actions": observed_actions,
        "tool_arguments": accumulator.tool_arguments,
        "observed_arguments": observed_arguments,
        "success": http_error is None and not action_mismatch and not argument_mismatch,
        "policy_contract_success": policy_contract_success,
        "label_contract_conflict": label_contract_conflict,
        "http_error": http_error,
        "action_mismatch": action_mismatch,
        "argument_mismatch": argument_mismatch,
        "reasoning_detected": bool(
            accumulator.reasoning_content or "<think>" in accumulator.content
        ),
        "inference_metrics": metrics.model_dump(mode="json"),
    }


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(
        args.cases_file,
        runtime_policy_projection=args.runtime_policy_projection,
    )
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case.case_id in selected]
        missing = selected - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"unknown case_id values: {sorted(missing)}")
    headers = {"Authorization": f"Bearer {os.environ.get(args.api_key_env, 'not-needed')}"}
    timeout = httpx.Timeout(args.timeout_seconds)
    limits = httpx.Limits(max_connections=max(args.concurrency, 1))
    async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits) as client:
        for index in range(args.warmup):
            await run_one(
                client,
                base_url=args.base_url,
                case=cases[index % len(cases)],
                model=args.model,
                repetition=-(index + 1),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                thinking_enabled=args.thinking == "enabled",
            )

        semaphore = asyncio.Semaphore(args.concurrency)

        async def bounded(case: VLLMBenchmarkCase, repetition: int) -> dict[str, Any]:
            async with semaphore:
                return await run_one(
                    client,
                    base_url=args.base_url,
                    case=case,
                    model=args.model,
                    repetition=repetition,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    thinking_enabled=args.thinking == "enabled",
                )

        measured_started = time.perf_counter()
        runs = await asyncio.gather(
            *(
                bounded(case, repetition)
                for repetition in range(args.repetitions)
                for case in cases
            )
        )
        measured_duration_s = time.perf_counter() - measured_started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "runs.jsonl").open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run, ensure_ascii=False) + "\n")
    summary = summarize_benchmark_runs(runs)
    summary["wall_clock_seconds"] = round(measured_duration_s, 3)
    summary["request_throughput_per_second"] = round(
        len(runs) / measured_duration_s if measured_duration_s else 0.0,
        3,
    )
    report = {
        "schema_version": "vllm-http-benchmark.v1",
        "scope": "streaming HTTP inference; excludes model startup and tool execution",
        "model": args.model,
        "base_url": args.base_url,
        "thinking_mode": args.thinking,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "cases": len(cases),
        "runtime_policy_projection": args.runtime_policy_projection,
        "summary": summary,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases-file", type=Path, required=True)
    parser.add_argument(
        "--case-id",
        action="append",
        help="run only this case_id; repeat the option to select multiple cases",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--runtime-policy-projection",
        action="store_true",
        help="rebuild frozen cases with the current controller constraints and tool schemas",
    )
    args = parser.parse_args()
    if min(args.max_tokens, args.repetitions, args.concurrency) < 1:
        parser.error("max-tokens, repetitions and concurrency must be positive")
    if args.warmup < 0 or args.timeout_seconds <= 0:
        parser.error("warmup must be non-negative and timeout must be positive")
    report = asyncio.run(benchmark(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
