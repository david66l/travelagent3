"""Run one real bounded Agent Loop episode with production dependencies.

This is intentionally separate from the HTTP benchmark: it isolates policy,
tool, solver and verifier latency while still connecting the shared Redis
services used by production skills.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from agentic.integration import run_agent_branch  # noqa: E402
from agentic.runtime import initialize_agent_ledger  # noqa: E402
from core.redis_client import redis_client  # noqa: E402
from evaluation.performance_budget import evaluate_episode_performance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="北京")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--start-date", default="2026-10-20")
    parser.add_argument("--end-date", default="2026-10-22")
    parser.add_argument("--budget", type=float, default=5000)
    parser.add_argument("--episode-output", type=Path)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict:
    state = {
        "user_input": (
            f"Plan a {args.days}-day {args.city} cultural food trip, "
            f"budget CNY {args.budget:.0f}, {args.start_date} to {args.end_date}"
        ),
        "slots": {
            "destination": args.city,
            "travel_days": args.days,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "budget_range": args.budget,
            "interests": ["历史文化", "美食"],
        },
        "profile": {
            "destination": args.city,
            "travel_days": args.days,
            "travel_dates": f"{args.start_date}|{args.end_date}",
            "budget_range": args.budget,
            "travelers_count": 2,
            "interests": ["历史文化", "美食"],
        },
    }
    initialized = initialize_agent_ledger(state, mode="agent")
    await redis_client.connect()
    started = time.perf_counter()
    try:
        result = await run_agent_branch({**state, **initialized})
    finally:
        elapsed = time.perf_counter() - started
        await redis_client.disconnect()

    ledger = result.get("agent_ledger") or {}
    budget = ledger.get("budget") or {}
    episode = result.get("agent_episode") or {}
    if args.episode_output and episode:
        args.episode_output.parent.mkdir(parents=True, exist_ok=True)
        args.episode_output.write_text(
            json.dumps(episode, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {
        "elapsed_s": round(elapsed, 2),
        "status": result.get("agent_status"),
        "termination_reason": result.get("termination_reason"),
        "stage": result.get("stage"),
        "solve_status": result.get("solve_status"),
        "days": len(result.get("itinerary") or []),
        "validation_hard_pass": (result.get("validation_report") or {}).get("hard_pass"),
        "used_steps": budget.get("used_episode_steps"),
        "used_tool_calls": budget.get("used_tool_calls"),
        "used_solver_calls": budget.get("used_solver_calls"),
        "used_tokens": budget.get("used_tokens"),
        "used_latency_ms": budget.get("used_latency_ms"),
        "actions": [
            {
                "task": step.get("task_id"),
                "action": (step.get("action") or {}).get("action"),
                "source": (step.get("action") or {}).get("decision_source"),
                "policy_latency_ms": step.get("policy_latency_ms", 0),
                "action_latency_ms": step.get("action_latency_ms", 0),
            }
            for step in episode.get("steps") or []
        ],
        "failures": ledger.get("failures") or [],
        "performance_budget": (
            evaluate_episode_performance(episode).model_dump(mode="json") if episode else None
        ),
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run(parse_args())), ensure_ascii=False, indent=2))
