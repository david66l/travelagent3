"""Run one real-model ReAct itinerary episode through deterministic gates.

This script is intentionally checkpoint-agnostic.  It talks to any
OpenAI-compatible tool-calling endpoint, while the local application executes
the real knowledge, POI, route-matrix, CP-SAT and verifier implementations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

from agentic.action_executor import TravelActionExecutor
from agentic.loop import BoundedAgentLoop
from agentic.policy import ControllerFirstPolicy, NativeToolAgentPolicy, SelfRepairingAgentPolicy
from agentic.react import ReactTaskGraphPlanner
from agentic.state import AgentLedgerState, BudgetLedger, GoalLedger, TaskGraphController
from core.llm_client import LLMClient


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--model", required=True)
    parser.add_argument("--request", default="去上海玩2天，想看人文景点，节奏不要太赶")
    parser.add_argument("--destination", default="上海")
    parser.add_argument("--travel-days", type=int, default=2)
    parser.add_argument("--start-date")
    parser.add_argument("--origin")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--max-tool-calls", type=int, default=48)
    return parser.parse_args()


def _goal(args: argparse.Namespace) -> GoalLedger:
    hard_constraints: dict[str, Any] = {
        "destination": args.destination,
        "travel_days": args.travel_days,
        "intent_kind": "itinerary",
    }
    if args.start_date:
        hard_constraints["start_date"] = args.start_date
    if args.origin:
        hard_constraints["origin"] = args.origin
    return GoalLedger(
        original_request=args.request,
        success_definition=[
            "generate an executable itinerary",
            "pass deterministic hard-constraint validation",
            "wait for user confirmation",
        ],
        hard_constraints=hard_constraints,
        soft_preferences={"interests": ["人文"], "pace": "relaxed"},
    )


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    goal = _goal(args)
    graph = TaskGraphController().refresh_ready(ReactTaskGraphPlanner().plan(goal))
    ledger = AgentLedgerState(
        goal=goal,
        task_graph=graph,
        budget=BudgetLedger(
            max_episode_steps=args.max_steps,
            max_tool_calls=args.max_tool_calls,
            max_solver_calls=3,
            max_tokens=64_000,
            timeout_ms=300_000,
        ),
    )
    client = LLMClient(
        base_url=args.base_url,
        api_key=args.api_key,
        using_vllm=False,
    )
    listed_models = await client.client.models.list()
    available_model_ids = [str(item.id) for item in listed_models.data]
    if args.model not in available_model_ids:
        raise RuntimeError(
            f"requested model {args.model!r} is not served by {args.base_url}; "
            f"available models: {available_model_ids}"
        )
    model_policy = NativeToolAgentPolicy(
        client,
        model=args.model,
        temperature=0.1,
        max_tokens=256,
    )
    policy = ControllerFirstPolicy(SelfRepairingAgentPolicy(model_policy))
    result = await BoundedAgentLoop().run(
        ledger,
        policy=policy,
        executor=TravelActionExecutor(),
    )
    final_ledger = result.ledger
    current_artifacts = [
        item
        for item in final_ledger.artifacts.values()
        if item.goal_version == final_ledger.goal.goal_version
        and item.plan_version == final_ledger.task_graph.plan_version
    ]
    solver_result = next(
        (
            item.payload
            for item in reversed(current_artifacts)
            if item.artifact_type == "solver_result"
        ),
        {},
    )
    validation_report = next(
        (
            item.payload
            for item in reversed(current_artifacts)
            if item.artifact_type == "validation_report"
        ),
        {},
    )
    return {
        "status": result.status,
        "termination_reason": result.termination_reason,
        "served_models": available_model_ids,
        "termination_details": next(
            (
                item.payload
                for item in reversed(result.events)
                if item.event_type == "episode_terminated"
            ),
            {},
        ),
        "budget": final_ledger.budget.model_dump(mode="json"),
        "solver_status": solver_result.get("status"),
        "validation_hard_pass": validation_report.get("hard_pass"),
        "tasks": [
            {
                "task_id": task.task_id,
                "status": task.status,
                "attempts": task.attempts,
                "failure": task.failure,
            }
            for task in final_ledger.task_graph.tasks
        ],
        "decisions": [
            {
                "task_id": item.task_id,
                "action": item.action,
                "arguments": item.arguments,
                "outcome_status": item.outcome_status,
                "progress_made": item.progress_made,
            }
            for item in final_ledger.decision_history
        ],
        "artifact_types": sorted({item.artifact_type for item in current_artifacts}),
        "failures": [
            {
                "task_id": item.task_id,
                "code": item.code,
                "message": item.message,
                "attempted_strategy": item.attempted_strategy,
            }
            for item in final_ledger.failures
        ],
        "event_types": [item.event_type for item in result.events],
    }


def main() -> None:
    args = _arguments()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    result = asyncio.run(_run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
