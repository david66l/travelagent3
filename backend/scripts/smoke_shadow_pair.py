"""Create one real deterministic/Agent shadow pair and print its metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import date, timedelta

from agentic.runtime import initialize_agent_ledger
from agentic.shadow import (
    ShadowProvenance,
    record_deterministic_shadow_result,
    start_shadow_run,
)
from core.database import async_session_maker
from core.settings import settings
from evaluation.validator import ItineraryValidator
from graph.nodes import plan_node, retrieve_node, weather_check_node
from repositories.agentic_evaluation import AgenticEvaluationRepository


async def run(city: str, days: int, budget: float, timeout: float) -> int:
    start_date = date.today() + timedelta(days=14)
    end_date = start_date + timedelta(days=days - 1)
    slots = {
        "destination": city,
        "travel_days": days,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "budget_range": budget,
        "interests": ["history", "food"],
    }
    state = {
        "user_input": f"Plan {days} days in {city} for {budget}",
        "slots": slots,
        "profile": slots,
        "missing_slots": [],
        "feasibility_report": {"feasible": True},
    }
    state.update(initialize_agent_ledger(state, mode="shadow"))
    state.update(
        await start_shadow_run(
            state,
            provenance=ShadowProvenance(
                evaluation_source="synthetic_smoke",
                deployment_id=settings.agentic_deployment_id,
                batch_id="manual-smoke",
                release_gate_eligible=False,
            ),
            force_sample=True,
        )
    )
    scenario_id = state.get("shadow_scenario_id")
    if not scenario_id:
        print(json.dumps(state, ensure_ascii=False, default=str))
        return 2

    started = time.perf_counter()
    retrieval, weather = await asyncio.gather(retrieve_node(state), weather_check_node(state))
    state.update(retrieval)
    state.update(weather)
    state.update(await plan_node(state))
    state["validation_report"] = (
        ItineraryValidator()
        .validate(
            state.get("itinerary") or [],
            constraints={
                "travel_days": days,
                "total_budget": budget,
                "must_visit": [],
                "interests": slots["interests"],
            },
            facts=state.get("poi_candidates") or [],
        )
        .model_dump(mode="json")
    )
    await record_deterministic_shadow_result(
        state, latency_ms=int((time.perf_counter() - started) * 1000)
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        async with async_session_maker() as db:
            repo = AgenticEvaluationRepository(db)
            deterministic = await repo.get(scenario_id=scenario_id, mode="deterministic")
            agent = await repo.get(scenario_id=scenario_id, mode="agent")
        if agent and agent.status in {"completed", "failed"}:
            print(
                json.dumps(
                    {
                        "scenario_id": scenario_id,
                        "deterministic": {
                            "status": deterministic.status if deterministic else "missing",
                            "metrics": deterministic.metrics if deterministic else None,
                        },
                        "agent": {
                            "status": agent.status,
                            "metrics": agent.metrics,
                            "error": agent.error,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            return 0 if agent.status == "completed" else 1
        await asyncio.sleep(1)
    print(json.dumps({"scenario_id": scenario_id, "error": "timeout"}))
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Shanghai")
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--budget", type=float, default=3000)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.city, args.days, args.budget, args.timeout)))


if __name__ == "__main__":
    main()
