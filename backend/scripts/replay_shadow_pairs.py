"""Run labeled, resumable deterministic/Agent pairs at concurrency one."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from agentic.runtime import initialize_agent_ledger
from agentic.shadow import (
    ShadowProvenance,
    record_deterministic_shadow_result,
    start_shadow_run,
)
from core.database import async_session_maker
from core.settings import settings
from evaluation.agentic_eval import AgenticEvaluator, ReleaseGateConfig
from evaluation.shadow_replay import (
    AuthorizedReplayCase,
    load_authorized_replay_cases,
    replay_case_state,
    replay_scenario_id,
)
from evaluation.validator import ItineraryValidator
from graph.nodes import plan_node, retrieve_node, weather_check_node
from repositories.agentic_evaluation import AgenticEvaluationRepository


async def _records(scenario_id: str):
    async with async_session_maker() as db:
        repo = AgenticEvaluationRepository(db)
        return (
            await repo.get(scenario_id=scenario_id, mode="deterministic"),
            await repo.get(scenario_id=scenario_id, mode="agent"),
        )


async def _record_deterministic(case: AuthorizedReplayCase, state: dict) -> None:
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
                "travel_days": case.travel_days,
                "total_budget": case.budget,
                "must_visit": [],
                "interests": case.interests,
            },
            facts=state.get("poi_candidates") or [],
        )
        .model_dump(mode="json")
    )
    await record_deterministic_shadow_result(
        state, latency_ms=int((time.perf_counter() - started) * 1000)
    )


async def _run_case(
    case: AuthorizedReplayCase,
    *,
    deployment_id: str,
    batch_id: str,
    timeout: float,
) -> dict:
    scenario_id = replay_scenario_id(
        deployment_id=deployment_id,
        batch_id=batch_id,
        case_id=case.case_id,
    )
    deterministic, agent = await _records(scenario_id)
    if deterministic and agent and {deterministic.status, agent.status} == {"completed"}:
        return {"case_id": case.case_id, "scenario_id": scenario_id, "status": "skipped"}
    if agent and agent.status == "failed":
        return {
            "case_id": case.case_id,
            "scenario_id": scenario_id,
            "status": "failed_existing",
        }

    state = replay_case_state(case)
    initialized = initialize_agent_ledger(state, mode="shadow")
    initialized["agent_ledger"]["trajectory_id"] = scenario_id
    state.update(initialized)
    provenance = ShadowProvenance(
        evaluation_source="authorized_replay",
        deployment_id=deployment_id,
        batch_id=batch_id,
        source_case_id=case.case_id,
        release_gate_eligible=case.release_gate_eligible,
    )
    state.update(await start_shadow_run(state, provenance=provenance, force_sample=True))
    if state.get("shadow_status") not in {"running", "already_started"}:
        return {
            "case_id": case.case_id,
            "scenario_id": scenario_id,
            "status": str(state.get("shadow_status")),
        }
    if deterministic is None or deterministic.status != "completed":
        await _record_deterministic(case, state)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        deterministic, agent = await _records(scenario_id)
        if agent and agent.status == "failed":
            return {
                "case_id": case.case_id,
                "scenario_id": scenario_id,
                "status": "failed",
            }
        if deterministic and agent and {deterministic.status, agent.status} == {"completed"}:
            return {
                "case_id": case.case_id,
                "scenario_id": scenario_id,
                "status": "completed",
            }
        await asyncio.sleep(1)
    return {"case_id": case.case_id, "scenario_id": scenario_id, "status": "timeout"}


async def run(args: argparse.Namespace) -> dict:
    cases = load_authorized_replay_cases(args.cases_file)
    selected = cases[args.offset : args.offset + args.limit]
    results = []
    for index, case in enumerate(selected, start=1):
        result = await _run_case(
            case,
            deployment_id=args.deployment_id,
            batch_id=args.batch_id,
            timeout=args.timeout_seconds,
        )
        results.append(result)
        print(
            json.dumps(
                {"progress": f"{index}/{len(selected)}", **result},
                ensure_ascii=False,
            ),
            flush=True,
        )
        failures = sum(item["status"] not in {"completed", "skipped"} for item in results)
        if failures >= args.max_failures:
            break

    async with async_session_maker() as db:
        deterministic, agent = await AgenticEvaluationRepository(db).completed_pairs(
            limit=10000,
            evaluation_source="authorized_replay",
            deployment_id=args.deployment_id,
            batch_id=args.batch_id,
            release_gate_eligible=True,
        )
    report = None
    if deterministic:
        report = (
            AgenticEvaluator()
            .compare(
                deterministic,
                agent,
                gate=ReleaseGateConfig(minimum_paired_scenarios=300),
            )
            .model_dump(mode="json")
        )
        report["quality_gates_passed"] = report["release_eligible"]
        report["release_eligible"] = False
        report["canary_evidence"] = False
        report["evaluation_source"] = "authorized_replay"
        report["deployment_id"] = args.deployment_id
        report["batch_id"] = args.batch_id
    summary = {
        "schema_version": "authorized-shadow-replay-report.v1",
        "deployment_id": args.deployment_id,
        "batch_id": args.batch_id,
        "selected_cases": len(selected),
        "attempted_cases": len(results),
        "completed_this_run": sum(item["status"] == "completed" for item in results),
        "skipped_completed": sum(item["status"] == "skipped" for item in results),
        "failed_or_timed_out": sum(
            item["status"] not in {"completed", "skipped"} for item in results
        ),
        "stopped_early": len(results) < len(selected),
        "results": results,
        "batch_report": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-file", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--deployment-id", default=settings.agentic_deployment_id)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--max-failures", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.limit, args.timeout_seconds, args.max_failures) <= 0 or args.offset < 0:
        parser.error(
            "limit, timeout and max-failures must be positive; offset must be non-negative"
        )
    summary = asyncio.run(run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failed_or_timed_out"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
