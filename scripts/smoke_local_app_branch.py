"""Exercise the configured local checkpoint through the App's Agent branch."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.action_executor import TravelActionExecutor  # noqa: E402
from agentic.corpus_generation import build_curriculum_case  # noqa: E402
from agentic.environment import SnapshotToolExecutor  # noqa: E402
from agentic.integration import run_agent_branch  # noqa: E402
from agentic.runtime import initialize_agent_ledger  # noqa: E402
from core.settings import settings  # noqa: E402


async def run(scenario: int) -> dict:
    if settings.agentic_policy_backend != "local_checkpoint":
        raise RuntimeError("AGENTIC_POLICY_BACKEND must be local_checkpoint")
    task, snapshot = build_curriculum_case(scenario)
    initialized = initialize_agent_ledger(
        {
            "user_input": task.user_request,
            "slots": task.slots,
            "profile": task.profile,
            "missing_slots": task.missing_slots,
            "feasibility_report": task.feasibility_report,
        },
        mode="agent",
    )
    patch = await run_agent_branch(
        initialized,
        executor=TravelActionExecutor(SnapshotToolExecutor(snapshot)),
    )
    episode = patch.get("agent_episode") or {}
    return {
        "policy_backend": settings.agentic_policy_backend,
        "checkpoint": settings.agentic_local_checkpoint,
        "task_id": task.task_id,
        "agent_status": patch.get("agent_status"),
        "stage": patch.get("stage"),
        "termination_reason": patch.get("termination_reason"),
        "policy_name": episode.get("policy_name"),
        "policy_version": episode.get("policy_version"),
        "actions": [
            step.get("action", {}).get("action") for step in episode.get("steps", [])
        ],
        "hard_pass": (patch.get("validation_report") or {}).get("hard_pass"),
        "itinerary_days": len(patch.get("itinerary") or []),
        "content_hash": episode.get("content_hash"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.scenario)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
