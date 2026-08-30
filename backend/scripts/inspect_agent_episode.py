"""Print a compact task/action trace for one persisted Agent episode."""

from __future__ import annotations

import argparse
import asyncio
import json

from sqlalchemy import select

from core.database import async_session_maker
from models.agentic_evaluation import AgenticEvaluationRecord


async def run(scenario_id: str) -> None:
    async with async_session_maker() as db:
        record = (
            await db.execute(
                select(AgenticEvaluationRecord).where(
                    AgenticEvaluationRecord.scenario_id == scenario_id,
                    AgenticEvaluationRecord.mode == "agent",
                )
            )
        ).scalar_one()
    episode = record.episode or {}
    steps = []
    for step in episode.get("steps") or []:
        steps.append(
            {
                "task": step["task_id"],
                "action": step["action"]["action"],
                "verification": step["verification"],
                "observations": [
                    {
                        "tool": item["tool"],
                        "ok": item["ok"],
                        "error": (item.get("error") or {}).get("code"),
                        "message": (item.get("error") or {}).get("message"),
                    }
                    for item in step["observations"]
                ],
            }
        )
    tasks = ((episode.get("final_state") or {}).get("task_graph") or {}).get("tasks") or []
    print(json.dumps({"steps": steps, "tasks": tasks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_id")
    asyncio.run(run(parser.parse_args().scenario_id))
