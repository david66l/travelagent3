"""Run diverse user scenarios through the real ReAct loop and retain full episodes."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.policy import NativeToolAgentPolicy  # noqa: E402
from agentic.sft_dataset import EpisodeCandidate  # noqa: E402
from core.llm_client import LLMClient  # noqa: E402
from core.redis_client import redis_client  # noqa: E402
from core.settings import settings  # noqa: E402
from evaluation.native_react_training_corpus import (  # noqa: E402
    build_native_react_training_cases,
)
from evaluate_full_agent_loop import (  # noqa: E402
    _write_episode_candidates,
    build_report,
    evaluate_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--rollout-id", default="teacher-0")
    parser.add_argument("--policy-model")
    parser.add_argument("--policy-temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict:
    cases = build_native_react_training_cases(args.start_index, args.count)
    records_by_id: dict[str, dict] = {}
    candidates_by_id: dict[str, EpisodeCandidate] = {}
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        records_by_id = {item["case_id"]: item for item in report.get("records", [])}
    if args.resume and args.episode_output.exists():
        for line_number, line in enumerate(
            args.episode_output.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                candidate = EpisodeCandidate(**json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{args.episode_output}:{line_number}: {exc}") from exc
            candidates_by_id[candidate.scenario_id] = candidate

    model = args.policy_model or settings.llm_model
    policy = NativeToolAgentPolicy(
        LLMClient(),
        model=model,
        temperature=args.policy_temperature,
        max_tokens=256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    await redis_client.connect()
    try:
        for case in cases:
            if case.case_id in records_by_id:
                continue
            collector: list[EpisodeCandidate] = []
            record = await evaluate_case(
                case,
                policy=policy,
                rollout_id=args.rollout_id,
                episode_collector=collector,
            )
            records_by_id[case.case_id] = record
            for candidate in collector:
                candidates_by_id[candidate.scenario_id] = candidate
            selected = [records_by_id[item.case_id] for item in cases if item.case_id in records_by_id]
            args.output.write_text(
                json.dumps(
                    build_report(cases, selected, policy_model=model),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            _write_episode_candidates(
                args.episode_output,
                list(candidates_by_id.values()),
            )
            print(
                json.dumps(
                    {
                        "progress": f"{len(selected)}/{len(cases)}",
                        "case_id": case.case_id,
                        "passed": record["passed"],
                        "failures": record["failures"],
                        "tokens": record["total_tokens"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        await redis_client.disconnect()

    records = [records_by_id[case.case_id] for case in cases if case.case_id in records_by_id]
    report = build_report(cases, records, policy_model=model)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = parse_args()
    if args.start_index < 0 or args.count < 1:
        raise SystemExit("start-index must be non-negative and count must be positive")
    report = asyncio.run(run(args))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
