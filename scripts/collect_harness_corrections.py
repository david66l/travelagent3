"""Teacher continuations from auditable, real student train-state checkpoints.

Each output is a continuation only, not a teacher episode from the task origin.
Student prefix counters and evidence are preserved; private reasoning is not a target.
"""

import argparse
import asyncio
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import time

from scripts import evaluate_harness_base as base
from scripts.evaluate_harness_teacher import AuditableNativePolicy
from agentic.trajectory import AgentEpisode, _canonical_hash
from core.glm_tool_client import GLMToolClient

EXCLUDED_ERRORS = {None, "POLICY_ABORT"}
ENVIRONMENT_ERRORS = {"TOOL_TIMEOUT", "TOOL_EXECUTION_FAILED", "EXECUTOR_ERROR"}


def recovery_candidates(student_run):
    cases = base.load_cases(student_run / "cases.json")
    assert cases and all(c["split"] == "train" for c in cases), (
        "Do not mine dev/test targets"
    )
    selected, excluded = [], []
    for case in cases:
        folder = student_run / case["id"]
        episode = AgentEpisode(
            **json.loads((folder / "episode.json").read_text(encoding="utf-8"))
        )
        assert not base.EpisodeReplayVerifier().verify(episode)
        choices = []
        for step in episode.steps:
            if step.verification.get("error_code") in EXCLUDED_ERRORS:
                continue
            path = folder / f"state-after-{step.step_index:03d}.json"
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            assert snapshot["case_id"] == case["id"] and snapshot["split"] == "train"
            assert snapshot["source_group"] == case["source_group"]
            assert (
                _canonical_hash(snapshot["state"])
                == snapshot["state_after_hash"]
                == step.state_after_hash
            )
            state = base.AgentLedgerState(**snapshot["state"])
            if (
                state.task_graph.tasks[0].status != "ready"
                or state.budget.remaining_episode_steps < 2
            ):
                continue
            kind = (
                "environment_recovery"
                if snapshot["error_code"] in ENVIRONMENT_ERRORS
                else "student_decision_correction"
            )
            # Target the observed recovery mistake itself when available,
            # rather than always stopping at an earlier missing-candidate step.
            priority = 3 if kind == "environment_recovery" else 2
            if snapshot["error_code"] == "RESEARCH_EVIDENCE_INSUFFICIENT":
                priority = 1
            if (
                snapshot["error_code"] == "REPEATED_NO_PROGRESS_ACTION"
                and step.action.action == "search_current_info"
            ):
                priority = 0
            choices.append((priority, step.step_index, kind, path, snapshot))
        if not choices:
            excluded.append(
                {
                    "case": case["id"],
                    "reason": "no_eligible_executed_failure_checkpoint",
                }
            )
            continue
        _, index, kind, path, snapshot = min(choices, key=lambda c: c[:2])
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        selected.append(
            {
                "case": case,
                "step_index": index,
                "kind": kind,
                "snapshot_path": str(path),
                "snapshot": snapshot,
                "source_episode_sha256": base.sha(folder / "episode.json"),
                "source_snapshot_sha256": base.sha(path),
                "student_final_passed": summary["criterion_passed"],
            }
        )
    # Round-robin the preassigned train groups. One earliest eligible checkpoint
    # per source task; prefer model-decision errors over injected provider faults.
    groups = defaultdict(list)
    for candidate in selected:
        groups[candidate["case"]["source_group"]].append(candidate)
    ordered = []
    while any(groups.values()):
        for group in sorted(groups):
            if groups[group]:
                ordered.append(groups[group].pop(0))
    return ordered, excluded


def restore_branch(candidate, student_run):
    case, snapshot = candidate["case"], candidate["snapshot"]
    state = base.AgentLedgerState(**deepcopy(snapshot["state"]))
    backend = base.FrozenResearchExecutor(case)
    backend.counts = Counter(snapshot["provider_state"]["counts"])
    backend.first_info_query = snapshot["provider_state"]["first_info_query"]
    records = json.loads(
        (student_run / case["id"] / "tool-calls.json").read_text(encoding="utf-8")
    )
    count = snapshot["provider_state"]["tool_call_count"]
    assert 0 <= count <= len(records)
    backend.calls = deepcopy(records[:count])
    assert count == state.budget.used_tool_calls
    return state, backend


async def main(args):
    key = os.environ.get("ZAI_API_KEY")
    if not key:
        raise RuntimeError("ZAI_API_KEY is required in the cloud process environment")
    assert not args.output.exists(), "Use a new immutable continuation run"
    candidates, excluded = recovery_candidates(args.student_run)
    chosen = candidates[: args.max_cases]
    if args.case:
        assert set(args.case) <= {c["case"]["id"] for c in chosen}, (
            "Unknown selected source case"
        )
        chosen = [c for c in chosen if c["case"]["id"] in args.case]
    args.output.mkdir(parents=True)
    source_manifest = json.loads((args.student_run / "manifest.json").read_text())
    for name, expected in source_manifest["source_sha256"].items():
        assert base.sha(base.ROOT / name) == expected, name
    assert (
        base.sha(base.ROOT / "scripts/evaluate_harness_base.py")
        == source_manifest["script_sha256"]
    )
    base.dump(args.output / "cases.json", [c["case"] for c in chosen])
    base.dump(
        args.output / "selection.json",
        {
            "eligible": [
                {k: v for k, v in c.items() if k not in {"case", "snapshot"}}
                | {"case": c["case"]["id"]}
                for c in candidates
            ],
            "excluded": excluded,
            "selected_count": len(chosen),
            "rule": "one checkpoint per task: repeated current-info query, then research insufficiency, then other decision error, then environment recovery; earliest per priority; round-robin train families",
        },
    )
    manifest = {
        "version": "harness-teacher-corrections.v2",
        "model": args.model,
        "student_run": str(args.student_run),
        "student_manifest_sha256": base.sha(args.student_run / "manifest.json"),
        "harness_revision": base.HARNESS_REVISION,
        "judging_version": base.JUDGING_VERSION,
        "fixture_revision": base.FIXTURE_REVISION,
        "driver_sha256": base.sha(__file__),
        "source_sha256": source_manifest["source_sha256"],
        "split": "train",
        "decoding": {
            "temperature": 1,
            "max_tokens": args.max_tokens,
            "reasoning_effort": "max",
        },
        "request_timeout_seconds": args.request_timeout,
        "counter_contract": "preserve used steps/tools/solver/tokens; add teacher-only wall-time allowance",
        "teacher_extra_timeout_ms": args.teacher_seconds * 1000,
        "scope": "teacher suffix from student state; not from-start performance or RL",
    }
    manifest["support_source_sha256"] = {
        name: base.sha(base.ROOT / name)
        for name in [
            "scripts/evaluate_harness_teacher.py",
            "backend/src/core/glm_tool_client.py",
            "backend/src/core/inference_metrics.py",
        ]
    }
    manifest["endpoint"] = args.base_url
    base.dump(args.output / "manifest.json", manifest)
    semaphore = asyncio.Semaphore(args.concurrency)
    stopped = asyncio.Event()

    async def run(candidate):
        async with semaphore:
            if stopped.is_set():
                return
            case = candidate["case"]
            target = args.output / case["id"]
            target.mkdir()
            client = GLMToolClient(
                key, base_url=args.base_url, timeout=args.request_timeout
            )
            policy = AuditableNativePolicy(
                client, model=args.model, temperature=1, max_tokens=args.max_tokens
            )
            audited = base.AuditedPolicy(policy, target / "model-calls.jsonl")
            try:
                with base.frozen_reference_time(base.MOMENT):
                    state, backend = restore_branch(candidate, args.student_run)
                    original_budget = state.budget.model_dump(mode="json")
                    state.budget = state.budget.model_copy(
                        update={
                            "timeout_ms": state.budget.used_latency_ms
                            + args.teacher_seconds * 1000
                        }
                    )
                    base.dump(
                        target / "provenance.json",
                        {
                            "source_case": case["id"],
                            "source_group": case["source_group"],
                            "split": "train",
                            "source_step_index": candidate["step_index"],
                            "source_error": candidate["snapshot"]["error_code"],
                            "source_episode_sha256": candidate["source_episode_sha256"],
                            "source_snapshot_sha256": candidate[
                                "source_snapshot_sha256"
                            ],
                            "source_state_hash": candidate["snapshot"][
                                "state_after_hash"
                            ],
                            "kind": candidate["kind"],
                            "student_final_passed": candidate["student_final_passed"],
                            "initial_budget": original_budget,
                            "prefix_tool_calls": len(backend.calls),
                            "changed_initial_fields": ["budget.timeout_ms"],
                            "student_prefix_targets": 0,
                        },
                    )
                    recorder = base.EpisodeRecorder(
                        state,
                        environment_version=base.ARCHITECTURE_VERSION,
                        validator_version=base.VALIDATOR_VERSION,
                        policy_name=args.model,
                        policy_version=manifest["version"],
                    )
                    started = time.perf_counter()
                    await base.BoundedAgentLoop().run(
                        state,
                        policy=base.SelfRepairingAgentPolicy(audited),
                        executor=base.TravelActionExecutor(backend),
                        recorder=recorder,
                    )
                    episode = recorder.episode
                    assert not base.EpisodeReplayVerifier().verify(episode)
                    base.dump(target / "episode.json", episode.model_dump(mode="json"))
                    base.dump(target / "tool-calls.json", backend.calls)
                    row = base.judge(case, episode, audited.calls)
                    provider_failures = sum(
                        bool(c.get("generation", {}).get("provider_error"))
                        for c in audited.calls
                    )
                    row.update(
                        kind=candidate["kind"],
                        student_final_passed=candidate["student_final_passed"],
                        wall_seconds=time.perf_counter() - started,
                        provider_failures=provider_failures,
                        source_step_index=candidate["step_index"],
                        teacher_tool_calls=len(backend.calls)
                        - candidate["snapshot"]["provider_state"]["tool_call_count"],
                    )
                    base.dump(target / "summary.json", row)
                    if provider_failures:
                        stopped.set()
                    print(
                        json.dumps(
                            {
                                "event": "correction_done",
                                "case": case["id"],
                                "passed": row["criterion_passed"],
                                "kind": row["kind"],
                                "teacher_calls": row["model_calls"],
                                "provider_failures": provider_failures,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            finally:
                await client.aclose()

    await asyncio.gather(*(run(c) for c in chosen))
    summaries = [
        json.loads(
            (args.output / c["case"]["id"] / "summary.json").read_text(encoding="utf-8")
        )
        for c in chosen
        if (args.output / c["case"]["id"] / "summary.json").exists()
    ]
    base.dump(args.output / "summary.json", summaries)
    if stopped.is_set():
        raise RuntimeError(
            "Teacher provider failure; unstarted branches stopped; evidence retained"
        )
    print("CORRECTION_COLLECTION_COMPLETE", len(summaries), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--concurrency", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument("--teacher-seconds", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--case", action="append")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--base-url", default="https://open.bigmodel.cn/api/paas/v4")
    asyncio.run(main(parser.parse_args()))
