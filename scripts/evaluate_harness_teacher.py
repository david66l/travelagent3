"""GLM teacher diagnostic using the SAME loop, cases, tools and terminal judge.

No dataset generation or training. Each output directory is one immutable run;
completed episodes resume, interrupted episodes require a new run directory.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_harness_base as base
from agentic.policy_backends import NativeToolAgentPolicy
from core.glm_tool_client import GLMToolClient

VERSION = "harness-glm-teacher-diagnostic24.v3"


class AuditableNativePolicy(NativeToolAgentPolicy):
    @property
    def last_generation_audit(self):
        return self.client.last_generation_audit


def manifest_for(args, cases_path):
    files = [p for folder in ("backend/src/agentic", "backend/src/tools",
                              "backend/src/vrp_solver_service", "backend/src/evaluation")
             for p in (base.ROOT / folder).rglob("*.py")]
    files += [base.ROOT / p for p in (
        "backend/src/core/glm_tool_client.py", "backend/src/core/inference_metrics.py",
        "scripts/evaluate_harness_base.py", "scripts/evaluate_harness_teacher.py")]
    return {
        "version": VERSION, "architecture": base.ARCHITECTURE_VERSION,
        "harness_revision": base.HARNESS_REVISION, "judging_version": base.JUDGING_VERSION,
        "fixture_revision": base.FIXTURE_REVISION,
        "model": args.model, "endpoint": args.base_url, "cases_sha256": base.sha(cases_path),
        "research": "frozen synthetic fixtures, not live providers",
        "solver": "real TravelVRPSolver", "validator": "real ItineraryValidator",
        "parsing": "predeclared user-grounded slots; intent parser not evaluated",
        "decoding": {"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "max",
            "thinking": "enabled", "clear_thinking": True, "max_tokens": args.max_tokens,
            "tool_choice": "auto", "seed": None, "http_retries": 0,
            "context": "same projected state; no cross-step private reasoning history",
            "protocol_reminder": "additional user message repeats one native call per turn, no action hints"},
        "budget": {"episode_steps": 24, "tool_calls": 24, "solver_calls": 3,
                   "completion_tokens": 32000, "episode_timeout_seconds": args.episode_timeout,
                   "request_timeout_seconds": args.request_timeout},
        "concurrency": args.concurrency,
        "comparison_limit": "teacher uses thinking and longer latency budget; not equal compute",
        "source_sha256": {str(p.relative_to(base.ROOT)).replace('\\', '/'): base.sha(p) for p in files},
        "runtime": {"python": sys.version, **{p: importlib.metadata.version(p)
                    for p in ("httpx", "pydantic", "ortools")}},
    }


async def main(args):
    key = os.environ.get("ZAI_API_KEY")
    if not key:
        raise RuntimeError("Set ZAI_API_KEY in the process environment")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    base.settings.agentic_guard_mode = "enforce"
    cases = base.load_cases(getattr(args,"cases_file",None))
    selected = [c for c in cases if not args.case or c["id"] in args.case]
    if args.case and set(args.case) - {c["id"] for c in cases}:
        raise ValueError("Unknown case ID")
    if (out / "cases.json").exists():
        assert json.loads((out / "cases.json").read_text(encoding="utf-8")) == cases
    else:
        base.dump(out / "cases.json", cases)
    manifest = manifest_for(args, out / "cases.json")
    if (out / "manifest.json").exists():
        assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == manifest, \
            "Configuration/source changed; use a new run directory"
    base.dump(out / "manifest.json", manifest)
    for case in selected:
        target = out / case["id"]
        if target.exists() and any(target.iterdir()) and not (target / "summary.json").exists():
            raise RuntimeError(f"Interrupted evidence in {case['id']}; use a new run directory")
    semaphore = asyncio.Semaphore(args.concurrency)
    stopped = asyncio.Event()

    async def run_case(case):
        async with semaphore:
            target = out / case["id"]
            if (target / "summary.json").exists():
                previous = json.loads((target / "summary.json").read_text(encoding="utf-8"))
                if previous.get("provider_failures"):
                    stopped.set()
                return
            if stopped.is_set():
                return
            target.mkdir(exist_ok=True)
            print(json.dumps({"event": "case_start", "case": case["id"]}), flush=True)
            client = GLMToolClient(key, base_url=args.base_url, timeout=args.request_timeout)
            policy = AuditableNativePolicy(client, model=args.model, temperature=1, max_tokens=args.max_tokens)
            audited = base.AuditedPolicy(policy, target / "model-calls.jsonl")
            try:
                with base.frozen_reference_time(base.MOMENT):
                    state = base.initial(case)
                    state.budget = state.budget.model_copy(update={"timeout_ms": int(args.episode_timeout * 1000)})
                    backend = base.FrozenResearchExecutor(case)
                    recorder = base.EpisodeRecorder(state, environment_version=base.ARCHITECTURE_VERSION,
                        validator_version=base.VALIDATOR_VERSION, policy_name=args.model, policy_version=VERSION)
                    started = time.perf_counter()
                    await base.BoundedAgentLoop().run(state, policy=base.SelfRepairingAgentPolicy(audited),
                        executor=base.TravelActionExecutor(backend), recorder=recorder)
                    episode = recorder.episode
                    base.dump(target / "episode.json", episode.model_dump(mode="json"))
                    base.dump(target / "tool-calls.json", backend.calls)
                    errors = base.EpisodeReplayVerifier().verify(episode)
                    assert not errors, errors
                    summary = base.judge(case, episode, audited.calls)
                    provider_failures = [r for r in audited.calls if r["generation"].get("provider_error")]
                    summary.update(
                        provider_failures=len(provider_failures),
                        parse_or_schema_errors=sum(r.get("error", {}).get("type") == "PolicyOutputError" for r in audited.calls),
                        wall_seconds=time.perf_counter() - started,
                        reward=base.HierarchicalRewardEngine().score(episode).model_dump(mode="json"),
                        replay_errors=errors,
                        reasoning_tokens=sum(r["generation"].get("reasoning_tokens") or 0 for r in audited.calls),
                        reasoning_usage_known_calls=sum(r["generation"].get("reasoning_tokens") is not None for r in audited.calls),
                        usage_unknown_calls=sum(not r["generation"].get("usage_known") for r in audited.calls),
                        cancelled_requests=sum(bool(r["generation"].get("request_cancelled")) for r in audited.calls),
                    )
                    base.dump(target / "summary.json", summary)
                    if provider_failures:
                        stopped.set()
                    print(json.dumps({"event": "case_done", **{k: summary[k] for k in
                        ("case", "criterion_passed", "termination_reason", "actions", "wall_seconds", "provider_failures")}}, ensure_ascii=False), flush=True)
            finally:
                await client.aclose()

    await asyncio.gather(*(run_case(c) for c in selected))
    summaries = [json.loads((out / c["id"] / "summary.json").read_text(encoding="utf-8"))
                 for c in cases if (out / c["id"] / "summary.json").exists()]
    base.dump(out / "summary.json", summaries)
    if stopped.is_set():
        raise RuntimeError("Provider failure: remaining cases stopped; inspect private audit")
    print(f"EVALUATION_COMPLETE {len(summaries)}/{len(cases)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--base-url", default="https://open.bigmodel.cn/api/paas/v4")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--episode-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--case", action="append")
    parser.add_argument("--cases-file", type=Path)
    asyncio.run(main(parser.parse_args()))
