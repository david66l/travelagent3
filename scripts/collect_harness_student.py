"""Run the existing evaluator unchanged, adding recovery-state sidecars only."""

import argparse
import asyncio
from pathlib import Path

from scripts import evaluate_harness_base as base
from agentic.trajectory import redact_pii


class RecoverySnapshotRecorder(base.EpisodeRecorder):
    def __init__(self, *args, backend, output, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.output = Path(output)

    def record_step(self, **kwargs):
        super().record_step(**kwargs)
        step = self.episode.steps[-1]
        snapshot = {
            "schema_version": "harness-recovery-state.v1",
            "case_id": self.backend.case["id"],
            "split": self.backend.case["split"],
            "source_group": self.backend.case["source_group"],
            "step_index": step.step_index,
            "state": redact_pii(kwargs["state_after"].model_dump(mode="json")),
            "state_after_hash": step.state_after_hash,
            "provider_state": {
                "counts": dict(self.backend.counts),
                "first_info_query": self.backend.first_info_query,
                "tool_call_count": len(self.backend.calls),
            },
            "error_code": step.verification.get("error_code"),
        }
        base.dump(self.output / f"state-after-{step.step_index:03d}.json", snapshot)


async def main(args):
    cases = base.load_cases(args.cases_file)
    assert cases and all(c["split"] == "train" for c in cases), (
        "Only explicit train tasks"
    )
    assert not args.output.exists(), "Use an immutable new collection directory"
    original_executor, original_recorder = (
        base.FrozenResearchExecutor,
        base.EpisodeRecorder,
    )
    active = {}

    class CapturedExecutor(original_executor):
        def __init__(self, case):
            super().__init__(case)
            active["backend"] = self

    def recorder(*a, **kw):
        backend = active["backend"]
        return RecoverySnapshotRecorder(
            *a, backend=backend, output=args.output / backend.case["id"], **kw
        )

    base.FrozenResearchExecutor, base.EpisodeRecorder = CapturedExecutor, recorder
    try:
        await base.main(args)
    finally:
        base.FrozenResearchExecutor, base.EpisodeRecorder = (
            original_executor,
            original_recorder,
        )
    base.dump(
        args.output / "collection-manifest.json",
        {
            "schema_version": "harness-student-collection.v1",
            "driver_sha256": base.sha(__file__),
            "cases_sha256": base.sha(args.output / "cases.json"),
            "snapshot_timing": "after real action commit, before any subsequent policy decision",
            "state_hash": "EpisodeRecorder state_after_hash; redacted canonical JSON",
            "provider_state": "real fixture counters, first query and tool-call prefix length",
            "split": "train",
            "training_targets_present": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-file", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", action="store_true")
    parser.set_defaults(case=None, fixture_check=False)
    asyncio.run(main(parser.parse_args()))
