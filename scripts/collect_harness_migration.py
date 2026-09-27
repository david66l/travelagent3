"""Collect autonomous student trajectories in preflighted offline migration cases."""
import argparse
import asyncio
from pathlib import Path

from agentic.trajectory import redact_pii
from evaluation.migration_fixture import MigrationResearchExecutor, VERSION
from scripts import evaluate_harness_base as base


async def main(args):
    cases = base.load_cases(args.cases_file)
    assert all(c["split"] == "train" and c["dataset_version"] == VERSION for c in cases)
    assert not args.output.exists(), "Use a new immutable collection directory"
    original_executor, original_recorder = base.FrozenResearchExecutor, base.EpisodeRecorder
    active = {}

    class CapturedExecutor(MigrationResearchExecutor):
        def __init__(self, case):
            super().__init__(case)
            active["backend"] = self

    class MigrationRecorder(original_recorder):
        def record_step(self, **kwargs):
            super().record_step(**kwargs)
            backend = active["backend"]
            step = self.episode.steps[-1]
            base.dump(args.output / backend.case["id"] / f"state-after-{step.step_index:03d}.json", {
                "schema_version": "migration-recovery-state.v1",
                "case_id": backend.case["id"], "split": "train",
                "source_group": backend.case["source_group"], "step_index": step.step_index,
                "state": redact_pii(kwargs["state_after"].model_dump(mode="json")),
                "state_after_hash": step.state_after_hash,
                "provider_state": {"counts": dict(backend.counts),
                    "seen_fault_indices": sorted(backend._seen_faults),
                    "tool_call_count": len(backend.calls)},
                "provider_revision": VERSION,
                "error_code": step.verification.get("error_code"),
            })

    base.FrozenResearchExecutor, base.EpisodeRecorder = CapturedExecutor, MigrationRecorder
    try:
        await base.main(args)
    finally:
        base.FrozenResearchExecutor, base.EpisodeRecorder = original_executor, original_recorder
    base.dump(args.output / "collection-manifest.json", {
        "schema_version": "migration-student-collection.v1", "driver_sha256": base.sha(__file__),
        "cases_sha256": base.sha(args.output / "cases.json"), "provider_revision": VERSION,
        "source_groups": sorted(c["source_group"] for c in cases),
        "policy": "autonomous current SFT student; no scripted prefix or teacher feedback",
        "snapshot_timing": "after action commit, before next policy decision",
        "provider_restore_requirement": "Restore counters AND seen_fault_indices with MigrationResearchExecutor; legacy correction driver is not compatible",
        "training_targets_present": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-file", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", action="store_true")
    parser.set_defaults(case=None, fixture_check=False)
    asyncio.run(main(parser.parse_args()))
