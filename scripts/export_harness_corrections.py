"""Audit train-state lineage and export only verified teacher continuation actions."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from agentic.sft_dataset import SFTDatasetBuilder, EpisodeCandidate
from agentic.trajectory import _canonical_hash, redact_pii
from scripts.export_harness_sft import audit_episode, recorded_preflight_rejection, sha
from agentic.loop import NO_TOOL_ACTIONS


class CorrectionBuilder(SFTDatasetBuilder):
    def __init__(self, cases, audits):
        super().__init__()
        self.cases, self.audits = cases, audits

    def _split_group(self, candidate):
        return self.cases[candidate.scenario_id]["source_group"]

    def _split_for_group(self, group):
        return "train"

    def _review(self, candidate):
        errors = super()._review(candidate)
        missing_observations = [
            step
            for step in candidate.episode.steps
            if step.action.action not in NO_TOOL_ACTIONS and not step.observations
        ]
        if missing_observations and all(
            recorded_preflight_rejection(step) for step in missing_observations
        ):
            # Same existing export rule: a real preflight refusal has no provider
            # observation. Its failed action remains in evidence, never in labels.
            errors = [
                error for error in errors if error != "L2_TOOL_OBSERVATION_MISSING"
            ]
        audit = self.audits[candidate.scenario_id]
        if not audit["criterion_passed"]:
            errors.append("L3_INDEPENDENT_TASK_CRITERION_FAILED")
        if audit["reward"]["gate_status"] != "passed":
            errors.append("L3_TERMINAL_REWARD_GATE_FAILED")
        if audit["provider_failures"]:
            errors.append("L1_PROVIDER_FAILURE")
        return errors

    @staticmethod
    def _examples(candidate, split, quality_label):
        examples = SFTDatasetBuilder._examples(candidate, split, quality_label)
        # A wrapper repair includes extra feedback not present in the original
        # recorded step context; do not silently train a different conditioning.
        return [
            ex
            for ex in examples
            if candidate.episode.steps[ex.step_index].action.repair_attempts == 0
        ]


def main(args):
    assert not args.output.exists(), "Use a new immutable export version"
    cases = json.loads((args.teacher_run / "cases.json").read_text(encoding="utf-8"))
    assert cases and all(c["split"] == "train" for c in cases)
    index = {c["id"]: c for c in cases}
    audits, candidates, lineage, model_records = {}, [], {}, {}
    for case in cases:
        name = case["id"]
        folder = args.teacher_run / name
        provenance = json.loads(
            (folder / "provenance.json").read_text(encoding="utf-8")
        )
        source = args.student_run / name
        snapshot_path = (
            source / f"state-after-{provenance['source_step_index']:03d}.json"
        )
        assert sha(source / "episode.json") == provenance["source_episode_sha256"]
        assert sha(snapshot_path) == provenance["source_snapshot_sha256"]
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        assert (
            snapshot["split"] == "train"
            and snapshot["source_group"] == case["source_group"]
        )
        episode, audit = audit_episode(args.teacher_run, case)
        raw = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        assert audit["criterion_passed"] == raw["criterion_passed"]
        actual = deepcopy(episode.initial_state)
        actual["budget"]["timeout_ms"] = snapshot["state"]["budget"]["timeout_ms"]
        assert (
            _canonical_hash(actual)
            == snapshot["state_after_hash"]
            == provenance["source_state_hash"]
        )
        assert (
            episode.initial_state["budget"]["used_episode_steps"]
            == snapshot["state"]["budget"]["used_episode_steps"]
        )
        records = json.loads((folder / "tool-calls.json").read_text(encoding="utf-8"))
        prefix = json.loads((source / "tool-calls.json").read_text(encoding="utf-8"))[
            : provenance["prefix_tool_calls"]
        ]
        assert records[: len(prefix)] == prefix
        assert (
            len(episode.steps)
            == episode.final_state["budget"]["used_episode_steps"]
            - episode.initial_state["budget"]["used_episode_steps"]
        )
        audit.update(
            source_error=provenance["source_error"],
            kind=provenance["kind"],
            student_final_passed=provenance["student_final_passed"],
            teacher_tool_calls=raw["teacher_tool_calls"],
            wall_seconds=raw["wall_seconds"],
            initial_state_lineage_verified=True,
            prefix_tool_calls_verified=True,
        )
        audits[name] = audit
        lineage[name] = provenance
        model_records[name] = {
            r["action"]["action_id"]: r
            for r in map(
                json.loads,
                (folder / "model-calls.jsonl").read_text(encoding="utf-8").splitlines(),
            )
            if r.get("action")
        }
        candidates.append(
            EpisodeCandidate(
                scenario_id=name,
                source="teacher",
                template_family=case["source_group"],
                city=case["slots"]["destination"],
                episode=episode,
            )
        )
    builder = CorrectionBuilder(index, audits)
    result = builder.build(candidates)
    lookup = {c.scenario_id: c for c in candidates}
    for ex in result.examples:
        step = lookup[ex.scenario_id].episode.steps[ex.step_index]
        call = model_records[ex.scenario_id][step.action.action_id]
        assert json.loads(ex.messages[1].content) == redact_pii(call["state"]), (
            ex.example_id
        )
        assert len(ex.messages) == 3 and ex.messages[-1].content is None
        assert ex.split == "train"
    builder.export(result, args.output)
    report = {
        "version": "harness-correction-export.v1",
        "teacher_run": str(args.teacher_run),
        "student_run": str(args.student_run),
        "audit_sha256": sha(Path(__file__)),
        "teacher_manifest_sha256": sha(args.teacher_run / "manifest.json"),
        "teacher_cases_sha256": sha(args.teacher_run / "cases.json"),
        "student_manifest_sha256": sha(args.student_run / "manifest.json"),
        "cases": audits,
        "lineage": lineage,
        "student_prefix_target_count": 0,
        "dev_target_count": 0,
        "test_target_count": 0,
        "target_context_matches_actual_teacher_call": True,
        "scope": "verified continuation targets; not from-start teacher performance",
        "accepted_episodes": result.manifest.accepted_episodes,
        "exported_examples": result.manifest.exported_examples,
    }
    (args.output / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result.manifest.model_dump_json(indent=2))
    print(
        json.dumps(
            {
                "teacher_successes": sum(
                    a["criterion_passed"] for a in audits.values()
                ),
                "source_failed_tasks": sum(
                    not a["student_final_passed"] for a in audits.values()
                ),
                "source_failed_tasks_teacher_recovered": sum(
                    not a["student_final_passed"] and a["criterion_passed"]
                    for a in audits.values()
                ),
                "accepted_episodes": result.manifest.accepted_episodes,
                "exported_examples": result.manifest.exported_examples,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-run", type=Path, required=True)
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
