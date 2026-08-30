"""Build a leakage-safe online GRPO corpus from audited native ReAct episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_seed(scenario_id: str) -> int:
    return int(hashlib.sha256(scenario_id.encode("utf-8")).hexdigest()[:8], 16)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _snapshot_source(source: str, ok: bool, is_fallback: bool) -> str:
    if not ok:
        return "unavailable"
    if is_fallback:
        return "fallback"
    return source if source in {"api", "built_in", "fallback", "unavailable"} else "api"


def _difficulty(episode: dict[str, Any]) -> str:
    final_state = episode.get("final_state") or {}
    failures = len(final_state.get("failures") or [])
    policy_steps = sum(
        (step.get("action") or {}).get("decision_source") != "controller"
        for step in episode.get("steps") or []
    )
    score = failures * 2 + policy_steps + max(0, len(episode.get("steps") or []) - 8)
    return "L1" if score <= 1 else "L2" if score <= 3 else "L3" if score <= 6 else "L4"


def _episode_row(
    candidate: dict[str, Any], *, task_id: str, seed: int
) -> dict[str, Any]:
    episode = candidate["episode"]
    initial_goal = (episode.get("initial_state") or {}).get("goal") or {}
    final_goal = (episode.get("final_state") or {}).get("goal") or initial_goal
    hard = dict(final_goal.get("hard_constraints") or {})
    soft = dict(final_goal.get("soft_preferences") or {})
    tool_responses: dict[str, list[dict[str, Any]]] = {}
    for step in episode.get("steps") or []:
        for observation in step.get("observations") or []:
            error = observation.get("error") or {}
            tool_responses.setdefault(observation["tool"], []).append(
                {
                    "data": observation.get("data"),
                    "data_source": _snapshot_source(
                        str(observation.get("source") or ""),
                        bool(observation.get("ok")),
                        bool(observation.get("is_fallback")),
                    ),
                    "confidence": observation.get("confidence", 1.0),
                    "is_fallback": bool(observation.get("is_fallback")),
                    "fallback_reason": error.get("message"),
                    "latency_ms": observation.get("latency_ms", 0),
                    "error_code": error.get("code"),
                    "retryable": bool(error.get("retryable")),
                }
            )
    reports = [
        artifact.get("payload") or {}
        for artifact in ((episode.get("final_state") or {}).get("artifacts") or {}).values()
        if artifact.get("artifact_type") == "validation_report"
    ]
    content_hash = str(episode.get("content_hash") or "")
    capability = final_goal.get("capability") or {}
    return {
        "task": {
            "task_id": task_id,
            "template_family": candidate["template_family"],
            "difficulty": _difficulty(episode),
            "seed": seed,
            "user_request": final_goal.get("original_request") or "Travel planning request",
            "slots": hard,
            "profile": soft,
            "missing_slots": list(final_goal.get("missing_information") or []),
            "feasibility_report": {
                "feasible": capability.get("status") == "solvable",
                "status": capability.get("status"),
                "reasons": list(capability.get("evidence") or []),
                "actionable_alternatives": capability.get("actionable_alternatives"),
                "alternatives": list(capability.get("alternatives") or []),
            },
        },
        "snapshot": {
            "environment_version": episode["environment_version"],
            "snapshot_version": "episode-" + content_hash[:16],
            "state_id": "trajectory-" + content_hash[:16],
            "tool_responses": tool_responses,
            "hidden_test_facts": {
                "source_content_hash": content_hash,
                "validation_report": reports[-1] if reports else None,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, action="append", required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_rows = _read_jsonl(args.reviews)
    reviews = {row["scenario_id"]: row for row in review_rows}
    if len(reviews) != len(review_rows):
        raise ValueError("duplicate scenario_id in review file")

    candidates: dict[str, dict[str, Any]] = {}
    for path in args.episodes:
        for raw in _read_jsonl(path):
            scenario_id = str(raw.get("scenario_id") or "")
            if not scenario_id or not raw.get("episode"):
                raise ValueError(f"invalid episode candidate in {path}")
            if scenario_id in candidates:
                raise ValueError(f"duplicate episode candidate: {scenario_id}")
            candidates[scenario_id] = raw

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    excluded = Counter()
    family_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    selected_scenarios: set[str] = set()
    for scenario_id, review in sorted(reviews.items()):
        if not review.get("accepted"):
            excluded["sft_rejected"] += 1
            continue
        split = review.get("split")
        if split == "test":
            excluded["frozen_test"] += 1
            continue
        if split not in split_rows:
            raise ValueError(f"unsupported accepted split for {scenario_id}: {split}")
        if review.get("quality_label") != "validated_plan":
            excluded["safe_termination_reserved_for_targeted_curriculum"] += 1
            continue
        candidate = candidates.get(scenario_id)
        if candidate is None:
            raise ValueError(f"accepted review has no episode candidate: {scenario_id}")
        if candidate["episode"].get("trajectory_id") != review.get("trajectory_id"):
            raise ValueError(f"review/episode trajectory mismatch: {scenario_id}")
        payload = _episode_row(
            candidate,
            task_id=f"native-grpo-{hashlib.sha256(scenario_id.encode()).hexdigest()[:16]}",
            seed=_stable_seed(scenario_id),
        )
        payload["snapshot"]["hidden_test_facts"]["corpus_provenance"] = {
            "scenario_id": scenario_id,
            "sft_split": split,
            "quality_label": review.get("quality_label"),
            "source": candidate["source"],
        }
        split_rows[split].append(payload)
        family_counts[split][candidate["template_family"]] += 1
        selected_scenarios.add(scenario_id)

    if not split_rows["train"] or not split_rows["validation"]:
        raise ValueError("native ReAct GRPO corpus requires non-empty train and validation splits")
    if selected_scenarios & {
        row["scenario_id"] for row in review_rows if row.get("split") == "test"
    }:
        raise AssertionError("frozen test leakage detected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        _write_jsonl(args.output_dir / f"{split}.jsonl", rows)

    manifest = {
        "schema_version": "native-react-grpo.v1",
        "source_reviews": str(args.reviews),
        "source_episode_files": [str(path) for path in args.episodes],
        "selection_rule": (
            "accepted validated-plan SFT train/validation episodes only; "
            "safe terminations reserved for a targeted recovery curriculum; frozen test excluded"
        ),
        "rollout_contract": "fresh_ledger_no_teacher_prefix.v1",
        "counts": {split: len(rows) for split, rows in split_rows.items()},
        "excluded": dict(sorted(excluded.items())),
        "template_family_counts": {
            split: dict(sorted(counts.items())) for split, counts in family_counts.items()
        },
        "split_sha256": {
            split: _sha256(args.output_dir / f"{split}.jsonl") for split in split_rows
        },
        "frozen_test_in_training": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
