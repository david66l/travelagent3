"""Export privacy-minimized challenger observations from Agent episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.trajectory import AgentEpisode, EpisodeReplayVerifier  # noqa: E402


def _decision_id(trajectory_id: str, step_index: int) -> str:
    material = f"{trajectory_id}:{step_index}".encode("utf-8")
    return f"shadow-{hashlib.sha256(material).hexdigest()[:48]}"


def _episode_payload(row: dict[str, Any]) -> dict[str, Any]:
    episode = row.get("episode")
    return episode if isinstance(episode, dict) else row


def export_observations(rows: list[dict[str, Any]]) -> tuple[list[dict], dict]:
    observations = []
    episodes = 0
    for row in rows:
        episode = AgentEpisode(**_episode_payload(row))
        replay_errors = EpisodeReplayVerifier().verify(episode)
        if replay_errors:
            raise ValueError(
                f"invalid episode {episode.trajectory_id}: {', '.join(replay_errors)}"
            )
        episodes += 1
        for step in episode.steps:
            shadow = step.action.shadow_trace
            if shadow is None:
                continue
            route = step.action.route_trace
            candidate_route = shadow.route_trace
            observations.append(
                {
                    "schema_version": "policy-shadow-observation.v1",
                    "decision_id": _decision_id(episode.trajectory_id, step.step_index),
                    "step_index": step.step_index,
                    "family": (
                        route.family
                        if route is not None
                        else candidate_route.family
                        if candidate_route is not None
                        else "unknown"
                    ),
                    "champion_action": step.action.action,
                    "challenger_model": shadow.candidate_model,
                    "challenger_status": shadow.status,
                    "challenger_action": shadow.action,
                    "action_divergent": (
                        shadow.status == "completed"
                        and shadow.action != step.action.action
                    ),
                    "challenger_error_code": shadow.error_code,
                    "champion_latency_ms": (
                        step.action.inference_metrics.request_latency_ms
                        if step.action.inference_metrics is not None
                        else None
                    ),
                    "challenger_latency_ms": (
                        shadow.inference_metrics.request_latency_ms
                        if shadow.inference_metrics is not None
                        else None
                    ),
                    "champion_outcome_observed": True,
                    "challenger_outcome_observed": False,
                    "release_gate_eligible": False,
                }
            )
    status_counts = Counter(row["challenger_status"] for row in observations)
    family_counts = Counter(row["family"] for row in observations)
    manifest = {
        "schema_version": "policy-shadow-observation-manifest.v1",
        "episodes": episodes,
        "decisions": len(observations),
        "challenger_status_counts": dict(sorted(status_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "action_divergences": sum(row["action_divergent"] for row in observations),
        "challenger_failures": status_counts.get("failed", 0),
        "contains_raw_context": False,
        "release_gate_eligible": False,
        "reason": "challenger actions were observed but not executed",
    }
    return observations, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.episodes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observations, manifest = export_observations(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "observations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in observations),
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
