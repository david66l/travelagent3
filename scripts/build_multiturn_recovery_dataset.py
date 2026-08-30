"""Build SFT recovery examples that match TRL's multi-turn tool history."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT, policy_prompt_payload  # noqa: E402
from agentic.policy_actions import policy_action_schemas  # noqa: E402
from agentic.sft_dataset import (  # noqa: E402
    DatasetManifest,
    EpisodeCandidate,
    SFTExample,
    SFTMessage,
    SFTToolCall,
    SFTToolFunction,
)
from agentic.training import load_jsonl  # noqa: E402


def _tool_call(action: Any) -> SFTToolCall:
    return SFTToolCall(
        function=SFTToolFunction(name=action.action, arguments=action.arguments)
    )


def _transition_content(first_step: Any, next_context: Any) -> str:
    """Mirror TRLTravelEnvironment._render_transition for a failed first call."""
    payload = {
        "done": False,
        "last_transition": {
            "action": first_step.action.action,
            "observations": [
                {
                    "ok": item.ok,
                    "tool": item.tool,
                    "error_code": item.error.code if item.error else None,
                    "is_fallback": item.is_fallback,
                }
                for item in first_step.observations
            ],
            "verification": first_step.verification,
        },
        "policy_state": policy_prompt_payload(next_context),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_multiturn_example(
    candidate: EpisodeCandidate,
    *,
    split: str,
    expected_error: str = "UPSTREAM_TIMEOUT",
    require_adaptation: bool = False,
    example_prefix: str = "multiturn-recovery",
) -> SFTExample:
    policy_steps = [
        step
        for step in candidate.episode.steps
        if step.action.decision_source != "controller"
        and step.task_id == "search_candidates"
        and step.action.action == "search_pois"
    ]
    if len(policy_steps) != 2:
        raise ValueError(f"{candidate.scenario_id} does not contain one recovery pair")
    first, second = policy_steps
    first_error = (first.verification or {}).get("error_code")
    if first_error != expected_error:
        raise ValueError(
            f"{candidate.scenario_id} first action is not {expected_error} recovery"
        )
    if not any(observation.ok for observation in second.observations):
        raise ValueError(f"{candidate.scenario_id} recovery action produced no grounded observation")
    if require_adaptation and first.action.arguments == second.action.arguments:
        raise ValueError(f"{candidate.scenario_id} recovery did not change its strategy")
    initial_content = json.dumps(
        {"policy_state": policy_prompt_payload(first.context)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return SFTExample(
        example_id=f"{example_prefix}:{candidate.episode.trajectory_id}:{second.step_index}",
        scenario_id=candidate.scenario_id,
        trajectory_id=candidate.episode.trajectory_id,
        step_index=second.step_index,
        split=split,
        quality_label="validated_plan",
        source=candidate.source,
        environment_version=candidate.episode.environment_version,
        policy_name=candidate.episode.policy_name,
        policy_version=candidate.episode.policy_version,
        messages=[
            SFTMessage(role="system", content=AGENT_TOOL_POLICY_SYSTEM_PROMPT),
            SFTMessage(role="user", content=initial_content),
            SFTMessage(role="assistant", tool_calls=[_tool_call(first.action)]),
            SFTMessage(
                role="tool",
                name=first.action.action,
                content=_transition_content(first, second.context),
            ),
            SFTMessage(role="assistant", tool_calls=[_tool_call(second.action)]),
        ],
        tools=policy_action_schemas(first.context.allowed_actions),
    )


def _family(candidate: EpisodeCandidate) -> tuple[str, ...]:
    return tuple(candidate.episode.steps[0].context.soft_preferences.get("interests") or ["unknown"])


def _split_candidates(
    candidates: list[EpisodeCandidate],
) -> dict[str, list[EpisodeCandidate]]:
    families: dict[tuple[str, ...], list[EpisodeCandidate]] = defaultdict(list)
    for candidate in candidates:
        families[_family(candidate)].append(candidate)
    result: dict[str, list[EpisodeCandidate]] = defaultdict(list)
    for family in sorted(families):
        ordered = sorted(families[family], key=lambda item: item.scenario_id)
        for index, candidate in enumerate(ordered):
            bucket = index % 20
            split = "validation" if bucket < 3 else ("test" if bucket < 6 else "train")
            result[split].append(candidate)
    return result


def build(
    candidates_file: Path,
    official_train_file: Path,
    replay_dataset_dir: Path,
    output_dir: Path,
) -> DatasetManifest:
    official_train = load_jsonl(official_train_file)
    recovery_ids = {
        row["scenario_id"]
        for row in official_train
        if any(
            failure.get("code") == "UPSTREAM_TIMEOUT" and failure.get("retryable")
            for failure in json.loads(row["messages"][1]["content"]).get("failure_summary")
            or []
        )
    }
    candidates = [
        EpisodeCandidate(**row)
        for row in load_jsonl(candidates_file)
        if row["scenario_id"] in recovery_ids
    ]
    if {item.scenario_id for item in candidates} != recovery_ids:
        raise ValueError("not every official-train recovery scenario has a source episode")
    split_candidates = _split_candidates(candidates)

    output: dict[str, list[SFTExample]] = {}
    for split in ("train", "validation", "test"):
        multiturn = [
            build_multiturn_example(candidate, split=split)
            for candidate in split_candidates[split]
        ]
        replay_rows = [
            SFTExample(**row)
            for row in load_jsonl(replay_dataset_dir / f"{split}.jsonl")
            if row["example_id"].startswith("repair:replay:")
        ][: len(multiturn)]
        replay = []
        for item in replay_rows:
            updated = deepcopy(item.model_dump(mode="json"))
            updated["example_id"] = "multiturn:" + updated["example_id"]
            replay.append(SFTExample(**updated))
        if len(replay) != len(multiturn):
            raise ValueError(f"insufficient replay examples for {split}")
        output[split] = sorted([*multiturn, *replay], key=lambda item: item.example_id)

    scenario_sets = {
        split: {item.scenario_id for item in rows} for split, rows in output.items()
    }
    if any(
        scenario_sets[left] & scenario_sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("multi-turn repair split leakage detected")

    version = "sft-multiturn-repair-" + hashlib.sha256(
        json.dumps(
            {split: [item.example_id for item in rows] for split, rows in output.items()},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    all_rows = [item for rows in output.values() for item in rows]
    manifest = DatasetManifest(
        dataset_version=version,
        candidate_episodes=len(all_rows),
        accepted_episodes=len(all_rows),
        rejected_episodes=0,
        exported_examples=len(all_rows),
        split_examples={split: len(rows) for split, rows in output.items()},
        source_episodes=dict(Counter(item.source for item in all_rows)),
        quality_episodes=dict(Counter(item.quality_label for item in all_rows)),
        rejection_codes={},
        environment_versions=sorted({item.environment_version for item in all_rows}),
        policy_versions=sorted(
            {f"{item.policy_name}:{item.policy_version}" for item in all_rows}
        ),
        split_group_overlap=False,
        excluded_policy_steps=0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in output.items():
        (output_dir / f"{split}.jsonl").write_text(
            "\n".join(item.model_dump_json() for item in rows) + "\n",
            encoding="utf-8",
        )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "derivation.json").write_text(
        json.dumps(
            {
                "official_source_split": "train only",
                "official_validation_or_test_used": False,
                "multi_turn_recovery_examples": len(candidates),
                "replay_ratio": "1:1",
                "history_contract": "TRL tool-call loop compatible",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-file", type=Path, required=True)
    parser.add_argument("--official-train-file", type=Path, required=True)
    parser.add_argument("--replay-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        build(
            args.candidates_file,
            args.official_train_file,
            args.replay_dataset_dir,
            args.output_dir,
        ).model_dump_json(indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
