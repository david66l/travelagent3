"""Rebuild and deduplicate SFT projection from completed teacher episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_teacher_distillation import (
    deduplicate_sft_result,
    load_holdout_contract,
    model_payload_hash,
)

from agentic.sft_dataset import EpisodeCandidate, SFTDatasetBuilder


def repair(
    output_dir: Path,
    *,
    forbidden_holdout_dir: Path | None = None,
) -> dict:
    chosen_path = output_dir / "chosen_episodes.jsonl"
    if not chosen_path.is_file():
        raise FileNotFoundError(f"missing chosen episodes: {chosen_path}")
    chosen = [
        EpisodeCandidate(**json.loads(line))
        for line in chosen_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_result = SFTDatasetBuilder().build(chosen)
    result, duplicates_dropped, conflicts = deduplicate_sft_result(raw_result)
    prompt_hashes = [
        model_payload_hash(
            [message.model_dump(mode="json") for message in example.messages[:-1]],
            example.tools,
        )
        for example in result.examples
    ]
    _, holdout_hashes = load_holdout_contract(forbidden_holdout_dir)
    overlap = set(prompt_hashes) & holdout_hashes
    errors: list[str] = []
    if raw_result.manifest.rejected_episodes:
        errors.append("SFT_CHOSEN_EPISODE_REJECTED")
    if len(prompt_hashes) != len(set(prompt_hashes)):
        errors.append("MODEL_PAYLOAD_DUPLICATE_AFTER_DEDUP")
    if overlap:
        errors.append("FROZEN_HOLDOUT_PAYLOAD_OVERLAP")

    SFTDatasetBuilder().export(result, output_dir / "sft")
    (output_dir / "sft_label_conflicts.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in conflicts)
        + ("\n" if conflicts else ""),
        encoding="utf-8",
    )
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("model_payload_label_conflicts", None)
    manifest.update(
        {
            "status": "passed" if not errors else "rejected",
            "sft_examples": len(result.examples),
            "raw_sft_examples": len(raw_result.examples),
            "model_payload_duplicates_dropped": duplicates_dropped,
            "model_payload_label_conflicts_quarantined": len(conflicts),
            "model_payload_conflict_rows_quarantined": sum(
                len(item["example_ids"]) for item in conflicts
            ),
            "unique_model_payloads": len(set(prompt_hashes)),
            "model_payloads": len(prompt_hashes),
            "frozen_holdout_payload_overlap": len(overlap),
            "sft_manifest": result.manifest.model_dump(mode="json"),
            "raw_sft_manifest": raw_result.manifest.model_dump(mode="json"),
            "errors": errors,
            "sft_projection_repaired": True,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forbidden-holdout-dir", type=Path)
    args = parser.parse_args()
    manifest = repair(
        args.output_dir,
        forbidden_holdout_dir=args.forbidden_holdout_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
