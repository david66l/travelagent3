"""Build audited GRPO-B0 action data from rollout-group JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_dataset import GRPODatasetBuilder, GRPOGroupCandidate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    candidates: list[GRPOGroupCandidate] = []
    for line_number, line in enumerate(
        args.input.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            candidates.append(GRPOGroupCandidate(**json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{args.input}:{line_number}: {exc}") from exc
    result = GRPODatasetBuilder().build(candidates)
    GRPODatasetBuilder.export(result, args.output_dir)
    print(result.manifest.model_dump_json(indent=2))
    return 0 if result.manifest.accepted_groups else 2


if __name__ == "__main__":
    raise SystemExit(main())
