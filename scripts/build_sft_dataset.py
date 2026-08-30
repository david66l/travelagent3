"""Audit Agent Loop episode candidates and export policy SFT JSONL splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.sft_dataset import EpisodeCandidate, SFTDatasetBuilder  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        nargs="+",
        help="One or more EpisodeCandidate JSONL files",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=16)
    args = parser.parse_args()

    candidates: list[EpisodeCandidate] = []
    for input_path in args.input:
        for line_number, line in enumerate(
            input_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                candidates.append(EpisodeCandidate(**json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{input_path}:{line_number}: {exc}") from exc

    builder = SFTDatasetBuilder(max_steps=args.max_steps)
    result = builder.build(candidates)
    builder.export(result, args.output_dir)
    print(result.manifest.model_dump_json(indent=2))
    return 0 if result.manifest.rejected_episodes == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
