"""Compare normalized deterministic and agent JSONL runs.

Each input line must conform to ``evaluation.agentic_eval.EvaluationRun``.
The command exits non-zero when the configured promotion gate is not met.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = REPO_ROOT / "backend" / "src"
sys.path.insert(0, str(BACKEND_SRC))

from evaluation.agentic_eval import AgenticEvaluator, EvaluationRun, ReleaseGateConfig  # noqa: E402


def _read_jsonl(path: Path) -> list[EvaluationRun]:
    runs: list[EvaluationRun] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            runs.append(EvaluationRun(**json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-paired-scenarios", type=int, default=300)
    args = parser.parse_args()

    report = AgenticEvaluator().compare(
        _read_jsonl(args.deterministic),
        _read_jsonl(args.agent),
        gate=ReleaseGateConfig(minimum_paired_scenarios=args.minimum_paired_scenarios),
    )
    payload = report.model_dump_json(indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.release_eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
