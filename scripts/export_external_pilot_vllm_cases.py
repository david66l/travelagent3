"""Project rich external Pilot cases into the existing vLLM action benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from evaluation.external_benchmark import ExternalBenchmarkCase  # noqa: E402
from evaluation.inference_benchmark import VLLMBenchmarkCase  # noqa: E402


def load_external_cases(path: Path) -> list[ExternalBenchmarkCase]:
    return [
        ExternalBenchmarkCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def project(case: ExternalBenchmarkCase) -> VLLMBenchmarkCase:
    if not case.annotations:
        raise ValueError(f"case {case.case_id} has no draft action annotation")
    expected_action = (
        case.adjudication.primary_action
        if case.adjudication is not None
        else case.annotations[0].primary_action
    )
    return VLLMBenchmarkCase(
        case_id=case.case_id,
        family=case.group_keys.request_family,
        messages=case.messages,
        tools=case.tools,
        allowed_actions=case.allowed_actions,
        expected_action=expected_action,
        # Rich semantic argument rules are evaluated separately. Exact dict
        # equality would incorrectly reject equivalent questions/tradeoffs.
        expected_arguments=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        help="Repeat to merge multiple benchmark splits into one inference suite.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "ml/agentic/datasets/external-benchmark-v1/ai-assisted-pilot-v1/vllm-cases.jsonl"
        ),
    )
    args = parser.parse_args()
    inputs = args.input or [
        Path("ml/agentic/datasets/external-benchmark-v1/ai-assisted-pilot-v1/cases.jsonl")
    ]
    external = [case for path in inputs for case in load_external_cases(path)]
    projected = [project(case) for case in external]
    case_ids = [case.case_id for case in projected]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("input benchmark splits contain duplicate case IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(case.model_dump_json() for case in projected) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(projected), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
