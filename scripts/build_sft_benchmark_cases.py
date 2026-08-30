"""Project held-out SFT examples into exact-action/argument HTTP benchmark cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.sft_dataset import SFTExample  # noqa: E402
from evaluation.inference_benchmark import VLLMBenchmarkCase  # noqa: E402


def build(source_file: Path, output_file: Path) -> dict[str, int | str]:
    cases = []
    for line in source_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        example = SFTExample(**json.loads(line))
        decision = example.messages[-1].tool_calls[0].function
        messages = [
            message.model_dump(exclude_none=True, exclude={"tool_calls"})
            for message in example.messages[:-1]
        ]
        allowed = [str(tool["function"]["name"]) for tool in example.tools]
        cases.append(
            VLLMBenchmarkCase(
                case_id=example.example_id,
                messages=messages,
                tools=example.tools,
                allowed_actions=allowed,
                expected_action=decision.name,
                expected_arguments=decision.arguments,
                family=decision.name,
            )
        )
    if not cases:
        raise ValueError("SFT benchmark source is empty")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("SFT benchmark case IDs are not unique")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(case.model_dump_json() for case in cases) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "sft-exact-tool-benchmark.v1",
        "source_file": str(source_file),
        "cases": len(cases),
        "unique_case_ids": len(cases),
    }
    (output_file.parent / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_file, args.output_file), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
