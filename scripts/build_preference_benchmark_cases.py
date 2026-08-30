"""Project verified preference holdouts into exact tool-decision benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.distillation import TeacherPreferencePair  # noqa: E402
from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT  # noqa: E402
from evaluation.inference_benchmark import VLLMBenchmarkCase  # noqa: E402


def build(source_file: Path, output_file: Path) -> dict[str, object]:
    cases: list[VLLMBenchmarkCase] = []
    for line in source_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pair = TeacherPreferencePair(**json.loads(line))
        calls = pair.chosen.get("tool_calls") or []
        if len(calls) != 1:
            raise ValueError(f"chosen response must contain one tool call: {pair.pair_id}")
        decision = calls[0]["function"]
        cases.append(
            VLLMBenchmarkCase(
                case_id=pair.pair_id,
                messages=[
                    {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
                    *pair.messages[1:],
                ],
                tools=pair.tools,
                allowed_actions=[str(tool["function"]["name"]) for tool in pair.tools],
                expected_action=decision["name"],
                expected_arguments=(
                    decision.get("arguments") or {}
                    if pair.family in {"search", "recovery"}
                    else None
                ),
                family=pair.family,
            )
        )
    if not cases:
        raise ValueError("preference benchmark source is empty")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("preference benchmark case IDs are not unique")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(case.model_dump_json() for case in cases) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "verified-preference-exact-tool-benchmark.v1",
        "source_file": str(source_file),
        "cases": len(cases),
        "family_counts": dict(Counter(case.family for case in cases)),
        "exact_argument_families": ["search", "recovery"],
        "action_only_families": ["clarification", "tradeoff"],
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
