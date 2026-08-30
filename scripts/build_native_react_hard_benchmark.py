"""Build and audit the 200-case frozen Native ReAct hard benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from evaluation.native_react_hard_benchmark import write_benchmark  # noqa: E402


PROMPT_KEYS = {"user_input", "revision_input", "feedback", "user_query"}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _extract_prompts(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        if value.get("role") == "user" and isinstance(value.get("content"), str):
            yield value["content"]
        for key, item in value.items():
            if key in PROMPT_KEYS and isinstance(item, str):
                yield item
            else:
                yield from _extract_prompts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _extract_prompts(item)


def collect_forbidden_prompts(roots: list[Path]) -> tuple[list[str], dict[str, int]]:
    prompts: set[str] = set()
    files = 0
    invalid_lines = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            files += 1
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_lines += 1
                        continue
                    prompts.update(
                        item.strip() for item in _extract_prompts(value) if item.strip()
                    )
    return sorted(prompts), {
        "jsonl_files_scanned": files,
        "unique_prompts_extracted": len(prompts),
        "invalid_jsonl_lines": invalid_lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contamination-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    prompts, scan = collect_forbidden_prompts(args.contamination_root)
    manifest = write_benchmark(
        args.output_dir,
        forbidden_prompts=prompts,
        git_commit=_git_commit(),
    )
    manifest["contamination_scan"] = scan
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
