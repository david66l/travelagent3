"""Validate external benchmark contracts without exposing sealed case content."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from evaluation.external_benchmark import (  # noqa: E402
    ExternalBenchmarkCase,
    ForbiddenCorpusDocument,
    audit_external_benchmark,
    audit_training_contamination,
)


def read_jsonl(path: Path) -> list[ExternalBenchmarkCase]:
    return [
        ExternalBenchmarkCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _strings(value, *, prefix: str = "root"):
    if isinstance(value, str):
        if len(value.strip()) >= 8:
            yield prefix, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, prefix=f"{prefix}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, prefix=f"{prefix}.{key}")


def read_forbidden_documents(paths: list[Path]) -> list[ForbiddenCorpusDocument]:
    documents = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            for field, text in _strings(value):
                documents.append(
                    ForbiddenCorpusDocument(
                        document_id=f"{path.name}:{line_number}:{field}", text=text
                    )
                )
    return documents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--sealed-test", type=Path, required=True)
    parser.add_argument(
        "--forbidden-corpus",
        type=Path,
        action="append",
        default=[],
        help="Repeat for every SFT/DPO/GRPO/holdout JSONL that external data must not match.",
    )
    parser.add_argument("--sealed-access-events", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sealed_access_events < 0:
        parser.error("--sealed-access-events must be non-negative")
    cases = read_jsonl(args.dev) + read_jsonl(args.sealed_test)
    report = audit_external_benchmark(
        cases, sealed_access_events=args.sealed_access_events
    )
    documents = read_forbidden_documents(args.forbidden_corpus)
    contamination = audit_training_contamination(cases, documents)
    report["training_contamination"] = contamination
    report["gates"]["forbidden_corpora_registered"] = bool(
        args.forbidden_corpus
    ) and bool(documents)
    report["gates"]["training_contamination"] = contamination["passed"]
    report["passed"] = all(report["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": report["passed"], "gates": report["gates"]}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
