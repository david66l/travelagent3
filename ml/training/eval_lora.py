"""Evaluate LoRA adapter quality against a JSONL holdout set (M4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL with prompt/completion fields")
    parser.add_argument("--adapter", default="travel-plan-v1")
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--min-match-rate", type=float, default=0.8)
    parser.add_argument("--gate", action="store_true", help="Exit 1 when below min-match-rate")
    args = parser.parse_args()

    rows = _load_jsonl(args.dataset)
    sample_count = len(rows)
    exact_match = sum(
        1 for row in rows if row.get("completion") and row.get("prediction") == row.get("completion")
    )
    match_rate = exact_match / sample_count if sample_count else 0.0

    print(f"adapter={args.adapter} samples={sample_count} exact_match_rate={match_rate:.3f}")
    if sample_count < args.min_samples:
        print(f"WARN: fewer than {args.min_samples} samples — expand dataset before production gate")
        if args.gate:
            raise SystemExit(1)

    if args.gate and match_rate < args.min_match_rate:
        print(f"FAIL: match_rate {match_rate:.3f} < {args.min_match_rate}")
        raise SystemExit(1)

    tracking_uri = __import__("os").environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            with mlflow.start_run(run_name=f"eval-{args.adapter}"):
                mlflow.log_param("adapter", args.adapter)
                mlflow.log_metric("eval_samples", sample_count)
                mlflow.log_metric("exact_match_rate", match_rate)
        except ImportError:
            pass


if __name__ == "__main__":
    main()
