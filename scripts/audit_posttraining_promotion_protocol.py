"""Audit the frozen post-training promotion protocol without reading protected cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from evaluation.posttraining_promotion_protocol import (  # noqa: E402
    LineageRecord,
    PromotionAccessEvent,
    PromotionProtocol,
    PromotionReadiness,
    audit_access_log,
    audit_lineage_isolation,
    audit_protocol,
    audit_readiness,
)


DEFAULT_PROTOCOL = ROOT / "evals" / "promotion-val-v1" / "protocol.json"
DEFAULT_PROTOCOL_SHA256 = ROOT / "evals" / "promotion-val-v1" / "protocol.sha256"
DEFAULT_READINESS = ROOT / "evals" / "promotion-val-v1" / "readiness.json"


def read_jsonl(path: Path, model_type):
    return [
        model_type.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-sha256", type=Path, default=DEFAULT_PROTOCOL_SHA256)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--lineage", type=Path)
    parser.add_argument("--access-log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-dataset-ready", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_bytes = args.protocol.read_bytes()
    actual_protocol_file_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    expected_protocol_file_sha256 = args.protocol_sha256.read_text(encoding="utf-8").split()[0]
    protocol = PromotionProtocol.model_validate_json(
        protocol_bytes.decode("utf-8")
    )
    protocol_audit = audit_protocol(protocol)
    protocol_audit["file_sha256"] = actual_protocol_file_sha256
    protocol_audit["frozen_file_sha256_matches"] = (
        actual_protocol_file_sha256 == expected_protocol_file_sha256
    )
    protocol_audit["passed"] = bool(
        protocol_audit["passed"] and protocol_audit["frozen_file_sha256_matches"]
    )
    readiness = PromotionReadiness.model_validate_json(
        args.readiness.read_text(encoding="utf-8")
    )
    readiness_audit = audit_readiness(
        readiness, protocol_file_sha256=actual_protocol_file_sha256
    )

    lineage_audit = None
    if args.lineage:
        lineage_audit = audit_lineage_isolation(read_jsonl(args.lineage, LineageRecord))

    access_audit = None
    if args.access_log:
        access_audit = audit_access_log(
            read_jsonl(args.access_log, PromotionAccessEvent),
            max_campaigns_per_family=(
                protocol.access_policy.promotion_campaigns_per_experiment_family
            ),
        )

    dataset_ready = bool(
        readiness_audit["dataset_ready"]
        and lineage_audit
        and lineage_audit["passed"]
    )
    access_log_valid = access_audit is None or access_audit["passed"]
    report = {
        "schema_version": "travelagent-posttraining-promotion-readiness.v1",
        "protocol": protocol_audit,
        "readiness": readiness_audit,
        "lineage": lineage_audit,
        "access_log": access_audit,
        "access_log_valid": access_log_valid,
        "dataset_ready": dataset_ready,
        "h006_gpu_training_authorized": bool(
            dataset_ready
            and access_log_valid
            and readiness.h006_gpu_training_authorized
        ),
        "protected_case_payloads_read": False,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not protocol_audit["passed"]:
        return 2
    if not readiness_audit["passed"] or not access_log_valid:
        return 2
    if args.require_dataset_ready and not dataset_ready:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
