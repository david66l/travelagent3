"""Append one hash-chained promotion access event after fail-closed checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from evaluation.posttraining_promotion_protocol import (  # noqa: E402
    ACCESS_SCHEMA_VERSION,
    PromotionAccessEvent,
    PromotionProtocol,
    PromotionReadiness,
    access_event_hash,
    audit_access_log,
    audit_protocol,
    audit_readiness,
)


BASE_DIR = ROOT / "evals" / "promotion-val-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_events(path: Path) -> list[PromotionAccessEvent]:
    if not path.exists():
        return []
    return [
        PromotionAccessEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", choices=("campaign_opened", "campaign_closed"), required=True)
    parser.add_argument("--experiment-family", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--frozen-run-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=BASE_DIR / "protocol.json")
    parser.add_argument("--protocol-sha256", type=Path, default=BASE_DIR / "protocol.sha256")
    parser.add_argument("--readiness", type=Path, default=BASE_DIR / "readiness.json")
    parser.add_argument("--access-log", type=Path, default=BASE_DIR / "access-log.jsonl")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_sha256 = _sha256(args.protocol)
    frozen_sha256 = args.protocol_sha256.read_text(encoding="utf-8").split()[0]
    if protocol_sha256 != frozen_sha256:
        raise RuntimeError("promotion protocol hash drifted; refuse access")
    protocol = PromotionProtocol.model_validate_json(args.protocol.read_text(encoding="utf-8"))
    if not audit_protocol(protocol)["passed"]:
        raise RuntimeError("promotion protocol audit failed")
    readiness = PromotionReadiness.model_validate_json(args.readiness.read_text(encoding="utf-8"))
    readiness_audit = audit_readiness(readiness, protocol_file_sha256=protocol_sha256)
    if not readiness_audit["passed"]:
        raise RuntimeError("promotion readiness audit failed")
    if args.event == "campaign_opened" and not readiness.h006_gpu_training_authorized:
        raise RuntimeError("promotion campaign cannot open before H-006 is explicitly authorized")

    events = _read_events(args.access_log)
    existing = audit_access_log(
        events,
        max_campaigns_per_family=protocol.access_policy.promotion_campaigns_per_experiment_family,
    )
    if not existing["passed"]:
        raise RuntimeError("existing promotion access log is invalid")
    open_families = set(existing["currently_open_families"])
    opened_count = int(existing["campaigns_opened"].get(args.experiment_family, 0))
    if args.event == "campaign_opened":
        if args.experiment_family in open_families:
            raise RuntimeError("campaign is already open")
        if opened_count >= protocol.access_policy.promotion_campaigns_per_experiment_family:
            raise RuntimeError("promotion access budget is exhausted for this experiment family")
    elif args.experiment_family not in open_families:
        raise RuntimeError("campaign cannot close before it is opened")

    payload = {
        "schema_version": ACCESS_SCHEMA_VERSION,
        "sequence": len(events),
        "event_type": args.event,
        "experiment_family": args.experiment_family,
        "split": "promotion-val-v1",
        "protocol_sha256": protocol_sha256,
        "frozen_run_manifest_sha256": _sha256(args.frozen_run_manifest),
        "actor": args.actor,
        "reason": args.reason,
        "previous_event_sha256": events[-1].event_sha256 if events else "GENESIS",
    }
    payload["event_sha256"] = access_event_hash(payload)
    event = PromotionAccessEvent.model_validate(payload)
    args.access_log.parent.mkdir(parents=True, exist_ok=True)
    with args.access_log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(event.model_dump_json() + "\n")
        handle.flush()
    print(json.dumps(event.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
