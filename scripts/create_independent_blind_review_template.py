"""Create a human-only worksheet for a quarantined SFT blind-review packet.

This command intentionally accepts the packet and never opens the withheld key,
source corpus, derivation report, or training data.  A reviewer fills the
per-record dispositions after inspecting the packet; the release audit performs
the later key comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _read_packet(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not str(value.get("review_id") or "").strip():
            raise ValueError(f"packet line {line_number} has no valid review_id")
        rows.append(value)
    ids = [str(row["review_id"]) for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("packet review IDs must be non-empty and unique")
    return rows


def build_template(packet_path: Path, output_path: Path, reviewer: str) -> dict[str, Any]:
    """Write a pending review template without reading any labels or source data."""

    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("reviewer must be non-empty")
    packet = _read_packet(packet_path)
    template = {
        "status": "pending",
        "review_protocol": "packet-only-no-key-no-source-label",
        "reviewer": reviewer,
        "review_started_at": datetime.now(UTC).isoformat(),
        "attestation": "",
        "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        "reviewed_records": len(packet),
        "matched_pairs": None,
        "rejected_candidates": None,
        "failed_review_ids": [],
        "record_dispositions": [
            {"review_id": str(row["review_id"]), "disposition": ""}
            for row in packet
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    args = parser.parse_args()
    template = build_template(args.packet, args.output, args.reviewer)
    print(
        json.dumps(
            {
                "status": template["status"],
                "reviewed_records": template["reviewed_records"],
                "packet_sha256": template["packet_sha256"],
                "output": args.output.as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
