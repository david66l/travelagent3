import hashlib
import json
from pathlib import Path

from scripts.audit_tradeoff_grounding_sft import _validate_independent_blind_review


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(dataset_dir: Path) -> tuple[dict, list[dict]]:
    packet = [{"review_id": f"r-{index:03d}"} for index in range(120)]
    key = [
        (
            {"review_id": row["review_id"], "expected_action": "propose_tradeoff"}
            if index < 90
            else {
                "review_id": row["review_id"],
                "expected_disposition": "reject",
            }
        )
        for index, row in enumerate(packet)
    ]
    _write_jsonl(dataset_dir / "blind_review_packet.jsonl", packet)
    _write_jsonl(dataset_dir / "blind_review_key.jsonl", key)
    packet_sha256 = hashlib.sha256(
        (dataset_dir / "blind_review_packet.jsonl").read_bytes()
    ).hexdigest()
    review = {
        "status": "passed",
        "review_protocol": "packet-only-no-key-no-source-label",
        "reviewer": "independent-human-reviewer",
        "attestation": "I reviewed only the blind packet and submitted each disposition.",
        "packet_sha256": packet_sha256,
        "reviewed_records": 120,
        "matched_pairs": 30,
        "rejected_candidates": 30,
        "failed_review_ids": [],
        "record_dispositions": [
            {
                "review_id": row["review_id"],
                "disposition": "accept" if index < 90 else "reject",
            }
            for index, row in enumerate(packet)
        ],
    }
    return review, packet


def test_human_blind_review_requires_complete_record_level_agreement(tmp_path):
    review, _ = _fixture(tmp_path)

    integrity, errors = _validate_independent_blind_review(tmp_path, review)

    assert errors == []
    assert integrity["packet_records"] == 120
    assert integrity["submitted_record_dispositions"] == 120
    assert integrity["expected_rejected_candidates"] == 30
    assert integrity["record_disposition_match"] is True


def test_human_blind_review_rejects_summary_only_or_mismatched_dispositions(tmp_path):
    review, _ = _fixture(tmp_path)
    review["record_dispositions"][0]["disposition"] = "reject"

    integrity, errors = _validate_independent_blind_review(tmp_path, review)

    assert integrity["record_disposition_mismatch_count"] == 1
    assert "BLIND_REVIEW_RECORD_DISPOSITION_MISMATCH" in errors

    review.pop("record_dispositions")
    _, summary_only_errors = _validate_independent_blind_review(tmp_path, review)
    assert "BLIND_REVIEW_RECORD_DISPOSITIONS_MISSING" in summary_only_errors
