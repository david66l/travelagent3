"""Authorize a quarantined tradeoff SFT only after transport and render gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.chat_template_contract import (  # noqa: E402
    AGENT_CHAT_TEMPLATE_KWARGS,
    AGENT_CHAT_TEMPLATE_SHA256,
    AGENT_RENDER_PROTOCOL_VERSION,
    install_agent_chat_template,
    validate_agent_chat_template,
)
from agentic.training import (  # noqa: E402
    load_jsonl,
    preflight_sft_dataset,
    preflight_sft_model,
    preflight_sft_termination_boundaries,
    to_rendered_prompt_completion,
)


SCHEMA_VERSION = "tradeoff-grounding-sft-readiness.v1"


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object on line {line_number}: {path}")
        rows.append(value)
    return rows


def _validate_independent_blind_review(
    dataset_dir: Path,
    blind_review: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Verify a human packet review against the withheld record-level key.

    The reviewer is shown only ``blind_review_packet.jsonl``.  This verifier is
    the first component allowed to open ``blind_review_key.jsonl`` after the
    reviewer submits a complete per-record disposition.  A summary-only human
    attestation cannot authorize training: it would be too easy to claim the
    right aggregate counts without actually inspecting the packet.
    """

    packet_path = dataset_dir / "blind_review_packet.jsonl"
    key_path = dataset_dir / "blind_review_key.jsonl"
    packet = _read_jsonl(packet_path)
    key = _read_jsonl(key_path)
    errors: list[str] = []

    packet_ids = [str(row.get("review_id") or "") for row in packet]
    key_by_id = {str(row.get("review_id") or ""): row for row in key}
    if not packet_ids or len(packet_ids) != len(set(packet_ids)):
        errors.append("BLIND_PACKET_ID_INVALID")
    if set(packet_ids) != set(key_by_id) or len(key_by_id) != len(key):
        errors.append("BLIND_REVIEW_KEY_PACKET_MISMATCH")

    expected_dispositions = {
        review_id: (
            "reject" if row.get("expected_disposition") == "reject" else "accept"
        )
        for review_id, row in key_by_id.items()
    }
    submitted = blind_review.get("record_dispositions")
    submitted_by_id: dict[str, str] = {}
    malformed_records = 0
    if not isinstance(submitted, list):
        errors.append("BLIND_REVIEW_RECORD_DISPOSITIONS_MISSING")
        submitted = []
    for row in submitted:
        if not isinstance(row, dict):
            malformed_records += 1
            continue
        review_id = str(row.get("review_id") or "")
        disposition = str(row.get("disposition") or "")
        if not review_id or disposition not in {"accept", "reject"}:
            malformed_records += 1
            continue
        if review_id in submitted_by_id:
            malformed_records += 1
            continue
        submitted_by_id[review_id] = disposition
    if malformed_records:
        errors.append("BLIND_REVIEW_RECORD_DISPOSITIONS_MALFORMED")
    if set(submitted_by_id) != set(packet_ids):
        errors.append("BLIND_REVIEW_RECORD_COVERAGE_INCOMPLETE")

    mismatched_ids = sorted(
        review_id
        for review_id, disposition in submitted_by_id.items()
        if expected_dispositions.get(review_id) != disposition
    )
    if mismatched_ids:
        errors.append("BLIND_REVIEW_RECORD_DISPOSITION_MISMATCH")

    expected_rejected = sum(
        disposition == "reject" for disposition in expected_dispositions.values()
    )
    reported_failed = blind_review.get("failed_review_ids")
    human_identity = str(blind_review.get("reviewer") or "").strip()
    attestation = str(blind_review.get("attestation") or "").strip()
    summary_pass = (
        blind_review.get("status") == "passed"
        and blind_review.get("review_protocol") == "packet-only-no-key-no-source-label"
        and blind_review.get("packet_sha256") == _sha256_file(packet_path)
        and bool(human_identity)
        and bool(attestation)
        and blind_review.get("reviewed_records") == len(packet)
        and blind_review.get("matched_pairs") == 30
        and blind_review.get("rejected_candidates") == expected_rejected
        and reported_failed == []
    )
    if not summary_pass:
        errors.append("INDEPENDENT_BLIND_REVIEW_ATTESTATION_INVALID")

    integrity = {
        "packet_sha256": _sha256_file(packet_path),
        "packet_records": len(packet),
        "key_records": len(key),
        "expected_rejected_candidates": expected_rejected,
        "submitted_record_dispositions": len(submitted_by_id),
        "record_disposition_mismatch_count": len(mismatched_ids),
        # Individual IDs are deliberately not printed: audit output may be
        # shared more broadly than the quarantined reviewer materials.
        "record_disposition_match": not mismatched_ids,
        "human_attestation_present": bool(human_identity and attestation),
    }
    return integrity, errors


def _arrow_roundtrip(dataset_dir: Path, tokenizer: Any) -> dict[str, Any]:
    from datasets import Dataset

    rows_checked = 0
    exact_rows = 0
    exact_prompt_fingerprints = 0
    split_counts: dict[str, int] = {}
    feature_schemas: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        before = to_rendered_prompt_completion(
            load_jsonl(dataset_dir / f"{split}.jsonl"),
            tokenizer,
            chat_template_kwargs=AGENT_CHAT_TEMPLATE_KWARGS,
        )
        arrow = Dataset.from_list(before)
        after = arrow.to_list()
        split_counts[split] = len(after)
        feature_schemas[split] = str(arrow.features)
        if len(before) != len(after):
            continue
        for expected, observed in zip(before, after, strict=True):
            rows_checked += 1
            if _canonical(expected) == _canonical(observed):
                exact_rows += 1
            expected_prompt = _sha256_text(expected["prompt"])
            observed_prompt = _sha256_text(observed["prompt"])
            if expected_prompt == observed_prompt:
                exact_prompt_fingerprints += 1
    return {
        "rows_checked": rows_checked,
        "exact_rows": exact_rows,
        "whole_row_equality_rate": exact_rows / rows_checked if rows_checked else 0.0,
        "exact_prompt_fingerprints": exact_prompt_fingerprints,
        "prompt_fingerprint_equality_rate": (
            exact_prompt_fingerprints / rows_checked if rows_checked else 0.0
        ),
        "split_counts": split_counts,
        "feature_schemas": feature_schemas,
    }


def audit(
    dataset_dir: Path,
    tokenizer_source: str,
    output_path: Path,
    *,
    max_length: int,
) -> dict[str, Any]:
    derivation = _read_json(dataset_dir / "derivation_report.json")
    blind_review = _read_json(dataset_dir / "independent_blind_review.json")
    errors: list[str] = []
    if derivation.get("status") != "passed":
        errors.append("DERIVATION_GATE_FAILED")
    if derivation.get("official_validation_or_test_used") is not False:
        errors.append("OFFICIAL_EVALUATION_DATA_WAS_USED")
    if derivation.get("source_train_template_profile") != "strict-counterfactual-v1":
        errors.append("SOURCE_TRAIN_PROFILE_NOT_ELIGIBLE")
    if int(derivation.get("strict_counterfactual_pair_count") or 0) < 30:
        errors.append("STRICT_COUNTERFACTUAL_PAIR_COVERAGE_INSUFFICIENT")
    quality = derivation.get("quality_gate") or {}
    required_quality = {
        "source_overlap_zero": True,
        "source_validation_or_test_files_read": 0,
        "exactly_one_tool_call_rate": 1.0,
        "schema_valid_rate": 1.0,
        "reason_grounding_rate": 1.0,
        # v2 teaches only model-owned ``reason``.  Controller options are
        # checked as an authority invariant instead of pretending the model
        # generated them.
        "tradeoff_model_owned_schema_rate": 1.0,
        "tradeoff_controller_authority_present_rate": 1.0,
        "private_label_leak_count": 0,
        "exact_model_visible_duplicate_count": 0,
        "reason_diversity_pass": True,
    }
    for name, expected in required_quality.items():
        if quality.get(name) != expected:
            errors.append(f"QUALITY_GATE_FAILED:{name}")

    blind_integrity, blind_errors = _validate_independent_blind_review(
        dataset_dir,
        blind_review,
    )
    errors.extend(blind_errors)
    blind_pass = not blind_errors
    if not blind_pass:
        errors.append("INDEPENDENT_BLIND_REVIEW_FAILED")

    dataset_preflight = preflight_sft_dataset(
        dataset_dir,
        minimum_train_examples=1,
        require_dependencies=True,
    )
    if not dataset_preflight.ready:
        errors.extend(f"DATASET_PREFLIGHT:{item}" for item in dataset_preflight.errors)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    install_agent_chat_template(tokenizer)
    validate_agent_chat_template(tokenizer.chat_template)
    model_preflight = preflight_sft_model(
        dataset_dir,
        tokenizer,
        max_length=max_length,
    )
    boundary_preflight = preflight_sft_termination_boundaries(dataset_dir, tokenizer)
    if not model_preflight.ready:
        errors.extend(f"MODEL_PREFLIGHT:{item}" for item in model_preflight.errors)
    if not boundary_preflight.ready:
        errors.extend(f"BOUNDARY_PREFLIGHT:{item}" for item in boundary_preflight.errors)

    arrow = _arrow_roundtrip(dataset_dir, tokenizer)
    if arrow["whole_row_equality_rate"] != 1.0:
        errors.append("ARROW_WHOLE_ROW_ROUNDTRIP_MISMATCH")
    if arrow["prompt_fingerprint_equality_rate"] != 1.0:
        errors.append("ARROW_PROMPT_FINGERPRINT_MISMATCH")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not errors else "rejected",
        "training_authorized": not errors,
        "authorization_scope": "quarantined-targeted-sft-only",
        "dataset_version": derivation.get("dataset_version"),
        "dataset_dir": dataset_dir.as_posix(),
        "tokenizer_source": tokenizer_source,
        "render_contract": {
            "protocol": AGENT_RENDER_PROTOCOL_VERSION,
            "chat_template_sha256": AGENT_CHAT_TEMPLATE_SHA256,
            "chat_template_kwargs": AGENT_CHAT_TEMPLATE_KWARGS,
        },
        "dataset_preflight": dataset_preflight.model_dump(mode="json"),
        "model_preflight": model_preflight.model_dump(mode="json"),
        "termination_boundary_preflight": boundary_preflight.model_dump(mode="json"),
        "arrow_transport": arrow,
        "derivation_quality_gate": quality,
        "independent_blind_review": blind_review,
        "independent_blind_review_integrity": blind_integrity,
        "production_promotion_authorized": False,
        "production_blockers": [
            "HUMAN_BLIND_REVIEW_REQUIRED",
            "INTERNAL_DEV_AND_TRAIN_SHADOW_MODEL_EVAL_REQUIRED",
            "FORMAL_VALIDATION_REQUIRED",
            "ACTUAL_VLLM_HTTP_PARITY_REQUIRED",
        ],
        "errors": sorted(set(errors)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=6144)
    args = parser.parse_args()
    report = audit(
        args.dataset_dir,
        args.tokenizer,
        args.output,
        max_length=args.max_length,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
