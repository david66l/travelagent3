import json
from pathlib import Path

from evaluation.posttraining_promotion_protocol import (
    ACCESS_SCHEMA_VERSION,
    LINEAGE_SCHEMA_VERSION,
    LineageRecord,
    PromotionAccessEvent,
    PromotionProtocol,
    PromotionReadiness,
    access_event_hash,
    audit_access_log,
    audit_lineage_isolation,
    audit_protocol,
    audit_readiness,
)


ROOT = Path(__file__).resolve().parents[4]
PROTOCOL_PATH = ROOT / "evals" / "promotion-val-v1" / "protocol.json"
READINESS_PATH = ROOT / "evals" / "promotion-val-v1" / "readiness.json"


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _lineage(record_id: str, split: str, source: str) -> LineageRecord:
    return LineageRecord(
        schema_version=LINEAGE_SCHEMA_VERSION,
        record_id=record_id,
        split=split,
        source_state_id=source,
        source_hash=_sha(f"source:{source}"),
        template_lineage=f"template:{source}",
        tool_snapshot_hash=_sha(f"tool:{source}"),
        generator_model="frozen-policy",
        generator_prompt_hash=_sha(f"prompt:{source}"),
        parent_trajectory_id=f"parent:{source}",
    )


def _event(
    sequence: int,
    event_type: str,
    *,
    previous: str,
    family: str = "H-006",
) -> PromotionAccessEvent:
    payload = {
        "schema_version": ACCESS_SCHEMA_VERSION,
        "sequence": sequence,
        "event_type": event_type,
        "experiment_family": family,
        "split": "promotion-val-v1",
        "protocol_sha256": _sha("protocol"),
        "frozen_run_manifest_sha256": _sha("run-manifest"),
        "actor": "release-reviewer",
        "reason": "frozen promotion campaign",
        "previous_event_sha256": previous,
    }
    payload["event_sha256"] = access_event_hash(payload)
    return PromotionAccessEvent.model_validate(payload)


def test_frozen_protocol_is_valid_but_training_remains_locked():
    protocol = PromotionProtocol.model_validate_json(PROTOCOL_PATH.read_text(encoding="utf-8"))
    readiness = PromotionReadiness.model_validate_json(
        READINESS_PATH.read_text(encoding="utf-8")
    )

    result = audit_protocol(protocol)
    readiness_result = audit_readiness(
        readiness, protocol_file_sha256=readiness.protocol_sha256
    )

    assert result["passed"] is True
    assert readiness_result["prerequisites"]["protocol_locked"] is True
    assert readiness_result["prerequisites"]["promotion_dataset_authored"] is False
    assert readiness_result["h006_gpu_training_authorized"] is False


def test_protocol_rejects_premature_h006_authorization():
    value = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    value["h006_gpu_training_authorized"] = True

    try:
        PromotionReadiness.model_validate(value)
    except ValueError as error:
        assert "H-006 cannot be authorized" in str(error)
    else:
        raise AssertionError("premature H-006 authorization was accepted")


def test_readiness_rejects_protocol_hash_drift():
    readiness = PromotionReadiness.model_validate_json(
        READINESS_PATH.read_text(encoding="utf-8")
    )

    result = audit_readiness(readiness, protocol_file_sha256="0" * 64)

    assert result["passed"] is False
    assert result["prerequisites"]["protocol_hash_matches"] is False


def test_lineage_audit_detects_protected_source_reuse_without_exposing_ids():
    train = _lineage("train-row", "train-hard-donor", "shared-state")
    protected = _lineage("promotion-row", "promotion-val-v1", "shared-state")

    result = audit_lineage_isolation([train, protected])

    assert result["passed"] is False
    assert result["source_state_id_overlaps"]
    assert "shared-state" not in json.dumps(result)


def test_lineage_audit_passes_independent_sources():
    result = audit_lineage_isolation(
        [
            _lineage("train-row", "train-hard-donor", "train-state"),
            _lineage("promotion-row", "promotion-val-v1", "promotion-state"),
        ]
    )

    assert result["passed"] is True


def test_access_log_hash_chain_and_one_campaign_budget():
    opened = _event(0, "campaign_opened", previous="GENESIS")
    closed = _event(1, "campaign_closed", previous=opened.event_sha256)

    assert audit_access_log([opened, closed])["passed"] is True

    reopened = _event(2, "campaign_opened", previous=closed.event_sha256)
    result = audit_access_log([opened, closed, reopened])
    assert result["passed"] is False
    assert "campaign_budget_exceeded:H-006" in result["errors"]


def test_access_log_detects_tampering():
    opened = _event(0, "campaign_opened", previous="GENESIS")
    tampered = opened.model_copy(update={"reason": "changed after opening"})

    result = audit_access_log([tampered])

    assert result["passed"] is False
    assert "event_hash_mismatch:0" in result["errors"]
