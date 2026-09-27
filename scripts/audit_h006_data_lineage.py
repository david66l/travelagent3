"""Audit H-006 train-only candidates against forbidden post-training donors.

This audit is deliberately asymmetric.  It may read explicitly supplied old
training/internal-development GRPO corpora, but it never opens promotion or
sealed case payloads.  Those protected sets must be represented by a separate
hash-only lineage registry before GPU training can be authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from evaluation.external_benchmark import (  # noqa: E402
    character_ngrams,
    jaccard_similarity,
    normalize_text,
)
from evaluation.posttraining_promotion_protocol import (  # noqa: E402
    TRAIN_ELIGIBLE_SPLITS,
    LineageRecord,
    canonical_hash,
)


SCHEMA_VERSION = "h006-data-lineage-audit.v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision(row: dict[str, Any]) -> dict[str, Any]:
    value = (
        (row.get("snapshot") or {}).get("hidden_test_facts") or {}
    ).get("grpo_decision_state")
    if not isinstance(value, dict):
        raise ValueError("GRPO row is missing grpo_decision_state")
    return value


def _source_id(row: dict[str, Any]) -> str:
    hidden = (row.get("snapshot") or {}).get("hidden_test_facts") or {}
    decision = hidden.get("grpo_decision_state")
    lineage = hidden.get("training_lineage")
    value = str(
        (decision.get("source_task_id") if isinstance(decision, dict) else None)
        or (lineage.get("source_state_id") if isinstance(lineage, dict) else None)
        or (row.get("task") or {}).get("task_id")
        or ""
    ).strip()
    if not value:
        raise ValueError("GRPO decision is missing source_task_id")
    return value


def _tool_snapshot_hash(row: dict[str, Any]) -> str:
    tools = (row.get("snapshot") or {}).get("tool_responses") or {}
    return canonical_hash(tools)


def _source_hashes(row: dict[str, Any]) -> set[str]:
    hidden = (row.get("snapshot") or {}).get("hidden_test_facts") or {}
    values = {
        str(hidden.get("source_content_hash") or "").strip(),
        canonical_hash(
            {
                "task": row.get("task") or {},
                "state_id": (row.get("snapshot") or {}).get("state_id"),
                "tool_responses": (row.get("snapshot") or {}).get("tool_responses")
                or {},
            }
        ),
    }
    return {value for value in values if value}


def _visible_prompt(row: dict[str, Any]) -> str:
    hidden = (row.get("snapshot") or {}).get("hidden_test_facts") or {}
    decision = hidden.get("grpo_decision_state")
    if not isinstance(decision, dict):
        return str((row.get("task") or {}).get("user_request") or "").strip()
    messages = list(decision.get("prompt_messages") or [])
    if not messages:
        raise ValueError("GRPO decision is missing prompt_messages")
    payload = json.loads(str(messages[-1].get("content") or "{}"))
    state = payload.get("policy_state") or {}
    request = str(state.get("original_request") or "")
    reports = [
        item
        for item in state.get("relevant_artifacts") or []
        if isinstance(item, dict) and item.get("artifact_type") == "validation_report"
    ]
    violations = reports[-1].get("violations") or [] if reports else []
    evidence = " ".join(
        str(item.get("message") or "") for item in violations if isinstance(item, dict)
    )
    capability = state.get("capability") or {}
    alternatives = " ".join(
        str(item) for item in capability.get("alternatives") or [] if isinstance(item, str)
    )
    return " ".join((request, evidence, alternatives)).strip()


def _near_duplicate_audit(
    candidates: Iterable[tuple[str, str]],
    forbidden: Iterable[tuple[str, str]],
    *,
    threshold: float,
) -> dict[str, Any]:
    if not 0 < threshold <= 1:
        raise ValueError("similarity threshold must be in (0, 1]")
    forbidden_rows = [
        (row_id, normalize_text(text), character_ngrams(text))
        for row_id, text in forbidden
        if normalize_text(text)
    ]
    exact_index: dict[str, list[int]] = defaultdict(list)
    ngram_index: dict[str, set[int]] = defaultdict(set)
    for index, (_, normalized, grams) in enumerate(forbidden_rows):
        exact_index[normalized].append(index)
        for gram in grams:
            ngram_index[gram].add(index)

    exact_matches = 0
    near_matches = 0
    max_similarity = 0.0
    findings: list[dict[str, Any]] = []
    for candidate_id, text in candidates:
        normalized = normalize_text(text)
        grams = character_ngrams(text)
        possible: set[int] = set()
        for gram in grams:
            possible.update(ngram_index.get(gram, set()))
        best_index = None
        best = 0.0
        for index in possible:
            similarity = jaccard_similarity(grams, forbidden_rows[index][2])
            if similarity > best:
                best = similarity
                best_index = index
        max_similarity = max(max_similarity, best)
        kind = None
        if normalized and normalized in exact_index:
            kind = "exact"
            best = 1.0
            best_index = exact_index[normalized][0]
            exact_matches += 1
        elif best_index is not None and best >= threshold:
            kind = "near_duplicate"
            near_matches += 1
        if kind is not None and best_index is not None:
            findings.append(
                {
                    "candidate_id_sha256": canonical_hash(candidate_id),
                    "forbidden_id_sha256": canonical_hash(forbidden_rows[best_index][0]),
                    "type": kind,
                    "similarity": round(best, 8),
                }
            )
    return {
        "passed": exact_matches == 0 and near_matches == 0,
        "threshold": threshold,
        "candidate_prompts": sum(1 for _ in candidates)
        if not isinstance(candidates, list)
        else len(candidates),
        "forbidden_prompts": len(forbidden_rows),
        "exact_matches": exact_matches,
        "near_duplicate_matches": near_matches,
        "max_similarity": round(max_similarity, 8),
        "finding_hashes": sorted(
            canonical_hash(item) for item in findings
        ),
        "privacy_note": "Findings contain hashes and similarity only, never prompt text.",
    }


def audit(
    candidate_dir: Path,
    legacy_h001_train: Path,
    forbidden_grpo: list[Path],
    *,
    protected_lineage: Path | None = None,
    similarity_threshold: float = 0.82,
) -> dict[str, Any]:
    candidate_rows = [
        *_read_jsonl(candidate_dir / "train.jsonl"),
        *_read_jsonl(candidate_dir / "validation.jsonl"),
    ]
    lineage = [
        LineageRecord.model_validate(item)
        for item in _read_jsonl(candidate_dir / "lineage.jsonl")
    ]
    lineage_by_record = {item.record_id: item for item in lineage}
    candidate_ids = [str((row.get("task") or {}).get("task_id") or "") for row in candidate_rows]
    candidate_source_ids = {_source_id(row) for row in candidate_rows}
    embedded_matches = 0
    for row, record_id in zip(candidate_rows, candidate_ids, strict=True):
        record = lineage_by_record.get(record_id)
        embedded = (
            (row.get("snapshot") or {}).get("hidden_test_facts") or {}
        ).get("training_lineage")
        if (
            record is not None
            and isinstance(embedded, dict)
            and embedded.get("source_state_id") == record.source_state_id
            and embedded.get("source_hash") == record.source_hash
            and embedded.get("tool_snapshot_hash") == record.tool_snapshot_hash
        ):
            embedded_matches += 1

    forbidden_rows: list[tuple[str, dict[str, Any]]] = []
    forbidden_file_manifest = []
    for path in forbidden_grpo:
        rows = _read_jsonl(path)
        forbidden_rows.extend((path.name, row) for row in rows)
        forbidden_file_manifest.append(
            {"name": path.name, "sha256": _sha256(path), "rows": len(rows)}
        )
    forbidden_source_ids = {_source_id(row) for _, row in forbidden_rows}
    forbidden_tool_hashes = {_tool_snapshot_hash(row) for _, row in forbidden_rows}
    forbidden_source_hashes = {
        value for _, row in forbidden_rows for value in _source_hashes(row)
    }

    candidate_tool_hashes = {_tool_snapshot_hash(row) for row in candidate_rows}
    candidate_source_hashes = {record.source_hash for record in lineage}
    candidate_source_overlap = candidate_source_ids & forbidden_source_ids
    candidate_tool_overlap = candidate_tool_hashes & forbidden_tool_hashes
    candidate_hash_overlap = candidate_source_hashes & forbidden_source_hashes

    legacy_rows = _read_jsonl(legacy_h001_train)
    legacy_source_ids = {
        str(row.get("trajectory_id") or row.get("scenario_id") or "").strip()
        for row in legacy_rows
    } - {""}
    legacy_forbidden_overlap = legacy_source_ids & forbidden_source_ids

    candidate_prompts = [
        (record_id, _visible_prompt(row))
        for record_id, row in zip(candidate_ids, candidate_rows, strict=True)
    ]
    forbidden_prompts = [
        (
            f"{file_name}:{(row.get('task') or {}).get('task_id')}",
            _visible_prompt(row),
        )
        for file_name, row in forbidden_rows
    ]
    prompt_audit = _near_duplicate_audit(
        candidate_prompts,
        forbidden_prompts,
        threshold=similarity_threshold,
    )

    protected_records: list[LineageRecord] = []
    if protected_lineage is not None:
        protected_records = [
            LineageRecord.model_validate(item) for item in _read_jsonl(protected_lineage)
        ]
    protected_source_ids = {item.source_state_id for item in protected_records}
    protected_source_hashes = {item.source_hash for item in protected_records}
    protected_tool_hashes = {item.tool_snapshot_hash for item in protected_records}
    protected_splits = Counter(item.split for item in protected_records)
    required_protected_splits = {"promotion-val-v1", "sealed-160"}
    protected_registry_complete = required_protected_splits.issubset(protected_splits)
    protected_overlap = bool(
        candidate_source_ids & protected_source_ids
        or candidate_source_hashes & protected_source_hashes
        or candidate_tool_hashes & protected_tool_hashes
    )

    train_lineage = [
        item
        for item in lineage
        if item.split in TRAIN_ELIGIBLE_SPLITS and item.split != "train-shadow"
    ]
    shadow_lineage = [item for item in lineage if item.split == "train-shadow"]
    train_sources = {item.source_state_id for item in train_lineage}
    shadow_sources = {item.source_state_id for item in shadow_lineage}
    gates = {
        "candidate_rows_registered": bool(candidate_rows),
        "candidate_lineage_complete": len(lineage) == len(candidate_rows),
        "candidate_record_ids_unique": len(candidate_ids) == len(set(candidate_ids)) == len(lineage_by_record),
        "embedded_lineage_exact": embedded_matches == len(candidate_rows),
        "one_candidate_per_source": len(candidate_source_ids) == len(candidate_rows),
        "train_shadow_source_isolation": not (train_sources & shadow_sources),
        "forbidden_source_id_overlap_zero": not candidate_source_overlap,
        "forbidden_source_hash_overlap_zero": not candidate_hash_overlap,
        "forbidden_tool_snapshot_overlap_zero": not candidate_tool_overlap,
        "forbidden_prompt_overlap_zero": prompt_audit["passed"],
        "legacy_h001_direct_reuse_blocked": bool(legacy_forbidden_overlap),
        "protected_hash_registry_complete": protected_registry_complete,
        "protected_hash_overlap_zero": protected_registry_complete and not protected_overlap,
    }
    train_only_gate_names = [
        name for name in gates if not name.startswith("protected_")
    ]
    train_only_passed = all(gates[name] for name in train_only_gate_names)
    gpu_training_authorized = train_only_passed and all(gates.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if train_only_passed else "rejected",
        "train_only_donor_rollout_authorized": train_only_passed,
        "h006_gpu_training_authorized": gpu_training_authorized,
        "gates": gates,
        "counts": {
            "candidate_rows": len(candidate_rows),
            "candidate_sources": len(candidate_source_ids),
            "train_sources": len(train_sources),
            "train_shadow_sources": len(shadow_sources),
            "forbidden_rows": len(forbidden_rows),
            "forbidden_sources": len(forbidden_source_ids),
            "legacy_h001_rows": len(legacy_rows),
            "legacy_h001_sources": len(legacy_source_ids),
            "legacy_h001_forbidden_source_overlaps": len(legacy_forbidden_overlap),
            "candidate_forbidden_source_overlaps": len(candidate_source_overlap),
            "candidate_forbidden_source_hash_overlaps": len(candidate_hash_overlap),
            "candidate_forbidden_tool_snapshot_overlaps": len(candidate_tool_overlap),
            "protected_lineage_records": len(protected_records),
        },
        "prompt_similarity": prompt_audit,
        "legacy_h001": {
            "direct_payload_reuse_allowed": not legacy_forbidden_overlap,
            "decision": (
                "block payload reuse; retain checkpoint/contract as teacher reference only"
                if legacy_forbidden_overlap
                else "eligible only after current-contract revalidation"
            ),
            "overlap_source_hashes": sorted(
                canonical_hash(item) for item in legacy_forbidden_overlap
            ),
        },
        "forbidden_files": forbidden_file_manifest,
        "protected_registry": {
            "path": str(protected_lineage) if protected_lineage else None,
            "payloads_read": False,
            "required_splits": sorted(required_protected_splits),
            "registered_split_counts": dict(sorted(protected_splits.items())),
            "complete": protected_registry_complete,
        },
        "blocking_reasons": [
            name for name, passed in gates.items() if not passed
        ],
        "privacy_note": (
            "Promotion and sealed payloads were not opened. Overlap evidence exposes only "
            "counts and canonical hashes."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--legacy-h001-train", type=Path, required=True)
    parser.add_argument("--forbidden-grpo", type=Path, action="append", default=[])
    parser.add_argument("--protected-lineage", type=Path)
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.candidate_dir,
        args.legacy_h001_train,
        args.forbidden_grpo,
        protected_lineage=args.protected_lineage,
        similarity_threshold=args.similarity_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["train_only_donor_rollout_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
