"""Re-score stored verifier-repair rollouts with the strict reason-quality gate.

This is an offline audit: it never samples the model and never opens validation
or frozen-test files beyond the explicitly supplied corpus.  It also runs a
deterministic adversarial suite to prove that phrase copying, generic reasons,
English wrappers, and private action names cannot obtain full reason credit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPO_ROOT), str(REPO_ROOT / "backend" / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402
from agentic.reason_quality import (  # noqa: E402
    build_grounded_repair_reason,
    verifier_reason_quality_checks,
)


SCHEMA_VERSION = "verifier-reason-quality-offline-audit.v1"
QUALITY_KEYS = (
    "grounding_match",
    "reason_specificity_match",
    "reason_language_match",
    "reason_public_language",
    "reason_action_rationale_match",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(payload)
    return rows


def _contract(row: GRPOCorpusRow) -> dict[str, Any]:
    value = row.snapshot.hidden_test_facts.get("grpo_decision_state")
    if not isinstance(value, dict):
        raise ValueError(f"missing decision contract: {row.task.task_id}")
    return value


def _visible_evidence(contract: dict[str, Any]) -> str:
    for message in reversed(list(contract.get("prompt_messages") or [])):
        try:
            payload = json.loads(str(message.get("content") or "{}"))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            continue
        state = payload.get("policy_state")
        if not isinstance(state, dict):
            continue
        reports = [
            item
            for item in state.get("relevant_artifacts") or []
            if isinstance(item, dict) and item.get("artifact_type") == "validation_report"
        ]
        violations = reports[-1].get("violations") or [] if reports else []
        if violations and isinstance(violations[0], dict):
            evidence = str(violations[0].get("message") or "").strip()
            if evidence:
                return evidence
    phrases = [str(item) for item in contract.get("grounding_phrases") or []]
    return "；".join(phrases)


def _reason_arguments(rollout: dict[str, Any]) -> dict[str, Any]:
    raw = rollout.get("model_raw_arguments")
    if isinstance(raw, dict):
        return raw
    observed = rollout.get("observed_arguments")
    return observed if isinstance(observed, dict) else {}


def _quality(contract: dict[str, Any], reason: Any) -> dict[str, bool]:
    return verifier_reason_quality_checks(
        reason=reason,
        target_action=str(contract.get("target_action") or ""),
        grounding_phrases=[
            str(item) for item in contract.get("grounding_phrases") or []
        ],
        evidence=_visible_evidence(contract),
    )


def _adversarial_suite(contract: dict[str, Any]) -> dict[str, bool]:
    target = str(contract.get("target_action") or "")
    phrases = [str(item) for item in contract.get("grounding_phrases") or []]
    evidence = _visible_evidence(contract)
    full_reason = build_grounded_repair_reason(evidence, target)
    generic = {
        "retry_solve": "当前安排存在问题，建议进行一次有界重算。",
        "propose_tradeoff": "当前安排存在冲突，需要您选择调整方案。",
        "abort": "当前安排存在问题且无法继续，因此停止规划。",
    }[target]
    attacks = {
        "single_phrase_copy": phrases[0] if phrases else "当前安排有冲突",
        "generic_action_rationale": generic,
        "english_wrapper": (
            "Verification failed because " + " and ".join(phrases) + ", please retry."
        ),
        "evidence_without_action_rationale": evidence,
        "private_action_name": f"{full_reason} 调用 {target}。",
    }
    return {
        name: all(_quality(contract, reason).values())
        for name, reason in attacks.items()
    }


def _rate(count: int, total: int) -> float:
    return round(count / total, 8) if total else 0.0


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    return {
        "rollouts": total,
        "action_accuracy": _rate(sum(item["action_match"] for item in records), total),
        "strict_reason_quality_rate": _rate(
            sum(item["strict_reason_quality"] for item in records), total
        ),
        "strict_full_success_rate": _rate(
            sum(item["strict_full_success"] for item in records), total
        ),
        **{
            f"{key}_rate": _rate(
                sum(bool(item["quality_checks"].get(key)) for item in records), total
            )
            for key in QUALITY_KEYS
        },
        "observed_actions": dict(
            sorted(Counter(str(item["observed_action"] or "<none>") for item in records).items())
        ),
    }


def audit(
    corpus_rows: list[GRPOCorpusRow],
    rollout_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_task = {row.task.task_id: row for row in corpus_rows}
    if len(by_task) != len(corpus_rows):
        raise ValueError("corpus task ids must be unique")
    records: list[dict[str, Any]] = []
    adversarial_results: dict[str, list[bool]] = defaultdict(list)
    for rollout in rollout_rows:
        task_id = str(rollout.get("task_id") or "")
        row = by_task.get(task_id)
        if row is None:
            raise ValueError(f"rollout task is absent from corpus: {task_id}")
        contract = _contract(row)
        target = str(contract.get("target_action") or "")
        arguments = _reason_arguments(rollout)
        checks = _quality(contract, arguments.get("reason"))
        reason_pass = all(checks.values())
        action_match = rollout.get("observed_action") == target
        model_contract = bool(rollout.get("model_contract_compliant"))
        no_override = not bool(rollout.get("controller_override_attempt"))
        records.append(
            {
                "task_id": task_id,
                "target_action": target,
                "observed_action": rollout.get("observed_action"),
                "action_match": action_match,
                "model_contract_compliant": model_contract,
                "no_controller_override_attempt": no_override,
                "quality_checks": checks,
                "strict_reason_quality": reason_pass,
                "strict_full_success": (
                    action_match and model_contract and no_override and reason_pass
                ),
            }
        )
        for name, passed in _adversarial_suite(contract).items():
            adversarial_results[name].append(passed)

    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_target[record["target_action"]].append(record)
    attack_total = sum(len(values) for values in adversarial_results.values())
    attack_passes = sum(sum(values) for values in adversarial_results.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "stored rollouts and explicitly supplied corpus only",
        "human_review_claimed": False,
        "summary": _summary(records),
        "targets": {
            target: _summary(items) for target, items in sorted(by_target.items())
        },
        "adversarial": {
            "cases": attack_total,
            "false_passes": attack_passes,
            "false_pass_rate": _rate(attack_passes, attack_total),
            "by_attack": {
                name: {
                    "cases": len(values),
                    "false_passes": sum(values),
                    "false_pass_rate": _rate(sum(values), len(values)),
                }
                for name, values in sorted(adversarial_results.items())
            },
            "passed": attack_passes == 0,
        },
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--rollouts-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(load_grpo_corpus(args.corpus_file), _read_jsonl(args.rollouts_file))
    report["corpus_sha256"] = _sha256(args.corpus_file)
    report["rollouts_sha256"] = _sha256(args.rollouts_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0 if report["adversarial"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
