"""Validate that an Agent Policy candidate is backed by reproducible evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(manifest_path: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path.cwd()).resolve()
    manifest = _load_json(manifest_path)
    errors: list[str] = []

    if manifest.get("status") != "offline_qualified_online_unreleased":
        errors.append("candidate must remain offline_qualified_online_unreleased")
    if manifest.get("credit_assignment_claim") != "trajectory-level only":
        errors.append("credit assignment claim must remain trajectory-level only")

    checkpoint = manifest.get("checkpoint")
    evidence = manifest.get("promotion_evidence", {})
    for gate_name in ("small_gate", "holdout_gate"):
        relative_path = evidence.get(gate_name)
        if not relative_path:
            errors.append(f"missing promotion_evidence.{gate_name}")
            continue
        gate_path = root / relative_path
        if not gate_path.is_file():
            errors.append(f"missing evidence file: {relative_path}")
            continue
        gate = _load_json(gate_path)
        if gate.get("promoted") is not True:
            errors.append(f"gate did not promote candidate: {relative_path}")
        if gate.get("after_checkpoint") != checkpoint:
            errors.append(f"gate checkpoint mismatch: {relative_path}")
        if gate.get("gate_errors"):
            errors.append(f"gate contains errors: {relative_path}")
        protocol = gate.get("paired_contract", {}).get("seed_protocol")
        if protocol != manifest["offline_evaluation"]["paired_seed_protocol"]:
            errors.append(f"seed protocol mismatch: {relative_path}")

    report_relative_path = manifest.get("offline_evaluation", {}).get("report")
    report_path = root / report_relative_path if report_relative_path else None
    if report_path is None or not report_path.is_file():
        errors.append(f"missing comparison report: {report_relative_path}")
    else:
        report = _load_json(report_path)
        arm = next(
            (item for item in report.get("arms", []) if item.get("checkpoint") == checkpoint),
            None,
        )
        if arm is None:
            errors.append("comparison report does not contain candidate checkpoint")
        else:
            expected = manifest["offline_evaluation"]
            for key in ("tasks", "success_rate", "mean_reward"):
                if arm.get(key) != expected.get(key):
                    errors.append(f"comparison metric mismatch: {key}")
            if report.get("contract", {}).get("seed_protocol") != expected.get(
                "paired_seed_protocol"
            ):
                errors.append("comparison seed protocol mismatch")

    smoke = manifest.get("app_branch_smoke")
    if smoke:
        smoke_relative_path = smoke.get("report")
        smoke_path = root / smoke_relative_path if smoke_relative_path else None
        if smoke_path is None or not smoke_path.is_file():
            errors.append(f"missing App branch smoke report: {smoke_relative_path}")
        else:
            smoke_report = _load_json(smoke_path)
            if smoke_report.get("checkpoint") != checkpoint:
                errors.append("App branch smoke checkpoint mismatch")
            for key in ("hard_pass", "itinerary_days"):
                if smoke_report.get(key) != smoke.get(key):
                    errors.append(f"App branch smoke mismatch: {key}")
            if smoke_report.get("agent_status") != smoke.get("status"):
                errors.append("App branch smoke mismatch: status")

    result = {
        "schema_version": "agent-policy-candidate-validation.v1",
        "valid": not errors,
        "candidate": checkpoint,
        "status": manifest.get("status"),
        "errors": errors,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = validate(args.manifest, repo_root=args.repo_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
