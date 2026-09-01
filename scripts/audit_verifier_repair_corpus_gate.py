"""Run the full train/validation Arrow, route, surface, fingerprint, and reset gate."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import (  # noqa: E402
    VERIFIER_REPAIR_ACTIONS_BY_ROUTE,
    load_grpo_corpus,
)
from agentic.trl_environment import build_trl_environment_factories  # noqa: E402
from scripts.audit_model_curriculum import (  # noqa: E402
    transport_trl_environment_rows,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_rows = {
        split: load_grpo_corpus(args.corpus_dir / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    source_rows = [*split_rows["train"], *split_rows["validation"]]
    transported, transport_evidence = transport_trl_environment_rows(source_rows)
    factories = build_trl_environment_factories("react")
    if "decision_verifier_repair" in factories:
        raise RuntimeError("legacy broad verifier-repair route is still registered")

    surface_evidence = {}
    for route, expected_actions in VERIFIER_REPAIR_ACTIONS_BY_ROUTE.items():
        environment = factories[route](audit_enabled=False)
        public_actions = {
            name
            for name, _ in inspect.getmembers(environment, predicate=inspect.ismethod)
            if name not in {"reset", "get_reward"} and not name.startswith("_")
        }
        surface_evidence[route] = {
            "expected_actions": list(expected_actions),
            "public_actions": sorted(public_actions),
            "exact": public_actions == set(expected_actions),
        }
    if not all(item["exact"] for item in surface_evidence.values()):
        raise RuntimeError("a route public tool surface differs from its action contract")

    route_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    reset_count = 0
    for source_row, transported_row in zip(source_rows, transported, strict=True):
        route = transported_row["environment"]
        expected_actions = VERIFIER_REPAIR_ACTIONS_BY_ROUTE.get(route)
        if expected_actions is None:
            raise RuntimeError(f"unexpected verifier-repair route: {route}")
        decision_state = source_row.snapshot.hidden_test_facts["grpo_decision_state"]
        target = str(decision_state["target_action"])
        if tuple(decision_state["review_allowed_actions"]) != expected_actions:
            raise RuntimeError(
                f"metadata action contract differs from route for {source_row.task.task_id}"
            )
        environment = factories[route](audit_enabled=False)
        try:
            if environment.reset(**transported_row) != "":
                raise RuntimeError("authoritative decision replay returned a duplicate prompt")
            rendered_state = json.loads(transported_row["prompt"][-1]["content"])
            replayed_actions = tuple(
                rendered_state["policy_state"].get("allowed_actions") or []
            )
            if replayed_actions != expected_actions:
                raise RuntimeError(
                    f"replayed action contract differs for {source_row.task.task_id}"
                )
            reset_count += 1
            route_counts[route] += 1
            target_counts[target] += 1
        finally:
            environment.get_reward()

    manifest_path = args.corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "manifest_preflight_ready": manifest["preflight"]["ready"] is True,
        "legacy_broad_route_absent": "decision_verifier_repair" not in factories,
        "all_route_surfaces_exact": all(
            item["exact"] for item in surface_evidence.values()
        ),
        "arrow_roundtrip_exact": transport_evidence["whole_row_equality"] is True,
        "authority_decode_exact": transport_evidence["decoded_authority_equality"]
        is True,
        "fingerprints_exact": transport_evidence["fingerprint_equality"] is True,
        "prompts_exact": transport_evidence["authoritative_prompt_equality"] is True,
        "all_rows_reset": reset_count == len(source_rows),
    }
    report = {
        "schema_version": "verifier-repair-full-corpus-gate.v1",
        "scope": "train and validation only; frozen test rows were not loaded",
        "passed": all(checks.values()),
        "checks": checks,
        "corpus": {
            "path": str(args.corpus_dir),
            "manifest_sha256": _sha256(manifest_path),
            "corpus_content_sha256": manifest["corpus_content_sha256"],
            "split_sha256": manifest["split_sha256"],
            "rows": {split: len(rows) for split, rows in split_rows.items()},
        },
        "transport": transport_evidence,
        "reset_rows": reset_count,
        "route_counts": dict(sorted(route_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "surfaces": surface_evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
