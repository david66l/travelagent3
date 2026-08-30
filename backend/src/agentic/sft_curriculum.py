"""Dataset-level audits for policy SFT curriculum design."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from agentic.policy import minimize_controller_hydrated_payload


SPLITS = ("train", "validation", "test")
CONTROLLER_ARGUMENT_LURES = {
    "$oid",
    "amap_minutes",
    "candidate_poi_ids",
    "city",
    "constraints",
    "dist_matrix",
    "facts",
    "itineraries",
    "itinerary",
    "matrices",
    "maxItems",
    "max_results",
    "poi_ids",
    "pois",
    "tc_matrix",
    "trusted_city",
}
DECISION_ACTIONS = {
    "abort",
    "ask_user",
    "propose_tradeoff",
    "search_pois",
}


def load_sft_dataset(dataset_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {split: _load_jsonl(dataset_dir / f"{split}.jsonl") for split in SPLITS}


def audit_sft_dataset(dataset_dir: Path) -> dict[str, Any]:
    rows_by_split = load_sft_dataset(dataset_dir)
    rows = [row for split in SPLITS for row in rows_by_split[split]]
    action_counts: Counter[str] = Counter()
    split_action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    quality_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    split_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_singleton_rows: Counter[str] = Counter()
    split_singleton_empty_rows: Counter[str] = Counter()
    target_errors: Counter[str] = Counter()
    lure_rows: Counter[str] = Counter()
    lure_keys: dict[str, Counter[str]] = defaultdict(Counter)
    singleton_rows = 0
    singleton_empty_rows = 0
    decision_rows = 0
    context_lengths: list[int] = []
    exact_signatures: Counter[str] = Counter()
    template_signatures: Counter[str] = Counter()
    split_scenarios: dict[str, set[str]] = defaultdict(set)

    all_schema_properties: set[str] = set()
    for row in rows:
        for tool in row.get("tools") or []:
            function = tool.get("function") or {}
            parameters = function.get("parameters") or {}
            all_schema_properties.update((parameters.get("properties") or {}).keys())
    argument_like_keys = all_schema_properties | CONTROLLER_ARGUMENT_LURES

    for split, split_rows in rows_by_split.items():
        for row in split_rows:
            action, arguments = target_call(row)
            context = policy_context(row)
            schema = target_schema(row, action)
            properties = set((schema.get("properties") or {}).keys())
            required = set(schema.get("required") or [])
            action_counts[action] += 1
            split_action_counts[split][action] += 1
            quality_counts[str(row.get("quality_label") or "unknown")] += 1
            source_counts[str(row.get("source") or "unknown")] += 1
            family = action_family(action, context)
            family_counts[family] += 1
            split_family_counts[split][family] += 1
            split_scenarios[split].add(str(row.get("scenario_id") or ""))
            decision_rows += int(action in DECISION_ACTIONS or bool(context.get("failure_summary")))
            context_lengths.append(len(json.dumps(context, ensure_ascii=False)))

            allowed = list(context.get("allowed_actions") or [])
            if len(allowed) == 1:
                singleton_rows += 1
                split_singleton_rows[split] += 1
                if not properties and not required:
                    singleton_empty_rows += 1
                    split_singleton_empty_rows[split] += 1

            unknown = set(arguments) - properties
            missing = required - set(arguments)
            for key in sorted(unknown):
                target_errors[f"unknown_argument:{action}:{key}"] += 1
            for key in sorted(missing):
                target_errors[f"missing_argument:{action}:{key}"] += 1

            visible_keys = collect_keys(context)
            lures = (visible_keys & argument_like_keys) - properties
            if lures:
                lure_rows[action] += 1
                lure_keys[action].update(lures)

            exact_signatures[
                _hash({"messages": row.get("messages"), "tools": row.get("tools")})
            ] += 1
            template_signatures[
                _hash(normalize_template({"context": context, "action": action}))
            ] += 1

    total = len(rows)
    overlaps = {
        f"{left}:{right}": len(split_scenarios[left] & split_scenarios[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    repeated_exact = sum(count - 1 for count in exact_signatures.values() if count > 1)
    largest_template_cluster = max(template_signatures.values(), default=0)
    return {
        "schema_version": "agent-policy-sft-curriculum-audit.v1",
        "dataset_dir": str(dataset_dir),
        "rows": total,
        "split_rows": {split: len(rows_by_split[split]) for split in SPLITS},
        "action_counts": dict(action_counts.most_common()),
        "split_action_counts": {
            split: dict(counts.most_common()) for split, counts in split_action_counts.items()
        },
        "action_family_counts": dict(family_counts.most_common()),
        "quality_counts": dict(quality_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
        "curriculum": {
            "decision_rows": decision_rows,
            "decision_row_rate": _rate(decision_rows, total),
            "singleton_rows": singleton_rows,
            "singleton_row_rate": _rate(singleton_rows, total),
            "singleton_empty_argument_rows": singleton_empty_rows,
            "singleton_empty_argument_rate": _rate(singleton_empty_rows, total),
        },
        "split_curriculum": {
            split: {
                "action_family_counts": dict(split_family_counts[split].most_common()),
                "decision_rows": sum(
                    split_family_counts[split][family]
                    for family in ("search", "recovery", "clarification", "tradeoff")
                ),
                "decision_row_rate": _rate(
                    sum(
                        split_family_counts[split][family]
                        for family in ("search", "recovery", "clarification", "tradeoff")
                    ),
                    len(rows_by_split[split]),
                ),
                "singleton_rows": split_singleton_rows[split],
                "singleton_empty_argument_rows": split_singleton_empty_rows[split],
            }
            for split in SPLITS
        },
        "protocol": {
            "target_argument_errors": dict(target_errors.most_common()),
            "rows_with_controller_argument_lures_by_action": dict(lure_rows.most_common()),
            "controller_argument_lure_keys_by_action": {
                action: dict(keys.most_common()) for action, keys in sorted(lure_keys.items())
            },
        },
        "diversity": {
            "exact_duplicate_rows": repeated_exact,
            "normalized_template_count": len(template_signatures),
            "largest_normalized_template_cluster": largest_template_cluster,
            "largest_normalized_template_cluster_rate": _rate(largest_template_cluster, total),
            "scenario_overlap": overlaps,
        },
        "context_characters": _distribution(context_lengths),
    }


def target_call(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    messages = row.get("messages") or []
    if not messages:
        raise ValueError("SFT row has no messages")
    calls = messages[-1].get("tool_calls") or []
    if len(calls) != 1:
        raise ValueError("SFT row must end with exactly one tool call")
    function = calls[0].get("function") or {}
    return str(function.get("name") or ""), dict(function.get("arguments") or {})


def policy_context(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    user_messages = [item for item in messages[:-1] if item.get("role") == "user"]
    if not user_messages:
        raise ValueError("SFT row has no policy context")
    content = user_messages[-1].get("content")
    if not isinstance(content, str):
        raise ValueError("SFT policy context is not JSON text")
    return json.loads(content)


def target_schema(row: dict[str, Any], action: str) -> dict[str, Any]:
    for tool in row.get("tools") or []:
        function = tool.get("function") or {}
        if function.get("name") == action:
            return dict(function.get("parameters") or {})
    raise ValueError(f"target action {action!r} has no supplied schema")


def action_family(action: str, context: dict[str, Any]) -> str:
    if context.get("failure_summary"):
        return "recovery"
    if action == "search_pois":
        return "search"
    if action == "ask_user":
        return "clarification"
    if action in {"abort", "propose_tradeoff"}:
        return "tradeoff"
    return "controller_transition"


def minimize_row_context(row: dict[str, Any]) -> dict[str, Any]:
    """Apply the same deterministic-turn projection used by online policy inference."""
    copied = json.loads(json.dumps(row, ensure_ascii=False))
    for message in copied.get("messages") or []:
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        context = json.loads(message["content"])
        current = context.get("current_subtask") or {}
        for controller_field in {
            "artifact_refs",
            "depends_on",
            "invalidates_on",
            "required",
            "required_facts",
            "success_criteria",
            "updated_at",
            "verifier_evidence_refs",
        }:
            current.pop(controller_field, None)
        context["current_subtask"] = current
        minimized = minimize_controller_hydrated_payload(context)
        message["content"] = json.dumps(
            minimized,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return copied


def collect_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in collect_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in collect_keys(item)}
    if isinstance(value, str):
        # Controller fact names also occur as values, for example
        # ``{"key": "candidate_poi_ids"}`` and ``required_facts`` entries.
        return {value}
    return set()


def normalize_template(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_template(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [normalize_template(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return "<NUMBER>"
    if isinstance(value, str):
        normalized = re.sub(r"\d+", "<N>", value)
        return normalized if len(normalized) < 80 else "<LONG_TEXT>"
    return f"<{type(value).__name__}>"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rate(value: int, total: int) -> float:
    return round(value / total, 6) if total else 0.0


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "min": ordered[0],
        "median": median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }
