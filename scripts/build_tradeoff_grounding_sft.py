"""Build a train-only, source-disjoint grounding SFT set for verifier tradeoffs.

Only the explicitly supplied GRPO *train file* is opened.  Official validation
and test files are intentionally outside this command's input surface.  The
source states are deterministically divided into optimization, internal-dev,
and train-shadow groups before examples are materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402
from agentic.policy_actions import validate_policy_arguments  # noqa: E402
from agentic.reason_quality import verifier_reason_quality_checks  # noqa: E402
from agentic.sft_dataset import DatasetManifest, SFTExample  # noqa: E402
from scripts.build_verifier_repair_sft_warmstart import (  # noqa: E402
    _decision_example,
)


SCHEMA_VERSION = "react-verifier-tradeoff-grounding-sft.v3"
EXPECTED_DECISION_SCHEMA = "react-verifier-repair-decision.v5"
MIN_REASON_UNIQUE_RATE = 0.40
MAX_REASON_REPEAT_RATE = 0.03
_PRIVATE_MARKERS = (
    "propose_tradeoff",
    "retry_solve",
    "policy_state",
    "hidden_test_facts",
    "snapshot_version",
    "verifier-repair-",
    "native-grpo-",
    "artifact:",
    "fact:",
)
_VOLATILE_ID = re.compile(r"[0-9a-f]{8,}", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _contract(row: GRPOCorpusRow) -> dict[str, Any]:
    value = row.snapshot.hidden_test_facts.get("grpo_decision_state")
    if not isinstance(value, dict):
        raise ValueError(f"missing decision contract: {row.task.task_id}")
    if value.get("schema_version") != EXPECTED_DECISION_SCHEMA:
        raise ValueError(
            f"unsupported decision schema for {row.task.task_id}: "
            f"{value.get('schema_version')!r}"
        )
    return value


def _visible_state(row: GRPOCorpusRow) -> dict[str, Any]:
    messages = list(_contract(row).get("prompt_messages") or [])
    if not messages:
        raise ValueError(f"empty decision prompt: {row.task.task_id}")
    payload = json.loads(str(messages[-1].get("content") or "{}"))
    state = payload.get("policy_state")
    if not isinstance(state, dict):
        raise ValueError(f"missing visible policy state: {row.task.task_id}")
    return state


def _violation(row: GRPOCorpusRow) -> tuple[str, str]:
    reports = [
        item
        for item in _visible_state(row).get("relevant_artifacts") or []
        if isinstance(item, dict) and item.get("artifact_type") == "validation_report"
    ]
    violations = (reports[-1].get("violations") or []) if reports else []
    if not violations:
        raise ValueError(f"missing visible verifier violation: {row.task.task_id}")
    code = str(violations[0].get("code") or "").strip()
    message = str(violations[0].get("message") or "").strip()
    if not code or not message:
        raise ValueError(f"incomplete visible verifier violation: {row.task.task_id}")
    return code, message


def _target(row: GRPOCorpusRow) -> str:
    return str(_contract(row).get("target_action") or "")


def _source_id(row: GRPOCorpusRow) -> str:
    value = str(_contract(row).get("source_task_id") or "").strip()
    if not value:
        raise ValueError(f"missing source task id: {row.task.task_id}")
    return value


def _stable_source_ids(rows: list[GRPOCorpusRow]) -> list[str]:
    return sorted(
        {_source_id(row) for row in rows},
        key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value),
    )


def _partition_source_ids(rows: list[GRPOCorpusRow]) -> dict[str, list[str]]:
    source_ids = _stable_source_ids(rows)
    if len(source_ids) < 6:
        raise ValueError("at least six source states are required for three-way splitting")
    optimization_count = len(source_ids) * 2 // 3
    remaining = len(source_ids) - optimization_count
    dev_count = remaining // 2
    return {
        "optimization": source_ids[:optimization_count],
        "internal_dev": source_ids[optimization_count : optimization_count + dev_count],
        "train_shadow": source_ids[optimization_count + dev_count :],
    }


def _rows_by_source(rows: list[GRPOCorpusRow]) -> dict[str, list[GRPOCorpusRow]]:
    grouped: dict[str, list[GRPOCorpusRow]] = defaultdict(list)
    for row in rows:
        grouped[_source_id(row)].append(row)
    return {
        source_id: sorted(items, key=lambda item: item.task.task_id)
        for source_id, items in grouped.items()
    }


def _tradeoffs(rows: list[GRPOCorpusRow]) -> list[GRPOCorpusRow]:
    return [row for row in rows if _target(row) == "propose_tradeoff"]


def _matched_retention_row(
    rows: list[GRPOCorpusRow],
    target: str,
) -> GRPOCorpusRow:
    tradeoff_codes = {_violation(row)[0] for row in _tradeoffs(rows)}
    candidates = [
        row
        for row in rows
        if _target(row) == target and _violation(row)[0] in tradeoff_codes
    ]
    if not candidates:
        raise ValueError(
            f"source {_source_id(rows[0])} has no {target} row matched to a tradeoff violation"
        )
    return sorted(candidates, key=lambda item: (_violation(item)[0], item.task.task_id))[0]


def _strict_counterfactual_triplets(rows: list[GRPOCorpusRow]) -> list[list[GRPOCorpusRow]]:
    """Return fail-closed one-per-action strict counterfactual groups.

    A balanced action curriculum must preserve the original decision boundary:
    sampling a tradeoff independently of its retry/abort counterparts would
    reintroduce class priors without retaining the controlled contrast.
    """

    grouped: dict[str, list[GRPOCorpusRow]] = defaultdict(list)
    for row in rows:
        group_id = str(_contract(row).get("counterfactual_group_id") or "").strip()
        if group_id:
            grouped[group_id].append(row)
    triplets: list[list[GRPOCorpusRow]] = []
    expected = {"abort", "propose_tradeoff", "retry_solve"}
    for group_id, members in grouped.items():
        by_action = {_target(row): row for row in members}
        if len(members) != 3 or set(by_action) != expected:
            continue
        triplets.append(
            [by_action[action] for action in sorted(expected)]
        )
    return sorted(
        triplets,
        key=lambda members: _digest(
            [_source_id(members[0]), _contract(members[0])["counterfactual_group_id"]]
        ),
    )


def _balanced_strict_optimization_rows(
    by_source: dict[str, list[GRPOCorpusRow]],
    optimization_sources: list[str],
) -> list[GRPOCorpusRow]:
    """Select 300 source-balanced rows (100 examples per decision action).

    The previous tradeoff-grounding selection retained every tradeoff row but
    only a small number of abort/retry rows.  That is suitable for auditing
    tradeoff wording but creates a 240/30/30 optimization prior.  This targeted
    curriculum keeps two complete strict triples from every source, then adds
    one or two members from a third triple.  The deterministic extra schedule
    gives every action exactly 20 extra rows, while each source contributes
    seven or eight rows and every source/action cell contains two or three.
    This avoids the previous 6-versus-9 source weighting artifact.
    """

    if len(optimization_sources) != 40:
        raise ValueError("balanced strict selection requires exactly 40 optimization sources")
    selected: list[GRPOCorpusRow] = []
    # First 20 sources receive two extras: 7 RT, 7 RA, 6 TA.  The remaining
    # 20 receive one extra: 6 R, 7 T, 7 A.  Global extras are therefore
    # exactly R=20, T=20, A=20.
    paired_extras = (
        [("retry_solve", "propose_tradeoff")] * 7
        + [("retry_solve", "abort")] * 7
        + [("propose_tradeoff", "abort")] * 6
    )
    single_extras = (
        [("retry_solve",)] * 6
        + [("propose_tradeoff",)] * 7
        + [("abort",)] * 7
    )
    extra_schedule = [*paired_extras, *single_extras]
    source_contributions: Counter[str] = Counter()
    source_action_counts: dict[str, Counter[str]] = {}
    for source_index, source_id in enumerate(optimization_sources):
        triplets = _strict_counterfactual_triplets(by_source[source_id])
        if len(triplets) < 3:
            raise ValueError(
                f"source {source_id} has fewer than 3 strict counterfactual triples"
            )
        source_rows = [row for triplet in triplets[:2] for row in triplet]
        third_by_action = {_target(row): row for row in triplets[2]}
        source_rows.extend(
            third_by_action[action] for action in extra_schedule[source_index]
        )
        selected.extend(source_rows)
        source_contributions[source_id] = len(source_rows)
        source_action_counts[source_id] = Counter(_target(row) for row in source_rows)
    counts = Counter(_target(row) for row in selected)
    if len(selected) != 300 or counts != Counter(
        {"abort": 100, "propose_tradeoff": 100, "retry_solve": 100}
    ):
        raise ValueError("balanced strict selection did not produce a 100/100/100 action mix")
    if Counter(source_contributions.values()) != Counter({7: 20, 8: 20}):
        raise ValueError("balanced strict selection did not produce 20x7 and 20x8 sources")
    if any(
        set(action_counts) != {"abort", "propose_tradeoff", "retry_solve"}
        or any(count not in {2, 3} for count in action_counts.values())
        for action_counts in source_action_counts.values()
    ):
        raise ValueError("balanced strict selection has a source/action cell outside 2..3")
    return sorted(selected, key=lambda item: item.task.task_id)


def _hard_case_optimization_rows(
    by_source: dict[str, list[GRPOCorpusRow]],
    optimization_sources: list[str],
) -> list[GRPOCorpusRow]:
    """Select 120 evidence-rich positives with a retry-first action mix.

    The quota is counted by distinct decision states, not duplicated examples:
    72 retry, 36 tradeoff, and 12 abort anchors.  Every optimization source
    contributes exactly three examples so no source is overweighted.
    """

    if len(optimization_sources) != 40:
        raise ValueError("hard-case selection requires exactly 40 optimization sources")
    quotas = {"retry_solve": 72, "propose_tradeoff": 36, "abort": 12}
    selected: list[GRPOCorpusRow] = []
    source_contributions: Counter[str] = Counter()
    for source_index, source_id in enumerate(optimization_sources):
        per_target = {
            target: sorted(
                (row for row in by_source[source_id] if _target(row) == target),
                key=lambda item: (_violation(item)[0], item.task.task_id),
            )
            for target in quotas
        }
        source_quota = (
            {"retry_solve": 1, "propose_tradeoff": 1, "abort": 1}
            if source_index < 12
            else (
                {"retry_solve": 2, "propose_tradeoff": 1, "abort": 0}
                if source_index < 36
                else {"retry_solve": 3, "propose_tradeoff": 0, "abort": 0}
            )
        )
        for target, count in source_quota.items():
            if len(per_target[target]) < count:
                raise ValueError(
                    f"source {source_id} has fewer than {count} distinct {target} states"
                )
            selected.extend(per_target[target][:count])
            source_contributions[source_id] += count
    counts = Counter(_target(row) for row in selected)
    if len(selected) != 120 or counts != Counter(quotas):
        raise ValueError("hard-case selection did not produce the declared action mix")
    if len({row.task.task_id for row in selected}) != len(selected):
        raise ValueError("hard-case selection contains duplicate decision states")
    if set(source_contributions.values()) != {3} or len(source_contributions) != 40:
        raise ValueError("hard-case selection does not weight every source equally")
    return sorted(selected, key=lambda item: item.task.task_id)


def _select_rows(
    rows: list[GRPOCorpusRow],
    *,
    balanced_actions: bool = False,
    hard_case_actions: bool = False,
) -> tuple[dict[str, list[GRPOCorpusRow]], dict[str, list[str]]]:
    by_source = _rows_by_source(rows)
    partitions = _partition_source_ids(rows)
    selected: dict[str, list[GRPOCorpusRow]] = {}

    optimization_sources = partitions["optimization"]
    if balanced_actions and hard_case_actions:
        raise ValueError("balanced and hard-case action selection are mutually exclusive")
    if hard_case_actions:
        selected["optimization"] = _hard_case_optimization_rows(
            by_source,
            optimization_sources,
        )
    elif balanced_actions:
        selected["optimization"] = _balanced_strict_optimization_rows(
            by_source,
            optimization_sources,
        )
    else:
        optimization = [
            row
            for source_id in optimization_sources
            for row in _tradeoffs(by_source[source_id])
        ]
        retention_count = min(30, len(optimization_sources))
        retry_sources = optimization_sources[:retention_count]
        abort_sources = list(reversed(optimization_sources))[:retention_count]
        optimization.extend(
            _matched_retention_row(by_source[source_id], "retry_solve")
            for source_id in retry_sources
        )
        optimization.extend(
            _matched_retention_row(by_source[source_id], "abort")
            for source_id in abort_sources
        )
        selected["optimization"] = sorted(optimization, key=lambda item: item.task.task_id)

    for partition in ("internal_dev", "train_shadow"):
        held_out: list[GRPOCorpusRow] = []
        for source_id in partitions[partition]:
            source_rows = by_source[source_id]
            held_out.extend(_tradeoffs(source_rows))
            held_out.append(_matched_retention_row(source_rows, "retry_solve"))
            held_out.append(_matched_retention_row(source_rows, "abort"))
        selected[partition] = sorted(held_out, key=lambda item: item.task.task_id)
    return selected, partitions


def _assistant_call(example: SFTExample) -> tuple[str, dict[str, Any]]:
    calls = example.messages[-1].tool_calls
    if len(calls) != 1:
        raise ValueError(f"example must have exactly one tool call: {example.example_id}")
    function = calls[0].function
    return function.name, dict(function.arguments)


def _training_example(row: GRPOCorpusRow, split: str) -> SFTExample:
    return _decision_example(row, split).model_copy(
        update={
            "policy_name": "verified-tradeoff-grounding-teacher",
            "policy_version": SCHEMA_VERSION,
        }
    )


def _normalize_text(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _normalized_payload(example: SFTExample) -> str:
    payload = json.dumps(
        {
            "messages": [item.model_dump(mode="json") for item in example.messages],
            "tools": example.tools,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload = _VOLATILE_ID.sub("[ID]", payload)
    payload = _NUMBER.sub("[N]", payload)
    return "".join(payload.casefold().split())


def _quality_checks(
    examples: dict[str, list[SFTExample]],
    rows_by_task: dict[str, GRPOCorpusRow],
    partitions: dict[str, list[str]],
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    payload_hashes: set[str] = set()
    normalized_hashes: set[str] = set()
    reasons: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    total = 0
    schema_valid = 0
    grounded = 0
    reason_quality_passed = 0
    supported = 0
    covered = 0
    private_leaks = 0
    rationale_first = 0

    partition_sets = {key: set(value) for key, value in partitions.items()}
    overlap = bool(
        partition_sets["optimization"] & partition_sets["internal_dev"]
        or partition_sets["optimization"] & partition_sets["train_shadow"]
        or partition_sets["internal_dev"] & partition_sets["train_shadow"]
    )
    if overlap:
        errors.append("SOURCE_STATE_OVERLAP")

    for split_examples in examples.values():
        for example in split_examples:
            total += 1
            action, arguments = _assistant_call(example)
            action_counts[action] += 1
            row = rows_by_task[example.scenario_id]
            contract = _contract(row)
            expected_action = _target(row)
            if action != expected_action:
                errors.append(f"ACTION_MISMATCH:{example.example_id}")
            try:
                validate_policy_arguments(action, arguments)
            except (TypeError, ValueError):
                errors.append(f"SCHEMA_INVALID:{example.example_id}")
            else:
                schema_valid += 1

            reason = str(arguments.get("reason") or "")
            reasons[reason] += 1
            phrases = [str(item) for item in contract.get("grounding_phrases") or []]
            _, evidence = _violation(row)
            if not _normalize_text(reason).startswith(_normalize_text(evidence)):
                rationale_first += 1
            else:
                errors.append(f"EVIDENCE_PREFIX_SHORTCUT:{example.example_id}")
            reason_checks = verifier_reason_quality_checks(
                reason=reason,
                target_action=action,
                grounding_phrases=phrases,
                evidence=evidence,
            )
            if reason_checks["grounding_match"]:
                grounded += 1
            else:
                errors.append(f"UNGROUNDED_REASON:{example.example_id}")
            if all(reason_checks.values()):
                reason_quality_passed += 1
            else:
                failed = ",".join(
                    sorted(name for name, passed in reason_checks.items() if not passed)
                )
                errors.append(f"REASON_QUALITY_INVALID:{example.example_id}:{failed}")

            if action == "propose_tradeoff":
                expected = [str(item) for item in contract.get("supervised_options") or []]
                if "options" not in arguments:
                    supported += 1
                else:
                    errors.append(f"CONTROLLER_FIELD_IN_COMPLETION:{example.example_id}")
                if expected:
                    covered += 1
                else:
                    errors.append(f"CONTROLLER_AUTHORITY_MISSING:{example.example_id}")

            code, _ = _violation(row)
            user_visible = json.dumps(arguments, ensure_ascii=False).casefold()
            forbidden = [*_PRIVATE_MARKERS, code.casefold()]
            if any(marker.casefold() in user_visible for marker in forbidden):
                private_leaks += 1
                errors.append(f"PRIVATE_LABEL_LEAK:{example.example_id}")

            payload = {
                "messages": [item.model_dump(mode="json") for item in example.messages],
                "tools": example.tools,
            }
            payload_hashes.add(_digest(payload))
            normalized_hashes.add(_digest(_normalized_payload(example)))

    exact_duplicates = total - len(payload_hashes)
    near_duplicates = total - len(normalized_hashes)
    near_duplicate_rate = near_duplicates / total if total else 0.0
    if exact_duplicates:
        errors.append(f"EXACT_MODEL_VISIBLE_DUPLICATES:{exact_duplicates}")
    if near_duplicate_rate >= 0.05:
        warnings.append(f"NORMALIZED_NEAR_DUPLICATE_RATE:{near_duplicate_rate:.6f}")

    tradeoff_total = action_counts["propose_tradeoff"]
    nonempty_reasons = {reason: count for reason, count in reasons.items() if reason}
    reason_unique_count = len(nonempty_reasons)
    reason_total = sum(nonempty_reasons.values())
    reason_unique_rate = reason_unique_count / reason_total if reason_total else 0.0
    reason_max_repeat_count = max(nonempty_reasons.values(), default=0)
    reason_max_repeat_rate = reason_max_repeat_count / reason_total if reason_total else 1.0
    report = {
        "source_overlap_zero": not overlap,
        "source_validation_or_test_files_read": 0,
        "exactly_one_tool_call_rate": 1.0,
        "schema_valid_rate": schema_valid / total if total else 0.0,
        "reason_grounding_rate": grounded / total if total else 0.0,
        "reason_quality_rate": reason_quality_passed / total if total else 0.0,
        "reason_rationale_first_rate": rationale_first / total if total else 0.0,
        "tradeoff_model_owned_schema_rate": (
            supported / tradeoff_total if tradeoff_total else 0.0
        ),
        "tradeoff_controller_authority_present_rate": (
            covered / tradeoff_total if tradeoff_total else 0.0
        ),
        "private_label_leak_count": private_leaks,
        "exact_model_visible_duplicate_count": exact_duplicates,
        "normalized_near_duplicate_count": near_duplicates,
        "normalized_near_duplicate_rate": near_duplicate_rate,
        "unique_model_visible_payloads": len(payload_hashes),
        "reason_unique_count": reason_unique_count,
        "reason_unique_rate": reason_unique_rate,
        "reason_max_repeat_count": reason_max_repeat_count,
        "reason_max_repeat_rate": reason_max_repeat_rate,
        "reason_diversity_pass": (
            reason_unique_rate >= MIN_REASON_UNIQUE_RATE
            and reason_max_repeat_rate <= MAX_REASON_REPEAT_RATE
        ),
        "action_counts": dict(sorted(action_counts.items())),
    }
    return report, sorted(set(errors)), warnings


def _matched_pair_records(rows: list[GRPOCorpusRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[GRPOCorpusRow]] = defaultdict(list)
    for row in rows:
        contract = _contract(row)
        group_id = str(contract.get("counterfactual_group_id") or "")
        # V9+ groups are strict counterfactual triples.  The legacy fallback
        # keeps historical fixtures readable but is explicitly marked and can
        # never satisfy the promotion audit.
        key = group_id or _violation(row)[0]
        grouped[(_source_id(row), key)].append(row)
    records: list[dict[str, Any]] = []
    for (source_id, group_key), items in sorted(grouped.items()):
        actions = {_target(item) for item in items}
        if "propose_tradeoff" not in actions or len(actions) < 2:
            continue
        selected = sorted(items, key=lambda item: (_target(item), item.task.task_id))
        tradeoff = next(item for item in selected if _target(item) == "propose_tradeoff")
        is_strict = bool(_contract(tradeoff).get("counterfactual_group_id"))
        contrasts = [item for item in selected if _target(item) != "propose_tradeoff"]
        if not is_strict:
            contrasts = contrasts[:1]
        for contrast in contrasts:
            records.append(
                {
                    "source_task_id": source_id,
                    "violation_code": _violation(tradeoff)[0],
                    "counterfactual_group_id": group_key if is_strict else None,
                    "strict_counterfactual": is_strict,
                    "members": [tradeoff, contrast],
                }
            )
    return records


def _rejected_examples(
    tradeoff_rows: list[GRPOCorpusRow],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    defect_types = ("forbidden_options", "ungrounded_reason")
    for index, row in enumerate(sorted(tradeoff_rows, key=lambda item: item.task.task_id)):
        contract = _contract(row)
        _, reason = _violation(row)
        prompt = list(contract.get("prompt_messages") or [])
        output.append(
            {
                "rejected_id": f"rejected:wrong-action:{row.task.task_id}",
                "source_task_id": _source_id(row),
                "prompt_messages": prompt,
                "tools": _decision_example(row, "train").model_dump(mode="json")["tools"],
                "candidate": {
                    "name": "abort",
                    "arguments": {"reason": reason},
                },
                "rejection_codes": ["WRONG_ACTION_ABORT"],
            }
        )
        defect = defect_types[index % len(defect_types)]
        if defect == "forbidden_options":
            arguments = {"reason": reason, "options": ["忽略当前限制并继续原计划"]}
            codes = ["CONTROLLER_FIELD_IN_COMPLETION"]
        else:
            arguments = {"reason": "行程需要调整，请选择。"}
            codes = ["REASON_NOT_GROUNDED"]
        output.append(
            {
                "rejected_id": f"rejected:{defect}:{row.task.task_id}",
                "source_task_id": _source_id(row),
                "prompt_messages": prompt,
                "tools": _decision_example(row, "train").model_dump(mode="json")["tools"],
                "candidate": {"name": "propose_tradeoff", "arguments": arguments},
                "rejection_codes": codes,
            }
        )
    return output


def _candidate_message(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _blind_review(
    tradeoff_rows: list[GRPOCorpusRow],
    all_optimization_rows: list[GRPOCorpusRow],
    rejected_examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_pairs = _matched_pair_records(all_optimization_rows)
    pairs: list[dict[str, Any]] = []
    used_task_ids: set[str] = set()
    strict_pairs = [item for item in all_pairs if item["strict_counterfactual"]]
    desired_contrasts = ("abort", "retry_solve") if strict_pairs else (None,)
    for contrast_target in desired_contrasts:
        candidates = sorted(
            (
                item
                for item in (strict_pairs if strict_pairs else all_pairs)
                if contrast_target is None or _target(item["members"][1]) == contrast_target
            ),
            key=lambda item: _digest(
                [item["source_task_id"], item["counterfactual_group_id"], contrast_target]
            ),
        )
        for candidate in candidates:
            member_ids = {member.task.task_id for member in candidate["members"]}
            if member_ids & used_task_ids:
                continue
            pairs.append(candidate)
            used_task_ids.update(member_ids)
            if (
                len(pairs) == 30
                if contrast_target is None
                else sum(_target(item["members"][1]) == contrast_target for item in pairs) == 15
            ):
                break
    if len(pairs) != 30:
        raise ValueError("blind-review strict counterfactual pair selection is incomplete")
    paired_task_ids = {
        member.task.task_id for pair in pairs for member in pair["members"]
    }
    positives = [row for row in tradeoff_rows if row.task.task_id not in paired_task_ids]
    positives = sorted(positives, key=lambda item: _digest(item.task.task_id))[:30]
    if len(positives) != 30:
        raise ValueError("blind-review positives overlap matched-boundary records")

    def opaque_id(kind: str, source: Any) -> str:
        return "r-" + _digest([kind, source])[:20]

    packet: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for row in positives:
        example = _decision_example(row, "train")
        review_id = opaque_id("accepted", row.task.task_id)
        packet.append(
            {
                "review_id": review_id,
                "prompt_messages": [item.model_dump(mode="json") for item in example.messages[:-1]],
                "tools": example.tools,
                "candidate": example.messages[-1].model_dump(mode="json"),
            }
        )
        key.append(
            {
                "review_id": review_id,
                "expected_action": _target(row),
                "source_task_id": _source_id(row),
                "violation_code": _violation(row)[0],
                "review_category": "accepted",
            }
        )
    for pair in pairs:
        pair_id = opaque_id("pair", [pair["source_task_id"], pair["violation_code"]])
        for row in pair["members"]:
            example = _decision_example(row, "train")
            review_id = opaque_id("boundary", row.task.task_id)
            packet.append(
                {
                    "review_id": review_id,
                    "prompt_messages": [
                        item.model_dump(mode="json") for item in example.messages[:-1]
                    ],
                    "tools": example.tools,
                    "candidate": example.messages[-1].model_dump(mode="json"),
                }
            )
            key.append(
                {
                    "review_id": review_id,
                    "expected_action": _target(row),
                "source_task_id": _source_id(row),
                "violation_code": _violation(row)[0],
                "pair_id": pair_id,
                "review_category": "matched-boundary",
                }
            )
    selected_rejections = sorted(
        rejected_examples,
        key=lambda item: _digest(item["rejected_id"]),
    )[:30]
    for rejected in selected_rejections:
        candidate = dict(rejected["candidate"])
        review_id = opaque_id("rejected", rejected["rejected_id"])
        packet.append(
            {
                "review_id": review_id,
                "prompt_messages": list(rejected["prompt_messages"]),
                "tools": list(rejected["tools"]),
                "candidate": _candidate_message(candidate["name"], dict(candidate["arguments"])),
            }
        )
        key.append(
            {
                "review_id": review_id,
                "expected_disposition": "reject",
                "rejection_codes": list(rejected["rejection_codes"]),
                "source_task_id": str(rejected["source_task_id"]),
                "review_category": "rejected-candidate",
            }
        )
    packet = sorted(packet, key=lambda item: _digest(item["review_id"]))
    return packet, sorted(key, key=lambda item: item["review_id"])


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for item in rows
        ),
        encoding="utf-8",
    )


def _optimization_balance_report(
    rows: list[GRPOCorpusRow],
    examples: list[SFTExample],
) -> tuple[dict[str, Any], list[str]]:
    """Audit train-only source/action balance and rationale template diversity."""
    errors: list[str] = []
    source_counts = Counter(_source_id(row) for row in rows)
    source_action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    action_counts: Counter[str] = Counter()
    prefixes: dict[str, Counter[str]] = defaultdict(Counter)
    for row, example in zip(rows, examples, strict=True):
        action, arguments = _assistant_call(example)
        source_action_counts[_source_id(row)][action] += 1
        action_counts[action] += 1
        prefixes[action][str(arguments.get("reason") or "").split("：", 1)[0]] += 1

    source_distribution = Counter(source_counts.values())
    cell_distribution = Counter(
        count for values in source_action_counts.values() for count in values.values()
    )
    prefix_metrics = {
        action: {
            "unique": len(values),
            "max_count": max(values.values(), default=0),
            "max_rate": max(values.values(), default=0) / action_counts[action]
            if action_counts[action]
            else 1.0,
        }
        for action, values in sorted(prefixes.items())
    }
    if action_counts != Counter(
        {"abort": 100, "propose_tradeoff": 100, "retry_solve": 100}
    ):
        errors.append("OPTIMIZATION_ACTION_BALANCE_INVALID")
    if source_distribution != Counter({7: 20, 8: 20}):
        errors.append("OPTIMIZATION_SOURCE_BALANCE_INVALID")
    if set(cell_distribution) - {2, 3}:
        errors.append("OPTIMIZATION_SOURCE_ACTION_CELL_INVALID")
    if any(
        metrics["unique"] < 4 or metrics["max_rate"] > 0.25
        for metrics in prefix_metrics.values()
    ):
        errors.append("OPTIMIZATION_RATIONALE_PREFIX_DIVERSITY_INVALID")
    return {
        "action_counts": dict(sorted(action_counts.items())),
        "source_contribution_distribution": dict(sorted(source_distribution.items())),
        "source_action_cell_distribution": dict(sorted(cell_distribution.items())),
        "rationale_prefix_metrics": prefix_metrics,
        "passed": not errors,
    }, errors


def build(
    source_train: Path,
    output_dir: Path,
    *,
    balanced_actions: bool = False,
    hard_case_actions: bool = False,
) -> dict[str, Any]:
    if source_train.name != "train.jsonl":
        raise ValueError("source input must be the explicit train.jsonl file")
    rows = load_grpo_corpus(source_train)
    if not rows:
        raise ValueError("source train corpus is empty")
    task_ids = [row.task.task_id for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("source train task ids are not unique")
    source_manifest_path = source_train.parent / "manifest.json"
    source_template_profile: str | None = None
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(source_manifest, dict):
            raise ValueError("source corpus manifest must be a JSON object")
        profile = source_manifest.get("train_template_profile")
        source_template_profile = str(profile) if profile is not None else None

    if (
        balanced_actions or hard_case_actions
    ) and source_template_profile != "strict-counterfactual-v1":
        raise ValueError("balanced or hard-case actions require a strict-counterfactual source corpus")
    selected, partitions = _select_rows(
        rows,
        balanced_actions=balanced_actions,
        hard_case_actions=hard_case_actions,
    )
    split_map = {
        "optimization": "train",
        "internal_dev": "validation",
        "train_shadow": "test",
    }
    examples = {
        partition: [
            _training_example(row, split_map[partition])
            for row in selected[partition]
        ]
        for partition in split_map
    }
    rows_by_task = {row.task.task_id: row for row in rows}
    quality, errors, warnings = _quality_checks(examples, rows_by_task, partitions)
    optimization_balance: dict[str, Any] | None = None
    if balanced_actions:
        optimization_balance, balance_errors = _optimization_balance_report(
            selected["optimization"], examples["optimization"]
        )
        errors.extend(balance_errors)

    rows_by_source = _rows_by_source(rows)
    review_pool = (
        [
            row
            for source_id in partitions["optimization"]
            for row in rows_by_source[source_id]
        ]
        if hard_case_actions
        else selected["optimization"]
    )
    tradeoff_train = _tradeoffs(review_pool)
    rejected = _rejected_examples(tradeoff_train)
    blind_packet, blind_key = _blind_review(
        tradeoff_train,
        review_pool,
        rejected,
    )
    if len(rejected) != 2 * len(tradeoff_train):
        errors.append("REJECTED_EXAMPLE_COUNT_MISMATCH")
    if len(blind_packet) != 120 or len(blind_key) != 120:
        errors.append("BLIND_REVIEW_PACKET_INCOMPLETE")
    packet_payload_hashes = {
        _digest(
            {
                "prompt_messages": item["prompt_messages"],
                "tools": item["tools"],
                "candidate": item["candidate"],
            }
        )
        for item in blind_packet
    }
    if len(packet_payload_hashes) != len(blind_packet):
        errors.append("BLIND_REVIEW_PACKET_DUPLICATE_RECORDS")
    if len({item["review_id"] for item in blind_packet}) != len(blind_packet):
        errors.append("BLIND_REVIEW_PACKET_ID_COLLISION")

    all_examples = [item for values in examples.values() for item in values]
    dataset_version = "tradeoff-grounding-sft-" + _digest(
        {
            "source_train_sha256": _sha256(source_train),
            "example_ids": [item.example_id for item in all_examples],
            "schema_version": SCHEMA_VERSION,
        }
    )[:16]
    manifest = DatasetManifest(
        dataset_version=dataset_version,
        created_at=datetime.now(UTC),
        candidate_episodes=len(all_examples),
        accepted_episodes=len(all_examples),
        rejected_episodes=0,
        exported_examples=len(all_examples),
        split_examples={
            "train": len(examples["optimization"]),
            "validation": len(examples["internal_dev"]),
            "test": len(examples["train_shadow"]),
        },
        source_episodes={"synthetic": len(all_examples)},
        quality_episodes=dict(Counter(item.quality_label for item in all_examples)),
        rejection_codes={},
        environment_versions=sorted({item.environment_version for item in all_examples}),
        policy_versions=[SCHEMA_VERSION],
        split_group_overlap=not quality["source_overlap_zero"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "train.jsonl", examples["optimization"])
    _write_jsonl(output_dir / "validation.jsonl", examples["internal_dev"])
    _write_jsonl(output_dir / "test.jsonl", examples["train_shadow"])
    _write_jsonl(output_dir / "internal_dev_grpo.jsonl", selected["internal_dev"])
    _write_jsonl(output_dir / "train_shadow_grpo.jsonl", selected["train_shadow"])
    _write_jsonl(output_dir / "rejected_examples.jsonl", rejected)
    _write_jsonl(output_dir / "blind_review_packet.jsonl", blind_packet)
    _write_jsonl(output_dir / "blind_review_key.jsonl", blind_key)
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    matched_pairs = _matched_pair_records(review_pool)
    strict_counterfactual_pair_count = sum(
        bool(item.get("strict_counterfactual")) for item in matched_pairs
    )
    if (
        source_template_profile == "strict-counterfactual-v1"
        and strict_counterfactual_pair_count < 30
    ):
        errors.append("STRICT_COUNTERFACTUAL_PAIR_COVERAGE_INSUFFICIENT")
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not errors else "rejected",
        "training_authorized": False,
        "training_blocker": "INDEPENDENT_BLIND_REVIEW_PENDING",
        "dataset_version": dataset_version,
        "source_train": source_train.as_posix(),
        "source_train_sha256": _sha256(source_train),
        "source_train_template_profile": source_template_profile,
        "optimization_action_selection": (
            "strict-hard-case-retry72-tradeoff36-abort12-v1"
            if hard_case_actions
            else (
                "strict-balanced-100-per-action-v1"
                if balanced_actions
                else "tradeoff-grounding-retention-v1"
            )
        ),
        "source_files_read": [source_train.as_posix()],
        "official_validation_or_test_used": False,
        "partition_semantics": {
            "train": "optimization",
            "validation": "train-internal-dev",
            "test": "untouched-train-shadow",
        },
        "source_state_counts": {key: len(value) for key, value in partitions.items()},
        "split_counts": manifest.split_examples,
        "behavior_eval_corpora": {
            "internal_dev": "internal_dev_grpo.jsonl",
            "train_shadow": "train_shadow_grpo.jsonl",
        },
        "quality_gate": quality,
        "optimization_balance": optimization_balance,
        "matched_boundary_pair_count": len(matched_pairs),
        "strict_counterfactual_pair_count": strict_counterfactual_pair_count,
        "rejected_example_count": len(rejected),
        "blind_review": {
            "status": "pending",
            "positive_examples": 30,
            "matched_pairs": 30,
            "rejected_candidates": 30,
            "packet_records": len(blind_packet),
            "packet": "blind_review_packet.jsonl",
            "key": "blind_review_key.jsonl",
        },
        "warnings": warnings,
        "errors": sorted(set(errors)),
    }
    (output_dir / "derivation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--balanced-actions",
        action="store_true",
        help=(
            "Select a strict 100/100/100 optimization mix. Intended for a "
            "targeted action-prior repair; requires strict-counterfactual input."
        ),
    )
    parser.add_argument(
        "--hard-case-actions",
        action="store_true",
        help=(
            "Select 120 distinct strict decision states with a 72/36/12 "
            "retry/tradeoff/abort mix for reason-quality repair."
        ),
    )
    args = parser.parse_args()
    report = build(
        args.source_train,
        args.output_dir,
        balanced_actions=args.balanced_actions,
        hard_case_actions=args.hard_case_actions,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
