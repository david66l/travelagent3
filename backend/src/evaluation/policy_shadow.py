"""Paired champion/challenger policy evaluation for Shadow promotion.

The evaluator accepts normalized HTTP benchmark rows as well as future live
policy-decision records.  It deliberately separates quality gates from Canary
eligibility: sealed benchmarks and authorized replay may validate a candidate,
but only live Shadow evidence may promote it.
"""

from __future__ import annotations

from collections import defaultdict
from math import comb, sqrt
from statistics import fmean
from typing import Any, Literal

from pydantic import BaseModel, Field


EvidenceSource = Literal["sealed_benchmark", "authorized_replay", "live_shadow"]


class PolicyShadowGateConfig(BaseModel):
    """Conservative gates for champion/challenger policy promotion."""

    minimum_paired_decisions: int = Field(default=300, ge=1)
    minimum_challenger_success_rate: float = Field(default=0.98, ge=0, le=1)
    noninferiority_margin: float = Field(default=0.01, ge=0, le=1)
    maximum_http_error_rate: float = Field(default=0.01, ge=0, le=1)
    maximum_p95_latency_ratio: float = Field(default=1.25, ge=1)


class PolicyShadowGateCheck(BaseModel):
    code: str
    passed: bool
    actual: float | int | str
    expected: str


class PolicySideSummary(BaseModel):
    decisions: int
    successful_decisions: int
    success_rate: float
    policy_contract_successes: int
    policy_contract_success_rate: float
    http_errors: int
    http_error_rate: float
    mean_latency_ms: float
    p95_latency_ms: float


class PolicyFamilyComparison(BaseModel):
    family: str
    paired_decisions: int
    champion_success_rate: float
    challenger_success_rate: float
    success_rate_delta: float
    champion_only_successes: int
    challenger_only_successes: int
    action_divergence_rate: float


class PolicyShadowReport(BaseModel):
    schema_version: Literal["policy-shadow-comparison.v1"] = "policy-shadow-comparison.v1"
    evidence_source: EvidenceSource
    champion: PolicySideSummary
    challenger: PolicySideSummary
    input_champion_rows: int
    input_challenger_rows: int
    paired_decisions: int
    excluded_label_contract_conflicts: int
    champion_only_successes: int
    challenger_only_successes: int
    both_successes: int
    both_failures: int
    success_rate_delta: float
    success_rate_delta_ci95: tuple[float, float]
    mcnemar_exact_pvalue: float
    action_divergences: int
    action_divergence_rate: float
    p95_latency_ratio: float
    family_comparisons: list[PolicyFamilyComparison]
    quality_gates_passed: bool
    canary_evidence: bool
    release_eligible: bool
    checks: list[PolicyShadowGateCheck]


def compare_policy_shadow_runs(
    champion_rows: list[dict[str, Any]],
    challenger_rows: list[dict[str, Any]],
    *,
    evidence_source: EvidenceSource,
    gate: PolicyShadowGateConfig | None = None,
) -> PolicyShadowReport:
    """Compare exact paired policy decisions and produce promotion evidence."""
    config = gate or PolicyShadowGateConfig()
    champion_by_key = _index_rows(champion_rows, side="champion")
    challenger_by_key = _index_rows(challenger_rows, side="challenger")
    if set(champion_by_key) != set(challenger_by_key):
        missing_challenger = sorted(set(champion_by_key) - set(challenger_by_key))
        missing_champion = sorted(set(challenger_by_key) - set(champion_by_key))
        raise ValueError(
            "policy shadow scenario sets differ; "
            f"missing challenger={missing_challenger[:5]}, "
            f"missing champion={missing_champion[:5]}"
        )

    paired = [(champion_by_key[key], challenger_by_key[key]) for key in sorted(champion_by_key)]
    excluded_conflicts = sum(
        _label_contract_conflict(champion) or _label_contract_conflict(challenger)
        for champion, challenger in paired
    )
    eligible = [
        (champion, challenger)
        for champion, challenger in paired
        if not (_label_contract_conflict(champion) or _label_contract_conflict(challenger))
    ]
    if not eligible:
        raise ValueError("no contract-consistent policy shadow pairs found")
    if evidence_source == "live_shadow":
        _validate_live_shadow_provenance(eligible)

    champion_successes = [_contract_success(champion) for champion, _ in eligible]
    challenger_successes = [_contract_success(challenger) for _, challenger in eligible]
    champion_only = sum(
        champion and not challenger
        for champion, challenger in zip(champion_successes, challenger_successes, strict=True)
    )
    challenger_only = sum(
        challenger and not champion
        for champion, challenger in zip(champion_successes, challenger_successes, strict=True)
    )
    both_successes = sum(
        champion and challenger
        for champion, challenger in zip(champion_successes, challenger_successes, strict=True)
    )
    both_failures = len(eligible) - champion_only - challenger_only - both_successes
    delta = (challenger_only - champion_only) / len(eligible)
    delta_ci = _paired_difference_ci95(
        challenger_only=challenger_only,
        champion_only=champion_only,
        sample_size=len(eligible),
    )
    champion_summary = _side_summary([row for row, _ in eligible])
    challenger_summary = _side_summary([row for _, row in eligible])
    latency_ratio = _safe_ratio(challenger_summary.p95_latency_ms, champion_summary.p95_latency_ms)
    action_divergences = sum(
        _observed_action(champion) != _observed_action(challenger)
        for champion, challenger in eligible
    )
    family_comparisons = _family_comparisons(eligible)
    canary_evidence = evidence_source == "live_shadow"
    checks = [
        PolicyShadowGateCheck(
            code="MINIMUM_PAIRED_DECISIONS",
            passed=len(eligible) >= config.minimum_paired_decisions,
            actual=len(eligible),
            expected=f">= {config.minimum_paired_decisions}",
        ),
        PolicyShadowGateCheck(
            code="CHALLENGER_SUCCESS_RATE",
            passed=(
                challenger_summary.policy_contract_success_rate
                >= config.minimum_challenger_success_rate
            ),
            actual=challenger_summary.policy_contract_success_rate,
            expected=f">= {config.minimum_challenger_success_rate}",
        ),
        PolicyShadowGateCheck(
            code="PAIRED_NONINFERIORITY_CI95",
            passed=delta_ci[0] >= -config.noninferiority_margin,
            actual=delta_ci[0],
            expected=f">= -{config.noninferiority_margin}",
        ),
        PolicyShadowGateCheck(
            code="CHALLENGER_HTTP_ERROR_RATE",
            passed=challenger_summary.http_error_rate <= config.maximum_http_error_rate,
            actual=challenger_summary.http_error_rate,
            expected=f"<= {config.maximum_http_error_rate}",
        ),
        PolicyShadowGateCheck(
            code="P95_LATENCY_RATIO",
            passed=latency_ratio <= config.maximum_p95_latency_ratio,
            actual=latency_ratio,
            expected=f"<= {config.maximum_p95_latency_ratio}x champion",
        ),
        PolicyShadowGateCheck(
            code="LIVE_SHADOW_EVIDENCE",
            passed=canary_evidence,
            actual=evidence_source,
            expected="evidence_source=live_shadow",
        ),
    ]
    quality_gates_passed = all(
        check.passed for check in checks if check.code != "LIVE_SHADOW_EVIDENCE"
    )
    return PolicyShadowReport(
        evidence_source=evidence_source,
        champion=champion_summary,
        challenger=challenger_summary,
        input_champion_rows=len(champion_rows),
        input_challenger_rows=len(challenger_rows),
        paired_decisions=len(eligible),
        excluded_label_contract_conflicts=excluded_conflicts,
        champion_only_successes=champion_only,
        challenger_only_successes=challenger_only,
        both_successes=both_successes,
        both_failures=both_failures,
        success_rate_delta=delta,
        success_rate_delta_ci95=delta_ci,
        mcnemar_exact_pvalue=_mcnemar_exact_pvalue(champion_only, challenger_only),
        action_divergences=action_divergences,
        action_divergence_rate=action_divergences / len(eligible),
        p95_latency_ratio=latency_ratio,
        family_comparisons=family_comparisons,
        quality_gates_passed=quality_gates_passed,
        canary_evidence=canary_evidence,
        release_eligible=quality_gates_passed and canary_evidence,
        checks=checks,
    )


def _index_rows(rows: list[dict[str, Any]], *, side: str) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError(f"{side} row is missing case_id")
        key = (case_id, int(row.get("repetition", 0)))
        if key in result:
            raise ValueError(f"duplicate {side} policy shadow key: {key}")
        result[key] = row
    if not result:
        raise ValueError(f"{side} policy shadow rows are empty")
    return result


def _validate_live_shadow_provenance(
    paired: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    for champion, challenger in paired:
        for side, row in (("champion", champion), ("challenger", challenger)):
            valid = (
                row.get("evaluation_source") == "live_shadow"
                and row.get("release_gate_eligible") is True
                and row.get("outcome_observed") is True
            )
            if not valid:
                raise ValueError(
                    "live Shadow promotion requires provenance and executed outcomes; "
                    f"invalid {side} row for {row.get('case_id')}"
                )


def _contract_success(row: dict[str, Any]) -> bool:
    if "policy_contract_success" in row:
        return bool(row["policy_contract_success"])
    return bool(row.get("success"))


def _label_contract_conflict(row: dict[str, Any]) -> bool:
    return bool(row.get("label_contract_conflict", False))


def _http_error(row: dict[str, Any]) -> bool:
    return bool(row.get("http_error"))


def _latency_ms(row: dict[str, Any]) -> float:
    metrics = row.get("inference_metrics") or {}
    return max(0.0, float(metrics.get("request_latency_ms") or 0.0))


def _observed_action(row: dict[str, Any]) -> str:
    actions = row.get("observed_actions") or []
    return str(actions[0]) if actions else "[NO_ACTION]"


def _policy_family(row: dict[str, Any]) -> str:
    """Prefer runtime route family, otherwise map the frozen expected action."""
    explicit = row.get("route_family") or row.get("policy_route_family")
    if explicit:
        return str(explicit)
    expected_action = str(row.get("expected_action") or "")
    if expected_action == "ask_user":
        return "clarification"
    if expected_action == "search_pois":
        return "search"
    if expected_action in {"propose_tradeoff", "abort"}:
        return "tradeoff"
    return str(row.get("family") or "complex")


def _side_summary(rows: list[dict[str, Any]]) -> PolicySideSummary:
    successes = sum(bool(row.get("success")) for row in rows)
    contract_successes = sum(_contract_success(row) for row in rows)
    http_errors = sum(_http_error(row) for row in rows)
    latencies = [_latency_ms(row) for row in rows]
    return PolicySideSummary(
        decisions=len(rows),
        successful_decisions=successes,
        success_rate=successes / len(rows),
        policy_contract_successes=contract_successes,
        policy_contract_success_rate=contract_successes / len(rows),
        http_errors=http_errors,
        http_error_rate=http_errors / len(rows),
        mean_latency_ms=fmean(latencies),
        p95_latency_ms=_percentile(latencies, 0.95),
    )


def _family_comparisons(
    paired: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[PolicyFamilyComparison]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for champion, challenger in paired:
        champion_family = _policy_family(champion)
        challenger_family = _policy_family(challenger)
        if champion_family != challenger_family:
            raise ValueError(
                f"paired family mismatch for {champion.get('case_id')}: "
                f"{champion_family} != {challenger_family}"
            )
        grouped[champion_family].append((champion, challenger))

    results = []
    for family, rows in sorted(grouped.items()):
        champion_successes = [_contract_success(champion) for champion, _ in rows]
        challenger_successes = [_contract_success(challenger) for _, challenger in rows]
        champion_only = sum(
            champion and not challenger
            for champion, challenger in zip(champion_successes, challenger_successes, strict=True)
        )
        challenger_only = sum(
            challenger and not champion
            for champion, challenger in zip(champion_successes, challenger_successes, strict=True)
        )
        divergences = sum(
            _observed_action(champion) != _observed_action(challenger)
            for champion, challenger in rows
        )
        champion_rate = sum(champion_successes) / len(rows)
        challenger_rate = sum(challenger_successes) / len(rows)
        results.append(
            PolicyFamilyComparison(
                family=family,
                paired_decisions=len(rows),
                champion_success_rate=champion_rate,
                challenger_success_rate=challenger_rate,
                success_rate_delta=challenger_rate - champion_rate,
                champion_only_successes=champion_only,
                challenger_only_successes=challenger_only,
                action_divergence_rate=divergences / len(rows),
            )
        )
    return results


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.999999)))
    return ordered[index]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else float("inf")
    return numerator / denominator


def _wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    spread = (
        z * sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    )
    return (max(0.0, center - spread), min(1.0, center + spread))


def _paired_difference_ci95(
    *, challenger_only: int, champion_only: int, sample_size: int
) -> tuple[float, float]:
    """Conservative Newcombe-style interval for a paired proportion difference."""
    challenger_interval = _wilson_interval(challenger_only, sample_size)
    champion_interval = _wilson_interval(champion_only, sample_size)
    return (
        max(-1.0, challenger_interval[0] - champion_interval[1]),
        min(1.0, challenger_interval[1] - champion_interval[0]),
    )


def _mcnemar_exact_pvalue(champion_only: int, challenger_only: int) -> float:
    discordant = champion_only + challenger_only
    if discordant == 0:
        return 1.0
    tail = min(champion_only, challenger_only)
    probability = sum(comb(discordant, index) for index in range(tail + 1)) / (2**discordant)
    return min(1.0, 2 * probability)


__all__ = [
    "PolicyShadowGateConfig",
    "PolicyShadowReport",
    "compare_policy_shadow_runs",
]
