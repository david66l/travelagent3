"""Tests for paired champion/challenger policy Shadow evaluation."""

import pytest

from evaluation.policy_shadow import (
    PolicyShadowGateConfig,
    compare_policy_shadow_runs,
)


def _row(
    index: int,
    *,
    success: bool = True,
    action: str = "search_pois",
    latency_ms: float = 100.0,
    conflict: bool = False,
    http_error: str | None = None,
    family: str = "search",
) -> dict:
    return {
        "case_id": f"case-{index:03d}",
        "repetition": 0,
        "family": family,
        "expected_action": action,
        "success": success,
        "policy_contract_success": success,
        "label_contract_conflict": conflict,
        "observed_actions": [action] if action else [],
        "http_error": http_error,
        "inference_metrics": {"request_latency_ms": latency_ms},
    }


def _live(row: dict) -> dict:
    return {
        **row,
        "evaluation_source": "live_shadow",
        "release_gate_eligible": True,
        "outcome_observed": True,
    }


def test_policy_shadow_requires_exact_pair_alignment():
    with pytest.raises(ValueError, match="scenario sets differ"):
        compare_policy_shadow_runs(
            [_row(1)],
            [_row(2)],
            evidence_source="sealed_benchmark",
        )


def test_policy_shadow_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate champion"):
        compare_policy_shadow_runs(
            [_row(1), _row(1)],
            [_row(1)],
            evidence_source="sealed_benchmark",
        )


def test_policy_shadow_excludes_contract_conflicts_and_counts_discordance():
    champion = [
        _row(1, success=True),
        _row(2, success=True),
        _row(3, success=False),
        _row(4, conflict=True),
    ]
    challenger = [
        _row(1, success=True),
        {
            **_row(2, success=False, action="abort"),
            "expected_action": "search_pois",
        },
        _row(3, success=True),
        _row(4, conflict=True),
    ]

    report = compare_policy_shadow_runs(
        champion,
        challenger,
        evidence_source="sealed_benchmark",
        gate=PolicyShadowGateConfig(minimum_paired_decisions=3),
    )

    assert report.paired_decisions == 3
    assert report.excluded_label_contract_conflicts == 1
    assert report.champion_only_successes == 1
    assert report.challenger_only_successes == 1
    assert report.both_successes == 1
    assert report.both_failures == 0
    assert report.mcnemar_exact_pvalue == 1.0
    assert report.action_divergences == 1
    assert report.canary_evidence is False
    assert report.release_eligible is False


def test_policy_shadow_live_evidence_can_pass_after_sufficient_pairs():
    champion = [_live(_row(index)) for index in range(500)]
    challenger = [_live(_row(index, latency_ms=105.0)) for index in range(500)]

    report = compare_policy_shadow_runs(
        champion,
        challenger,
        evidence_source="live_shadow",
    )

    assert report.quality_gates_passed is True
    assert report.canary_evidence is True
    assert report.release_eligible is True
    assert all(check.passed for check in report.checks)


def test_policy_shadow_blocks_latency_and_http_regressions():
    champion = [_live(_row(index)) for index in range(500)]
    challenger = [
        _live(
            _row(
                index,
                success=index != 0,
                latency_ms=150.0,
                http_error="timeout" if index == 0 else None,
            )
        )
        for index in range(500)
    ]

    report = compare_policy_shadow_runs(
        champion,
        challenger,
        evidence_source="live_shadow",
        gate=PolicyShadowGateConfig(
            maximum_http_error_rate=0.001,
            maximum_p95_latency_ratio=1.25,
        ),
    )

    failed = {check.code for check in report.checks if not check.passed}
    assert "CHALLENGER_HTTP_ERROR_RATE" in failed
    assert "P95_LATENCY_RATIO" in failed
    assert report.quality_gates_passed is False
    assert report.release_eligible is False


def test_policy_shadow_rejects_live_label_without_executed_outcome_provenance():
    with pytest.raises(ValueError, match="provenance and executed outcomes"):
        compare_policy_shadow_runs(
            [_row(1)],
            [_row(1)],
            evidence_source="live_shadow",
            gate=PolicyShadowGateConfig(minimum_paired_decisions=1),
        )


def test_policy_shadow_reports_family_slices():
    champion = [
        _row(1, family="search"),
        {**_row(2, family="recovery"), "route_family": "recovery"},
    ]
    challenger = [
        _row(1, family="search"),
        {
            **_row(2, family="recovery", success=False, action="abort"),
            "route_family": "recovery",
        },
    ]

    report = compare_policy_shadow_runs(
        champion,
        challenger,
        evidence_source="authorized_replay",
        gate=PolicyShadowGateConfig(minimum_paired_decisions=2),
    )

    assert [row.family for row in report.family_comparisons] == ["recovery", "search"]
    recovery = report.family_comparisons[0]
    assert recovery.champion_only_successes == 1
    assert recovery.action_divergence_rate == 1.0
