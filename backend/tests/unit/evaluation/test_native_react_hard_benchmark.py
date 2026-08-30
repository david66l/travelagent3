import hashlib
import json

from evaluation.native_react_hard_benchmark import (
    DEV_CITIES,
    FAMILIES,
    audit_cases,
    benchmark_hash,
    build_cases,
    existing_full_loop_prompts,
    write_benchmark,
)


def test_benchmark_has_200_cluster_aware_statistical_units():
    cases = build_cases()
    audit = audit_cases(cases, forbidden_prompts=existing_full_loop_prompts())

    assert audit["passed"] is True
    assert audit["counts"]["total"] == 200
    assert audit["counts"]["split"] == {"dev": 40, "test": 160}
    assert audit["counts"]["family"] == {family: 20 for family in FAMILIES}
    assert audit["counts"]["cities"] == 20
    assert audit["counts"]["semantic_clusters"] == 40
    assert set(audit["split_isolation"]["dev_cities"]) == DEV_CITIES
    assert audit["split_isolation"]["city_overlap"] == []


def test_benchmark_is_deterministic_and_never_training_eligible():
    first = build_cases()
    second = build_cases()

    assert benchmark_hash(first) == benchmark_hash(second)
    assert all(case.metadata.frozen_for_training for case in first)
    assert all(not case.metadata.eligible_for_independent_human_benchmark_claim for case in first)
    assert sum(case.metadata.fault_spec is not None for case in first) == 20


def test_expected_slots_only_require_information_present_in_each_prompt_variant():
    cases = build_cases()
    by_id = {case.case.case_id: case.case for case in cases}

    assert "travelers_count" not in by_id["nrhb-v2-poi-grounding-001-beijing"].expected_slots
    assert "total_budget" not in by_id["nrhb-v2-current-information-002-shanghai"].expected_slots
    assert by_id["nrhb-v2-accessibility-003-guangzhou"].expected_slots["travelers_count"] == 2
    revision = by_id["nrhb-v2-revision-004-chengdu"]
    assert revision.expected_revision_soft == {"travelers_count": 4}


def test_prompt_budget_and_expected_budget_never_drift():
    for benchmark_case in build_cases():
        expected = benchmark_case.case.expected_slots.get("total_budget")
        if expected is None:
            continue
        assert f"{int(expected)}" in benchmark_case.case.user_input


def test_accessibility_budgets_do_not_accidentally_trigger_feasibility_gate():
    daily_cost = {"北京": 800, "上海": 900, "广州": 700, "成都": 600}
    cases = [
        row
        for row in build_cases()
        if row.metadata.split == "dev" and row.metadata.family == "accessibility"
    ]

    for benchmark_case in cases:
        slots = benchmark_case.case.expected_slots
        if "total_budget" not in slots:
            continue
        travelers = int(slots.get("travelers_count") or 2)
        minimum = daily_cost[benchmark_case.metadata.city] * int(slots["travel_days"])
        minimum *= travelers * 0.6
        assert float(slots["total_budget"]) >= minimum


def test_fault_cases_score_observable_recovery_not_one_intent_synonym():
    stale_case = next(
        row
        for row in build_cases()
        if row.metadata.family == "tool_recovery"
        and row.metadata.fault_spec
        and row.metadata.fault_spec.fault_type == "stale_data"
    )

    assert "information_needs" not in stale_case.case.expected_slots
    assert "search_current_info" in stale_case.case.required_actions
    assert "current_info_search" in stale_case.case.required_artifacts


def test_duplicate_prompt_and_training_overlap_fail_the_gate():
    cases = build_cases()
    cases[1].case.user_input = cases[0].case.user_input

    audit = audit_cases(cases, forbidden_prompts=[cases[2].case.user_input])

    assert audit["passed"] is False
    assert audit["gates"]["unique_normalized_prompts"] is False
    assert audit["gates"]["no_forbidden_exact_overlap"] is False


def test_writer_freezes_split_files_and_hashes(tmp_path):
    manifest = write_benchmark(tmp_path, git_commit="abc123")

    assert manifest["git_commit"] == "abc123"
    for name, expected_hash in manifest["files"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected_hash
    dev = [
        json.loads(line)
        for line in (tmp_path / "dev.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    test = [
        json.loads(line)
        for line in (tmp_path / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(dev) == 40
    assert len(test) == 160
