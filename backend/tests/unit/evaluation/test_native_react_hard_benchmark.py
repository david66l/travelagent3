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

    assert "travelers_count" not in by_id["nrhb-v1-poi-grounding-001-beijing"].expected_slots
    assert "total_budget" not in by_id["nrhb-v1-current-information-002-shanghai"].expected_slots
    assert by_id["nrhb-v1-accessibility-003-guangzhou"].expected_slots["travelers_count"] == 2
    revision = by_id["nrhb-v1-revision-004-chengdu"]
    assert revision.expected_revision_hard == {"travelers_count": 4}


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
