from evaluation.external_benchmark import BenchmarkSource, normalized_prompt_hash
from scripts.build_ai_assisted_external_pilot import build_cases, build_manifest


def test_ai_pilot_builds_30_schema_valid_diverse_cases():
    cases = build_cases()
    manifest = build_manifest(cases)

    assert len(cases) == 30
    assert manifest["unique_case_ids"] == 30
    assert manifest["unique_normalized_prompts"] == 30
    assert manifest["eligible_for_external_claim"] is False
    assert manifest["independent_human_authors"] == 0
    assert manifest["double_annotated_cases"] == 0
    assert len({normalized_prompt_hash(case) for case in cases}) == 30


def test_ai_pilot_provenance_never_claims_human_independence():
    cases = build_cases()

    assert all(case.provenance.authoring_method == "simulated" for case in cases)
    assert all(case.provenance.template_independent is False for case in cases)
    assert all(len(case.annotations) == 1 for case in cases)
    assert all(case.split.value == "dev" for case in cases)


def test_failure_contract_and_source_counts_are_explicit():
    cases = build_cases()
    failures = [case for case in cases if case.source == BenchmarkSource.TOOL_FAILURE]

    assert len(failures) == 6
    assert all(case.fault_injection is not None for case in failures)
    assert sum(case.source == BenchmarkSource.LONG_CONTEXT_REPLAN for case in cases) == 3
    assert sum(case.source == BenchmarkSource.AUTHORIZED_REAL_OR_SIMULATED for case in cases) == 21


def test_cases_use_the_production_policy_prompt_contract():
    cases = build_cases()

    assert all(
        [message["role"] for message in case.messages] == ["system", "user"] for case in cases
    )
    payloads = [__import__("json").loads(case.messages[1]["content"]) for case in cases]
    assert all(payload["original_request"] for payload in payloads)
    assert all(payload["hard_constraints"] for payload in payloads)
    assert all(payload["allowed_actions"] for payload in payloads)
