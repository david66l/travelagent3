from scripts.build_ai_assisted_external_pilot import build_cases
from scripts.export_external_pilot_vllm_cases import project
from evaluation.external_benchmark import Adjudication


def test_projection_preserves_action_benchmark_contract():
    external = build_cases()
    projected = [project(case) for case in external]

    assert len(projected) == 30
    assert [case.case_id for case in projected] == [case.case_id for case in external]
    assert all(case.expected_action is not None for case in projected)
    assert all(case.expected_action in case.allowed_actions for case in projected)
    assert all(case.tools for case in projected)
    assert all(case.expected_arguments is None for case in projected)
    assert {case.family for case in projected} == {
        "clarification",
        "search",
        "tradeoff",
        "recovery",
        "long_context_replan",
    }


def test_projection_prefers_adjudicated_action():
    external = build_cases()[0]
    external.adjudication = Adjudication(
        adjudicator_id="c",
        primary_action="abort",
        reason="Final adjudicated boundary",
    )

    assert project(external).expected_action == "abort"
