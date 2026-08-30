from evaluation.external_benchmark import ForbiddenCorpusDocument
from scripts.audit_ai_assisted_external_pilot import audit_pilot
from scripts.build_ai_assisted_external_pilot import build_cases


def test_pilot_audit_passes_calibration_but_never_external_claim():
    report = audit_pilot(
        build_cases(),
        [ForbiddenCorpusDocument(document_id="train:1", text="完全无关的训练请求文本")],
    )

    assert report["passed"] is True
    assert report["status"] == "passed_for_schema_calibration"
    assert report["eligible_for_external_claim"] is False
    assert all(report["gates"].values())


def test_pilot_audit_blocks_training_duplicate():
    cases = build_cases()
    prompt = next(message["content"] for message in cases[0].messages if message["role"] == "user")

    report = audit_pilot(
        cases, [ForbiddenCorpusDocument(document_id="train:duplicate", text=prompt)]
    )

    assert report["passed"] is False
    assert report["gates"]["no_training_contamination"] is False
