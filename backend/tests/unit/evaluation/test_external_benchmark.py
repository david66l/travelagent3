import pytest

from evaluation.external_benchmark import (
    Adjudication,
    BenchmarkSource,
    BenchmarkSplit,
    ExpectedTermination,
    ExternalBenchmarkCase,
    FaultInjection,
    ForbiddenCorpusDocument,
    GroupKeys,
    HardConstraint,
    IndependentAnnotation,
    OutcomeRubric,
    Provenance,
    audit_split_isolation,
    audit_training_contamination,
    cohens_kappa,
)


def _case(
    case_id: str,
    *,
    split: BenchmarkSplit,
    prompt: str,
    city: str,
    fixture_value: str,
    source: BenchmarkSource = BenchmarkSource.HUMAN_ORIGINAL_CONSTRAINT,
) -> ExternalBenchmarkCase:
    action = "propose_tradeoff"
    return ExternalBenchmarkCase(
        case_id=case_id,
        split=split,
        source=source,
        difficulty="L3",
        provenance=Provenance(
            authoring_method="human_original",
            permission_basis="benchmark author consent",
            deidentified=True,
            template_independent=True,
            author_group="author-a",
        ),
        group_keys=GroupKeys(
            request_family="budget_conflict",
            city_cluster=city,
            date_pattern="weekend",
            constraint_combo="budget+must_visit",
        ),
        messages=[{"role": "user", "content": prompt}],
        tool_fixture={"inventory": fixture_value},
        allowed_actions=[action, "abort"],
        hard_constraints=[
            HardConstraint(
                constraint_id="budget",
                kind="budget",
                description="Total cost must remain within the cap",
                verifier="budget_cap",
                params={"cap": 1000},
            )
        ],
        acceptable_tradeoff_categories=["reduce_optional_item"],
        expected_termination=ExpectedTermination.TRADEOFF,
        outcome_rubric=OutcomeRubric(
            success="Offers a feasible explicit tradeoff",
            partial_success="Finds conflict but omits one explanation",
            failure="Claims all constraints can be met",
            safe_termination="Aborts with the conflict identified",
        ),
        max_steps=6,
        annotations=[
            IndependentAnnotation(
                annotator_id="a", primary_action=action, allowed_actions=[action]
            ),
            IndependentAnnotation(
                annotator_id="b", primary_action=action, allowed_actions=[action]
            ),
        ],
    )


def test_split_audit_detects_normalized_prompt_leakage():
    dev = _case(
        "ext-v1-dev-1",
        split=BenchmarkSplit.DEV,
        prompt="上海 周末，预算 1000！",
        city="east",
        fixture_value="dev",
    )
    test = _case(
        "ext-v1-test-1",
        split=BenchmarkSplit.SEALED_TEST,
        prompt="上海周末预算1000",
        city="south",
        fixture_value="test",
    )

    result = audit_split_isolation([dev], [test])

    assert result["passed"] is False
    assert result["overlap_counts"]["normalized_prompts"] == 1


def test_split_audit_passes_independent_cases():
    dev = _case(
        "ext-v1-dev-1",
        split=BenchmarkSplit.DEV,
        prompt="上海周末预算冲突",
        city="east",
        fixture_value="dev",
    )
    test = _case(
        "ext-v1-test-1",
        split=BenchmarkSplit.SEALED_TEST,
        prompt="成都工作日无障碍路线",
        city="southwest",
        fixture_value="test",
    )
    test.hard_constraints[0].params = {"cap": 2000}

    assert audit_split_isolation([dev], [test])["passed"] is True


def test_cohens_kappa_handles_agreement_and_invalid_input():
    assert cohens_kappa(["search", "clarify"], ["search", "clarify"]) == 1.0
    assert cohens_kappa(["search", "search"], ["clarify", "clarify"]) == 0.0
    with pytest.raises(ValueError, match="paired non-empty"):
        cohens_kappa([], [])


def test_tool_failure_requires_fault_injection():
    with pytest.raises(ValueError, match="require fault_injection"):
        _case(
            "ext-v1-failure-1",
            split=BenchmarkSplit.DEV,
            prompt="搜索超时后恢复",
            city="east",
            fixture_value="timeout",
            source=BenchmarkSource.TOOL_FAILURE,
        )

    case = _case(
        "ext-v1-failure-2",
        split=BenchmarkSplit.DEV,
        prompt="搜索超时后恢复",
        city="east",
        fixture_value="timeout",
    )
    payload = case.model_dump()
    payload["source"] = BenchmarkSource.TOOL_FAILURE
    payload["fault_injection"] = FaultInjection(
        fault_type="timeout", trigger_step=1, recoverable=True
    )
    assert ExternalBenchmarkCase.model_validate(payload).fault_injection is not None


def test_unresolved_annotation_conflict_is_explicit():
    case = _case(
        "ext-v1-test-2",
        split=BenchmarkSplit.SEALED_TEST,
        prompt="预算冲突",
        city="north",
        fixture_value="x",
    )
    case.annotations[1].primary_action = "abort"
    assert case.adjudication is None
    case.adjudication = Adjudication(
        adjudicator_id="c",
        primary_action="propose_tradeoff",
        reason="A feasible relaxation exists",
    )
    assert case.adjudication.primary_action == "propose_tradeoff"


def test_training_contamination_redacts_sealed_case_identity():
    dev = _case(
        "ext-v1-dev-contamination",
        split=BenchmarkSplit.DEV,
        prompt="请安排上海周末预算冲突行程",
        city="east",
        fixture_value="dev",
    )
    sealed = _case(
        "ext-v1-sealed-contamination",
        split=BenchmarkSplit.SEALED_TEST,
        prompt="请安排成都周末无障碍行程",
        city="southwest",
        fixture_value="sealed",
    )
    documents = [
        ForbiddenCorpusDocument(document_id="train:1", text="请安排上海周末预算冲突行程"),
        ForbiddenCorpusDocument(document_id="train:2", text="请安排成都周末无障碍行程"),
    ]

    result = audit_training_contamination([dev, sealed], documents)

    assert result["passed"] is False
    assert result["exact_matches"] == 2
    assert result["dev_findings"][0]["case_id"] == dev.case_id
    assert result["sealed_finding_hashes"]
    assert sealed.case_id not in str(result)


def test_training_contamination_detects_near_duplicate():
    case = _case(
        "ext-v1-dev-near",
        split=BenchmarkSplit.DEV,
        prompt="上海三天亲子旅行预算两千元并且必须去博物馆",
        city="east",
        fixture_value="dev",
    )
    documents = [
        ForbiddenCorpusDocument(
            document_id="sft:9",
            text="上海三天亲子旅行，预算两千元，并且必须去博物馆，谢谢。",
        )
    ]

    result = audit_training_contamination([case], documents)

    assert result["near_duplicate_matches"] == 1
    assert result["exact_matches"] == 0
