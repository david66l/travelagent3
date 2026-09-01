"""Verifier-repair GRPO rows must replay production review states safely."""

import json
import inspect
from pathlib import Path

import pytest

from agentic.environment import environment_fingerprint
from agentic.grpo_training import (
    MIN_DECISION_STATE_COMPLETION_LENGTH,
    VERIFIER_REPAIR_ACTIONS_BY_ROUTE,
    VERIFIER_REPAIR_ROUTE_BY_TARGET,
    _budget_probe_action,
    load_grpo_corpus,
    minimum_completion_length_floor,
    preflight_grpo_corpus,
    to_trl_environment_rows,
)
from agentic.policy_actions import (
    PolicyArgumentValidationError,
    model_visible_policy_actions,
    policy_action_schema,
)
from agentic.reason_quality import build_grounded_repair_reason
from agentic.trl_environment import (
    TRLReactEnvironment,
    build_trl_environment_factories,
    canonical_trl_tool_schemas,
)
from evaluation.validator import VALIDATOR_HARD_VIOLATION_CODES
from scripts.audit_model_curriculum import (
    rollout_action_rows,
    transport_trl_environment_rows,
)
from scripts.build_verifier_repair_grpo_corpus import (
    _FLEXIBILITY_PATTERNS,
    _ACTION_LEXICAL_CUES,
    _NEUTRAL_DIVERSE_TRAIN_TEMPLATES,
    _STRICT_COUNTERFACTUAL_TRAIN_TEMPLATES,
    _RL_CHALLENGE_TRAIN_TEMPLATES,
    _SEMANTIC_DIVERSE_TRAIN_TEMPLATES,
    _TEMPLATES,
    _strict_counterfactual_sources,
    _prepare_variant,
)


def _source():
    return load_grpo_corpus(
        Path("ml/agentic/datasets/native-react-grpo-v1/train.jsonl")
    )[0]


def _teacher_reason(row):
    contract = row.snapshot.hidden_test_facts["grpo_decision_state"]
    transition = json.loads(contract["prompt_messages"][-1]["content"])
    reports = [
        item
        for item in transition["policy_state"].get("relevant_artifacts") or []
        if item.get("artifact_type") == "validation_report"
    ]
    evidence = reports[-1]["violations"][0]["message"]
    return build_grounded_repair_reason(evidence, contract["target_action"])


def test_semantic_diverse_train_templates_are_balanced_and_holdout_safe():
    targets = [item["target_action"] for item in _SEMANTIC_DIVERSE_TRAIN_TEMPLATES]
    assert targets.count("retry_solve") == 3
    assert targets.count("propose_tradeoff") == 3
    assert targets.count("abort") == 3

    train_text = json.dumps(_SEMANTIC_DIVERSE_TRAIN_TEMPLATES, ensure_ascii=False)
    holdout_phrases = {
        phrase
        for split in ("validation", "test")
        for template in _TEMPLATES[split]
        for phrase in template["grounding_phrases"]
    }
    assert all(phrase not in train_text for phrase in holdout_phrases)


def test_rl_challenge_templates_are_balanced_unseen_and_holdout_safe():
    targets = [item["target_action"] for item in _RL_CHALLENGE_TRAIN_TEMPLATES]
    assert targets.count("retry_solve") == 3
    assert targets.count("propose_tradeoff") == 3
    assert targets.count("abort") == 3

    challenge_text = json.dumps(_RL_CHALLENGE_TRAIN_TEMPLATES, ensure_ascii=False)
    sft_text = json.dumps(
        [*_TEMPLATES["train"], *_SEMANTIC_DIVERSE_TRAIN_TEMPLATES],
        ensure_ascii=False,
    )
    holdout_phrases = {
        phrase
        for split in ("validation", "test")
        for template in _TEMPLATES[split]
        for phrase in template["grounding_phrases"]
    }
    challenge_phrases = {
        phrase
        for template in _RL_CHALLENGE_TRAIN_TEMPLATES
        for phrase in template["grounding_phrases"]
    }
    assert all(phrase not in sft_text for phrase in challenge_phrases)
    assert all(phrase not in challenge_text for phrase in holdout_phrases)


def test_neutral_diverse_train_templates_are_balanced_and_do_not_leak_actions():
    targets = [item["target_action"] for item in _NEUTRAL_DIVERSE_TRAIN_TEMPLATES]
    assert targets.count("retry_solve") == 6
    assert targets.count("propose_tradeoff") == 6
    assert targets.count("abort") == 6

    surface = "\n".join(
        "\n".join(
            [
                template["request_suffix"],
                template["violation"],
                *template["grounding_phrases"],
            ]
        )
        for template in _NEUTRAL_DIVERSE_TRAIN_TEMPLATES
    ).casefold()
    assert all(cue.casefold() not in surface for cue in _ACTION_LEXICAL_CUES)


def test_strict_counterfactual_triples_change_only_failed_constraint_handling():
    templates = [
        item
        for item in _STRICT_COUNTERFACTUAL_TRAIN_TEMPLATES
        if item["case_id"] == "budget-175y"
    ]
    assert {item["target_action"] for item in templates} == {
        "retry_solve",
        "propose_tradeoff",
        "abort",
    }
    rows = [
        _prepare_variant(_source(), split="train", template=template, ordinal=index)
        for index, template in enumerate(templates)
    ]
    contracts = [row.snapshot.hidden_test_facts["grpo_decision_state"] for row in rows]

    assert len({row.task.user_request for row in rows}) == 1
    assert len(
        {
            json.dumps(
                {
                    key: value
                    for key, value in row.task.slots.items()
                    if key != "constraint_flexibility"
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for row in rows
        }
    ) == 1
    assert len({_violation_message(row) for row in rows}) == 1
    assert len({contract["counterfactual_group_id"] for contract in contracts}) == 1
    assert [
        contract["flexibility_pattern"] for contract in contracts
    ] == ["counterfactual:budget-175y"] * 3
    assert {
        contract["target_action"] for contract in contracts
    } == {"retry_solve", "propose_tradeoff", "abort"}
    retry_contract = next(contract for contract in contracts if contract["target_action"] == "retry_solve")
    assert retry_contract["expected_arguments"] == {}
    assert retry_contract["controller_arguments"] == {"strategy": "greedy"}


def test_strict_counterfactual_sources_fail_closed_without_concrete_activities():
    complete = _source()
    incomplete = complete.model_copy(deep=True)
    incomplete.snapshot.tool_responses["solve_itinerary"][0].data["days"] = []

    assert _strict_counterfactual_sources([complete, incomplete]) == [complete]


def _violation_message(row):
    response = row.snapshot.tool_responses["validate_itinerary"][0]
    return response.data["hard_violations"][0]["message"]


def test_real_violation_codes_do_not_reveal_the_target_action():
    templates = [
        *_TEMPLATES["train"],
        *_TEMPLATES["validation"],
        *_TEMPLATES["test"],
        *_SEMANTIC_DIVERSE_TRAIN_TEMPLATES,
        *_RL_CHALLENGE_TRAIN_TEMPLATES,
        *_NEUTRAL_DIVERSE_TRAIN_TEMPLATES,
    ]
    targets_by_code: dict[str, set[str]] = {}
    for template in templates:
        code = template["violation_code"]
        targets_by_code.setdefault(code, set()).add(template["target_action"])

    synthetic_label_codes = {
        "REPAIRABLE_SOLVER_FAILURE",
        "ACTIONABLE_TRADEOFF",
        "NECESSARY_ABORT",
    }
    assert not synthetic_label_codes.intersection(targets_by_code)
    assert set(targets_by_code) <= VALIDATOR_HARD_VIOLATION_CODES
    assert any(len(targets) > 1 for targets in targets_by_code.values())


def test_neutral_flexibility_patterns_do_not_encode_an_action_label():
    templates = [
        *_TEMPLATES["train"],
        *_TEMPLATES["validation"],
        *_TEMPLATES["test"],
        *_SEMANTIC_DIVERSE_TRAIN_TEMPLATES,
        *_RL_CHALLENGE_TRAIN_TEMPLATES,
        *_NEUTRAL_DIVERSE_TRAIN_TEMPLATES,
    ]
    targets_by_pattern: dict[str, set[str]] = {}
    for template in templates:
        targets_by_pattern.setdefault(template["flexibility_pattern"], set()).add(
            template["target_action"]
        )

    assert set(targets_by_pattern) == set(_FLEXIBILITY_PATTERNS)
    assert all(
        targets == {"retry_solve", "propose_tradeoff", "abort"}
        for targets in targets_by_pattern.values()
    )
    visible_contracts = json.dumps(_FLEXIBILITY_PATTERNS, ensure_ascii=False)
    for forbidden in (
        "target_action",
        "solver_retry_authorized",
        "stop_if_unresolved",
        "should_abort",
        "should_tradeoff",
    ):
        assert forbidden not in visible_contracts


def test_verifier_repair_variants_are_unique_and_prompt_target_is_hidden():
    source = _source()
    rows = [
        _prepare_variant(
            source,
            split="validation",
            template=template,
            ordinal=index,
        )
        for index, template in enumerate(_TEMPLATES["validation"])
    ]

    assert len({row.task.task_id for row in rows}) == len(_TEMPLATES["validation"])
    assert len({environment_fingerprint(row.task, row.snapshot) for row in rows}) == len(
        _TEMPLATES["validation"]
    )
    assert {
        row.snapshot.hidden_test_facts["grpo_decision_state"]["target_action"]
        for row in rows
    } == {"retry_solve", "propose_tradeoff", "abort"}
    for row in rows:
        state = row.snapshot.hidden_test_facts["grpo_decision_state"]
        prompt = state["prompt_messages"]
        assert [message["role"] for message in prompt] == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
            "tool",
        ]
        assert "target_action" not in json.dumps(prompt, ensure_ascii=False)
        assert "constraint-flexibility.v1" in json.dumps(prompt, ensure_ascii=False)
        assert state["source_task_id"] == source.task.task_id
        visible = json.dumps(prompt, ensure_ascii=False)
        assert any(phrase in visible for phrase in state["grounding_phrases"])


def test_verifier_repair_rows_survive_dataset_roundtrip_and_replay_authority():
    pytest.importorskip("datasets")
    source = _source()
    templates = [
        next(
            template
            for template in _TEMPLATES["validation"]
            if template["target_action"] == target
        )
        for target in ("retry_solve", "propose_tradeoff", "abort")
    ]
    rows = [
        _prepare_variant(
            source,
            split="validation",
            template=template,
            ordinal=index,
        )
        for index, template in enumerate(templates)
    ]
    converted = to_trl_environment_rows(rows)

    transported, evidence = transport_trl_environment_rows(rows)

    assert transported == converted
    assert evidence["authority_payload_encoding"] == "canonical-json.v1"
    assert evidence["task_feature"] == "Value('string')"
    assert evidence["snapshot_feature"] == "Value('string')"
    assert evidence["whole_row_equality"] is True
    assert evidence["decoded_authority_equality"] is True
    assert evidence["fingerprint_equality"] is True
    assert evidence["authoritative_prompt_equality"] is True
    assert {
        row.snapshot.hidden_test_facts["grpo_decision_state"]["target_action"]
        for row in rows
    } == {"retry_solve", "propose_tradeoff", "abort"}
    factories = build_trl_environment_factories("react")
    for row, source_item, item in zip(rows, converted, transported, strict=True):
        assert isinstance(item["task"], str)
        assert isinstance(item["snapshot"], str)
        assert item["authority_payload_encoding"] == "canonical-json.v1"
        for source_message, transported_message in zip(
            source_item["prompt"], item["prompt"], strict=True
        ):
            assert transported_message == source_message
        environment = factories[item["environment"]](audit_enabled=False)
        assert environment.reset(**item) == ""
        initial = json.loads(item["prompt"][-1]["content"])
        target = row.snapshot.hidden_test_facts["grpo_decision_state"]["target_action"]
        assert initial["policy_state"]["current_subtask"]["task_id"] == "review_itinerary"
        assert target in initial["policy_state"]["allowed_actions"]
        environment.get_reward()

        missing_marker = dict(item)
        missing_marker.pop("authority_payload_encoding")
        rejected = factories[item["environment"]](audit_enabled=False)
        with pytest.raises(
            ValueError, match="canonical authority payloads require"
        ):
            rejected.reset(**missing_marker)


def test_preflight_rejects_verifier_repair_row_without_flexibility_contract(tmp_path):
    source = _source()
    train = _prepare_variant(
        source,
        split="validation",
        template=_TEMPLATES["validation"][0],
        ordinal=0,
    )
    train.task.slots = {
        key: value
        for key, value in train.task.slots.items()
        if key != "constraint_flexibility"
    }
    validation = _prepare_variant(
        source,
        split="validation",
        template=_TEMPLATES["validation"][1],
        ordinal=1,
    )
    for name, row in (("train", train), ("validation", validation)):
        (tmp_path / f"{name}.jsonl").write_text(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    report = preflight_grpo_corpus(
        tmp_path,
        minimum_train_tasks=1,
        require_dependencies=False,
    )

    assert report.ready is False
    assert any(
        error.startswith("VERIFIER_REPAIR_FLEXIBILITY_CONTRACT_INVALID:train:")
        for error in report.errors
    )


def test_preflight_rejects_unknown_verifier_repair_decision_schema(tmp_path):
    source = _source()
    train = _prepare_variant(
        source,
        split="validation",
        template=_TEMPLATES["validation"][0],
        ordinal=0,
    )
    train.snapshot.hidden_test_facts["grpo_decision_state"]["schema_version"] = (
        "react-verifier-repair-decision.v999"
    )
    validation = _prepare_variant(
        source,
        split="validation",
        template=_TEMPLATES["validation"][1],
        ordinal=1,
    )
    for name, row in (("train", train), ("validation", validation)):
        (tmp_path / f"{name}.jsonl").write_text(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    report = preflight_grpo_corpus(
        tmp_path,
        minimum_train_tasks=1,
        require_dependencies=False,
    )

    assert report.ready is False
    assert any(
        error.startswith("VERIFIER_REPAIR_DECISION_SCHEMA_INVALID:train:")
        for error in report.errors
    )


def test_validation_is_a_balanced_implicit_counterfactual_grid():
    templates = _TEMPLATES["validation"]
    targets_by_code: dict[str, set[str]] = {}
    messages_by_code: dict[str, set[str]] = {}
    for template in templates:
        code = template["violation_code"]
        targets_by_code.setdefault(code, set()).add(template["target_action"])
        messages_by_code.setdefault(code, set()).add(template["violation"])

    assert all(
        targets == {"retry_solve", "propose_tradeoff", "abort"}
        for targets in targets_by_code.values()
    )
    assert all(len(messages) == 1 for messages in messages_by_code.values())
    direct_action_cues = {
        "重试",
        "重算",
        "再算",
        "换策略",
        "选择",
        "选项",
        "停止",
        "终止",
        "结束规划",
        "不要继续",
        "不要编造",
    }
    assert all(
        not any(cue in template["request_suffix"] for cue in direct_action_cues)
        for template in templates
    )


def test_hidden_target_mutation_cannot_rewrite_contract_or_prompt_and_fails_reward():
    row = _prepare_variant(
        _source(),
        split="validation",
        template=_TEMPLATES["validation"][0],
        ordinal=0,
    )
    mutated = row.model_copy(deep=True)
    original_state = row.snapshot.hidden_test_facts["grpo_decision_state"]
    mutated_state = mutated.snapshot.hidden_test_facts["grpo_decision_state"]
    original_contract = row.task.slots["constraint_flexibility"]
    original_prompt = original_state["prompt_messages"]
    mutated_state["target_action"] = "abort"

    assert mutated.task.slots["constraint_flexibility"] == original_contract
    assert mutated_state["prompt_messages"] == original_prompt

    with pytest.raises(
        ValueError,
        match="review action contract does not match route",
    ):
        to_trl_environment_rows([mutated])


def test_verifier_repair_reset_rejects_prompt_state_tampering():
    row = _prepare_variant(
        _source(),
        split="validation",
        template=_TEMPLATES["validation"][0],
        ordinal=0,
    )
    converted = to_trl_environment_rows([row])[0]
    converted["prompt"][-1]["content"] = "{}"
    environment = build_trl_environment_factories("react")[converted["environment"]](
        audit_enabled=False
    )

    with pytest.raises(ValueError, match="replay prompt is not authoritative"):
        environment.reset(**converted)


@pytest.mark.parametrize(
    "tamper",
    ["tool_content", "tool_call_name", "tool_call_arguments"],
)
def test_verifier_repair_reset_rejects_early_prompt_tampering(tamper):
    row = _prepare_variant(
        _source(),
        split="validation",
        template=_TEMPLATES["validation"][0],
        ordinal=0,
    )
    converted = to_trl_environment_rows([row])[0]
    if tamper == "tool_content":
        converted["prompt"][3]["content"] = "TAMPERED_EARLY_TOOL_EVIDENCE"
    elif tamper == "tool_call_name":
        converted["prompt"][2]["tool_calls"][0]["function"]["name"] = "abort"
    else:
        converted["prompt"][2]["tool_calls"][0]["function"]["arguments"] = {
            "topic": "tampered"
        }
    environment = build_trl_environment_factories("react")[converted["environment"]](
        audit_enabled=False
    )

    with pytest.raises(ValueError, match="replay prompt is not authoritative"):
        environment.reset(**converted)


def test_verifier_repair_environment_exposes_the_production_review_action_space(
    tmp_path, monkeypatch
):
    row = _prepare_variant(
        _source(),
        split="validation",
        template=_TEMPLATES["validation"][0],
        ordinal=0,
    )
    converted = to_trl_environment_rows([row])[0]
    assert (
        minimum_completion_length_floor([row])
        == MIN_DECISION_STATE_COMPLETION_LENGTH
    )
    environment = build_trl_environment_factories("react")[converted["environment"]]()
    audit_path = tmp_path / "verifier-replay-audit.jsonl"
    monkeypatch.setenv("AGENTIC_GRPO_AUDIT_PATH", str(audit_path))

    environment.reset(**converted)

    expected = set(
        row.snapshot.hidden_test_facts["grpo_decision_state"]["review_allowed_actions"]
    )
    exposed = {
        name
        for name, _ in inspect.getmembers(environment, predicate=inspect.ismethod)
        if name not in {"reset", "get_reward"} and not name.startswith("_")
    }
    state = json.loads(
        row.snapshot.hidden_test_facts["grpo_decision_state"]["prompt_messages"][-1][
            "content"
        ]
    )["policy_state"]
    assert set(
        model_visible_policy_actions(tuple(exposed), capability=state["capability"])
    ) == expected
    assert tuple(
        row.snapshot.hidden_test_facts["grpo_decision_state"][
            "review_allowed_actions"
        ]
    ) == VERIFIER_REPAIR_ACTIONS_BY_ROUTE[converted["environment"]]
    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["event"] == "reset"
    assert records[0]["rollout_contract"] == "verified_decision_state_replay.v1"
    ready = next(item for item in records if item["event"] == "decision_replay_ready")
    assert ready["credited_step_start"] > 0
    assert ready["credited_step_start"] == len(ready["steps"])


@pytest.mark.parametrize("target", ["retry_solve", "propose_tradeoff", "abort"])
def test_verifier_repair_routes_expose_only_the_state_authorized_tools(target):
    template = next(
        item for item in _TEMPLATES["validation"] if item["target_action"] == target
    )
    row = _prepare_variant(
        _source(),
        split="validation",
        template=template,
        ordinal=0,
    )
    converted = to_trl_environment_rows([row])[0]
    route = VERIFIER_REPAIR_ROUTE_BY_TARGET[target]
    factories = build_trl_environment_factories("react")

    assert converted["environment"] == route
    assert "decision_verifier_repair" not in factories
    environment = factories[route](audit_enabled=False)
    public_tools = {
        name
        for name, _ in inspect.getmembers(environment, predicate=inspect.ismethod)
        if name not in {"reset", "get_reward"} and not name.startswith("_")
    }
    state = json.loads(
        row.snapshot.hidden_test_facts["grpo_decision_state"]["prompt_messages"][-1][
            "content"
        ]
    )["policy_state"]
    assert set(
        model_visible_policy_actions(tuple(public_tools), capability=state["capability"])
    ) == set(VERIFIER_REPAIR_ACTIONS_BY_ROUTE[route])


def test_verifier_repair_route_selection_rejects_action_set_mismatch():
    row = _prepare_variant(
        _source(),
        split="validation",
        template=next(
            item
            for item in _TEMPLATES["validation"]
            if item["target_action"] == "propose_tradeoff"
        ),
        ordinal=0,
    )
    row.snapshot.hidden_test_facts["grpo_decision_state"][
        "review_allowed_actions"
    ] = ["propose_tradeoff"]

    with pytest.raises(
        ValueError,
        match="review action contract does not match route",
    ):
        to_trl_environment_rows([row])


def test_verifier_repair_reset_rejects_cross_route_replay():
    row = _prepare_variant(
        _source(),
        split="validation",
        template=next(
            item
            for item in _TEMPLATES["validation"]
            if item["target_action"] == "retry_solve"
        ),
        ordinal=0,
    )
    converted = to_trl_environment_rows([row])[0]
    wrong_environment = build_trl_environment_factories("react")[
        "decision_verifier_repair_abort"
    ](audit_enabled=False)

    with pytest.raises(ValueError, match="does not match route"):
        wrong_environment.reset(**converted)
    wrong_environment.get_reward()


def test_verifier_repair_trl_schema_and_runtime_rejection_share_production_contract(
    tmp_path, monkeypatch
):
    row = _prepare_variant(
        _source(),
        split="validation",
        template=next(
            item
            for item in _TEMPLATES["validation"]
            if item["target_action"] == "retry_solve"
        ),
        ordinal=0,
    )
    converted = to_trl_environment_rows([row])[0]
    audit_path = tmp_path / "strict-tool-rejection.jsonl"
    monkeypatch.setenv("AGENTIC_GRPO_AUDIT_PATH", str(audit_path))
    environment = build_trl_environment_factories("react")[converted["environment"]]()
    environment.reset(**converted)

    assert canonical_trl_tool_schemas([environment.retry_solve]) == [
        policy_action_schema("retry_solve")
    ]
    before = len(environment._require_session().recorder.episode.steps)
    raw_arguments = {
        "strategy": "greedy",
        "reason": "当前路线仍可重排",
        "candidates": ["untrusted-poi"],
    }
    with pytest.raises(
        PolicyArgumentValidationError,
        match="UNEXPECTED_ARGUMENT:candidates",
    ):
        environment.retry_solve(**raw_arguments)
    after = len(environment._require_session().recorder.episode.steps)
    assert environment.get_reward() == -1.0
    reward = environment.rollout_record.reward

    assert after == before
    assert reward.audit_metrics["decision_single_policy_call"] is True
    assert reward.audit_metrics["decision_no_policy_call_rejections"] is False
    assert reward.audit_metrics["decision_policy_call_attempt_count"] == 1
    assert reward.audit_metrics["decision_policy_call_rejection_count"] == 1
    assert reward.audit_metrics["decision_policy_argument_rejection_count"] == 1
    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    rejected = next(item for item in records if item["event"] == "policy_argument_rejected")
    assert rejected["task_id"] == row.task.task_id
    assert rejected["initial_state_fingerprint"] == converted["initial_state_fingerprint"]
    assert rejected["submitted"] == {
        "action": "retry_solve",
        "arguments": raw_arguments,
    }
    assert rejected["rejection"]["parsed_tool_call"] == {
        "type": "function",
        "function": {"name": "retry_solve", "arguments": raw_arguments},
    }
    assert {
        key: rejected["rejection"][key]
        for key in (
            "parse_valid",
            "schema_valid",
            "rejection_code",
            "execution_attempted",
            "execution_valid",
            "execution_status",
        )
    } == {
        "parse_valid": True,
        "schema_valid": False,
        "rejection_code": "UNEXPECTED_ARGUMENT:candidates",
        "execution_attempted": False,
        "execution_valid": None,
        "execution_status": "not_attempted",
    }
    assert rejected["rejection"]["raw_arguments_truncated"] is False
    assert rejected["rejection"]["redacted_argument_paths"] == []
    assert rejected["rejection"]["validation_errors"][0] == {
        "type": "extra_forbidden",
        "loc": ["candidates"],
        "msg": "Extra inputs are not permitted",
    }


@pytest.mark.parametrize("tail_schema_valid", [False, True])
def test_verifier_repair_reward_hard_fails_and_does_not_execute_a_tail_call(
    tail_schema_valid,
):
    row = _prepare_variant(
        _source(),
        split="validation",
        template=next(
            item
            for item in _TEMPLATES["validation"]
            if item["target_action"] == "retry_solve"
        ),
        ordinal=0,
    )
    converted = to_trl_environment_rows([row])[0]
    environment = build_trl_environment_factories("react")[converted["environment"]](
        audit_enabled=False
    )
    environment.reset(**converted)
    contract = row.snapshot.hidden_test_facts["grpo_decision_state"]
    environment.retry_solve(reason=contract["grounding_phrases"][0])
    after_first = len(environment._require_session().recorder.episode.steps)

    with pytest.raises(ValueError, match="TOOL_CALL_CARDINALITY_INVALID"):
        if tail_schema_valid:
            environment.propose_tradeoff(
                reason=contract["grounding_phrases"][0],
            )
        else:
            environment.ask_user(
                reason=contract["grounding_phrases"][0],
                options=["提高预算"],
            )

    after = len(environment._require_session().recorder.episode.steps)
    assert after == after_first
    assert environment.get_reward() == -1.0
    reward = environment.rollout_record.reward
    assert reward.audit_metrics["decision_single_policy_call"] is False
    assert reward.audit_metrics["decision_no_policy_call_rejections"] is False
    assert reward.audit_metrics["decision_policy_call_attempt_count"] == 2
    assert reward.audit_metrics["decision_policy_call_rejection_count"] == 1


def test_verifier_repair_invalid_first_and_valid_second_executes_neither():
    row = _prepare_variant(
        _source(),
        split="validation",
        template=next(
            item
            for item in _TEMPLATES["validation"]
            if item["target_action"] == "retry_solve"
        ),
        ordinal=0,
    )
    converted = to_trl_environment_rows([row])[0]
    environment = build_trl_environment_factories("react")[converted["environment"]](
        audit_enabled=False
    )
    environment.reset(**converted)
    before = len(environment._require_session().recorder.episode.steps)

    with pytest.raises(PolicyArgumentValidationError):
        environment.retry_solve(
            strategy="greedy",
            reason="可重排",
            candidates=["untrusted-poi"],
        )
    with pytest.raises(ValueError, match="TOOL_CALL_CARDINALITY_INVALID"):
        environment.retry_solve(reason="可重排")

    assert len(environment._require_session().recorder.episode.steps) == before
    assert environment.get_reward() == -1.0


def test_verifier_repair_zero_tool_calls_is_a_hard_failure():
    row = _prepare_variant(
        _source(),
        split="validation",
        template=next(
            item
            for item in _TEMPLATES["validation"]
            if item["target_action"] == "retry_solve"
        ),
        ordinal=0,
    )
    converted = to_trl_environment_rows([row])[0]
    environment = build_trl_environment_factories("react")[converted["environment"]](
        audit_enabled=False
    )
    environment.reset(**converted)

    assert environment.get_reward() == -1.0
    reward = environment.rollout_record.reward
    assert reward.audit_metrics["decision_single_policy_call"] is False
    assert reward.audit_metrics["decision_policy_call_attempt_count"] == 0


def test_each_verifier_repair_target_passes_only_with_grounded_arguments():
    source = _source()
    rows = [
        _prepare_variant(
            source,
            split="validation",
            template=template,
            ordinal=index,
        )
        for index, template in enumerate(_TEMPLATES["validation"])
    ]
    factories = build_trl_environment_factories("react")
    for row in rows:
        converted = to_trl_environment_rows([row])[0]
        environment = factories[converted["environment"]](audit_enabled=False)
        environment.reset(**converted)
        contract = row.snapshot.hidden_test_facts["grpo_decision_state"]
        target = contract["target_action"]
        reason = _teacher_reason(row)
        if target == "retry_solve":
            environment.retry_solve(reason=reason)
        elif target == "propose_tradeoff":
            environment.propose_tradeoff(
                reason=reason,
            )
        else:
            environment.abort(reason=reason)

        assert environment.get_reward() == 1.0
        rollout = environment.rollout_record
        assert rollout.reward.audit_metrics["decision_target_action"] == target
        assert rollout.reward.components.task == 1.0
        assert rollout.reward.components.constraint == 1.0
        assert rollout_action_rows(rollout, policy_inference_metrics=[{}])[0][
            "action"
        ] == target


def test_verifier_repair_reward_gives_verified_partial_argument_credit():
    row = _prepare_variant(
        _source(),
        split="validation",
        template=_TEMPLATES["validation"][0],
        ordinal=0,
    )
    converted = to_trl_environment_rows([row])[0]
    environment = build_trl_environment_factories("react")[converted["environment"]](
        audit_enabled=False
    )
    environment.reset(**converted)

    environment.retry_solve(reason="没有引用可见的验证证据")

    score = environment.get_reward()
    reward = environment.rollout_record.reward
    assert 0 < score < 1
    assert reward.gate_status == "task_failed"
    assert reward.audit_metrics["verified_partial_credit"] is True
    assert reward.audit_metrics["decision_action_match"] is True
    assert reward.audit_metrics["decision_expected_arguments_match"] is True
    assert reward.audit_metrics["decision_controller_arguments_match"] is True
    assert reward.audit_metrics["decision_grounding_match"] is False


def test_tradeoff_options_cannot_substitute_for_a_grounded_reason():
    row = _prepare_variant(
        _source(),
        split="validation",
        template=next(
            item
            for item in _TEMPLATES["validation"]
            if item["target_action"] == "propose_tradeoff"
        ),
        ordinal=0,
    )
    converted = to_trl_environment_rows([row])[0]
    environment = build_trl_environment_factories("react")[converted["environment"]](
        audit_enabled=False
    )
    environment.reset(**converted)
    contract = row.snapshot.hidden_test_facts["grpo_decision_state"]
    environment._decision_contract["grounding_phrases"] = [
        contract["supervised_options"][0]
    ]

    environment.propose_tradeoff(
        reason="需要您确认一个调整方案",
    )

    score = environment.get_reward()
    reward = environment.rollout_record.reward
    assert score < 1
    assert reward.gate_status == "task_failed"
    assert reward.audit_metrics["decision_grounding_match"] is False


def test_tradeoff_raw_controller_field_is_audited_but_not_executed():
    row = _prepare_variant(
        _source(),
        split="validation",
        template=next(
            item
            for item in _TEMPLATES["validation"]
            if item["target_action"] == "propose_tradeoff"
        ),
        ordinal=0,
    )
    converted = to_trl_environment_rows([row])[0]
    environment = build_trl_environment_factories("react")[converted["environment"]](
        audit_enabled=False
    )
    environment.reset(**converted)
    contract = row.snapshot.hidden_test_facts["grpo_decision_state"]
    expected = list(contract["supervised_options"])
    reason = _teacher_reason(row)

    environment.propose_tradeoff(
        reason=reason,
        options=[*expected, "当前状态未授权的额外方案"],
    )

    # Serving remains safe through controller hydration, but the raw model
    # attempted an out-of-contract field and therefore cannot earn reward.
    assert environment.get_reward() == -1.0
    rollout = environment.rollout_record
    assert rollout is not None
    assert rollout.reward.audit_metrics["decision_system_step_valid"] is True
    assert rollout.reward.audit_metrics["decision_step_valid"] is False
    step = [
        item
        for item in rollout.episode.steps
        if item.action.action == "propose_tradeoff"
    ][-1]
    assert step.action.arguments == {"reason": reason, "options": expected}
    assert step.action.model_arguments == {
        "reason": reason,
        "options": [*expected, "当前状态未授权的额外方案"],
    }
    assert step.action.controller_override_attempt is True
    assert step.action.model_contract_compliant is False
    assert step.action.controller_hydration_exact is True


def test_completion_budget_probe_uses_only_model_owned_tradeoff_arguments():
    row = _prepare_variant(
        _source(),
        split="validation",
        template=next(
            item
            for item in _TEMPLATES["validation"]
            if item["target_action"] == "propose_tradeoff"
        ),
        ordinal=0,
    )
    converted = to_trl_environment_rows([row])[0]
    environment = build_trl_environment_factories("react")[converted["environment"]](
        audit_enabled=False
    )
    # A verified replay prompt already ends with the exact review state, so
    # reset returns an empty suffix for TRL.  Read the immutable final replay
    # message rather than treating that empty suffix as a malformed state.
    environment.reset(**converted)
    policy_state = json.loads(
        row.snapshot.hidden_test_facts["grpo_decision_state"]["prompt_messages"][-1][
            "content"
        ]
    )["policy_state"]

    action, arguments = _budget_probe_action(row, policy_state)

    assert action == "propose_tradeoff"
    assert set(arguments) == {"reason"}
    environment.propose_tradeoff(**arguments)
    assert environment.get_reward() == 1.0
    step = [
        item
        for item in environment.rollout_record.episode.steps
        if item.action.action == "propose_tradeoff"
    ][-1]
    assert step.action.model_arguments == arguments
    assert step.action.arguments["options"] == policy_state["capability"]["alternatives"]
    assert step.action.controller_override_attempt is False
    assert step.action.model_contract_compliant is True
    assert step.action.controller_hydration_exact is True


def test_real_full_loop_materializes_capability_and_passes_all_three_targets():
    source = _source()
    for index, template in enumerate(_TEMPLATES["validation"]):
        row = _prepare_variant(
            source,
            split="validation",
            template=template,
            ordinal=index,
        )
        full_row = row.model_copy(deep=True)
        contract = full_row.snapshot.hidden_test_facts.pop("grpo_decision_state")
        environment = TRLReactEnvironment(audit_enabled=False)
        rendered = environment.reset(
            task=full_row.task.model_dump(mode="json"),
            snapshot=full_row.snapshot.model_dump(mode="json"),
        )
        for item in contract["prefix_actions"]:
            rendered = environment._act(item["action"], item["arguments"])

        state = json.loads(rendered)["policy_state"]
        capability = state["capability"]
        target = contract["target_action"]
        reason = contract["grounding_phrases"][0]
        if target == "retry_solve":
            assert capability["status"] == "solvable"
            environment.retry_solve(reason=reason)
        elif target == "propose_tradeoff":
            assert capability["status"] == "infeasible"
            assert capability["actionable_alternatives"] is True
            environment.propose_tradeoff(
                reason=reason,
            )
        else:
            assert capability["status"] == "infeasible"
            assert capability["actionable_alternatives"] is False
            environment.abort(reason=reason)

        assert environment.get_reward() > 0
        reward = environment.rollout_record.reward
        assert reward.gate_status == "passed"
        assert reward.audit_metrics["termination_argument_mismatch"] is False
        if target == "retry_solve":
            assert reward.audit_metrics["hard_pass"] is True
