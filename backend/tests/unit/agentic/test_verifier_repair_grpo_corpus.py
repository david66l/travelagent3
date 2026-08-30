"""Verifier-repair GRPO rows must replay production review states safely."""

import json
from pathlib import Path

from agentic.environment import environment_fingerprint
from agentic.grpo_training import load_grpo_corpus, to_trl_environment_rows
from agentic.trl_environment import build_trl_environment_factories
from scripts.audit_model_curriculum import rollout_action_rows
from scripts.build_verifier_repair_grpo_corpus import _TEMPLATES, _prepare_variant


def _source():
    return load_grpo_corpus(
        Path("ml/agentic/datasets/native-react-grpo-v1/train.jsonl")
    )[0]


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

    assert len({row.task.task_id for row in rows}) == 3
    assert len({environment_fingerprint(row.task, row.snapshot) for row in rows}) == 3
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
        assert state["source_task_id"] == source.task.task_id
        visible = json.dumps(prompt, ensure_ascii=False)
        assert any(phrase in visible for phrase in state["grounding_phrases"])


def test_verifier_repair_environment_exposes_the_production_review_action_space():
    row = _prepare_variant(
        _source(),
        split="validation",
        template=_TEMPLATES["validation"][0],
        ordinal=0,
    )
    converted = to_trl_environment_rows([row])[0]
    environment = build_trl_environment_factories("react")[converted["environment"]]()

    environment.reset(**converted)

    expected = set(
        row.snapshot.hidden_test_facts["grpo_decision_state"]["review_allowed_actions"]
    )
    exposed = {
        name
        for name in expected
        if callable(getattr(environment, name, None))
    }
    assert exposed == expected


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
        reason = contract["grounding_phrases"][0]
        if target == "retry_solve":
            environment.retry_solve(strategy="greedy", reason=reason)
        elif target == "propose_tradeoff":
            environment.propose_tradeoff(reason=reason, options=["调整一个冲突约束"])
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

    environment.retry_solve(strategy="greedy", reason="没有引用可见的验证证据")

    score = environment.get_reward()
    reward = environment.rollout_record.reward
    assert 0 < score < 1
    assert reward.gate_status == "task_failed"
    assert reward.audit_metrics["verified_partial_credit"] is True
    assert reward.audit_metrics["decision_action_match"] is True
    assert reward.audit_metrics["decision_expected_arguments_match"] is True
    assert reward.audit_metrics["decision_grounding_match"] is False
