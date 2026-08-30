"""Tests for Agentic GRPO corpus gates and TRL row conversion."""

import json
import inspect
from pathlib import Path

from agentic.grpo_training import (
    GRPOCorpusRow,
    MIN_STATEFUL_COMPLETION_LENGTH,
    estimate_stateful_completion_budget,
    episode_to_grpo_corpus_row,
    load_grpo_corpus,
    preflight_grpo_corpus,
    tool_result_suffix_ids,
    to_trl_environment_rows,
)
from agentic.trl_environment import TRL_ENVIRONMENT_FACTORIES, build_trl_environment_factories
from agentic.environment import TravelAgentEnvironment
from tests.unit.agentic.test_environment import FirstAllowedPolicy
from tests.unit.agentic.test_environment import _snapshot, _task


def _write(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_environment_rows_keep_snapshot_out_of_model_prompt():
    row = GRPOCorpusRow(task=_task(), snapshot=_snapshot())

    converted = to_trl_environment_rows([row])[0]

    assert [message["role"] for message in converted["prompt"]] == ["system", "user"]
    assert converted["prompt"][-1]["content"] == row.task.user_request
    assert "hidden_test_facts" not in json.dumps(converted["prompt"])
    assert converted["snapshot"]["hidden_test_facts"] == {"closed_pois": []}
    assert converted["initial_state_fingerprint"]
    assert converted["environment"] == "search"
    assert converted["rollout_contract"] == "fresh_ledger_no_teacher_prefix.v1"


def test_environment_row_starts_a_fresh_policy_driven_ledger():
    converted = to_trl_environment_rows([GRPOCorpusRow(task=_task(), snapshot=_snapshot())])[0]
    environment = TRL_ENVIRONMENT_FACTORIES[converted["environment"]]()

    initial = json.loads(environment.reset(**converted))

    assert environment.execution_mode == "policy_driven"
    assert initial["policy_state"]["original_request"] == _task().user_request
    assert initial["policy_state"]["current_subtask"]["task_id"] == "capability_check"
    assert environment._session.recorder.episode.steps == []
    environment.get_reward()


def test_environment_rows_route_policy_tool_schema_by_task_type():
    normal = GRPOCorpusRow(task=_task(), snapshot=_snapshot())
    missing_task = _task().model_copy(update={"task_id": "missing", "missing_slots": ["budget"]})
    missing = GRPOCorpusRow(task=missing_task, snapshot=_snapshot())
    infeasible_task = _task().model_copy(
        update={"task_id": "infeasible", "feasibility_report": {"feasible": False}}
    )
    infeasible = GRPOCorpusRow(task=infeasible_task, snapshot=_snapshot())
    current_task = _task().model_copy(
        update={
            "task_id": "current",
            "slots": {**_task().slots, "information_needs": ["opening_hours"]},
        }
    )
    current = GRPOCorpusRow(task=current_task, snapshot=_snapshot())

    converted = to_trl_environment_rows([normal, missing, infeasible, current])

    assert [row["environment"] for row in converted] == [
        "search",
        "clarification",
        "tradeoff",
        "search_current",
    ]


def test_react_environment_matches_production_hybrid_decision_boundary():
    factories = build_trl_environment_factories("react")
    environment = factories["search"]()

    assert environment.execution_mode == "react"
    assert callable(environment.retrieve_city_knowledge)
    assert callable(environment.retry_solve)
    assert callable(environment.propose_tradeoff)
    assert callable(environment.abort)
    assert not hasattr(environment, "search_current_info")
    assert not hasattr(environment, "search_transport")
    assert not hasattr(environment, "finalize_research")
    assert not hasattr(environment, "solve_itinerary")
    assert not hasattr(environment, "validate_itinerary")
    assert not hasattr(environment, "finish")

    converted = to_trl_environment_rows([GRPOCorpusRow(task=_task(), snapshot=_snapshot())])[0]
    initial = json.loads(environment.reset(**converted))
    assert initial["policy_state"]["current_subtask"]["task_id"] == "research_evidence"
    assert "search_current_info" not in initial["policy_state"]["allowed_actions"]
    assert "propose_tradeoff" not in initial["policy_state"]["allowed_actions"]
    assert environment._session.recorder.episode.steps == []
    assert "default" not in inspect.signature(environment.retrieve_city_knowledge).parameters
    transition = json.loads(environment.retrieve_city_knowledge(default=None))
    assert transition["policy_state"]["allowed_actions"]
    environment.get_reward()

    current = factories["search_current"]()
    assert callable(current.search_current_info)
    assert not hasattr(current, "search_transport")


def test_react_verifier_repair_state_replays_to_review_and_scores_grounded_choice():
    row = load_grpo_corpus(
        Path("ml/agentic/datasets/native-react-grpo-v1/train.jsonl")
    )[0]
    snapshot = row.snapshot.model_copy(deep=True)
    report = snapshot.tool_responses["validate_itinerary"][0].data
    report["hard_pass"] = False
    report["hard_violations"] = [
        {
            "code": "BUDGET_EXCEEDED",
            "message": "预算超出300元，当前排程不能通过硬约束校验",
        }
    ]
    snapshot.hidden_test_facts["grpo_decision_state"] = {
        "schema_version": "react-verifier-repair-decision.v1",
        "target_action": "propose_tradeoff",
        "expected_arguments": {},
        "grounding_phrases": ["预算超出300元"],
        "require_options": True,
        "prefix_actions": [
            {"action": "retrieve_city_knowledge", "arguments": {}},
            {
                "action": "search_pois",
                "arguments": {"keywords": row.task.profile.get("interests") or []},
            },
            {"action": "get_poi_detail", "arguments": {}},
        ],
    }
    converted = to_trl_environment_rows([GRPOCorpusRow(task=row.task, snapshot=snapshot)])[0]

    assert converted["environment"] == "decision_verifier_repair"
    environment = build_trl_environment_factories("react")[converted["environment"]]()
    initial = json.loads(environment.reset(**converted))
    assert initial["policy_state"]["current_subtask"]["task_id"] == "review_itinerary"
    assert "propose_tradeoff" in initial["policy_state"]["allowed_actions"]
    terminal = json.loads(
        environment.propose_tradeoff(
            reason="预算超出300元，建议调整预算或减少一个景点",
            options=["提高预算300元", "减少一个景点"],
        )
    )

    assert terminal["done"] is True
    assert terminal["decision_complete"] is True
    assert environment.get_reward() == 1.0
    assert environment.rollout_record.reward.gate_status == "passed"


def test_react_verifier_repair_state_rejects_ungrounded_arguments():
    row = load_grpo_corpus(
        Path("ml/agentic/datasets/native-react-grpo-v1/train.jsonl")
    )[0]
    snapshot = row.snapshot.model_copy(deep=True)
    report = snapshot.tool_responses["validate_itinerary"][0].data
    report["hard_pass"] = False
    report["hard_violations"] = [
        {
            "code": "BUDGET_EXCEEDED",
            "message": "预算超出300元，当前排程不能通过硬约束校验",
        }
    ]
    snapshot.hidden_test_facts["grpo_decision_state"] = {
        "schema_version": "react-verifier-repair-decision.v1",
        "target_action": "propose_tradeoff",
        "grounding_phrases": ["预算超出300元"],
        "require_options": True,
        "prefix_actions": [
            {"action": "retrieve_city_knowledge", "arguments": {}},
            {
                "action": "search_pois",
                "arguments": {"keywords": row.task.profile.get("interests") or []},
            },
            {"action": "get_poi_detail", "arguments": {}},
        ],
    }
    converted = to_trl_environment_rows([GRPOCorpusRow(task=row.task, snapshot=snapshot)])[0]
    environment = build_trl_environment_factories("react")[converted["environment"]]()
    environment.reset(**converted)
    environment.propose_tradeoff(
        reason="我觉得换个方案更好",
        options=["随便改一下"],
    )

    score = environment.get_reward()
    assert 0 < score < 1
    assert environment.rollout_record.reward.gate_status == "task_failed"
    assert environment.rollout_record.reward.audit_metrics[
        "decision_grounding_match"
    ] is False


def test_react_environment_normalizes_arrow_null_tool_lists():
    converted = to_trl_environment_rows([GRPOCorpusRow(task=_task(), snapshot=_snapshot())])[0]
    converted["snapshot"]["tool_responses"]["search_current_info"] = None
    environment = build_trl_environment_factories("react")["search"]()

    initial = json.loads(environment.reset(**converted))

    assert initial["policy_state"]["original_request"] == _task().user_request
    assert environment._snapshot.tool_responses["search_current_info"] == []
    environment.get_reward()


def test_react_decision_state_replays_hidden_prefix_and_scores_one_tool_call():
    snapshot = _snapshot().model_copy(deep=True)
    snapshot.hidden_test_facts["grpo_decision_state"] = {
        "schema_version": "react-decision-state.v1",
        "target_action": "get_poi_detail",
        "prefix_actions": [
            {"action": "retrieve_city_knowledge", "arguments": {}},
            {"action": "search_pois", "arguments": {"keywords": ["museum"]}},
        ],
    }
    row = GRPOCorpusRow(task=_task(), snapshot=snapshot)
    converted = to_trl_environment_rows([row])[0]
    assert converted["environment"] == "decision_get_poi_detail"
    environment = build_trl_environment_factories("react")[converted["environment"]]()

    initial = json.loads(environment.reset(**converted))
    assert set(initial["policy_state"]["allowed_actions"]) == {
        "retrieve_city_knowledge",
        "get_poi_detail",
        "get_route_matrix",
    }
    terminal = json.loads(environment.get_poi_detail())

    assert terminal["done"] is True
    assert terminal["decision_complete"] is True
    assert environment.get_reward() == 1.0
    assert environment.rollout_record.reward.episode_reward == 1.0
    assert environment.rollout_record.reward.gate_status == "passed"


class _CharacterChatTokenizer:
    eos_token_id = 999

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        tokenize,
        return_dict,
        **_,
    ):
        assert tokenize is True
        assert return_dict is False
        tokens = []
        for message in messages:
            rendered = json.dumps(message, ensure_ascii=False, sort_keys=True)
            tokens.extend(ord(char) for char in rendered)
            tokens.append(self.eos_token_id)
        if add_generation_prompt:
            tokens.append(1000)
        return tokens


class _ConditionalThinkingTokenizer(_CharacterChatTokenizer):
    """Model Qwen's different rendering when a tool call is the final message."""

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        tokenize,
        return_dict,
        **_,
    ):
        assert tokenize is True
        assert return_dict is False
        tokens = []
        for index, message in enumerate(messages):
            if (
                message.get("role") == "assistant"
                and message.get("tool_calls")
                and index == len(messages) - 1
            ):
                tokens.append(777)
            rendered = json.dumps(message, ensure_ascii=False, sort_keys=True)
            tokens.extend(ord(char) for char in rendered)
            tokens.append(self.eos_token_id)
        if add_generation_prompt:
            tokens.append(1000)
        return tokens


def test_completion_budget_measures_real_stateful_tool_suffix():
    row = GRPOCorpusRow(task=_task(), snapshot=_snapshot())

    report = estimate_stateful_completion_budget(
        [row],
        _CharacterChatTokenizer(),
        TRL_ENVIRONMENT_FACTORIES,
    )

    assert report.sampled_tasks == 1
    assert report.max_tool_result_tokens > 100
    assert report.minimum_completion_length >= MIN_STATEFUL_COMPLETION_LENGTH
    assert report.max_observed_policy_turns == 11
    assert report.limiting_task_id == row.task.task_id
    assert report.limiting_environment == "search"


def test_tool_suffix_uses_full_assistant_boundary_when_prefix_render_changes():
    tokenizer = _ConditionalThinkingTokenizer()
    tool_messages = [{"role": "tool", "name": "search_pois", "content": '{"ok":true}'}]

    suffix = tool_result_suffix_ids(
        tokenizer,
        tool_messages=tool_messages,
        chat_template_kwargs={"enable_thinking": False},
    )

    assert suffix
    assert suffix[-1] == 1000
    assert 777 not in suffix


def test_completion_budget_does_not_write_rollout_audit(tmp_path, monkeypatch):
    audit_path = tmp_path / "rollouts.jsonl"
    monkeypatch.setenv("AGENTIC_GRPO_AUDIT_PATH", str(audit_path))

    estimate_stateful_completion_budget(
        [GRPOCorpusRow(task=_task(), snapshot=_snapshot())],
        _CharacterChatTokenizer(),
        TRL_ENVIRONMENT_FACTORIES,
    )

    assert not audit_path.exists()


def test_preflight_accepts_complete_non_overlapping_snapshot_corpus(tmp_path):
    train_task = _task()
    validation_task = _task().model_copy(update={"task_id": "validation-task", "seed": 99})
    _write(
        tmp_path / "train.jsonl",
        [
            {
                "task": train_task.model_dump(mode="json"),
                "snapshot": _snapshot().model_dump(mode="json"),
            }
        ],
    )
    validation_snapshot = _snapshot().model_copy(update={"state_id": "validation-state"})
    _write(
        tmp_path / "validation.jsonl",
        [
            {
                "task": validation_task.model_dump(mode="json"),
                "snapshot": validation_snapshot.model_dump(mode="json"),
            }
        ],
    )

    report = preflight_grpo_corpus(tmp_path, minimum_train_tasks=1, require_dependencies=False)

    assert report.ready is True
    assert report.train_tasks == 1
    assert report.validation_tasks == 1


def test_preflight_blocks_incomplete_infeasible_decision_contracts(tmp_path):
    invalid_reports = [
        {"feasible": False},
        {
            "feasible": False,
            "status": "unsupported",
            "reasons": ["required provider is unavailable"],
            "actionable_alternatives": True,
            "alternatives": ["use another provider"],
        },
        {
            "feasible": False,
            "status": "infeasible",
            "reasons": ["budget is too low"],
            "actionable_alternatives": True,
            "alternatives": [],
        },
        {
            "feasible": False,
            "status": "unsafe",
            "reasons": ["constraints conflict"],
            "actionable_alternatives": False,
            "alternatives": ["ignore the locked constraint"],
        },
        {
            "feasible": False,
            "status": "missing_tool",
            "reasons": ["live inventory is required"],
            "actionable_alternatives": True,
            "alternatives": ["change venue", "change venue"],
        },
    ]
    train_rows = []
    for index, feasibility_report in enumerate(invalid_reports):
        task = _task().model_copy(
            update={
                "task_id": f"invalid-infeasible-{index}",
                "seed": index,
                "feasibility_report": feasibility_report,
            }
        )
        snapshot = _snapshot().model_copy(update={"state_id": f"invalid-state-{index}"})
        train_rows.append(
            {
                "task": task.model_dump(mode="json"),
                "snapshot": snapshot.model_dump(mode="json"),
            }
        )
    _write(tmp_path / "train.jsonl", train_rows)

    validation_task = _task().model_copy(update={"task_id": "clean-validation", "seed": 99})
    validation_snapshot = _snapshot().model_copy(update={"state_id": "clean-validation-state"})
    _write(
        tmp_path / "validation.jsonl",
        [
            {
                "task": validation_task.model_dump(mode="json"),
                "snapshot": validation_snapshot.model_dump(mode="json"),
            }
        ],
    )

    report = preflight_grpo_corpus(tmp_path, minimum_train_tasks=1, require_dependencies=False)

    assert report.ready is False
    assert any(error.startswith("INFEASIBLE_STATUS_INVALID:") for error in report.errors)
    assert any(error.startswith("INFEASIBLE_ACTIONABLE_FLAG_MISSING:") for error in report.errors)
    assert any(error.startswith("INFEASIBLE_REASONS_EMPTY:") for error in report.errors)
    assert any(
        error.startswith("INFEASIBLE_ACTIONABLE_ALTERNATIVES_EMPTY:") for error in report.errors
    )
    assert any(
        error.startswith("INFEASIBLE_NONACTIONABLE_ALTERNATIVES_PRESENT:")
        for error in report.errors
    )
    assert any(error.startswith("INFEASIBLE_ALTERNATIVES_DUPLICATED:") for error in report.errors)


def test_preflight_blocks_split_leakage_missing_tools_and_pii(tmp_path):
    task = _task()
    task.user_request = "Call 13812345678"
    snapshot = _snapshot()
    snapshot.tool_responses.pop("validate_itinerary")
    row = {"task": task.model_dump(mode="json"), "snapshot": snapshot.model_dump(mode="json")}
    _write(tmp_path / "train.jsonl", [row])
    _write(tmp_path / "validation.jsonl", [row])

    report = preflight_grpo_corpus(tmp_path, minimum_train_tasks=2, require_dependencies=False)

    assert report.ready is False
    assert "TASK_ID_SPLIT_OVERLAP" in report.errors
    assert "INITIAL_STATE_SPLIT_OVERLAP" in report.errors
    assert any(error.startswith("PII_DETECTED") for error in report.errors)
    assert any(error.startswith("SNAPSHOT_TOOLS_MISSING") for error in report.errors)
    assert any(error.startswith("TRAIN_TASKS_BELOW_MINIMUM") for error in report.errors)


def test_preflight_blocks_unicode_replacement_character(tmp_path):
    train_task = _task()
    train_task.task_id = "broken-train"
    train_task.user_request = "Plan � trip"
    validation_task = _task()
    validation_task.task_id = "clean-validation"
    _write(
        tmp_path / "train.jsonl",
        [
            {
                "task": train_task.model_dump(mode="json"),
                "snapshot": _snapshot().model_dump(mode="json"),
            }
        ],
    )
    validation_snapshot = _snapshot()
    validation_snapshot.state_id = "validation-state"
    _write(
        tmp_path / "validation.jsonl",
        [
            {
                "task": validation_task.model_dump(mode="json"),
                "snapshot": validation_snapshot.model_dump(mode="json"),
            }
        ],
    )

    report = preflight_grpo_corpus(tmp_path, minimum_train_tasks=1, require_dependencies=False)

    assert any(error.startswith("TEXT_ENCODING_CORRUPT") for error in report.errors)


async def test_real_episode_can_become_isolated_grpo_snapshot():
    rollout = await TravelAgentEnvironment(_task(), _snapshot()).rollout(FirstAllowedPolicy())

    row = episode_to_grpo_corpus_row(
        rollout.episode,
        task_id="episode-task",
        template_family="normal-city-trip",
        seed=7,
    )

    assert row.task.task_id == "episode-task"
    assert row.task.feasibility_report["status"] == "solvable"
    assert row.snapshot.snapshot_version.startswith("episode-")
    assert row.snapshot.hidden_test_facts["source_content_hash"] == rollout.episode.content_hash
    assert set(row.snapshot.tool_responses) >= {
        "get_weather",
        "search_pois",
        "get_poi_detail",
        "get_route_matrix",
        "solve_itinerary",
        "validate_itinerary",
    }
