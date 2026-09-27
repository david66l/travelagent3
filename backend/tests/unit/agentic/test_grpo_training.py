"""Tests for Agentic GRPO corpus gates and TRL row conversion."""

import json

import pytest

from agentic.grpo_training import (
    AUTHORITY_PAYLOAD_ENCODING,
    GRPOCorpusRow,
    decode_authority_payload,
    encode_authority_payload,
    episode_to_grpo_corpus_row,
    preflight_grpo_corpus,
    tool_result_suffix_ids,
    to_trl_environment_rows,
)
from agentic.trl_environment import build_trl_environment_factories
from agentic.environment import TravelAgentEnvironment
from tests.unit.agentic.test_environment import FirstAllowedPolicy
from tests.unit.agentic.test_environment import _snapshot, _task


def _write(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


class _OneDecisionEnvironment:
    _single_decision_tool_contract = True

    def __init__(self):
        self.done = False
        self.executed = []
        self.rejections = []

    def retry_solve(self, **arguments):
        self.executed.append(("retry_solve", arguments))
        self.done = True
        return '{"done":true}'

    def ask_user(self, **arguments):
        self.executed.append(("ask_user", arguments))
        self.done = True
        return '{"done":true}'

    def _trl_tool_loop_done(self):
        return self.done

    def _reject_policy_call_batch(self, tool_calls, *, rejection_code):
        self.rejections.append((tool_calls, rejection_code))
        self.done = True


def test_environment_rows_keep_snapshot_out_of_model_prompt():
    row = GRPOCorpusRow(task=_task(), snapshot=_snapshot())

    converted = to_trl_environment_rows([row])[0]

    assert [message["role"] for message in converted["prompt"]] == ["system", "user"]
    assert converted["prompt"][-1]["content"] == row.task.user_request
    assert "hidden_test_facts" not in json.dumps(converted["prompt"])
    assert isinstance(converted["task"], str)
    assert isinstance(converted["snapshot"], str)
    assert converted["authority_payload_encoding"] == AUTHORITY_PAYLOAD_ENCODING
    assert json.loads(converted["task"]) == row.task.model_dump(mode="json")
    assert json.loads(converted["snapshot"])["hidden_test_facts"] == {"closed_pois": []}
    assert converted["task"] == encode_authority_payload(row.task.model_dump(mode="json"))
    assert converted["snapshot"] == encode_authority_payload(row.snapshot.model_dump(mode="json"))
    assert converted["initial_state_fingerprint"]
    assert converted["environment"] == "travel"
    assert converted["rollout_contract"] == "fresh_ledger_no_teacher_prefix.v1"


def _tool_call(name, arguments):
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


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

    assert [row["environment"] for row in converted] == ["travel"] * 4


def test_environment_exposes_the_full_shared_harness_tool_surface():
    from agentic.policy_actions import POLICY_ACTION_MODELS

    environment = build_trl_environment_factories()["travel"]()
    assert all(callable(getattr(environment, action)) for action in POLICY_ACTION_MODELS)
    assert not hasattr(environment, "retry_solve")


def test_react_environment_keeps_legacy_dict_snapshot_compatibility():
    task = _task().model_dump(mode="json")
    snapshot = _snapshot().model_dump(mode="json")
    snapshot["tool_responses"]["search_current_info"] = None
    environment = build_trl_environment_factories("react")["travel"]()

    initial = json.loads(environment.reset(task=task, snapshot=snapshot))

    assert initial["policy_state"]["original_request"] == _task().user_request
    assert environment._snapshot.tool_responses["search_current_info"] == []
    environment.get_reward()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{ "task_id": "not-canonical"}', "not canonical JSON"),
        ('["not-an-object"]', "must decode to a JSON object"),
        ('{"value":NaN}', "not valid strict JSON"),
    ],
)
def test_authority_payload_decoder_fails_closed(payload, message):
    with pytest.raises(ValueError, match=message):
        decode_authority_payload(payload, field="task")


def test_environment_reset_rejects_mixed_authority_payload_representations():
    converted = to_trl_environment_rows([GRPOCorpusRow(task=_task(), snapshot=_snapshot())])[0]
    converted["snapshot"] = json.loads(converted["snapshot"])
    environment = build_trl_environment_factories("react")["travel"]()

    with pytest.raises(ValueError, match="must use the same representation"):
        environment.reset(**converted)


def test_environment_reset_rejects_encoding_marker_on_legacy_dict_payloads():
    environment = build_trl_environment_factories("react")["travel"]()

    with pytest.raises(ValueError, match="must not declare an encoding"):
        environment.reset(
            task=_task().model_dump(mode="json"),
            snapshot=_snapshot().model_dump(mode="json"),
            authority_payload_encoding=AUTHORITY_PAYLOAD_ENCODING,
        )


def test_environment_reset_rejects_null_inside_canonical_snapshot():
    converted = to_trl_environment_rows([GRPOCorpusRow(task=_task(), snapshot=_snapshot())])[0]
    snapshot = decode_authority_payload(converted["snapshot"], field="snapshot")
    snapshot["tool_responses"]["search_current_info"] = None
    converted["snapshot"] = encode_authority_payload(snapshot)
    environment = build_trl_environment_factories("react")["travel"]()

    with pytest.raises(ValueError, match="search_current_info"):
        environment.reset(**converted)


def test_legacy_decision_prefix_is_rejected():
    row = GRPOCorpusRow(task=_task(), snapshot=_snapshot())
    row.snapshot.hidden_test_facts["grpo_decision_state"] = {"target_action": "abort"}
    with pytest.raises(ValueError, match="retired"):
        to_trl_environment_rows([row])


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
    observed_tools = {o.tool for step in rollout.episode.steps for o in step.observations}
    assert set(row.snapshot.tool_responses) == observed_tools
