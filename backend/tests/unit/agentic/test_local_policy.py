import json

import pytest

from agentic.local_policy import (
    LocalCheckpointAgentPolicy,
    _tokenizer_compatibility_kwargs,
    parse_local_tool_call,
)
from agentic.policy import PolicyOutputError


def test_tokenizer_compatibility_normalizes_legacy_extra_special_tokens(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"extra_special_tokens": ["<|im_start|>"]}),
        encoding="utf-8",
    )

    assert _tokenizer_compatibility_kwargs(tmp_path) == {"extra_special_tokens": {}}


def test_parse_local_tool_call_accepts_native_qwen_envelope():
    action, arguments = parse_local_tool_call(
        '<tool_call>\n{"name":"search_pois","arguments":{"keywords":["history"]}}\n</tool_call>'
    )
    assert action == "search_pois"
    assert arguments == {"keywords": ["history"]}


def test_parse_local_tool_call_accepts_plain_constrained_json():
    action, arguments = parse_local_tool_call(
        '{"name":"ask_user","arguments":{"question":"您的预算是多少？"}}'
    )

    assert action == "ask_user"
    assert arguments == {"question": "您的预算是多少？"}


def test_parse_local_tool_call_rejects_unstructured_prose():
    with pytest.raises(PolicyOutputError):
        parse_local_tool_call("I think we should search next.")


@pytest.mark.parametrize(
    "output",
    [
        (
            '<tool_call>{"name":"search_pois","arguments":{}}</tool_call>'
            '<tool_call>{"name":"ask_user","arguments":{"question":"预算？"}}'
            '</tool_call>'
        ),
        '先搜索 <tool_call>{"name":"search_pois","arguments":{}}</tool_call>',
        (
            '<tool_call>{"name":"search_pois","arguments":{}}</tool_call>'
            ' 然后继续搜索'
        ),
    ],
)
def test_parse_local_tool_call_rejects_multiple_calls_or_surrounding_prose(output):
    with pytest.raises(PolicyOutputError, match="one valid tool call"):
        parse_local_tool_call(output)


def test_multiple_tool_call_failure_keeps_only_sanitized_action_summary():
    output = (
        '<tool_call>{"name":"search_pois","arguments":{}}</tool_call>'
        '<tool_call>{"name":"ask_user","arguments":{"question":"预算？"}}</tool_call>'
    )

    with pytest.raises(PolicyOutputError) as captured:
        parse_local_tool_call(output)

    assert captured.value.output_summary == {
        "tool_call_count": 2,
        "actions": ["search_pois", "ask_user"],
    }


def test_parse_local_tool_call_accepts_one_allowlisted_qwen_terminator():
    action, arguments = parse_local_tool_call(
        '<tool_call>{"name":"search_pois","arguments":{}}</tool_call><|im_end|>'
    )

    assert action == "search_pois"
    assert arguments == {}


@pytest.mark.parametrize(
    "output",
    [
        '{"name":"search_pois","arguments":null}',
        '{"name":"search_pois","arguments":{},"extra":true}',
        '{"name":"search_pois","action":"ask_user","arguments":{}}',
        '[]',
    ],
)
def test_parse_local_tool_call_rejects_non_exact_json_shape(output):
    with pytest.raises(PolicyOutputError, match="tool call"):
        parse_local_tool_call(output)


def test_structured_processor_receives_state_scoped_json_schema():
    class Backend:
        def get_json_schema_logits_processor(self, schema_text):
            return json.loads(schema_text)

    policy = object.__new__(LocalCheckpointAgentPolicy)
    policy._structured_backend = Backend()
    policy.structured_decoding_mode = "json_schema"

    schema = policy._structured_logits_processor(["ask_user"])

    assert schema["oneOf"][0]["properties"]["name"] == {"const": "ask_user"}
    assert set(schema["oneOf"][0]["properties"]["arguments"]["properties"]) == {"question"}


def test_structured_processor_uses_reason_only_tradeoff_schema_and_authority_cache_key():
    class Backend:
        def __init__(self):
            self.schemas = []

        def get_json_schema_logits_processor(self, schema_text):
            schema = json.loads(schema_text)
            self.schemas.append(schema)
            return schema

    backend = Backend()
    policy = object.__new__(LocalCheckpointAgentPolicy)
    policy._structured_backend = backend
    policy._structured_processor_cache = {}
    policy.structured_decoding_mode = "json_schema"
    capability = {
        "status": "infeasible",
        "actionable_alternatives": True,
        "alternatives": ["提高总预算"],
    }

    tradeoff = policy._structured_logits_processor(
        ["propose_tradeoff", "abort"], capability=capability
    )
    no_authority = policy._structured_logits_processor(
        ["propose_tradeoff", "abort"], capability={}
    )

    assert set(tradeoff["oneOf"][0]["properties"]["arguments"]["properties"]) == {
        "reason"
    }
    assert [branch["properties"]["name"]["const"] for branch in no_authority["oneOf"]] == [
        "abort"
    ]
    assert len(backend.schemas) == 2


def test_structured_processor_can_preserve_qwen_tool_envelope(monkeypatch):
    pytest.importorskip(
        "outlines_core.json_schema",
        reason="agentic-training optional dependency",
    )

    class Backend:
        def get_regex_logits_processor(self, regex):
            return regex

    monkeypatch.setattr(
        "outlines_core.json_schema.build_regex_from_schema",
        lambda schema_text: "JSON_SCHEMA_REGEX",
    )
    policy = object.__new__(LocalCheckpointAgentPolicy)
    policy._structured_backend = Backend()
    policy.structured_decoding_mode = "qwen_tool_envelope"

    regex = policy._structured_logits_processor(["ask_user"])

    assert regex == r"<tool_call>\n(JSON_SCHEMA_REGEX)\n</tool_call>"


def test_structured_processor_reuses_compiled_index_and_resets_cursor():
    class Processor:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    class Backend:
        def __init__(self):
            self.compile_count = 0

        def get_json_schema_logits_processor(self, schema_text):
            self.compile_count += 1
            return Processor()

    backend = Backend()
    policy = object.__new__(LocalCheckpointAgentPolicy)
    policy._structured_backend = backend
    policy._structured_processor_cache = {}
    policy.structured_decoding_mode = "json_schema"

    first = policy._structured_logits_processor(["ask_user"])
    second = policy._structured_logits_processor(["ask_user"])

    assert second is first
    assert backend.compile_count == 1
    assert first.reset_count == 1


def test_structured_processor_requires_enabled_backend():
    policy = object.__new__(LocalCheckpointAgentPolicy)
    policy._structured_backend = None
    policy.structured_decoding_mode = "json_schema"

    with pytest.raises(RuntimeError, match="not enabled"):
        policy._structured_logits_processor(["ask_user"])


def test_policy_action_accepts_versioned_inference_metrics():
    from agentic.loop import PolicyAction

    action = PolicyAction(
        action="ask_user",
        inference_metrics={
            "model": "Qwen3-4B",
            "backend": "transformers",
            "thinking_mode": "disabled",
            "prompt_tokens": 100,
            "completion_tokens": 8,
            "request_latency_ms": 125.5,
        },
    )

    assert action.inference_metrics is not None
    assert action.inference_metrics.total_tokens == 108
    assert action.inference_metrics.schema_version == "policy-inference-metrics.v1"
