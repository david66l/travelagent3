import json

import pytest

from agentic.local_policy import LocalCheckpointAgentPolicy, parse_local_tool_call
from agentic.policy import PolicyOutputError


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


def test_structured_processor_can_preserve_qwen_tool_envelope(monkeypatch):
    pytest.importorskip("outlines_core", reason="agentic-training optional dependency")

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
