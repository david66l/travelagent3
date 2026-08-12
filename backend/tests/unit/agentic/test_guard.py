from __future__ import annotations

from agentic.guard import GuardContext, ToolGuard, tool_call_signature


def _call(name: str, arguments: str) -> dict:
    return {"id": "call-1", "function": {"name": name, "arguments": arguments}}


def test_shadow_mode_reports_but_allows_schema_violation() -> None:
    decision = ToolGuard(mode="shadow").evaluate_batch([_call("get_weather", "{}")])[0]

    assert decision.allowed is True
    assert decision.would_block is True
    assert decision.violations[0].code == "INVALID_ARGUMENTS"


def test_enforce_mode_blocks_disallowed_tool() -> None:
    decision = ToolGuard(mode="enforce").evaluate_batch(
        [_call("get_weather", '{"city":"北京"}')],
        GuardContext(allowed_tools={"get_route"}),
    )[0]

    assert decision.allowed is False
    assert {violation.code for violation in decision.violations} == {"TOOL_NOT_ALLOWED"}


def test_guard_detects_duplicate_and_call_budget() -> None:
    args = {"city": "北京"}
    signature = tool_call_signature("get_weather", args)
    decisions = ToolGuard(mode="enforce", max_calls=1).evaluate_batch(
        [
            _call("get_weather", '{"city":"北京"}'),
            _call("get_weather", '{"city":"北京"}'),
        ],
        GuardContext(previous_signatures={signature}),
    )

    first_codes = {violation.code for violation in decisions[0].violations}
    second_codes = {violation.code for violation in decisions[1].violations}
    assert "DUPLICATE_TOOL_CALL" in first_codes
    assert {"DUPLICATE_TOOL_CALL", "CALL_BUDGET_EXCEEDED"} <= second_codes


def test_guard_checks_grounded_city_only_when_context_provides_it() -> None:
    decision = ToolGuard(mode="enforce").evaluate_batch(
        [_call("get_weather", '{"city":"火星"}')],
        GuardContext(grounded_values={"city": {"北京"}}),
    )[0]

    assert decision.allowed is False
    assert decision.violations[0].code == "UNGROUNDED_ARGUMENT"


def test_off_mode_is_fully_compatible() -> None:
    decision = ToolGuard(mode="off").evaluate_batch([_call("unknown", "bad-json")])[0]

    assert decision.allowed is True
    assert decision.violations == []
