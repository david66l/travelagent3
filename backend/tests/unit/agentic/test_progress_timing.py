"""Solver telemetry cannot create business progress or be erased from evidence."""
from copy import deepcopy
from types import SimpleNamespace

from agentic.loop import BoundedAgentLoop, PolicyAction
from agentic.observations import ObservationEnvelope


def execution(payload, *, tool="solve_itinerary", artifact_type="solver_result"):
    return SimpleNamespace(
        task_id="solve", action=PolicyAction(action="solve_itinerary", arguments={}),
        outcome=SimpleNamespace(
            status="completed", facts=[],
            artifacts=[SimpleNamespace(artifact_type=artifact_type, payload=payload)],
            observations=[ObservationEnvelope(ok=True, tool=tool, data=payload, source="built_in", confidence=1)],
        ),
    )


def test_solver_time_alone_is_not_progress_and_raw_evidence_is_preserved():
    ledger = SimpleNamespace(decision_history=[])
    a = {"status": "optimal", "days": [{"cost": 100}], "solve_time_ms": 42}
    b = {**deepcopy(a), "solve_time_ms": 39}
    original = deepcopy(b)
    assert BoundedAgentLoop._record_decision(ledger, execution(a))
    assert not BoundedAgentLoop._record_decision(ledger, execution(b))
    assert ledger.decision_history[-1].progress_made is False
    assert b == original


def test_business_result_change_still_counts_as_progress():
    ledger = SimpleNamespace(decision_history=[])
    a = {"status": "optimal", "days": [{"cost": 100}], "solve_time_ms": 42}
    b = {"status": "optimal", "days": [{"cost": 90}], "solve_time_ms": 42}
    assert BoundedAgentLoop._record_decision(ledger, execution(a))
    assert BoundedAgentLoop._record_decision(ledger, execution(b))


def test_same_named_field_in_unrelated_tool_is_not_ignored():
    ledger = SimpleNamespace(decision_history=[])
    assert BoundedAgentLoop._record_decision(ledger, execution({"solve_time_ms": 42}, tool="search_current_info", artifact_type="search_result"))
    assert BoundedAgentLoop._record_decision(ledger, execution({"solve_time_ms": 39}, tool="search_current_info", artifact_type="search_result"))


def test_missing_solver_metric_does_not_create_progress():
    ledger = SimpleNamespace(decision_history=[])
    assert BoundedAgentLoop._record_decision(ledger, execution({"status": "optimal", "solve_time_ms": 42}))
    assert not BoundedAgentLoop._record_decision(ledger, execution({"status": "optimal"}))
