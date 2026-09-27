"""Recovery captures must retain real state and provider progress without hints."""

import json

import pytest

from scripts.build_harness_correction_tasks import build_correction_tasks
from scripts.collect_harness_student import RecoverySnapshotRecorder
from scripts.evaluate_harness_base import (
    BoundedAgentLoop,
    FrozenResearchExecutor,
    MOMENT,
    PolicyAction,
    TravelActionExecutor,
    frozen_reference_time,
    initial,
)
from agentic.trajectory import _canonical_hash


def test_new_tasks_preserve_training_family_and_constraints():
    cases = build_correction_tasks()
    assert len(cases) == len({c["id"] for c in cases}) == 24
    assert {c["split"] for c in cases} == {"train"}
    assert len({c["source_group"] for c in cases}) == 6
    for case in cases:
        assert str(case["slots"]["budget_range"]) in case["request"]
        assert case["slots"]["start_date"] in case["request"]
        assert all(p["name"] in case["request"] for p in case["pois"][:2])
        if case["scene_template"] == "combined_ticket_floor":
            prices = [p["ticket_price"] for p in case["pois"][:2]]
            assert max(prices) <= case["slots"]["budget_range"] < sum(prices)


@pytest.mark.asyncio
async def test_snapshot_matches_committed_state_and_fault_counter(tmp_path):
    case = next(c for c in build_correction_tasks() if c["family"] == "recovery")
    backend = FrozenResearchExecutor(case)
    actions = iter(["search_pois", "get_poi_detail"])

    class Policy:
        async def propose(self, context):
            return PolicyAction(action=next(actions))

    with frozen_reference_time(MOMENT):
        state = initial(case)
        recorder = RecoverySnapshotRecorder(
            state,
            backend=backend,
            output=tmp_path,
            environment_version="test",
            validator_version="test",
            policy_name="scripted",
            policy_version="test",
        )
        await BoundedAgentLoop().run(
            state,
            policy=Policy(),
            executor=TravelActionExecutor(backend),
            recorder=recorder,
            max_batches=2,
        )
    saved = json.loads((tmp_path / "state-after-001.json").read_text(encoding="utf-8"))
    assert (
        _canonical_hash(saved["state"])
        == saved["state_after_hash"]
        == recorder.episode.steps[-1].state_after_hash
    )
    assert saved["error_code"] == "TOOL_TIMEOUT"
    assert saved["provider_state"]["counts"]["get_poi_detail"] >= 1
    assert saved["provider_state"]["tool_call_count"] == len(backend.calls)
    assert saved["state"]["budget"]["used_episode_steps"] == 2

    from scripts.collect_harness_corrections import restore_branch

    folder = tmp_path / case["id"]
    folder.mkdir()
    (folder / "tool-calls.json").write_text(json.dumps(backend.calls), encoding="utf-8")
    candidate = {"case": case, "snapshot": saved}
    with frozen_reference_time(MOMENT):
        restored, restored_backend = restore_branch(candidate, tmp_path)
        assert restored.budget.model_dump() == state.budget.model_dump()
        assert restored_backend.counts == backend.counts
        assert restored_backend.calls == backend.calls
        outcome = await TravelActionExecutor(restored_backend).execute(
            task=restored.task_graph.tasks[0],
            action=PolicyAction(action="get_poi_detail"),
            ledger=restored,
        )
    assert outcome.status == "completed"
    assert any(a.artifact_type == "poi_detail_set" for a in outcome.artifacts)
