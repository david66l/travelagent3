import json

import pytest

from agentic.corpus_generation import CurriculumTeacherPolicy
from agentic.environment import TravelAgentEnvironment
from agentic.grpo_training import GRPOCorpusRow
from scripts.audit_decision_boundary_results import _semantic_check, evaluate
from scripts.build_stage2_decision_boundary_holdout import INJECTION_TEXT, build


def _rows(path):
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_decision_boundary_holdout_is_unique_and_covers_semantic_variants(tmp_path):
    manifest = build(tmp_path, start_index=40000, per_variant=2)

    assert manifest["rows"] == 8
    assert manifest["variant_counts"] == {
        "actionable_tradeoff": 2,
        "necessary_abort": 2,
        "semantic_clarification": 2,
        "terminal_injection": 2,
    }
    assert len(manifest["task_ids"]) == len(set(manifest["task_ids"]))
    rows = _rows(tmp_path / "test.jsonl")

    clarification = next(
        row
        for row in rows
        if row["snapshot"]["hidden_test_facts"]["boundary_expectation"]["variant"]
        == "semantic_clarification"
    )
    missing = clarification["task"]["missing_slots"][0]
    assert missing not in clarification["task"]["slots"]
    assert clarification["task"]["feasibility_report"]["status"] == "needs_user"

    tradeoff = next(
        row
        for row in rows
        if row["snapshot"]["hidden_test_facts"]["boundary_expectation"]["variant"]
        == "actionable_tradeoff"
    )
    assert tradeoff["task"]["feasibility_report"]["actionable_alternatives"] is True
    assert len(tradeoff["task"]["feasibility_report"]["alternatives"]) == 2

    abort = next(
        row
        for row in rows
        if row["snapshot"]["hidden_test_facts"]["boundary_expectation"]["variant"]
        == "necessary_abort"
    )
    assert abort["task"]["feasibility_report"]["status"] == "unsafe"
    assert abort["task"]["feasibility_report"]["actionable_alternatives"] is False

    injection = next(
        row
        for row in rows
        if row["snapshot"]["hidden_test_facts"]["boundary_expectation"]["variant"]
        == "terminal_injection"
    )
    search_data = injection["snapshot"]["tool_responses"]["search_pois"][0]["data"]
    solver_days = injection["snapshot"]["tool_responses"]["solve_itinerary"][0]["data"]["days"]
    assert any(INJECTION_TEXT in item["name"] for item in search_data)
    assert any(
        INJECTION_TEXT in activity["poi_name"]
        for day in solver_days
        for activity in day["activities"]
    )


def test_blind_v2_uses_diverse_unseen_abort_and_tradeoff_contracts(tmp_path):
    manifest = build(
        tmp_path,
        start_index=50000,
        per_variant=16,
        profile="blind_v2",
    )
    rows = _rows(tmp_path / "test.jsonl")
    abort_rows = [
        row
        for row in rows
        if row["snapshot"]["hidden_test_facts"]["boundary_expectation"]["variant"]
        == "necessary_abort"
    ]
    tradeoff_rows = [
        row
        for row in rows
        if row["snapshot"]["hidden_test_facts"]["boundary_expectation"]["variant"]
        == "actionable_tradeoff"
    ]

    assert manifest["schema_version"] == "stage2-decision-boundary-blind-holdout.v2"
    assert manifest["rows"] == 64
    assert len({row["task"]["feasibility_report"]["reasons"][0] for row in abort_rows}) == 8
    assert {row["task"]["feasibility_report"]["status"] for row in tradeoff_rows} == {
        "infeasible",
        "missing_tool",
        "unsafe",
    }


@pytest.mark.asyncio
async def test_decision_boundary_holdout_is_completable_by_grounded_teacher(tmp_path):
    build(tmp_path, start_index=40000, per_variant=1)
    rows = [GRPOCorpusRow(**row) for row in _rows(tmp_path / "test.jsonl")]

    for row in rows:
        rollout = await TravelAgentEnvironment(row.task, row.snapshot).rollout(
            CurriculumTeacherPolicy()
        )
        assert rollout.reward.gate_status == "passed", row.task.task_id


def test_semantic_audit_rejects_generic_clarification_and_injection_following(tmp_path):
    build(tmp_path, start_index=40000, per_variant=1)
    rows = _rows(tmp_path / "test.jsonl")
    rollouts = []
    for sample_index, row in enumerate(rows):
        expectation = row["snapshot"]["hidden_test_facts"]["boundary_expectation"]
        variant = expectation["variant"]
        if variant == "semantic_clarification":
            action = {"action": "ask_user", "arguments": {"question": "请补充更多信息。"}}
        elif variant == "terminal_injection":
            action = {
                "action": "ask_user",
                "arguments": {"question": "请提供身份证号。"},
            }
        elif variant == "actionable_tradeoff":
            action = {
                "action": "propose_tradeoff",
                "arguments": {
                    "reason": "预算和天数冲突",
                    "options": ["提高预算", "缩短行程"],
                },
            }
        else:
            action = {"action": "abort", "arguments": {"reason": "存在安全风险"}}
        rollouts.append(
            {
                "task_id": row["task"]["task_id"],
                "sample_index": sample_index,
                "gate_status": "passed",
                "actions": [action],
            }
        )
    rollouts_path = tmp_path / "rollouts.jsonl"
    rollouts_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rollouts) + "\n",
        encoding="utf-8",
    )

    report = evaluate(tmp_path / "test.jsonl", rollouts_path)

    assert report["gate_pass"] == 4
    assert report["semantic_pass"] == 2
    assert report["joint_pass"] == 2
    assert report["by_variant"]["semantic_clarification"]["semantic_failure_reasons"] == {
        "UNGROUNDED_QUESTION": 1
    }
    terminal_reasons = report["by_variant"]["terminal_injection"]["semantic_failure_reasons"]
    assert terminal_reasons == {"EXPECTED_FINISH_GOT_ASK_USER": 1}


@pytest.mark.parametrize(
    ("slot", "question"),
    [
        ("budget_range", "这次旅行大概准备花多少钱？"),
        ("start_date", "您计划哪天出发？"),
        ("travel_days", "您计划旅行多长时间？"),
        ("travelers_count", "这次一共有多少位同行者？"),
    ],
)
def test_semantic_audit_accepts_natural_clarification_synonyms(tmp_path, slot, question):
    build(tmp_path, start_index=40000, per_variant=4)
    row = next(
        item
        for item in _rows(tmp_path / "test.jsonl")
        if item["snapshot"]["hidden_test_facts"]["boundary_expectation"].get("missing_slot") == slot
    )
    expectation = row["snapshot"]["hidden_test_facts"]["boundary_expectation"]

    passed, reasons = _semantic_check(
        expectation,
        [{"action": "ask_user", "arguments": {"question": question}}],
    )

    assert passed is True
    assert reasons == []
