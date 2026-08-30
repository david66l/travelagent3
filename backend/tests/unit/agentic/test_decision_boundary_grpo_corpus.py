"""Decision-boundary GRPO data must be paired, split-safe and reward-separable."""

import json

from agentic.corpus_generation import build_curriculum_case
from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus
from scripts.build_decision_boundary_grpo_corpus import build, derive_boundary_pair
from scripts.build_focused_grpo_curriculum import _select_split


def _write_source(path, indices):
    rows = []
    for index in indices:
        task, snapshot = build_curriculum_case(index)
        rows.append(GRPOCorpusRow(task=task, snapshot=snapshot))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) for row in rows)
        + "\n",
        encoding="utf-8",
    )


def test_pair_changes_actionability_without_crossing_source_split():
    task, snapshot = build_curriculum_case(0)

    tradeoff, abort = derive_boundary_pair(GRPOCorpusRow(task=task, snapshot=snapshot))

    tradeoff_contract = tradeoff.task.feasibility_report
    abort_contract = abort.task.feasibility_report
    assert tradeoff_contract["actionable_alternatives"] is True
    assert tradeoff_contract["alternatives"]
    assert abort_contract["actionable_alternatives"] is False
    assert abort_contract["alternatives"] == []
    assert (
        tradeoff.snapshot.hidden_test_facts["decision_boundary_training"]["pair_id"]
        == abort.snapshot.hidden_test_facts["decision_boundary_training"]["pair_id"]
    )


def test_boundary_pairs_cover_infeasible_unsafe_and_missing_tool_contracts():
    task, snapshot = build_curriculum_case(0)
    source = GRPOCorpusRow(task=task, snapshot=snapshot)

    for kind in ("infeasible", "unsafe", "missing_tool"):
        tradeoff, abort = derive_boundary_pair(source, boundary_kind=kind)

        assert tradeoff.task.feasibility_report["status"] == kind
        assert abort.task.feasibility_report["status"] == kind
        assert tradeoff.task.feasibility_report["actionable_alternatives"] is True
        assert abort.task.feasibility_report["actionable_alternatives"] is False
        assert (
            abort.snapshot.hidden_test_facts["decision_boundary_training"]["boundary_kind"] == kind
        )


def test_unsafe_boundary_generation_uses_multiple_risk_scenarios():
    reasons = set()
    requests = set()
    eligible = 0
    for index in range(512):
        task, snapshot = build_curriculum_case(index)
        if task.missing_slots or not snapshot.tool_responses:
            continue
        _, abort = derive_boundary_pair(
            GRPOCorpusRow(task=task, snapshot=snapshot),
            boundary_kind="unsafe",
        )
        reasons.add(abort.task.feasibility_report["reasons"][0])
        requests.add(abort.task.user_request)
        eligible += 1
        if eligible == 32:
            break

    assert eligible == 32
    assert len(reasons) >= 6
    assert len(requests) == 32
    assert all("请直接终止" not in request for request in requests)


def test_focused_curriculum_selects_each_requested_cell_and_anchor():
    sources = []
    for index in range(64):
        task, snapshot = build_curriculum_case(index)
        if task.missing_slots or not snapshot.tool_responses:
            continue
        sources.append(GRPOCorpusRow(task=task, snapshot=snapshot))
        if len(sources) == 4:
            break

    rows = []
    for source, kind in zip(
        sources[:3],
        ("infeasible", "unsafe", "missing_tool"),
        strict=True,
    ):
        rows.extend(derive_boundary_pair(source, boundary_kind=kind))
    rows.append(sources[3])
    quotas = {
        (kind, variant): 1
        for kind in ("infeasible", "unsafe", "missing_tool")
        for variant in ("actionable_tradeoff", "necessary_abort")
    }

    selected = _select_split(rows, quotas=quotas, anchor_count=1, split="train")

    assert len(selected) == 7
    assert len({row.task.task_id for row in selected}) == 7


async def test_small_corpus_passes_preflight_and_bidirectional_reward_matrix(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    _write_source(source_dir / "train.jsonl", [0, 1, 2, 3, 6, 7])
    _write_source(source_dir / "validation.jsonl", [10, 11, 12, 13, 16, 17])

    manifest = await build(
        source_dir,
        output_dir,
        train_pairs=1,
        validation_pairs=1,
        train_anchors=4,
        validation_anchors=4,
        concurrency=4,
        minimum_train_tasks=1,
    )

    assert manifest["counts"] == {"train": 6, "validation": 6}
    assert manifest["preflight"]["ready"] is True
    assert manifest["boundary_actions"] == {"propose_tradeoff": 2, "abort": 2}
    assert manifest["boundary_kinds"] == {"infeasible": 4}
    assert manifest["reward_matrix"]["verified_rows"] == 4
    assert manifest["reward_matrix"]["correct_gate_pass"] == 4
    assert manifest["reward_matrix"]["opposite_gate_failed"] == 4
    assert manifest["reward_matrix"]["minimum_reward_gap"] >= 1.0

    split_by_pair = {}
    for split in ("train", "validation"):
        for row in load_grpo_corpus(output_dir / f"{split}.jsonl"):
            evidence = row.snapshot.hidden_test_facts.get("decision_boundary_training")
            if evidence:
                split_by_pair.setdefault(evidence["pair_id"], set()).add(split)
    assert split_by_pair
    assert all(len(splits) == 1 for splits in split_by_pair.values())
