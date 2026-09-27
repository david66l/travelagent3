from agentic.corpus_generation import build_curriculum_case
from agentic.grpo_training import GRPOCorpusRow
from scripts.select_h006_rollout_sources import select_rows


def test_select_rows_is_deterministic_unique_and_honors_exclusions():
    rows = []
    for index in range(12):
        task, snapshot = build_curriculum_case(index)
        snapshot.hidden_test_facts["grpo_decision_state"] = {
            "target_action": "retry_solve" if index % 2 else "abort"
        }
        rows.append(GRPOCorpusRow(task=task, snapshot=snapshot))

    selected = select_rows(
        rows,
        excluded_task_ids={rows[0].task.task_id},
        excluded_actions={"retry_solve"},
        count=4,
    )

    assert len({row.task.task_id for row in selected}) == 4
    assert rows[0].task.task_id not in {row.task.task_id for row in selected}
    assert all(
        row.snapshot.hidden_test_facts["grpo_decision_state"]["target_action"] == "abort"
        for row in selected
    )
    assert [row.task.task_id for row in selected] == [
        row.task.task_id
        for row in select_rows(
            rows,
            excluded_task_ids={rows[0].task.task_id},
            excluded_actions={"retry_solve"},
            count=4,
        )
    ]
