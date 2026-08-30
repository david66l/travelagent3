import json

from agentic.corpus_generation import build_curriculum_case
from agentic.distillation import TeacherPreferencePair
from agentic.grpo_training import GRPOCorpusRow
from scripts.build_preference_rollout_holdout import build


def test_preference_rollout_holdout_recovers_exact_environment_row(tmp_path):
    task, snapshot = build_curriculum_case(42)
    pair = TeacherPreferencePair(
        pair_id="pref-42",
        task_id=task.task_id,
        family="search",
        context_hash="context-42",
        messages=[{"role": "user", "content": "{}"}],
        tools=[],
        chosen={"role": "assistant", "content": "chosen"},
        rejected={"role": "assistant", "content": "rejected"},
        chosen_trajectory_id="chosen",
        rejected_trajectory_id="rejected",
        reason_codes=["VERIFIER_SUCCESS_OVER_FAILURE"],
        reward_margin=1.0,
    )
    preference_file = tmp_path / "preferences.jsonl"
    source_file = tmp_path / "corpus.jsonl"
    output_file = tmp_path / "holdout" / "test.jsonl"
    preference_file.write_text(pair.model_dump_json() + "\n", encoding="utf-8")
    source_file.write_text(
        GRPOCorpusRow(task=task, snapshot=snapshot).model_dump_json() + "\n",
        encoding="utf-8",
    )

    manifest = build(preference_file, [source_file], output_file)

    recovered = GRPOCorpusRow(**json.loads(output_file.read_text(encoding="utf-8")))
    assert recovered.task.task_id == task.task_id
    assert recovered.snapshot == snapshot
    assert manifest["cases"] == 1
    assert manifest["frozen_before_student_training"] is True
