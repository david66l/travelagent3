import json

from agentic.corpus_generation import build_curriculum_case
from agentic.grpo_training import GRPOCorpusRow
from scripts.build_adaptive_recovery_corpus import build, derive_adaptive_recovery


def _row(index: int) -> GRPOCorpusRow:
    task, snapshot = build_curriculum_case(index)
    return GRPOCorpusRow(task=task, snapshot=snapshot)


def _write(path, rows):
    path.write_text(
        "\n".join(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) for row in rows)
        + "\n",
        encoding="utf-8",
    )


def test_adaptive_recovery_requires_grounded_narrowing_after_observation():
    source = _row(7)

    derived = derive_adaptive_recovery(source)

    responses = derived.snapshot.tool_responses["search_pois"]
    target = source.task.slots["interests"][-1]
    assert responses[0].error_code == "QUERY_TOO_BROAD"
    assert target in responses[0].fallback_reason
    assert responses[1].expected_arguments == {"keywords": [target]}
    assert derived.snapshot.hidden_test_facts["adaptive_recovery"]["target_keywords"] == [target]
    assert source.snapshot.tool_responses["search_pois"][0].error_code == "UPSTREAM_TIMEOUT"


def test_builder_preserves_official_split_isolation(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    _write(source / "train.jsonl", [_row(7), _row(17)])
    _write(source / "validation.jsonl", [_row(27)])

    manifest = build(source, output, train_limit=2, validation_limit=1)

    assert manifest["counts"] == {"train": 2, "validation": 1}
    assert manifest["split_overlap"] == []
    assert not set(manifest["task_ids"]["train"]) & set(manifest["task_ids"]["validation"])
