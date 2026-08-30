import json

from agentic.corpus_generation import build_curriculum_case
from agentic.grpo_training import GRPOCorpusRow
from agentic.trl_environment import TRLSearchEnvironment
from scripts.build_stage3_multiturn_rl_corpus import build
from scripts.build_stage3_multiturn_rl_corpus import derive_multiturn_recovery


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_stage3_multiturn_rl_corpus_is_split_safe_and_verifiable(tmp_path):
    manifest = build(
        tmp_path,
        start_index=70000,
        train_count=8,
        validation_count=4,
        test_count=4,
    )

    assert manifest["counts"] == {"train": 8, "validation": 4, "test": 4}
    assert manifest["train_test_template_overlap"] is False
    assert set(manifest["initial_request_overlap"]) == {
        "train_validation",
        "train_test",
        "validation_test",
    }
    splits = {name: _rows(tmp_path / f"{name}.jsonl") for name in manifest["counts"]}
    ids = {name: {row["task"]["task_id"] for row in rows} for name, rows in splits.items()}
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])

    for rows in splits.values():
        for row in rows:
            facts = row["snapshot"]["hidden_test_facts"]["stage3_multiturn_recovery"]
            responses = row["snapshot"]["tool_responses"]["search_pois"]
            target = facts["target_keywords"]
            interests = row["task"]["slots"]["interests"]
            assert responses[0]["error_code"] == "QUERY_TOO_BROAD"
            assert responses[0]["retryable"] is True
            assert responses[0]["expected_arguments"] == {"keywords": interests[:2]}
            assert responses[0]["argument_match_mode"] == "context_tolerant_keywords"
            assert row["task"]["slots"]["destination"] in responses[0]["ignored_keyword_values"]
            assert target[0] in responses[0]["fallback_reason"]
            assert responses[1]["expected_arguments"] == {"keywords": target}
            assert responses[1]["argument_match_mode"] == "context_tolerant_keywords"
            if facts["cross_tool"]:
                weather = row["snapshot"]["tool_responses"]["get_weather"]
                assert weather[0]["error_code"] == "UPSTREAM_TIMEOUT"
                assert weather[0]["retryable"] is True


def test_stage3_snapshot_executes_the_demonstrated_recovery_contract():
    task, snapshot = build_curriculum_case(70007)
    row = derive_multiturn_recovery(
        GRPOCorpusRow(task=task, snapshot=snapshot),
        ordinal=0,
        message_template="删除“{drop}”，仅保留“{target}”。",
        cross_tool=False,
    )
    interests = row.task.slots["interests"][:2]
    target = row.snapshot.hidden_test_facts["stage3_multiturn_recovery"]["target_keywords"]
    environment = TRLSearchEnvironment()
    environment.reset(
        task=row.task.model_dump(mode="json"),
        snapshot=row.snapshot.model_dump(mode="json"),
    )

    first = json.loads(environment.search_pois(interests))
    second = json.loads(environment.search_pois(target))

    assert first["last_transition"]["verification"]["error_code"] == "QUERY_TOO_BROAD"
    assert first["policy_state"]["failure_summary"][-1]["code"] == "QUERY_TOO_BROAD"
    assert second["last_transition"]["verification"]["error_code"] is None
    environment.get_reward()
