import json

from scripts.build_stage2_hard_holdout import build


def test_hard_holdout_is_frozen_unique_and_covers_all_variants(tmp_path):
    manifest = build(tmp_path, start_index=30000, per_variant=2)

    assert manifest["rows"] == 6
    assert manifest["variant_counts"] == {
        "adaptive_recovery": 2,
        "cross_tool_recovery": 2,
        "priority_search": 2,
    }
    assert len(manifest["task_ids"]) == len(set(manifest["task_ids"]))
    rows = [
        json.loads(line)
        for line in (tmp_path / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cross_tool = next(row for row in rows if "cross-tool-recovery" in row["task"]["task_id"])
    weather = cross_tool["snapshot"]["tool_responses"]["get_weather"]
    search = cross_tool["snapshot"]["tool_responses"]["search_pois"]
    assert [item["error_code"] for item in weather] == ["UPSTREAM_TIMEOUT", None]
    assert [item["error_code"] for item in search] == [
        "QUERY_TOO_BROAD",
        None,
    ]
    target = cross_tool["snapshot"]["hidden_test_facts"]["cross_tool_recovery"]["target_keywords"]
    assert search[-1]["expected_arguments"] == {"keywords": target}
