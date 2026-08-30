from scripts.build_external_benchmark_pilot import build_assignments, build_manifest


def test_pilot_assignments_are_balanced_content_free_and_deterministic():
    first = build_assignments()
    second = build_assignments()
    manifest = build_manifest(first, 20260815)

    assert first == second
    assert manifest["assignments"] == 30
    assert manifest["source_counts"] == {
        "authorized_real_or_simulated": 12,
        "human_original_constraint": 9,
        "tool_failure": 6,
        "long_context_replan": 3,
    }
    assert manifest["author_group_counts"] == {
        "external-writer-a": 15,
        "external-writer-b": 15,
    }
    assert manifest["content_free"] is True
    assert all(item["submission_status"] == "awaiting_independent_author" for item in first)
    assert all("messages" not in item for item in first)


def test_only_failure_assignments_receive_faults():
    assignments = build_assignments()

    assert all(
        (item["fault_type"] is not None) == (item["source"] == "tool_failure")
        for item in assignments
    )
