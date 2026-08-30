from scripts.build_tradeoff_repair_dataset import _action, _split


def _row(index, action):
    return {
        "scenario_id": f"scenario-{index}",
        "messages": [
            {"role": "user", "content": "state"},
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": action, "arguments": {}}}],
            },
        ],
    }


def test_tradeoff_detection_and_internal_split_are_deterministic():
    rows = [_row(index, "propose_tradeoff") for index in range(40)]

    first = _split(rows)
    second = _split(list(reversed(rows)))

    assert _action(rows[0]) == "propose_tradeoff"
    assert {key: [row["scenario_id"] for row in value] for key, value in first.items()} == {
        key: [row["scenario_id"] for row in value] for key, value in second.items()
    }
    assert {key: len(value) for key, value in first.items()} == {
        "validation": 6,
        "test": 6,
        "train": 28,
    }
