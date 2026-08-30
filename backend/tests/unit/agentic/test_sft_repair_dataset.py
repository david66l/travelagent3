import json

from scripts.build_sft_repair_dataset import _is_recovery, _split_recovery


def _row(scenario: str, interests: list[str], recovery: bool) -> dict:
    context = {
        "soft_preferences": {"interests": interests},
        "failure_summary": (
            [
                {
                    "code": "UPSTREAM_TIMEOUT",
                    "retryable": True,
                }
            ]
            if recovery
            else []
        ),
    }
    return {
        "scenario_id": scenario,
        "messages": [{}, {"content": json.dumps(context)}],
    }


def test_recovery_detection_and_family_stratified_split():
    rows = [_row(f"family-a-{index:02d}", ["亲子", "公园"], True) for index in range(20)] + [
        _row("normal", ["亲子", "公园"], False)
    ]

    assert _is_recovery(rows[-1]) is False
    split = _split_recovery(rows[:-1])

    assert {name: len(items) for name, items in split.items()} == {
        "validation": 3,
        "test": 3,
        "train": 14,
    }
