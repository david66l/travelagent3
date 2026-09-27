"""Fresh task instances within the six already-reserved training families.

No holdout generator or holdout task content is an input to this builder.
These are train-only extensions, not new independent evaluation templates.
"""

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

from scripts.build_harness_training_pilot import build

VERSION = "harness-student-correction24.v1"
CITIES = [
    ("合肥", 31.82, 117.23),
    ("福州", 26.07, 119.30),
    ("南昌", 28.68, 115.86),
    ("太原", 37.87, 112.55),
]


def build_correction_tasks():
    cases = []
    for original in build():
        group = int(original["id"].split("-")[1])
        variant = int(original["id"].split("-")[2]) - 1
        city, lat, lng = CITIES[variant]
        old_city = original["slots"]["destination"]
        identifier = f"correction-{group:02d}-{variant + 1:02d}"
        replacements = {
            original["id"]: identifier,
            old_city: city,
            "青禾": "澄川",
            "望江": "知秋",
            "拾光": "星河",
            "映水": "栖风",
        }
        for field in ["start_date", "end_date"]:
            before = original["slots"][field]
            replacements[before] = (
                date.fromisoformat(before) + timedelta(days=45)
            ).isoformat()

        def replace(value):
            if isinstance(value, dict):
                return {k: replace(v) for k, v in value.items()}
            if isinstance(value, list):
                return [replace(v) for v in value]
            if isinstance(value, str):
                for before, after in replacements.items():
                    value = value.replace(before, after)
            return value

        case = replace(deepcopy(original))
        old_budget = case["slots"]["budget_range"]
        budget = 200 + 20 * variant if group == 3 else [720, 850, 950, 1100][variant]
        case["slots"]["budget_range"] = budget
        case["request"] = case["request"].replace(
            f"总预算{old_budget}元", f"总预算{budget}元"
        )
        for index, poi in enumerate(case["pois"]):
            poi.update(
                lat=lat + 0.002 * index,
                lng=lng + 0.002 * index,
                duration_minutes=[60, 75, 90, 105][variant],
            )
        if group == 3:
            case["pois"][0]["ticket_price"] = 115 + 10 * variant
            case["pois"][1]["ticket_price"] = 125 + 10 * variant
        for index, restaurant in enumerate(case["restaurants"]):
            restaurant.update(
                lat=lat + 0.001 * (index + 1), lng=lng + 0.001 * (index + 1)
            )
        case.update(
            dataset_version=VERSION,
            split="train",
            derivative_of_case=original["id"],
            training_targets_present=False,
        )
        # Preserve family ownership across every derivative, rather than giving
        # rewritten instances a new group that could later cross a split.
        assert case["source_group"] == original["source_group"]
        cases.append(case)
    old = build()
    assert not {p["name"] for c in cases for p in c["pois"]} & {
        p["name"] for c in old for p in c["pois"]
    }
    assert not {c["request"] for c in cases} & {c["request"] for c in old}
    return cases


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "ML/agentic/data/harness_student_correction24_v1"
    cases = build_correction_tasks()
    content = json.dumps(cases, ensure_ascii=False, indent=2).encode("utf-8")
    out.mkdir(parents=True, exist_ok=True)
    target = out / "cases.json"
    if target.exists() and target.read_bytes() != content:
        raise RuntimeError("Use a new immutable task version")
    target.write_bytes(content)
    manifest = {
        "version": VERSION,
        "count": len(cases),
        "split": "train",
        "source_groups": sorted({c["source_group"] for c in cases}),
        "cases_sha256": hashlib.sha256(content).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parent_builder_sha256": hashlib.sha256(
            (root / "scripts/build_harness_training_pilot.py").read_bytes()
        ).hexdigest(),
        "holdout_content_used": False,
        "training_targets_present": False,
        "limits": [
            "synthetic train-only instances of six existing training families",
            "not 24 independent templates; not a generalization score",
            "provider fault rules and meal-slot-only scope are unchanged",
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
