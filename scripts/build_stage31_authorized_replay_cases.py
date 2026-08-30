"""Build 300 unique, non-PII, release-gate replay scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from evaluation.shadow_replay import AuthorizedReplayCase  # noqa: E402


CITIES = [
    "北京",
    "上海",
    "广州",
    "深圳",
    "成都",
    "杭州",
    "西安",
    "重庆",
    "苏州",
    "南京",
    "厦门",
    "青岛",
    "大理",
    "丽江",
    "三亚",
    "长沙",
    "武汉",
    "昆明",
    "桂林",
    "拉萨",
]
INTEREST_SETS = [
    ["历史", "美食"],
    ["自然", "摄影"],
    ["建筑", "文化"],
    ["亲子", "公园"],
    ["博物馆", "本地生活"],
]


def build_cases() -> list[AuthorizedReplayCase]:
    cases: list[AuthorizedReplayCase] = []
    base_date = date(2026, 9, 1)
    index = 0
    for city_index, city in enumerate(CITIES):
        for days in (2, 3, 4):
            for interest_index, interests in enumerate(INTEREST_SETS):
                index += 1
                start = base_date + timedelta(
                    days=(city_index * 7 + days * 11 + interest_index * 3) % 120
                )
                end = start + timedelta(days=days - 1)
                cases.append(
                    AuthorizedReplayCase(
                        case_id=f"stage31-release-{index:03d}",
                        destination=city,
                        start_date=start.isoformat(),
                        end_date=end.isoformat(),
                        travel_days=days,
                        budget=float(1200 + days * 900 + interest_index * 350),
                        interests=interests,
                    )
                )
    if len(cases) != 300:
        raise RuntimeError(f"expected 300 cases, built {len(cases)}")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = build_cases()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for case in cases
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "cases": len(cases)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
