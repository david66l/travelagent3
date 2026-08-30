"""Build leakage-safe multi-turn recovery tasks for the Stage 3 RL track.

The corpus deliberately creates decisions that cannot be solved from the initial
prompt alone.  The policy must execute a search, read a retryable tool failure,
and adapt the next call to the grounded recovery instruction.  Half of the
tasks also contain an earlier weather timeout, which makes successful episodes
exercise recovery across two different tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.corpus_generation import build_curriculum_case  # noqa: E402
from agentic.environment import SnapshotToolResponse, environment_fingerprint  # noqa: E402
from agentic.grpo_training import GRPOCorpusRow  # noqa: E402


SCHEMA_VERSION = "stage3-multiturn-recovery-rl.v2"

_TRAIN_TEMPLATES = (
    "查询范围过宽。请删除关键词“{drop}”，仅保留用户已提供的“{target}”后重试。",
    "检索条件需要收窄：不要再使用“{drop}”，下一次只提交“{target}”。",
    "上次查询无法形成可靠候选；请用原偏好中的“{target}”作为唯一关键词重试。",
    "结果噪声过高。“{target}”仍有区分度，请只保留该词；不要自行补充新词。",
)

_BLIND_TEMPLATES = (
    "当前组合召回过泛。移除“{drop}”，重试参数中应仅包含“{target}”。",
    "工具建议进行单维检索：保留“{target}”，舍弃“{drop}”，不要增加其他条件。",
    "这组条件没有形成可用候选。去掉“{drop}”，只用已有偏好“{target}”重新查询。",
    "检索服务无法区分当前组合；下一次仅发送“{target}”，不要保留“{drop}”。",
    "候选噪声超过阈值。请删除“{drop}”，以“{target}”单独发起下一次检索。",
    "需要缩小查询维度：保留用户给出的“{target}”，移除“{drop}”及其他附加条件。",
)


def _next_retry_source(index: int) -> tuple[int, GRPOCorpusRow]:
    """Return a deterministic happy-path source with two search responses."""
    while True:
        task, snapshot = build_curriculum_case(index)
        responses = snapshot.tool_responses.get("search_pois") or []
        if len(responses) >= 2 and responses[0].retryable:
            return index + 1, GRPOCorpusRow(task=task, snapshot=snapshot)
        index += 1


def derive_multiturn_recovery(
    row: GRPOCorpusRow,
    *,
    ordinal: int,
    message_template: str,
    cross_tool: bool,
    target_index: int | None = None,
) -> GRPOCorpusRow:
    """Create one observable, exact-verifier recovery environment."""
    interests = [
        str(item)
        for item in (row.task.slots.get("interests") or row.task.profile.get("interests") or [])
    ]
    responses = row.snapshot.tool_responses.get("search_pois") or []
    if len(interests) < 2 or len(responses) < 2:
        raise ValueError("multi-turn recovery needs two interests and two search responses")

    # Balance the target position so the model cannot learn a first/last-word
    # shortcut.  The only reliable signal is the visible tool observation.
    target_index = ordinal % 2 if target_index is None else target_index
    if target_index not in {0, 1}:
        raise ValueError("target_index must be 0 or 1")
    target = interests[target_index]
    drop = interests[1 - target_index]
    ignored_keyword_values = [
        str(value)
        for value in (
            row.task.slots.get("destination"),
            row.task.slots.get("start_date"),
            row.task.slots.get("end_date"),
        )
        if value
    ]
    derived = row.model_copy(deep=True)
    suffix = "cross-tool" if cross_tool else "search-only"
    derived.task.task_id = f"{row.task.task_id}-stage3-{suffix}-{ordinal:05d}"
    derived.task.template_family = f"stage3-{suffix}-recovery"
    derived.task.difficulty = "L4" if cross_tool else "L3"
    derived.snapshot.environment_version = SCHEMA_VERSION
    derived.snapshot.snapshot_version = SCHEMA_VERSION
    derived.snapshot.state_id = f"{row.snapshot.state_id}-stage3-{suffix}-{ordinal:05d}"

    first = SnapshotToolResponse(
        data=None,
        data_source="unavailable",
        fallback_reason=message_template.format(target=target, drop=drop),
        error_code="QUERY_TOO_BROAD",
        retryable=True,
        # The first model-owned search is the ordinary grounded query.  Pin its
        # arguments so the snapshot returns the intended retryable failure;
        # otherwise the executor rejects the demonstrated call as an argument
        # mismatch before the policy can observe the recovery instruction.
        expected_arguments={"keywords": interests[:2]},
        argument_match_mode="context_tolerant_keywords",
        ignored_keyword_values=ignored_keyword_values,
    )
    second = responses[-1].model_copy(deep=True)
    second.expected_arguments = {"keywords": [target]}
    second.argument_match_mode = "context_tolerant_keywords"
    second.ignored_keyword_values = ignored_keyword_values
    derived.snapshot.tool_responses["search_pois"] = [first, second]

    if cross_tool:
        weather_success = derived.snapshot.tool_responses["get_weather"][-1].model_copy(
            deep=True
        )
        weather_timeout = SnapshotToolResponse(
            data=None,
            data_source="unavailable",
            fallback_reason=(
                "天气服务暂时超时；该失败可重试，请在当前子任务的剩余预算内再次调用。"
            ),
            error_code="UPSTREAM_TIMEOUT",
            retryable=True,
        )
        derived.snapshot.tool_responses["get_weather"] = [
            weather_timeout,
            weather_success,
        ]

    derived.snapshot.hidden_test_facts["stage3_multiturn_recovery"] = {
        "target_keywords": [target],
        "dropped_keyword": drop,
        "cross_tool": cross_tool,
        "source_task_id": row.task.task_id,
        "message_template_sha256": hashlib.sha256(message_template.encode()).hexdigest(),
        "argument_match_mode": "context_tolerant_keywords",
    }
    return derived


def _build_split(
    *,
    start_index: int,
    count: int,
    templates: tuple[str, ...],
    ordinal_offset: int,
) -> tuple[list[GRPOCorpusRow], int]:
    rows: list[GRPOCorpusRow] = []
    cursor = start_index
    for local_index in range(count):
        cursor, source = _next_retry_source(cursor)
        ordinal = ordinal_offset + local_index
        rows.append(
            derive_multiturn_recovery(
                source,
                ordinal=ordinal,
                message_template=templates[ordinal % len(templates)],
                cross_tool=bool(ordinal % 2),
            )
        )
    return rows, cursor


def _write_jsonl(path: Path, rows: list[GRPOCorpusRow]) -> str:
    path.write_text(
        "\n".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    output_dir: Path,
    *,
    start_index: int = 40000,
    train_count: int = 1024,
    validation_count: int = 128,
    test_count: int = 128,
) -> dict[str, Any]:
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("all split counts must be positive")
    if not set(_TRAIN_TEMPLATES).isdisjoint(_BLIND_TEMPLATES):
        raise ValueError("blind recovery templates overlap training templates")

    cursor = start_index
    train, cursor = _build_split(
        start_index=cursor,
        count=train_count,
        templates=_TRAIN_TEMPLATES,
        ordinal_offset=0,
    )
    validation, cursor = _build_split(
        start_index=cursor,
        count=validation_count,
        templates=_TRAIN_TEMPLATES,
        ordinal_offset=train_count,
    )
    # Blind tasks use fresh generator states and paraphrase templates that are
    # never present in train or validation.
    test, cursor = _build_split(
        start_index=cursor,
        count=test_count,
        templates=_BLIND_TEMPLATES,
        ordinal_offset=train_count + validation_count,
    )
    splits = {"train": train, "validation": validation, "test": test}

    ids = {name: {row.task.task_id for row in rows} for name, rows in splits.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if ids[left] & ids[right]:
            raise ValueError(f"task leakage detected between {left} and {right}")
    all_rows = [row for rows in splits.values() for row in rows]
    fingerprints = [environment_fingerprint(row.task, row.snapshot) for row in all_rows]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("duplicate environment fingerprints detected")

    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {name: _write_jsonl(output_dir / f"{name}.jsonl", rows) for name, rows in splits.items()}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": "train and validation for online GRPO; test is frozen evaluation only",
        "start_index": start_index,
        "next_unused_index": cursor,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "variant_counts": {
            name: dict(
                Counter(
                    "cross_tool"
                    if row.snapshot.hidden_test_facts["stage3_multiturn_recovery"]["cross_tool"]
                    else "search_only"
                    for row in rows
                )
            )
            for name, rows in splits.items()
        },
        "split_sha256": hashes,
        "train_validation_template_overlap": True,
        "train_test_template_overlap": False,
        # Generic itinerary requests may repeat across splits by design.  The
        # held-out variable is the post-failure decision state and wording, not
        # the initial travel request.  Record this explicitly rather than
        # overstating the blind-test boundary.
        "initial_request_overlap": {
            f"{left}_{right}": len(
                {row.task.user_request for row in splits[left]}
                & {row.task.user_request for row in splits[right]}
            )
            for left, right in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
        },
        "task_overlap": [],
        "environment_fingerprint_overlap": [],
        "reward_oracle": (
            "production Agent Loop + immutable snapshot argument contracts + hard verifier"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=40000)
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--validation-count", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=128)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.output_dir,
                start_index=args.start_index,
                train_count=args.train_count,
                validation_count=args.validation_count,
                test_count=args.test_count,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
