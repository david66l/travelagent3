"""Build paired decision-loop recovery tasks for argument change vs safe retry.

The same grounded search request is exposed to two different observable tool
failures. ``QUERY_TOO_BROAD`` requires changing the keyword arguments, while
``UPSTREAM_TIMEOUT`` requires one idempotent retry with the same arguments.
This prevents a recovery policy from learning the shortcut "always change" or
"always repeat" and gives turn-relative GRPO a genuine post-observation choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agentic.environment import SnapshotToolResponse, environment_fingerprint  # noqa: E402
from agentic.grpo_training import GRPOCorpusRow  # noqa: E402
from build_stage3_multiturn_rl_corpus import (  # noqa: E402
    _BLIND_TEMPLATES,
    _TRAIN_TEMPLATES,
    _next_retry_source,
    derive_multiturn_recovery,
)


SCHEMA_VERSION = "stage3-decision-loop-curriculum.v2"
DecisionScenario = Literal["change_arguments", "retry_same_arguments"]
EvidenceStyle = Literal["explicit_instruction", "diagnostic_evidence"]

_TRAIN_TIMEOUT_MESSAGES = (
    "搜索服务暂时超时；请求参数本身有效，请使用完全相同的关键词重试一次。",
    "上次请求因临时网络超时失败。不要修改关键词，在剩余预算内原样重试。",
)
_BLIND_TIMEOUT_MESSAGES = (
    "检索端出现瞬时超时，当前参数没有错误；下一轮保持关键词不变再次调用。",
    "服务短暂不可用，但查询条件有效。请勿改写参数，使用同一组关键词重试。",
    "这是可重试的上游超时，不是查询条件问题；保留原关键词重新发送。",
)

_TRAIN_DIAGNOSTIC_CHANGE_MESSAGES = (
    "检索诊断：关键词“{drop}”召回噪声过高，“{target}”仍保持稳定区分度；请求未产生候选。",
    "查询质量报告：“{drop}”对应结果过泛，“{target}”具有有效信号；本轮检索被拒绝。",
)
_BLIND_DIAGNOSTIC_CHANGE_MESSAGES = (
    "检索侧分析显示“{drop}”导致候选空间失焦，而“{target}”仍有明确相关性；当前请求没有返回结果。",
    "质量门禁拒绝了本次组合：“{drop}”带来大量无关召回，“{target}”的选择性正常。",
    "候选诊断中“{drop}”的噪声占比超标，“{target}”仍可用于有效检索；本轮没有可用候选。",
)
_TRAIN_DIAGNOSTIC_TIMEOUT_MESSAGES = (
    "检索诊断：参数校验已经通过，但上游在执行前发生瞬时超时，未生成候选。",
    "请求条件有效；服务在提交检索后临时中断，本轮没有返回结果。",
)
_BLIND_DIAGNOSTIC_TIMEOUT_MESSAGES = (
    "查询已通过合法性检查，候选生成前网络链路超时，因此没有结果返回。",
    "检索参数已被接受，但执行节点短暂失联，本轮未产生任何候选。",
    "服务确认请求条件有效；结果计算开始前发生瞬时故障，没有可消费的候选。",
)


def derive_decision_loop_case(
    row: GRPOCorpusRow,
    *,
    ordinal: int,
    scenario: DecisionScenario,
    evidence_style: EvidenceStyle,
    change_message_template: str,
    timeout_message: str,
    target_index: int | None = None,
) -> GRPOCorpusRow:
    """Derive one recovery task whose correct behavior depends on error semantics."""
    derived = derive_multiturn_recovery(
        row,
        ordinal=ordinal,
        message_template=change_message_template,
        cross_tool=False,
        target_index=target_index,
    )
    interests = [
        str(item)
        for item in (
            derived.task.slots.get("interests") or derived.task.profile.get("interests") or []
        )
    ][:2]
    if len(interests) < 2:
        raise ValueError("decision-loop recovery requires two grounded interests")

    target = derived.snapshot.hidden_test_facts["stage3_multiturn_recovery"]["target_keywords"]
    first = derived.snapshot.tool_responses["search_pois"][0]
    second = derived.snapshot.tool_responses["search_pois"][1]
    if scenario == "retry_same_arguments":
        first = SnapshotToolResponse(
            data=None,
            data_source="unavailable",
            fallback_reason=timeout_message,
            error_code="UPSTREAM_TIMEOUT",
            retryable=True,
            expected_arguments={"keywords": interests},
            argument_match_mode="context_tolerant_keywords",
            ignored_keyword_values=list(first.ignored_keyword_values),
        )
        second.expected_arguments = {"keywords": interests}
        expected_keywords = interests
        suffix = "retry-same"
    else:
        expected_keywords = list(target)
        suffix = "change-arguments"

    style_suffix = "explicit" if evidence_style == "explicit_instruction" else "diagnostic"
    derived.task.task_id = (
        f"{row.task.task_id}-decision-loop-{suffix}-{style_suffix}-{ordinal:05d}"
    )
    derived.task.template_family = f"decision-loop-{suffix}-{style_suffix}"
    derived.task.difficulty = "L4"
    derived.snapshot.environment_version = SCHEMA_VERSION
    derived.snapshot.snapshot_version = SCHEMA_VERSION
    derived.snapshot.state_id = f"{row.snapshot.state_id}-decision-loop-{suffix}-{ordinal:05d}"
    derived.snapshot.tool_responses["search_pois"] = [first, second]
    derived.snapshot.hidden_test_facts["decision_loop_curriculum"] = {
        "scenario": scenario,
        "evidence_style": evidence_style,
        "first_error_code": first.error_code,
        "initial_keywords": interests,
        "expected_recovery_keywords": expected_keywords,
        "requires_argument_change": scenario == "change_arguments",
        "source_task_id": row.task.task_id,
    }
    return derived


def _build_split(
    *,
    start_index: int,
    count: int,
    ordinal_offset: int,
    change_templates: tuple[str, ...],
    timeout_messages: tuple[str, ...],
    diagnostic_change_templates: tuple[str, ...],
    diagnostic_timeout_messages: tuple[str, ...],
) -> tuple[list[GRPOCorpusRow], int]:
    rows: list[GRPOCorpusRow] = []
    cursor = start_index
    for local_index in range(count):
        cursor, source = _next_retry_source(cursor)
        ordinal = ordinal_offset + local_index
        scenario: DecisionScenario = (
            "change_arguments" if ordinal % 2 == 0 else "retry_same_arguments"
        )
        evidence_style: EvidenceStyle = (
            "explicit_instruction"
            if (ordinal // 2) % 2 == 0
            else "diagnostic_evidence"
        )
        if evidence_style == "explicit_instruction":
            change_template = change_templates[ordinal % len(change_templates)]
            timeout_message = timeout_messages[ordinal % len(timeout_messages)]
        else:
            change_template = diagnostic_change_templates[
                ordinal % len(diagnostic_change_templates)
            ]
            timeout_message = diagnostic_timeout_messages[
                ordinal % len(diagnostic_timeout_messages)
            ]
        rows.append(
            derive_decision_loop_case(
                source,
                ordinal=ordinal,
                scenario=scenario,
                evidence_style=evidence_style,
                change_message_template=change_template,
                timeout_message=timeout_message,
            )
        )
    return rows, cursor


def _write_jsonl(path: Path, rows: list[GRPOCorpusRow]) -> str:
    path.write_text(
        "\n".join(
            json.dumps(
                row.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    output_dir: Path,
    *,
    start_index: int = 70000,
    train_count: int = 512,
    validation_count: int = 64,
    test_count: int = 128,
) -> dict[str, Any]:
    if min(train_count, validation_count, test_count) < 2:
        raise ValueError("each split needs at least two rows for paired scenarios")
    if not set(_TRAIN_TEMPLATES).isdisjoint(_BLIND_TEMPLATES):
        raise ValueError("change-argument blind templates overlap training")
    if not set(_TRAIN_TIMEOUT_MESSAGES).isdisjoint(_BLIND_TIMEOUT_MESSAGES):
        raise ValueError("timeout blind templates overlap training")
    if not set(_TRAIN_DIAGNOSTIC_CHANGE_MESSAGES).isdisjoint(
        _BLIND_DIAGNOSTIC_CHANGE_MESSAGES
    ):
        raise ValueError("diagnostic change blind templates overlap training")
    if not set(_TRAIN_DIAGNOSTIC_TIMEOUT_MESSAGES).isdisjoint(
        _BLIND_DIAGNOSTIC_TIMEOUT_MESSAGES
    ):
        raise ValueError("diagnostic timeout blind templates overlap training")

    cursor = start_index
    train, cursor = _build_split(
        start_index=cursor,
        count=train_count,
        ordinal_offset=0,
        change_templates=_TRAIN_TEMPLATES,
        timeout_messages=_TRAIN_TIMEOUT_MESSAGES,
        diagnostic_change_templates=_TRAIN_DIAGNOSTIC_CHANGE_MESSAGES,
        diagnostic_timeout_messages=_TRAIN_DIAGNOSTIC_TIMEOUT_MESSAGES,
    )
    validation, cursor = _build_split(
        start_index=cursor,
        count=validation_count,
        ordinal_offset=train_count,
        change_templates=_TRAIN_TEMPLATES,
        timeout_messages=_TRAIN_TIMEOUT_MESSAGES,
        diagnostic_change_templates=_TRAIN_DIAGNOSTIC_CHANGE_MESSAGES,
        diagnostic_timeout_messages=_TRAIN_DIAGNOSTIC_TIMEOUT_MESSAGES,
    )
    test, cursor = _build_split(
        start_index=cursor,
        count=test_count,
        ordinal_offset=train_count + validation_count,
        change_templates=_BLIND_TEMPLATES,
        timeout_messages=_BLIND_TIMEOUT_MESSAGES,
        diagnostic_change_templates=_BLIND_DIAGNOSTIC_CHANGE_MESSAGES,
        diagnostic_timeout_messages=_BLIND_DIAGNOSTIC_TIMEOUT_MESSAGES,
    )
    splits = {"train": train, "validation": validation, "test": test}

    ids = {name: {row.task.task_id for row in rows} for name, rows in splits.items()}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if ids[left] & ids[right]:
            raise ValueError(f"task leakage detected between {left} and {right}")
    fingerprints = [
        environment_fingerprint(row.task, row.snapshot) for rows in splits.values() for row in rows
    ]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("duplicate environment fingerprints detected")

    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        name: _write_jsonl(output_dir / f"{name}.jsonl", rows) for name, rows in splits.items()
    }
    scenario_counts = {
        name: dict(
            Counter(
                row.snapshot.hidden_test_facts["decision_loop_curriculum"]["scenario"]
                for row in rows
            )
        )
        for name, rows in splits.items()
    }
    evidence_style_counts = {
        name: dict(
            Counter(
                row.snapshot.hidden_test_facts["decision_loop_curriculum"][
                    "evidence_style"
                ]
                for row in rows
            )
        )
        for name, rows in splits.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": "controller-first online GRPO recovery decision curriculum",
        "start_index": start_index,
        "next_unused_index": cursor,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "scenario_counts": scenario_counts,
        "evidence_style_counts": evidence_style_counts,
        "split_sha256": hashes,
        "train_test_message_overlap": False,
        "task_overlap": [],
        "environment_fingerprint_overlap": [],
        "reward_contract": {
            "external_failure_turn": 0.0,
            "invalid_model_turn": -1.0,
            "successful_recovery": "inherits verified terminal and local turn credit",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=70000)
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--validation-count", type=int, default=64)
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
