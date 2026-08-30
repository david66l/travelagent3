"""Build an orthogonal, leakage-safe Stage 3 decision-loop curriculum.

V2 balanced marginal scenario counts but coupled each semantic stratum to one
city, one train template, and one recovery target position. V3 rotates factors
inside each stratum and reserves disjoint template families for train,
validation, and test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agentic.environment import environment_fingerprint  # noqa: E402
from build_stage3_decision_loop_curriculum import (  # noqa: E402
    derive_decision_loop_case,
)
from build_stage3_multiturn_rl_corpus import _next_retry_source  # noqa: E402


SCHEMA_VERSION = "stage3-decision-loop-curriculum.v3"
_STRATA = (
    ("change_arguments", "explicit_instruction"),
    ("retry_same_arguments", "explicit_instruction"),
    ("change_arguments", "diagnostic_evidence"),
    ("retry_same_arguments", "diagnostic_evidence"),
)

_TEMPLATES = {
    "train": {
        "change_arguments/explicit_instruction": (
            "删除“{drop}”，下一轮仅使用“{target}”。",
            "把查询收窄为“{target}”，不要继续携带“{drop}”。",
            "重试时保留“{target}”并移除“{drop}”。",
            "下一次只提交“{target}”，舍弃“{drop}”。",
        ),
        "retry_same_arguments/explicit_instruction": (
            "参数有效，仅发生瞬时超时；请使用完全相同的关键词重试。",
            "不要改变查询条件，上游恢复后原样重放本次请求。",
            "本次失败与参数无关，请保持关键词不变再次调用。",
            "查询已经通过校验；用同一组参数安全重试一次。",
        ),
        "change_arguments/diagnostic_evidence": (
            "召回诊断显示“{drop}”噪声过高，“{target}”仍有稳定区分度；本轮无候选。",
            "质量报告中“{drop}”使结果空间失焦，“{target}”保持有效相关性。",
            "“{drop}”对应结果过泛，“{target}”通过了选择性检查；组合查询被拒绝。",
            "候选分析发现主要噪声来自“{drop}”，“{target}”仍是可靠检索信号。",
            "检索熵因“{drop}”超过阈值，“{target}”的相关性分布仍然集中。",
            "画像匹配中“{drop}”覆盖面异常宽，“{target}”可以区分用户偏好。",
            "排序器无法处理“{drop}”引入的无关召回，“{target}”单项信号正常。",
            "查询审计将“{drop}”标为低选择性条件，“{target}”仍满足质量门槛。",
        ),
        "retry_same_arguments/diagnostic_evidence": (
            "输入校验已通过，候选计算前发生瞬时超时，本轮没有结果。",
            "查询条件被服务接受，但执行节点短暂中断，未生成候选。",
            "参数解析成功，上游网络随后超时，结果列表为空。",
            "请求已进入检索队列，计算阶段遇到可恢复故障。",
        ),
    },
    "validation": {
        "change_arguments/explicit_instruction": (
            "后续检索只保留“{target}”，并去掉“{drop}”。",
            "请以“{target}”作为唯一条件重试，不再发送“{drop}”。",
        ),
        "retry_same_arguments/explicit_instruction": (
            "条件无需修正；短暂故障结束后照原参数再请求一次。",
            "保持本轮查询内容不变，这是可以原样重试的服务超时。",
        ),
        "change_arguments/diagnostic_evidence": (
            "相关性检查表明“{drop}”稀释了候选，“{target}”仍提供清晰信号。",
            "“{drop}”造成召回边界扩散，“{target}”的匹配质量保持正常。",
            "搜索质量门将“{drop}”识别为噪声来源，“{target}”仍然有效。",
            "候选空间受“{drop}”干扰而失真，“{target}”的区分能力未下降。",
        ),
        "retry_same_arguments/diagnostic_evidence": (
            "请求内容通过检查，服务在实际计算时临时失联。",
            "查询被正常接收，候选返回前出现短时网络故障。",
        ),
    },
    "test": {
        "change_arguments/explicit_instruction": (
            "把下一次请求缩减为“{target}”，删除“{drop}”。",
            "不要再带“{drop}”，仅以“{target}”重新检索。",
            "重试参数中只能留下“{target}”，移除“{drop}”。",
        ),
        "retry_same_arguments/explicit_instruction": (
            "这是执行端瞬时失败，下一次应复用当前全部关键词。",
            "查询参数没有问题，请不作改写地重新发送。",
            "服务故障可恢复，保持原请求内容再调用一次。",
        ),
        "change_arguments/diagnostic_evidence": (
            "召回画像显示“{drop}”引起主题漂移，“{target}”仍与需求高度相关。",
            "结果分析把“{drop}”判为宽泛条件，“{target}”仍具备筛选价值。",
            "“{drop}”导致无关候选占比超标，“{target}”的精确信号仍在。",
            "检索边界因“{drop}”失去聚焦，“{target}”仍通过相关性门禁。",
            "候选分布被“{drop}”拉宽，“{target}”维持了有效选择性。",
            "质量模型认为“{drop}”贡献主要噪声，“{target}”仍可独立检索。",
        ),
        "retry_same_arguments/diagnostic_evidence": (
            "合法性检查完成后执行器暂时不可用，因此没有候选返回。",
            "服务已确认查询有效，但候选生成节点发生短暂故障。",
            "输入参数被接受，检索运行阶段遇到瞬时网络中断。",
        ),
    },
}


def _factor_index(local_index: int) -> int:
    """Rotate every stratum across every four-position source-city block."""
    return (local_index % 4 + local_index // 4) % 4


def _build_split(
    *, start_index: int, count: int, ordinal_offset: int, split: str
) -> tuple[list[Any], int]:
    if count < 32 or count % 32:
        raise ValueError("each v3 split count must be a multiple of 32")
    rows: list[Any] = []
    cursor = start_index
    occurrences: Counter[int] = Counter()
    for local_index in range(count):
        cursor, source = _next_retry_source(cursor)
        ordinal = ordinal_offset + local_index
        factor_index = _factor_index(local_index)
        scenario, evidence_style = _STRATA[factor_index]
        occurrence = occurrences[factor_index]
        occurrences[factor_index] += 1
        # Every stratum visits all four cities before changing target position.
        target_index = (occurrence // 4) % 2
        key = f"{scenario}/{evidence_style}"
        templates = _TEMPLATES[split][key]
        template_index = (occurrence // 8 + occurrence % 4) % len(templates)
        template = templates[template_index]
        row = derive_decision_loop_case(
            source,
            ordinal=ordinal,
            scenario=scenario,
            evidence_style=evidence_style,
            change_message_template=(
                template
                if scenario == "change_arguments"
                else _TEMPLATES[split][f"change_arguments/{evidence_style}"][
                    template_index
                    % len(_TEMPLATES[split][f"change_arguments/{evidence_style}"])
                ]
            ),
            timeout_message=(
                template
                if scenario == "retry_same_arguments"
                else _TEMPLATES[split][f"retry_same_arguments/{evidence_style}"][
                    template_index
                    % len(_TEMPLATES[split][f"retry_same_arguments/{evidence_style}"])
                ]
            ),
            target_index=target_index,
        )
        metadata = row.snapshot.hidden_test_facts["decision_loop_curriculum"]
        metadata.update(
            {
                "factor_schedule": "orthogonal-latin-rotation.v1",
                "target_position": target_index,
                "failure_template_id": f"{split}:{key}:{template_index}",
            }
        )
        row.snapshot.environment_version = SCHEMA_VERSION
        row.snapshot.snapshot_version = SCHEMA_VERSION
        row.snapshot.state_id += "-v3"
        row.task.template_family += "-v3"
        rows.append(row)
    return rows, cursor


def _write_jsonl(path: Path, rows: list[Any]) -> str:
    path.write_text(
        "\n".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coverage(rows: list[Any]) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        metadata = row.snapshot.hidden_test_facts["decision_loop_curriculum"]
        grouped[f"{metadata['scenario']}/{metadata['evidence_style']}"].append(row)
    output: dict[str, Any] = {}
    for stratum, items in sorted(grouped.items()):
        metadata = [item.snapshot.hidden_test_facts["decision_loop_curriculum"] for item in items]
        output[stratum] = {
            "tasks": len(items),
            "cities": dict(sorted(Counter(item.task.slots["destination"] for item in items).items())),
            "template_ids": dict(
                sorted(Counter(item["failure_template_id"] for item in metadata).items())
            ),
            "target_positions": dict(
                sorted(Counter(str(item["target_position"]) for item in metadata).items())
            ),
        }
    return output


def _validate_coverage(split: str, coverage: dict[str, Any]) -> None:
    for stratum, facts in coverage.items():
        if len(facts["cities"]) < 4:
            raise ValueError(f"{split}/{stratum} does not cover four cities")
        if len(facts["template_ids"]) < 2:
            raise ValueError(f"{split}/{stratum} does not cover multiple templates")
        if stratum.startswith("change_arguments/") and set(facts["target_positions"]) != {
            "0",
            "1",
        }:
            raise ValueError(f"{split}/{stratum} does not balance target position")


def build(
    output_dir: Path,
    *,
    start_index: int = 100000,
    train_count: int = 512,
    validation_count: int = 64,
    test_count: int = 128,
) -> dict[str, Any]:
    cursor = start_index
    splits: dict[str, list[Any]] = {}
    offset = 0
    for split, count in (
        ("train", train_count),
        ("validation", validation_count),
        ("test", test_count),
    ):
        splits[split], cursor = _build_split(
            start_index=cursor,
            count=count,
            ordinal_offset=offset,
            split=split,
        )
        offset += count

    template_ids = {
        split: {
            row.snapshot.hidden_test_facts["decision_loop_curriculum"]["failure_template_id"]
            for row in rows
        }
        for split, rows in splits.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if template_ids[left] & template_ids[right]:
            raise ValueError(f"template IDs overlap between {left} and {right}")
    fingerprints = [
        environment_fingerprint(row.task, row.snapshot)
        for rows in splits.values()
        for row in rows
    ]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("environment fingerprints overlap")

    coverage = {split: _coverage(rows) for split, rows in splits.items()}
    for split, facts in coverage.items():
        _validate_coverage(split, facts)
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        split: _write_jsonl(output_dir / f"{split}.jsonl", rows)
        for split, rows in splits.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": "orthogonal controller-first decision-loop curriculum",
        "factor_schedule": "orthogonal-latin-rotation.v1",
        "counts": {split: len(rows) for split, rows in splits.items()},
        "coverage": coverage,
        "split_sha256": hashes,
        "template_family_overlap": [],
        "environment_fingerprint_overlap": [],
        "next_unused_index": cursor,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=100000)
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
