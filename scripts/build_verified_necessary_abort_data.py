"""Build verifier-grounded SFT and DPO data for necessary safe termination.

The source must be the official GRPO training split.  Each derived task makes
the absence of acceptable alternatives explicit in the model-visible
capability contract.  A real rollout with ``abort`` is the positive; a second
rollout with ``propose_tradeoff`` is rejected by the reward gate.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.corpus_generation import CurriculumTeacherPolicy  # noqa: E402
from agentic.distillation import (  # noqa: E402
    TeacherPreferencePair,
    build_teacher_candidate,
    select_teacher_group,
)
from agentic.environment import TravelAgentEnvironment  # noqa: E402
from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402
from agentic.loop import PolicyAction, PolicyContext  # noqa: E402
from agentic.policy import ControllerFirstPolicy  # noqa: E402
from agentic.sft_dataset import EpisodeCandidate, SFTDatasetBuilder  # noqa: E402


DERIVATION_VERSION = "necessary-abort-decision.v1"
_SCENARIO_KINDS = (
    "fixed_closure",
    "fixed_schedule",
    "unsafe_access",
    "missing_required_tool",
    "exhausted_recovery",
)


class WrongTradeoffPolicy(CurriculumTeacherPolicy):
    """Offer an option even though the visible contract says none is acceptable."""

    async def propose(self, context: PolicyContext) -> PolicyAction:
        if (
            context.capability.get("status") in {"infeasible", "unsafe", "missing_tool"}
            and context.capability.get("actionable_alternatives") is False
            and "propose_tradeoff" in context.allowed_actions
        ):
            evidence = [str(item) for item in context.capability.get("evidence") or []]
            return PolicyAction(
                action="propose_tradeoff",
                arguments={
                    "reason": evidence[0] if evidence else "当前条件不可同时满足",
                    "options": ["放宽已经确认的硬约束"],
                },
            )
        return await super().propose(context)


def is_eligible(row: GRPOCorpusRow) -> bool:
    return bool(
        not row.task.missing_slots
        and row.task.feasibility_report.get("feasible", True) is False
        and row.task.feasibility_report.get("reasons")
    )


def derive_necessary_abort(row: GRPOCorpusRow) -> GRPOCorpusRow:
    if not is_eligible(row):
        raise ValueError("necessary-abort derivation requires a grounded infeasible task")
    derived = row.model_copy(deep=True)
    kind = _SCENARIO_KINDS[
        int(hashlib.sha256(row.task.task_id.encode()).hexdigest()[:8], 16)
        % len(_SCENARIO_KINDS)
    ]
    city = str(row.task.slots.get("destination") or "目的地")
    travel_date = str(row.task.slots.get("start_date") or "已确认日期")
    request, reason, status = _scenario_text(kind, city=city, travel_date=travel_date)
    derived.task.task_id = f"{row.task.task_id}-necessary-abort-{kind}"
    derived.task.template_family = f"{row.task.template_family}-necessary-abort-{kind}"
    derived.task.difficulty = "L4"
    derived.task.user_request = request
    derived.task.feasibility_report = {
        "feasible": False,
        "status": status,
        "reasons": [reason],
        "actionable_alternatives": False,
        "alternatives": [],
    }
    derived.snapshot.environment_version = DERIVATION_VERSION
    derived.snapshot.snapshot_version = DERIVATION_VERSION
    derived.snapshot.state_id = f"{row.snapshot.state_id}-necessary-abort"
    derived.snapshot.hidden_test_facts["necessary_abort_decision"] = {
        "expected_action": "abort",
        "source_task_id": row.task.task_id,
        "scenario_kind": kind,
        "capability_status": status,
        "actionable_alternatives": False,
    }
    return derived


def _scenario_text(kind: str, *, city: str, travel_date: str) -> tuple[str, str, str]:
    scenarios = {
        "fixed_closure": (
            f"我只在{travel_date}去{city}指定的专题馆，受信公告确认当天全天闭馆；日期和场馆都不能换，也不接受周边替代。",
            "受信开放日历确认指定场馆全天关闭，且日期、场馆和替代项均被锁定",
            "infeasible",
        ),
        "fixed_schedule": (
            f"请在{travel_date}于{city}完成两个固定活动：前一个结束后十分钟内必须到九十分钟车程外的后一个地点，时间和地点都不能改。",
            "冻结路线矩阵显示两项固定活动之间至少需要九十分钟，但可用衔接窗口只有十分钟",
            "infeasible",
        ),
        "unsafe_access": (
            f"{travel_date}{city}的山地景区因地质风险已被主管部门封闭；我仍要求越过封控进入，不接受延期或其他地点。",
            "主管部门的有效封闭通知构成安全硬门，且用户要求绕过封控",
            "unsafe",
        ),
        "missing_required_tool": (
            f"我在{travel_date}去{city}，行程必须以指定场馆的实时余票确认结果为准；当前唯一受信票务接口不可用，也不允许使用估算或替代场馆。",
            "完成任务所必需的实时票务能力不可用，且禁止使用未经验证的数据或替代目标",
            "missing_tool",
        ),
        "exhausted_recovery": (
            f"{travel_date}在{city}参加指定的季节限定活动；系统已经按两个不同关键词完成受控检索，均没有安全可用结果，活动和日期都不能换。",
            "两次受控检索均未找到安全可用结果，恢复预算已耗尽且用户锁定活动与日期",
            "infeasible",
        ),
    }
    try:
        return scenarios[kind]
    except KeyError as exc:
        raise ValueError(f"unknown necessary-abort scenario kind: {kind}") from exc


def _stable(rows: list[GRPOCorpusRow]) -> list[GRPOCorpusRow]:
    return sorted(rows, key=lambda row: hashlib.sha256(row.task.task_id.encode()).hexdigest())


def _first_policy_action(candidate: Any) -> str | None:
    for step in candidate.rollout.episode.steps:
        if step.action.decision_source != "controller":
            return step.action.action
    return None


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _split_preferences(
    pairs: list[TeacherPreferencePair],
) -> dict[str, list[TeacherPreferencePair]]:
    ordered = sorted(pairs, key=lambda item: _canonical_hash(item.pair_id))
    if len(ordered) < 3:
        raise ValueError("at least three preferences are required")
    validation = max(1, round(len(ordered) * 0.1))
    test = max(1, round(len(ordered) * 0.1))
    return {
        "validation": ordered[:validation],
        "test": ordered[validation : validation + test],
        "train": ordered[validation + test :],
    }


def _pilot_requests(path: Path | None) -> list[str]:
    if path is None:
        return []
    requests: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        for message in case.get("messages") or []:
            if message.get("role") != "user":
                continue
            try:
                payload = json.loads(message.get("content") or "{}")
            except json.JSONDecodeError:
                continue
            request = str(payload.get("original_request") or "").strip()
            if request:
                requests.append(request)
    return requests


def _normalize(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.casefold())


def _contamination_audit(rows: list[GRPOCorpusRow], pilot_path: Path | None) -> dict[str, Any]:
    training = [row.task.user_request for row in rows]
    pilot = _pilot_requests(pilot_path)
    normalized_pilot = {_normalize(item) for item in pilot}
    exact = [item for item in training if _normalize(item) in normalized_pilot]
    maximum = 0.0
    closest: dict[str, str] | None = None
    for train_request in training:
        for pilot_request in pilot:
            ratio = SequenceMatcher(
                None, _normalize(train_request), _normalize(pilot_request), autojunk=False
            ).ratio()
            if ratio > maximum:
                maximum = ratio
                closest = {"training": train_request, "pilot": pilot_request}
    return {
        "pilot_file": str(pilot_path) if pilot_path else None,
        "pilot_requests": len(pilot),
        "exact_normalized_overlap": len(exact),
        "max_sequence_similarity": round(maximum, 6),
        "closest_pair": closest,
        "passed": not exact,
    }


async def build(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = load_grpo_corpus(args.source_file)
    eligible = _stable([row for row in source_rows if is_eligible(row)])
    unique: dict[str, GRPOCorpusRow] = {}
    for source_row in eligible:
        derived = derive_necessary_abort(source_row)
        visible_key = _canonical_hash(
            {
                "original_request": derived.task.user_request,
                "slots": derived.task.slots,
                "profile": derived.task.profile,
                "missing_slots": derived.task.missing_slots,
                "feasibility_report": derived.task.feasibility_report,
            }
        )
        unique.setdefault(visible_key, derived)
    rows = list(unique.values())[: args.limit]
    if len(rows) < args.limit:
        raise ValueError(f"only {len(rows)} unique eligible source rows for limit={args.limit}")
    contamination = _contamination_audit(rows, args.forbidden_pilot_file)
    if not contamination["passed"]:
        raise ValueError("necessary-abort data overlaps the AI-assisted Pilot text")

    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(row: GRPOCorpusRow):
        async with semaphore:
            chosen_rollout, rejected_rollout = await asyncio.gather(
                TravelAgentEnvironment(row.task, row.snapshot).rollout(
                    ControllerFirstPolicy(CurriculumTeacherPolicy())
                ),
                TravelAgentEnvironment(row.task, row.snapshot).rollout(
                    ControllerFirstPolicy(WrongTradeoffPolicy())
                ),
            )
        family = "necessary_abort_" + str(
            row.snapshot.hidden_test_facts["necessary_abort_decision"]["scenario_kind"]
        )
        chosen = build_teacher_candidate(chosen_rollout, family=family, sample_index=0)
        rejected = build_teacher_candidate(
            rejected_rollout, family=family, sample_index=1
        )
        selection = select_teacher_group([chosen, rejected])
        if _first_policy_action(selection.chosen) != "abort":
            raise ValueError(f"{row.task.task_id}: verifier did not choose abort")
        if rejected.score.successful:
            raise ValueError(f"{row.task.task_id}: wrong tradeoff unexpectedly succeeded")
        if not rejected.rollout.reward.audit_metrics.get("termination_action_mismatch"):
            raise ValueError(f"{row.task.task_id}: missing termination mismatch evidence")
        if len(selection.preference_pairs) != 1:
            raise ValueError(f"{row.task.task_id}: expected one preference pair")
        return selection.chosen, rejected, selection.preference_pairs[0]

    results = await asyncio.gather(*(one(row) for row in rows))
    chosen = [item[0] for item in results]
    rejected = [item[1] for item in results]
    pairs = [item[2] for item in results]

    sft_candidates = [
        EpisodeCandidate(
            scenario_id=row.task.task_id,
            source="synthetic",
            template_family=row.task.template_family,
            city=str(row.task.slots.get("destination") or "unknown"),
            episode=candidate.rollout.episode,
        )
        for row, candidate in zip(rows, chosen, strict=True)
    ]
    sft_builder = SFTDatasetBuilder()
    sft_result = sft_builder.build(sft_candidates)
    if sft_result.manifest.rejected_episodes or sft_result.manifest.exported_examples != len(rows):
        raise ValueError(
            "SFT verifier export rejected necessary aborts: "
            f"{sft_result.manifest.model_dump(mode='json')}"
        )

    context_hashes = [pair.context_hash for pair in pairs]
    payload_hashes = [
        _canonical_hash({"messages": pair.messages, "tools": pair.tools}) for pair in pairs
    ]
    if len(context_hashes) != len(set(context_hashes)):
        raise ValueError("duplicate model-visible preference contexts")
    if len(payload_hashes) != len(set(payload_hashes)):
        raise ValueError("duplicate model-visible preference payloads")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = args.output_dir / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    _write_jsonl(corpus_dir / "train.jsonl", [row.model_dump(mode="json") for row in rows])
    sft_builder.export(sft_result, args.output_dir / "sft")
    preference_dir = args.output_dir / "preferences"
    preference_dir.mkdir(exist_ok=True)
    _write_jsonl(
        preference_dir / "chosen_candidates.jsonl",
        [item.model_dump(mode="json") for item in chosen],
    )
    _write_jsonl(
        preference_dir / "rejected_candidates.jsonl",
        [item.model_dump(mode="json") for item in rejected],
    )
    _write_jsonl(
        preference_dir / "preference_pairs.jsonl",
        [item.model_dump(mode="json") for item in pairs],
    )
    preference_splits = _split_preferences(pairs)
    for split, split_rows in preference_splits.items():
        _write_jsonl(
            preference_dir / f"{split}.jsonl",
            [item.model_dump(mode="json") for item in split_rows],
        )
    preference_version = "preference-necessary-abort-" + _canonical_hash(
        {
            split: [item.pair_id for item in split_rows]
            for split, split_rows in preference_splits.items()
        }
    )[:16]
    preference_manifest = {
        "schema_version": "verified-preference-dataset.v1",
        "status": "passed",
        "dataset_version": preference_version,
        "requires_verifier_success_over_failure": True,
        "unique_pairs": len(pairs),
        "unique_contexts": len(set(context_hashes)),
        "unique_model_payloads": len(set(payload_hashes)),
        "family_counts": dict(Counter(pair.family for pair in pairs)),
        "reason_counts": dict(Counter(reason for pair in pairs for reason in pair.reason_codes)),
        "split_counts": {
            split: len(split_rows) for split, split_rows in preference_splits.items()
        },
        "frozen_holdout_payload_overlap": 0,
        "pilot_exact_normalized_overlap": contamination["exact_normalized_overlap"],
        "errors": [],
    }
    (preference_dir / "manifest.json").write_text(
        json.dumps(preference_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "verified-necessary-abort-data.v1",
        "status": "passed",
        "source_file": str(args.source_file),
        "source_file_sha256": _file_sha256(args.source_file),
        "source_scope": "official GRPO train only",
        "eligible_source_tasks": len(eligible),
        "duplicate_visible_source_tasks_dropped": len(eligible) - len(unique),
        "derived_tasks": len(rows),
        "scenario_kind_counts": dict(
            Counter(
                row.snapshot.hidden_test_facts["necessary_abort_decision"]["scenario_kind"]
                for row in rows
            )
        ),
        "capability_status_counts": dict(
            Counter(str(row.task.feasibility_report.get("status")) for row in rows)
        ),
        "chosen_action_counts": dict(Counter(_first_policy_action(item) for item in chosen)),
        "rejected_action_counts": dict(Counter(_first_policy_action(item) for item in rejected)),
        "verified_preference_pairs": len(pairs),
        "preference_split_counts": {
            split: len(split_rows) for split, split_rows in preference_splits.items()
        },
        "preference_dataset_version": preference_version,
        "sft_manifest": sft_result.manifest.model_dump(mode="json"),
        "unique_preference_contexts": len(set(context_hashes)),
        "unique_model_payloads": len(set(payload_hashes)),
        "verifier_gate": "TERMINATION_ACTION_MISMATCH",
        "pilot_contamination_audit": contamination,
        "training_decision": "repair_shard_only_not_yet_merged_or_trained",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forbidden-pilot-file", type=Path)
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    if args.limit < 3 or args.concurrency < 1:
        parser.error("limit must be at least 3 and concurrency must be positive")
    manifest = asyncio.run(build(args))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
