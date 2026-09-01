"""Fail-closed quality checks for user-visible verifier-repair reasons.

The post-training reward must not treat one copied evidence substring as a
complete explanation.  This module keeps the checks deterministic and limited
to information already visible to the policy: the verifier message, grounding
phrases, and the selected public action.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_ASCII_LETTER = re.compile(r"[A-Za-z]")
_DAY_ANCHOR = re.compile(r"第\s*\d+\s*天")
_VALUE_ANCHOR = re.compile(r"\d+(?:\.\d+)?\s*(?:分钟|小时|元|天|公里|千米|%)")
_PRIVATE_MARKERS = (
    "policy_state",
    "hidden_test_facts",
    "controller",
    "decision_verifier_repair",
    "verifier",
    "verification",
    "propose_tradeoff",
    "retry_solve",
    "accept_itinerary",
    "abort",
    "verifier-repair-",
)
_ACTION_RATIONALE_CUES = {
    "retry_solve": (
        "重试",
        "重算",
        "重排",
        "重新规划",
        "重新求解",
        "调整顺序",
        "有界修复",
        "retry",
        "recompute",
        "reschedule",
    ),
    "propose_tradeoff": (
        "选择",
        "取舍",
        "确认",
        "放宽",
        "调整方案",
        "trade-off",
        "tradeoff",
        "choose",
        "confirm",
    ),
    "abort": (
        "停止",
        "终止",
        "结束规划",
        "无法继续",
        "不能继续",
        "无解",
        "没有安全",
        "abort",
        "stop",
        "cannot continue",
    ),
}

_RATIONALE_PREFIXES = {
    "retry_solve": (
        "该问题仍可在现有约束内修复，应先调整顺序并进行一次有界重算",
        "排程仍有可调整空间，应保持用户约束不变并重新求解",
        "当前冲突可以修复，应先重排活动并再次计算",
        "无需用户放宽要求，可用有界策略重算当前行程",
        "该冲突可在不改变现有要求的情况下重排，因此先重新规划一次",
        "现有约束仍留有修复余地，应调整活动次序后重试",
        "这是可恢复的排程冲突，可在保持要求不变时再次重算",
        "当前失败尚未到取舍或终止阶段，应先通过重排进行有界修复",
    ),
    "propose_tradeoff": (
        "现有要求无法同时满足，需要您从已授权的调整方案中作出选择",
        "继续规划需要放宽一项要求，请您确认接受哪一种调整方案",
        "这项冲突约束必须由您决定如何取舍后才能继续",
        "该约束只能经您同意后放宽，请先选择可接受的方案",
        "单靠重排无法同时满足这些要求，需要您选择允许放宽的约束",
        "继续执行涉及用户偏好取舍，请先确认可接受的调整",
        "可行路径不止一种但都需改变要求，应由您选择后再继续",
        "当前冲突需要用户授权调整，未确认取舍前不能继续规划",
    ),
    "abort": (
        "相关约束均已锁定且没有安全可行的调整空间，因此必须停止规划",
        "在不违背已锁定要求的前提下无法继续，只能终止本次规划",
        "现有硬约束无解且没有获准的替代方案，因此需要结束规划",
        "没有安全方式可以同时满足这些要求，应停止而不能编造结果",
        "所有可调整余地都已耗尽且不存在安全替代，应终止当前规划",
        "锁定要求之间不可调和且没有获准替代方案，只能停止规划",
        "继续将违反硬约束且无安全绕行路径，因此不能继续规划",
        "当前条件下不存在合规可行解，应结束规划而非返回不实方案",
    ),
}


def normalize_reason_text(value: Any) -> str:
    rendered = str(value or "")
    return "".join(character.casefold() for character in rendered if character.isalnum())


def repair_reason_rationale_prefixes(target_action: str) -> tuple[str, ...]:
    """Return the audited public rationale prefixes for one repair action."""
    try:
        return _RATIONALE_PREFIXES[target_action]
    except KeyError as exc:
        raise ValueError(
            f"unsupported verifier-repair action: {target_action}"
        ) from exc


def build_grounded_repair_reason(evidence: str, target_action: str) -> str:
    """Create a public teacher reason without model-owned controller fields.

    Put the action rationale before the copied verifier evidence.  Earlier
    curricula used ``evidence + rationale``; a continued adapter could then
    reproduce its old evidence-only completion and terminate before learning
    the new rationale.  Rationale-first wording removes that prefix shortcut
    while preserving every user-visible evidence anchor.
    """
    evidence = str(evidence or "").strip().rstrip("。.!！")
    if not evidence:
        raise ValueError("verifier evidence is required for a repair reason")
    suffixes = repair_reason_rationale_prefixes(target_action)
    index = int(
        hashlib.sha256(f"{target_action}:{evidence}".encode("utf-8")).hexdigest()[:8],
        16,
    ) % len(suffixes)
    rationale = suffixes[index]
    return f"{rationale}：{evidence}。"


def verifier_reason_quality_checks(
    *,
    reason: Any,
    target_action: str,
    grounding_phrases: list[str],
    evidence: str,
) -> dict[str, bool]:
    """Return independent quality gates used by reward and evaluation.

    Full grounding requires *all* declared evidence phrases.  Specificity also
    preserves visible day/value anchors, preventing generic summaries from
    passing.  Language alignment is intentionally lightweight and deterministic:
    a Chinese verifier message cannot be answered mainly in English with only
    copied Chinese place names.
    """

    rendered = str(reason or "").strip()
    normalized = normalize_reason_text(rendered)
    phrases = [str(item).strip() for item in grounding_phrases if str(item).strip()]
    phrase_match = bool(phrases) and all(
        normalize_reason_text(phrase) in normalized for phrase in phrases
    )

    anchors = [*_DAY_ANCHOR.findall(evidence), *_VALUE_ANCHOR.findall(evidence)]
    anchor_match = all(normalize_reason_text(anchor) in normalized for anchor in anchors)
    specificity_match = phrase_match and anchor_match

    reference_cjk = len(_CJK.findall(evidence))
    reason_cjk = len(_CJK.findall(rendered))
    reason_ascii = len(_ASCII_LETTER.findall(rendered))
    if reference_cjk >= 4:
        language_match = reason_cjk >= 4 and reason_cjk >= reason_ascii
    else:
        language_match = reason_ascii >= 4

    lowered = rendered.casefold()
    public_language = not any(marker in lowered for marker in _PRIVATE_MARKERS)
    rationale_cues = _ACTION_RATIONALE_CUES.get(target_action, ())
    rationale_match = bool(rationale_cues) and any(
        cue.casefold() in lowered for cue in rationale_cues
    )

    return {
        "grounding_match": phrase_match,
        "reason_specificity_match": specificity_match,
        "reason_language_match": language_match,
        "reason_public_language": public_language,
        "reason_action_rationale_match": rationale_match,
    }
