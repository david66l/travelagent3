"""Fail-closed quality checks for user-visible verifier-repair reasons.

The post-training reward must not treat one copied evidence substring as a
complete explanation.  This module keeps the checks deterministic and limited
to information already visible to the policy: the verifier message, grounding
phrases, and the selected public action.
"""

from __future__ import annotations

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
    "验证器",
    "控制器",
    "策略状态",
    "内部动作",
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

# One deterministic connector per action (H-001).  Every call site that
# recomputes a canonical reason for equality checks (SFT warm-start builder,
# DPO preference builder, DPO trainer validation, GRPO replay probe, reason
# audit) must agree, so the connector cannot fork by violation code unless
# every site threads the code through together.
_CANONICAL_RATIONALE_BY_ACTION: dict[str, str] = {
    "retry_solve": "该问题仍可在现有约束内修复，应先调整顺序并进行一次有界重算",
    "propose_tradeoff": "现有要求无法同时满足，需要您从已授权的调整方案中作出选择",
    "abort": "相关约束均已锁定且没有安全可行的调整空间，因此必须停止规划",
}

REPAIR_REASON_ASSEMBLY_VERSION = "repair-reason-assembly.v1"

# These patterns intentionally recognize only explicit action conclusions.
# Evidence-only text is allowed: the model selects the action and supplies the
# grounded detail, while the system owns the fixed action-to-rationale wording.
# A conservative detector avoids turning ordinary words such as "冲突" or
# "无解" into a guessed semantic label.
_EXPLICIT_ACTION_ASSERTIONS: dict[str, tuple[re.Pattern[str], ...]] = {
    "retry_solve": (
        re.compile(
            r"(?:仍|尚|还)?(?:可|可以|能够).{0,12}"
            r"(?:修复|重排|重算|重试|重新规划|重新求解)"
        ),
        re.compile(
            r"(?:无需|不需要).{0,12}(?:放宽|改变).{0,16}"
            r"(?:重排|重算|重试|规划)"
        ),
        re.compile(r"\bcan\s+(?:retry|recompute|reschedule)\b", re.IGNORECASE),
    ),
    "propose_tradeoff": (
        re.compile(
            r"(?:需要|请|必须由|应由).{0,16}(?:您|用户).{0,16}"
            r"(?:选择|确认|取舍)"
        ),
        re.compile(r"(?:需要|必须).{0,12}(?:放宽|调整).{0,16}(?:要求|约束|方案)"),
        re.compile(r"\b(?:user|you)\s+(?:must|need(?:s)?\s+to)\s+choose\b", re.IGNORECASE),
    ),
    "abort": (
        re.compile(r"(?:必须|只能|应当|需要).{0,8}(?:停止|终止|结束)"),
        re.compile(r"(?:无法|不能).{0,4}继续"),
        re.compile(r"(?:不存在|没有).{0,12}(?:安全|合规|可行).{0,8}(?:方案|路径|解)"),
        re.compile(r"\b(?:must\s+stop|cannot\s+continue)\b", re.IGNORECASE),
    ),
}

for _action, _canonical in _CANONICAL_RATIONALE_BY_ACTION.items():
    if _canonical not in _RATIONALE_PREFIXES[_action]:
        raise RuntimeError(
            "canonical repair rationale must come from the audited prefix pool: "
            f"{_action}"
        )


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


def canonical_repair_rationale(target_action: str) -> str:
    """Return the single deterministic rationale connector for a repair action.

    H-001 contract: the teacher reason must be a deterministic function of its
    inputs so the completion prefix stays learnable.  The earlier curriculum
    selected one of eight prefixes via ``sha256(action:evidence)``, which made
    the prefix tokens unpredictable by construction (measured prefix NLL 4.51
    vs evidence 0.057 on the v11 checkpoint) and left reason-quality SFT in a
    corner where SFT cannot converge, zero-variance GRPO cannot push, and the
    lexical eval only accepts licensed wording.  One canonical connector per
    action removes that entropy; user-visible variety is carried by the
    grounded evidence, not the connector.
    """
    try:
        return _CANONICAL_RATIONALE_BY_ACTION[target_action]
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
    while preserving every user-visible evidence anchor.  The connector is
    the canonical one for the action (see :func:`canonical_repair_rationale`).
    """
    evidence = str(evidence or "").strip().rstrip("。.!！")
    if not evidence:
        raise ValueError("verifier evidence is required for a repair reason")
    return f"{canonical_repair_rationale(target_action)}：{evidence}。"


def repair_reason_semantic_conflict(reason: Any, target_action: str) -> bool:
    """Detect an explicit action conclusion that contradicts ``target_action``.

    The detector is deliberately conservative.  It rejects a leading audited
    rationale for another action and clear action assertions, but does not try
    to infer intent from arbitrary evidence text.  Grounding and specificity
    remain separate deterministic checks.
    """

    repair_reason_rationale_prefixes(target_action)  # validate the target
    rendered = str(reason or "").strip()
    normalized = normalize_reason_text(rendered)
    for action, prefixes in _RATIONALE_PREFIXES.items():
        if action == target_action:
            continue
        if any(normalized.startswith(normalize_reason_text(prefix)) for prefix in prefixes):
            return True

    asserted_actions = {
        action
        for action, patterns in _EXPLICIT_ACTION_ASSERTIONS.items()
        if any(pattern.search(rendered) for pattern in patterns)
    }
    return bool(asserted_actions) and target_action not in asserted_actions


def _strip_matching_rationale_prefix(reason: str, target_action: str) -> str:
    for prefix in sorted(
        repair_reason_rationale_prefixes(target_action), key=len, reverse=True
    ):
        if reason.startswith(prefix):
            return reason[len(prefix) :].lstrip(" \t:：,，;；。.!！?？")
    return reason


def assemble_repair_reason(reason: Any, target_action: str) -> str:
    """Render one user-visible repair reason at the production boundary.

    The raw model text remains available in ``PolicyAction.model_arguments``.
    This function only replaces a matching leading rationale with the single
    system-owned connector; it never changes the selected action or evidence
    detail.  Explicit cross-action conclusions and private implementation
    wording fail closed before rendering.
    """

    rendered = str(reason or "").strip()
    if not rendered:
        raise ValueError("REPAIR_REASON_DETAIL_EMPTY")
    lowered = rendered.casefold()
    if any(marker in lowered for marker in _PRIVATE_MARKERS):
        raise ValueError("REPAIR_REASON_PRIVATE_CONTENT")
    if repair_reason_semantic_conflict(rendered, target_action):
        raise ValueError("REPAIR_REASON_ACTION_CONFLICT")
    detail = _strip_matching_rationale_prefix(rendered, target_action).strip()
    detail = detail.rstrip("。.!！?？")
    if not detail:
        raise ValueError("REPAIR_REASON_DETAIL_EMPTY")
    return f"{canonical_repair_rationale(target_action)}：{detail}。"


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


def verifier_reason_semantic_checks(
    *,
    reason: Any,
    target_action: str,
    grounding_phrases: list[str],
    evidence: str,
) -> dict[str, bool]:
    """Score model-owned evidence semantics without the system connector.

    This is the guard against connector-only false success: the model still
    has to ground its detail, preserve concrete anchors, use public language,
    and avoid explicitly arguing for a different action.
    """

    checks = verifier_reason_quality_checks(
        reason=reason,
        target_action=target_action,
        grounding_phrases=grounding_phrases,
        evidence=evidence,
    )
    checks.pop("reason_action_rationale_match")
    checks["reason_action_semantic_consistent"] = not repair_reason_semantic_conflict(
        reason, target_action
    )
    return checks
