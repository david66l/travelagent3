"""Turn-to-token advantage projection for the GRPO-R1 research baseline."""

from __future__ import annotations

from statistics import fmean, pstdev

from pydantic import BaseModel, Field


class TurnCreditProjectionReport(BaseModel):
    schema_version: str = "turn-credit-projection.v3"
    trajectories: int
    eligible_multiturn_trajectories: int
    model_turns: int
    locally_credited_turns: int
    effective_nonzero_credited_turns: int
    zero_variance_turn_buckets: int
    compared_turn_buckets: int
    invalid_model_turns: int
    external_failure_turns: int
    unmatched_model_turns: int
    alignment_rejected_trajectories: int
    extra_unmatched_model_turns: int
    invalid_action_positive_credit_count: int
    invalid_action_positive_credit_rate: float = Field(ge=0, le=1)
    zero_advantage_group_ratio: float = Field(ge=0, le=1)
    blend_weight: float = Field(ge=0, le=1)


def model_token_segments(mask: list[int]) -> list[tuple[int, int]]:
    """Return half-open spans for model generations separated by tool results."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate([*mask, 0]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index))
            start = None
    return spans


def project_turn_relative_advantages(
    trajectory_advantages: list[float],
    policy_turn_credits: list[list[float]],
    model_token_masks: list[list[int]],
    *,
    group_size: int,
    blend_weight: float = 0.5,
    epsilon: float = 1e-4,
    policy_turn_validities: list[list[str]] | None = None,
) -> tuple[list[list[float]], TurnCreditProjectionReport]:
    """Blend B0 trajectory advantage with group-relative policy-turn credit.

    A tool-result mask splits each completion into model-generated turns. Only
    rollouts with at least two model turns and two auditable policy-step credits
    receive R1 shaping. A normal tool trajectory has exactly one unmatched final
    assistant generation after its auditable policy calls. Unmatched generations
    are outside the deployed tool-policy scope and therefore receive zero credit.
    Trajectories with additional unmatched spans or fewer spans than policy
    records are rejected from local projection instead of relying on positional
    alignment that cannot be proven.
    """
    count = len(trajectory_advantages)
    if count != len(policy_turn_credits) or count != len(model_token_masks):
        raise ValueError("trajectory, credit and token-mask batch sizes must match")
    if policy_turn_validities is None:
        policy_turn_validities = [["valid"] * len(item) for item in policy_turn_credits]
    if count != len(policy_turn_validities):
        raise ValueError("turn validity batch size must match trajectories")
    if any(
        len(validities) != len(credits)
        for validities, credits in zip(policy_turn_validities, policy_turn_credits, strict=True)
    ):
        raise ValueError("turn validity rows must align with policy turn credits")
    allowed_validities = {"invalid", "external_failure", "valid"}
    if any(
        validity not in allowed_validities for row in policy_turn_validities for validity in row
    ):
        raise ValueError("unknown policy turn validity")
    if group_size < 2 or count % group_size:
        raise ValueError("batch size must be divisible by group_size")
    if not 0 <= blend_weight <= 1:
        raise ValueError("blend_weight must be in [0, 1]")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    segments = [model_token_segments(mask) for mask in model_token_masks]
    alignment_valid = [
        len(spans) >= len(credits) and len(spans) - len(credits) <= 1
        for spans, credits in zip(segments, policy_turn_credits, strict=True)
    ]
    eligible = [
        aligned and len(spans) >= 2 and len(credits) >= 2
        for aligned, spans, credits in zip(
            alignment_valid, segments, policy_turn_credits, strict=True
        )
    ]
    local: list[list[float | None]] = [[None] * len(spans) for spans in segments]
    zero_variance = 0
    compared_turn_buckets = 0
    for group_start in range(0, count, group_size):
        group_indexes = range(group_start, group_start + group_size)
        maximum_turns = max(
            (min(len(segments[index]), len(policy_turn_credits[index])) for index in group_indexes),
            default=0,
        )
        for turn_index in range(maximum_turns):
            indexes = [
                index
                for index in group_indexes
                if eligible[index]
                and turn_index < len(segments[index])
                and turn_index < len(policy_turn_credits[index])
            ]
            # Credits are already validity-gated: invalid=-1 and external
            # failure=0. Include those counterfactuals in group normalization
            # so a correct recovery turn can be ranked against the invalid
            # alternatives sampled from the same state. The overrides below
            # still guarantee that invalid/external tokens never receive a
            # positive projected advantage.
            if len(indexes) < 2 or not any(
                policy_turn_validities[index][turn_index] == "valid" for index in indexes
            ):
                continue
            compared_turn_buckets += 1
            values = [policy_turn_credits[index][turn_index] for index in indexes]
            mean = fmean(values)
            std = pstdev(values)
            if std <= epsilon:
                zero_variance += 1
                continue
            for index, value in zip(indexes, values, strict=True):
                local[index][turn_index] = (value - mean) / (std + epsilon)

    projected: list[list[float]] = []
    locally_credited_turns = 0
    effective_nonzero_credited_turns = 0
    invalid_model_turns = 0
    external_failure_turns = 0
    unmatched_model_turns = 0
    extra_unmatched_model_turns = 0
    invalid_positive = 0
    for base, mask, spans, local_turns, validities, aligned in zip(
        trajectory_advantages,
        model_token_masks,
        segments,
        local,
        policy_turn_validities,
        alignment_valid,
        strict=True,
    ):
        row = [0.0] * len(mask)
        if not aligned:
            unmatched_model_turns += max(0, len(spans) - len(validities))
            extra_unmatched_model_turns += max(0, len(spans) - len(validities) - 1)
            projected.append(row)
            continue
        for turn_index, (start, end) in enumerate(spans):
            if turn_index >= len(validities):
                # TRL emits one final assistant segment after the last tool
                # result. It is not a deployed policy action, so neither a
                # positive terminal reward nor an invalid-action penalty should
                # update it.
                unmatched_model_turns += 1
                continue
            validity = validities[turn_index]
            local_advantage = local_turns[turn_index]
            if validity == "invalid":
                value = -1.0
                invalid_model_turns += 1
                if value > epsilon:
                    invalid_positive += 1
            elif validity == "external_failure":
                value = 0.0
                external_failure_turns += 1
            elif local_advantage is None:
                value = base
            else:
                value = (1 - blend_weight) * base + blend_weight * local_advantage
                locally_credited_turns += 1
                if abs(local_advantage) > epsilon:
                    effective_nonzero_credited_turns += 1
            for token_index in range(start, end):
                row[token_index] = value
        projected.append(row)

    report = TurnCreditProjectionReport(
        trajectories=count,
        eligible_multiturn_trajectories=sum(eligible),
        model_turns=sum(len(item) for item in segments),
        locally_credited_turns=locally_credited_turns,
        effective_nonzero_credited_turns=effective_nonzero_credited_turns,
        zero_variance_turn_buckets=zero_variance,
        compared_turn_buckets=compared_turn_buckets,
        invalid_model_turns=invalid_model_turns,
        external_failure_turns=external_failure_turns,
        unmatched_model_turns=unmatched_model_turns,
        alignment_rejected_trajectories=sum(not item for item in alignment_valid),
        extra_unmatched_model_turns=extra_unmatched_model_turns,
        invalid_action_positive_credit_count=invalid_positive,
        invalid_action_positive_credit_rate=(
            invalid_positive / invalid_model_turns if invalid_model_turns else 0.0
        ),
        zero_advantage_group_ratio=(
            zero_variance / compared_turn_buckets if compared_turn_buckets else 0.0
        ),
        blend_weight=blend_weight,
    )
    return projected, report


__all__ = [
    "TurnCreditProjectionReport",
    "model_token_segments",
    "project_turn_relative_advantages",
]
