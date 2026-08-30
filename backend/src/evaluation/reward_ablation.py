"""Counterfactual six-component reward reports without weakening safety gates."""

from __future__ import annotations

from statistics import fmean
from typing import Iterable

from pydantic import BaseModel

from agentic.reward import EpisodeReward, RewardConfig


class RewardAblationRow(BaseModel):
    component: str
    episodes: int
    full_mean_reward: float
    ablated_mean_reward: float
    mean_delta: float
    ranking_inversions: int


class RewardAblationReport(BaseModel):
    schema_version: str = "reward-ablation.v1"
    reward_config_version: str
    note: str = (
        "Offline counterfactual only. Safety and hard-constraint gates remain authoritative; "
        "this report does not claim retrained-policy causality."
    )
    rows: list[RewardAblationRow]


def build_reward_ablation_report(
    rewards: Iterable[EpisodeReward | dict],
    *,
    config: RewardConfig | None = None,
) -> RewardAblationReport:
    parsed = [
        item if isinstance(item, EpisodeReward) else EpisodeReward(**item) for item in rewards
    ]
    if not parsed:
        raise ValueError("reward ablation requires at least one episode")
    cfg = config or RewardConfig()
    component_weights = {
        "task": cfg.task_weight,
        "constraint": cfg.constraint_weight,
        "format": cfg.format_weight,
        "tool": cfg.tool_weight,
        "grounding": cfg.grounding_weight,
        "efficiency": cfg.efficiency_weight,
    }
    full = [item.episode_reward for item in parsed]
    rows: list[RewardAblationRow] = []
    for component, weight in component_weights.items():
        counterfactual = [
            _gated_counterfactual(item, weight * getattr(item.components, component), cfg)
            for item in parsed
        ]
        rows.append(
            RewardAblationRow(
                component=component,
                episodes=len(parsed),
                full_mean_reward=round(fmean(full), 6),
                ablated_mean_reward=round(fmean(counterfactual), 6),
                mean_delta=round(fmean(counterfactual) - fmean(full), 6),
                ranking_inversions=_ranking_inversions(full, counterfactual),
            )
        )
    return RewardAblationReport(
        reward_config_version=cfg.config_version,
        rows=rows,
    )


def _gated_counterfactual(
    reward: EpisodeReward,
    removed_contribution: float,
    config: RewardConfig,
) -> float:
    if reward.gate_status == "unsafe":
        return config.unsafe_reward
    value = reward.episode_reward - removed_contribution
    if reward.gate_status in {"hard_constraint_failed", "task_failed"}:
        value = min(config.hard_failure_cap, value)
    return round(max(-1.0, min(1.0, value)), 6)


def _ranking_inversions(full: list[float], ablated: list[float]) -> int:
    inversions = 0
    for left in range(len(full)):
        for right in range(left + 1, len(full)):
            full_order = (full[left] > full[right]) - (full[left] < full[right])
            ablated_order = (ablated[left] > ablated[right]) - (ablated[left] < ablated[right])
            if full_order != ablated_order:
                inversions += 1
    return inversions


__all__ = ["RewardAblationReport", "RewardAblationRow", "build_reward_ablation_report"]
