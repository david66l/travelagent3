"""Hierarchical, anti-hacking rewards for Agent Loop episodes.

Six programmatic components drive the initial policy update.  ``quality`` is
recorded separately and defaults to audit-only (zero weight) until calibrated
against blinded human ratings.
"""

from __future__ import annotations

import json
from collections import Counter
from statistics import fmean
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from agentic.loop import NO_TOOL_ACTIONS
from agentic.trajectory import AgentEpisode, EpisodeReplayVerifier


REWARD_SCHEMA_VERSION = "agent-reward.v1"


class RewardConfig(BaseModel):
    config_version: str = "hierarchical-b0.v1"
    task_weight: float = Field(default=0.40, ge=0, le=1)
    constraint_weight: float = Field(default=0.40, ge=0, le=1)
    format_weight: float = Field(default=0.04, ge=0, le=1)
    tool_weight: float = Field(default=0.06, ge=0, le=1)
    grounding_weight: float = Field(default=0.06, ge=0, le=1)
    efficiency_weight: float = Field(default=0.04, ge=0, le=1)
    quality_weight: float = Field(default=0.0, ge=0, le=0.10)
    process_cap: float = Field(default=0.20, ge=0, le=0.20)
    unsafe_reward: float = Field(default=-1.0, ge=-1, le=0)
    hard_failure_cap: float = Field(default=-0.25, ge=-1, le=0)
    expected_max_steps: int = Field(default=16, ge=1)
    expected_max_tool_calls: int = Field(default=16, ge=1)

    @model_validator(mode="after")
    def validate_hierarchy(self) -> "RewardConfig":
        if self.task_weight + self.constraint_weight < 0.70:
            raise ValueError("terminal reward weights must sum to at least 0.70")
        process_weights = (
            self.format_weight + self.tool_weight + self.grounding_weight + self.efficiency_weight
        )
        if process_weights > self.process_cap + 1e-9:
            raise ValueError("process weights cannot exceed process_cap")
        if (
            self.task_weight + self.constraint_weight + process_weights + self.quality_weight
            > 1.0 + 1e-9
        ):
            raise ValueError("reward weights cannot sum above 1.0")
        return self


class RewardSafetySignals(BaseModel):
    security_violation: bool = False
    forged_fact: bool = False
    invalid_environment_mutation: bool = False
    reasons: list[str] = Field(default_factory=list)


class RewardComponents(BaseModel):
    task: float = Field(ge=-1, le=1)
    constraint: float = Field(ge=-1, le=1)
    format: float = Field(ge=-1, le=1)
    tool: float = Field(ge=-1, le=1)
    grounding: float = Field(ge=-1, le=1)
    efficiency: float = Field(ge=-1, le=1)
    quality: float = Field(default=0, ge=-1, le=1)


class TurnReward(BaseModel):
    step_index: int = Field(ge=0)
    action: str
    format: float = Field(ge=-1, le=1)
    tool: float = Field(ge=-1, le=1)
    grounding: float = Field(ge=-1, le=1)
    efficiency: float = Field(ge=-1, le=1)
    information_gain: bool = False
    duplicate_call: bool = False
    signals: list[str] = Field(default_factory=list)


class EpisodeReward(BaseModel):
    schema_version: str = REWARD_SCHEMA_VERSION
    trajectory_id: str
    reward_config_version: str
    gate_status: Literal["passed", "unsafe", "hard_constraint_failed", "task_failed"]
    gate_reasons: list[str] = Field(default_factory=list)
    components: RewardComponents
    terminal_reward: float
    process_reward: float
    quality_reward: float
    episode_reward: float = Field(ge=-1, le=1)
    turn_rewards: list[TurnReward]
    audit_metrics: dict[str, int | float | bool] = Field(default_factory=dict)


class HierarchicalRewardEngine:
    """Calculate reproducible outcome-first reward from auditable state."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self.replay = EpisodeReplayVerifier()

    def score(
        self,
        episode: AgentEpisode | dict[str, Any],
        *,
        safety: RewardSafetySignals | None = None,
        quality_score: float | None = None,
    ) -> EpisodeReward:
        parsed = episode if isinstance(episode, AgentEpisode) else AgentEpisode(**episode)
        safety = safety or RewardSafetySignals()
        replay_errors = self.replay.verify(parsed)
        automatic_forgery = _has_protected_policy_arguments(parsed)
        unsafe_reasons = [*safety.reasons]
        if replay_errors:
            unsafe_reasons.extend(f"REPLAY:{item}" for item in replay_errors)
        if safety.security_violation:
            unsafe_reasons.append("SECURITY_VIOLATION")
        if safety.forged_fact or automatic_forgery:
            unsafe_reasons.append("FORGED_FACT")
        if safety.invalid_environment_mutation:
            unsafe_reasons.append("INVALID_ENVIRONMENT_MUTATION")

        terminal_kind, report = _terminal_kind_and_report(parsed)
        turn_rewards = _turn_rewards(parsed, terminal_kind)
        components = RewardComponents(
            task=_task_reward(parsed, terminal_kind),
            constraint=_constraint_reward(terminal_kind, report),
            format=fmean(item.format for item in turn_rewards) if turn_rewards else -1.0,
            tool=fmean(item.tool for item in turn_rewards) if turn_rewards else -1.0,
            grounding=fmean(item.grounding for item in turn_rewards) if turn_rewards else -1.0,
            efficiency=_efficiency_reward(parsed, terminal_kind, turn_rewards, self.config),
            quality=_quality_component(quality_score),
        )
        terminal_reward = (
            self.config.task_weight * components.task
            + self.config.constraint_weight * components.constraint
        )
        raw_process = (
            self.config.format_weight * components.format
            + self.config.tool_weight * components.tool
            + self.config.grounding_weight * components.grounding
            + self.config.efficiency_weight * components.efficiency
        )
        process_reward = _clip(raw_process, -self.config.process_cap, self.config.process_cap)
        quality_reward = self.config.quality_weight * components.quality
        hard_failed_finish = _requested_finish(parsed) and not bool(
            report and report.get("hard_pass")
        )

        if unsafe_reasons:
            gate_status: Literal["passed", "unsafe", "hard_constraint_failed", "task_failed"] = (
                "unsafe"
            )
            total = self.config.unsafe_reward
        elif hard_failed_finish:
            gate_status = "hard_constraint_failed"
            total = min(
                self.config.hard_failure_cap,
                terminal_reward + process_reward + quality_reward,
            )
        elif terminal_kind == "failed":
            gate_status = "task_failed"
            total = min(
                self.config.hard_failure_cap,
                terminal_reward + process_reward + quality_reward,
            )
        else:
            gate_status = "passed"
            total = terminal_reward + process_reward + quality_reward

        observations = [item for step in parsed.steps for item in step.observations]
        duplicate_count = sum(item.duplicate_call for item in turn_rewards)
        return EpisodeReward(
            trajectory_id=parsed.trajectory_id,
            reward_config_version=self.config.config_version,
            gate_status=gate_status,
            gate_reasons=sorted(set(unsafe_reasons)),
            components=components,
            terminal_reward=round(terminal_reward, 6),
            process_reward=round(process_reward, 6),
            quality_reward=round(quality_reward, 6),
            episode_reward=round(_clip(total, -1.0, 1.0), 6),
            turn_rewards=turn_rewards,
            audit_metrics={
                "episode_steps": len(parsed.steps),
                "tool_observations": len(observations),
                "successful_observations": sum(item.ok for item in observations),
                "fallback_observations": sum(item.is_fallback for item in observations),
                "duplicate_calls": duplicate_count,
                "information_gain_steps": sum(item.information_gain for item in turn_rewards),
                "hard_pass": bool(report and report.get("hard_pass")),
                "quality_drives_training": self.config.quality_weight > 0,
            },
        )


def _turn_rewards(episode: AgentEpisode, terminal_kind: str) -> list[TurnReward]:
    signatures: Counter[str] = Counter()
    rewards: list[TurnReward] = []
    successful_terminal = terminal_kind in {"validated_plan", "clarification", "safe_termination"}
    for step in episode.steps:
        action = step.action.action
        signature = _canonical({"action": action, "arguments": step.action.arguments})
        duplicate = signatures[signature] > 0
        signatures[signature] += 1
        is_tool = action not in NO_TOOL_ACTIONS
        observations_valid = bool(step.observations) and all(
            item.tool_call_id and item.schema_version == episode.observation_schema_version
            for item in step.observations
        )
        action_valid = action in step.context.allowed_actions
        format_score = 1.0 if action_valid and (not is_tool or observations_valid) else -1.0

        state_changed = step.state_before_hash != step.state_after_hash
        successful_observation = any(item.ok for item in step.observations)
        information_gain = state_changed and (successful_observation or not is_tool)
        signals: list[str] = []
        if duplicate:
            signals.append("DUPLICATE_CALL")
        if is_tool and not step.observations:
            signals.append("OBSERVATION_MISSING")
        if is_tool and step.observations and not successful_observation:
            signals.append("TOOL_FAILED")
        if not information_gain:
            signals.append("NO_INFORMATION_GAIN")

        if not action_valid:
            tool_score = -1.0
        elif is_tool and successful_observation and information_gain and not duplicate:
            tool_score = 1.0
        elif is_tool and not successful_observation:
            tool_score = -0.5
        elif not is_tool and state_changed:
            tool_score = 0.5
        else:
            tool_score = 0.0

        protected = bool(set(step.action.arguments) & _protected_arguments())
        grounded = (
            True
            if not is_tool
            else _arguments_grounded(step.action.arguments, step.context.model_dump(mode="json"))
        )
        grounding_score = -1.0 if protected or not grounded else 1.0
        if protected:
            signals.append("PROTECTED_ARGUMENT_FORGERY")
        elif not grounded:
            signals.append("ARGUMENT_NOT_GROUNDED")

        if not successful_terminal:
            efficiency = 0.0
        elif duplicate or (is_tool and not information_gain):
            efficiency = -1.0
        else:
            efficiency = 1.0
        rewards.append(
            TurnReward(
                step_index=step.step_index,
                action=action,
                format=format_score,
                tool=tool_score,
                grounding=grounding_score,
                efficiency=efficiency,
                information_gain=information_gain,
                duplicate_call=duplicate,
                signals=signals,
            )
        )
    return rewards


def _terminal_kind_and_report(episode: AgentEpisode) -> tuple[str, dict[str, Any] | None]:
    final = episode.final_state or {}
    artifacts = list((final.get("artifacts") or {}).values())
    reports = [item for item in artifacts if item.get("artifact_type") == "validation_report"]
    report = (reports[-1].get("payload") or {}) if reports else None
    has_plan = any(
        item.get("artifact_type") in {"solver_result", "itinerary_draft"} for item in artifacts
    )
    if (
        report
        and report.get("hard_pass")
        and has_plan
        and episode.termination_reason
        in {
            "awaiting_user",
            "validated_finish",
        }
    ):
        return "validated_plan", report

    goal = final.get("goal") or episode.initial_state.get("goal") or {}
    missing = goal.get("missing_information") or []
    last_action = episode.steps[-1].action if episode.steps else None
    if (
        episode.termination_reason == "awaiting_user"
        and missing
        and last_action
        and last_action.action == "ask_user"
        and str(last_action.arguments.get("question") or "").strip()
    ):
        return "clarification", report
    capability = (goal.get("capability") or {}).get("status")
    if (
        capability in {"infeasible", "unsafe", "missing_tool"}
        and last_action
        and last_action.action in {"abort", "propose_tradeoff"}
    ):
        return "safe_termination", report
    return "failed", report


def _task_reward(episode: AgentEpisode, terminal_kind: str) -> float:
    if terminal_kind == "validated_plan":
        final = episode.final_state or {}
        tasks = (final.get("task_graph") or {}).get("tasks") or []
        autonomous = [
            task
            for task in tasks
            if task.get("required", True) and "ask_user" not in (task.get("allowed_actions") or [])
        ]
        if not autonomous:
            return 0.5
        closure = sum(task.get("status") in {"succeeded", "skipped"} for task in autonomous)
        return _clip(closure / len(autonomous), -1, 1)
    if terminal_kind == "clarification":
        return 1.0
    if terminal_kind == "safe_termination":
        return 0.8
    return -1.0


def _constraint_reward(terminal_kind: str, report: dict[str, Any] | None) -> float:
    if terminal_kind in {"clarification", "safe_termination"}:
        return 1.0
    if not report:
        return -1.0
    if report.get("hard_pass"):
        metrics = report.get("metrics") or {}
        budget_error = float(metrics.get("budget_error_rate") or 0)
        return _clip(1.0 - budget_error, 0, 1)
    violations = report.get("hard_violations") or []
    metrics = report.get("metrics") or {}
    budget_error = float(metrics.get("budget_error_rate") or 0)
    severity = min(1.0, 0.25 + 0.15 * len(violations) + budget_error)
    return -severity


def _efficiency_reward(
    episode: AgentEpisode,
    terminal_kind: str,
    turn_rewards: list[TurnReward],
    config: RewardConfig,
) -> float:
    if terminal_kind not in {"validated_plan", "clarification", "safe_termination"}:
        return 0.0
    duplicate_ratio = (
        sum(item.duplicate_call for item in turn_rewards) / len(turn_rewards) if turn_rewards else 1
    )
    tool_calls = sum(len(step.observations) for step in episode.steps)
    step_ratio = min(1.0, len(episode.steps) / config.expected_max_steps)
    call_ratio = min(1.0, tool_calls / config.expected_max_tool_calls)
    no_gain_ratio = (
        sum(not item.information_gain for item in turn_rewards) / len(turn_rewards)
        if turn_rewards
        else 1
    )
    return _clip(
        1.0 - duplicate_ratio - 0.35 * step_ratio - 0.35 * call_ratio - no_gain_ratio, -1, 1
    )


def _quality_component(value: float | None) -> float:
    if value is None:
        return 0.0
    return _clip(float(value), -1.0, 1.0)


def _requested_finish(episode: AgentEpisode) -> bool:
    return episode.termination_reason == "validated_finish" or any(
        step.action.action == "finish" for step in episode.steps
    )


def _has_protected_policy_arguments(episode: AgentEpisode) -> bool:
    protected = _protected_arguments()
    return any(set(step.action.arguments) & protected for step in episode.steps)


def _protected_arguments() -> set[str]:
    return {
        "constraints",
        "facts",
        "itinerary",
        "pois",
        "dist_matrix",
        "tc_matrix",
        "amap_minutes",
    }


def _arguments_grounded(arguments: dict[str, Any], context: dict[str, Any]) -> bool:
    if not arguments:
        return True
    grounded = _canonical(context).casefold()
    constants = {"auto", "greedy", "cpsat"}
    for name, value in arguments.items():
        if name in _protected_arguments():
            return False
        for leaf in _leaves(value):
            normalized = str(leaf).strip().casefold()
            if normalized and normalized not in constants and normalized not in grounded:
                return False
    return True


def _leaves(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _leaves(item)]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _leaves(item)]
    return [value]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
