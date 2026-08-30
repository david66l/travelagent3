"""Dependency-light corpus gates for stateful Agentic GRPO training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from agentic.environment import (
    EnvironmentSnapshot,
    EnvironmentTask,
    SnapshotToolResponse,
    environment_fingerprint,
)
from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT
from agentic.training import TrainingDependency, check_training_dependencies, load_jsonl
from agentic.trajectory import redact_pii
from agentic.trajectory import AgentEpisode, EpisodeReplayVerifier


class GRPOCorpusRow(BaseModel):
    task: EnvironmentTask
    snapshot: EnvironmentSnapshot


class GRPOPreflightReport(BaseModel):
    ready: bool
    train_tasks: int
    validation_tasks: int
    environment_versions: list[str]
    snapshot_versions: list[str]
    dependencies: list[TrainingDependency]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class GRPOCompletionBudgetReport(BaseModel):
    """Measured lower bound for one complete stateful production rollout."""

    sampled_tasks: int
    max_tool_result_tokens: int
    generated_action_reserve_tokens: int
    minimum_completion_length: int
    max_observed_policy_turns: int
    limiting_task_id: str | None = None
    limiting_environment: str | None = None


# TRL counts tool-result suffixes against max_completion_length. A value suited
# to a one-shot JSON call (for example 64 or 96) silently prevents a second
# policy decision after a retryable failure. This conservative static floor is
# also available to dependency-light/preflight-only invocations; the real
# tokenizer-based bound below is stricter when training starts.
MIN_STATEFUL_COMPLETION_LENGTH = 8192
_ACTION_TOKENS_PER_POLICY_TURN = 96
# A nominal full production episode now contains eleven policy-visible
# transitions: search and verifier review each have an explicit accept/repair
# decision. Keep several extra turns for bounded recovery during GRPO.
MIN_POLICY_DRIVEN_TOOL_ITERATIONS = 11
DEFAULT_POLICY_DRIVEN_TOOL_ITERATIONS = 16


# Weather and live transport/current-info evidence are intent-dependent. The
# five tools below are the invariant executable-plan path for a solvable task.
_FULL_PLAN_TOOLS = {
    "search_pois",
    "get_poi_detail",
    "get_route_matrix",
    "solve_itinerary",
    "validate_itinerary",
}


def episode_to_grpo_corpus_row(
    episode: AgentEpisode | dict[str, Any],
    *,
    task_id: str,
    template_family: str,
    seed: int,
) -> GRPOCorpusRow:
    """Convert one real, finalized trajectory into an immutable replay task.

    Observations are copied from the executed environment. Policy-only context
    and hidden validation facts are kept separate, so a rollout cannot inspect
    its answer through the prompt.
    """
    parsed = episode if isinstance(episode, AgentEpisode) else AgentEpisode(**episode)
    replay_errors = EpisodeReplayVerifier().verify(parsed)
    if replay_errors:
        raise ValueError("episode is not replayable: " + ",".join(replay_errors))
    if parsed.status == "running" or not parsed.content_hash:
        raise ValueError("episode must be finalized before snapshot export")

    initial_goal = parsed.initial_state.get("goal") or {}
    final_goal = (parsed.final_state or {}).get("goal") or initial_goal
    hard = dict(final_goal.get("hard_constraints") or {})
    soft = dict(final_goal.get("soft_preferences") or {})
    tool_responses: dict[str, list[SnapshotToolResponse]] = {}
    for step in parsed.steps:
        for observation in step.observations:
            source = _snapshot_source(observation.source, observation.ok, observation.is_fallback)
            tool_responses.setdefault(observation.tool, []).append(
                SnapshotToolResponse(
                    data=observation.data,
                    data_source=source,
                    confidence=observation.confidence,
                    is_fallback=observation.is_fallback,
                    fallback_reason=(observation.error.message if observation.error else None),
                    latency_ms=observation.latency_ms,
                    error_code=(observation.error.code if observation.error else None),
                    retryable=(observation.error.retryable if observation.error else False),
                )
            )

    validation_reports = [
        artifact.get("payload") or {}
        for artifact in ((parsed.final_state or {}).get("artifacts") or {}).values()
        if artifact.get("artifact_type") == "validation_report"
    ]
    difficulty = _episode_difficulty(parsed)
    row = GRPOCorpusRow(
        task=EnvironmentTask(
            task_id=task_id,
            template_family=template_family,
            difficulty=difficulty,
            seed=seed,
            user_request=str(final_goal.get("original_request") or "Travel planning request"),
            slots=hard,
            profile=soft,
            missing_slots=list(final_goal.get("missing_information") or []),
            feasibility_report={
                "feasible": (final_goal.get("capability") or {}).get("status") == "solvable",
                "status": (final_goal.get("capability") or {}).get("status"),
                "reasons": list((final_goal.get("capability") or {}).get("evidence") or []),
                "actionable_alternatives": (final_goal.get("capability") or {}).get(
                    "actionable_alternatives"
                ),
                "alternatives": list(
                    (final_goal.get("capability") or {}).get("alternatives") or []
                ),
            },
        ),
        snapshot=EnvironmentSnapshot(
            environment_version=parsed.environment_version,
            snapshot_version="episode-" + str(parsed.content_hash)[:16],
            state_id="trajectory-" + str(parsed.content_hash)[:16],
            tool_responses=tool_responses,
            hidden_test_facts={
                "source_content_hash": parsed.content_hash,
                "validation_report": validation_reports[-1] if validation_reports else None,
            },
        ),
    )
    # Defense in depth: direct identifiers must never enter an RL corpus.
    payload = row.model_dump(mode="json")
    if redact_pii(payload) != payload:
        raise ValueError("episode-derived GRPO row contains PII")
    return row


def _snapshot_source(source: str, ok: bool, is_fallback: bool) -> str:
    if not ok:
        return "unavailable"
    if is_fallback:
        return "fallback"
    if source in {"api", "built_in", "fallback", "unavailable"}:
        return source
    return "api"


def _episode_difficulty(episode: AgentEpisode) -> str:
    failures = len((episode.final_state or {}).get("failures") or [])
    policy_steps = sum(step.action.decision_source != "controller" for step in episode.steps)
    score = failures * 2 + policy_steps + max(0, len(episode.steps) - 8)
    if score <= 1:
        return "L1"
    if score <= 3:
        return "L2"
    if score <= 6:
        return "L3"
    return "L4"


def load_grpo_corpus(path: Path) -> list[GRPOCorpusRow]:
    return [GRPOCorpusRow(**row) for row in load_jsonl(path)]


def to_trl_environment_rows(rows: list[GRPOCorpusRow]) -> list[dict[str, Any]]:
    """Build fresh-rollout rows without assistant or tool trajectory prefixes."""
    converted = []
    for row in rows:
        decision_state = row.snapshot.hidden_test_facts.get("grpo_decision_state")
        replay_prompt = (
            decision_state.get("prompt_messages")
            if isinstance(decision_state, dict)
            else None
        )
        prompt = (
            replay_prompt
            if isinstance(replay_prompt, list) and replay_prompt
            else [
                {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": row.task.user_request},
            ]
        )
        converted.append(
            {
            "prompt": prompt,
            "task": row.task.model_dump(mode="json"),
            "snapshot": row.snapshot.model_dump(mode="json"),
            "environment": _environment_route(row.task, row.snapshot),
            "task_id": row.task.task_id,
            "difficulty": row.task.difficulty,
            "initial_state_fingerprint": environment_fingerprint(row.task, row.snapshot),
            "rollout_contract": (
                "verified_decision_state_replay.v1"
                if replay_prompt
                else "fresh_ledger_no_teacher_prefix.v1"
            ),
        }
        )
    return converted


def estimate_stateful_completion_budget(
    rows: list[GRPOCorpusRow],
    tokenizer: Any,
    environment_factories: dict[str, Callable[..., Any]],
    *,
    max_samples: int = 12,
) -> GRPOCompletionBudgetReport:
    """Measure cumulative tool suffixes across a complete production rollout.

    The estimate deliberately exercises the same production-backed environment
    classes used by GRPO. Retry snapshots are prioritized because recovery
    results add more state and may consume more completion budget than the
    nominal nine-decision success path.
    """
    selected = _completion_budget_samples(rows, max_samples=max_samples)
    max_tokens = 0
    max_observed_policy_turns = 0
    limiting_task_id: str | None = None
    limiting_environment: str | None = None
    for row in selected:
        route = _environment_route(row.task, row.snapshot)
        # Budget measurement must not pollute the training rollout audit.
        environment = environment_factories[route](audit_enabled=False)
        reset_succeeded = False
        try:
            rendered = environment.reset(
                task=row.task.model_dump(mode="json"),
                snapshot=row.snapshot.model_dump(mode="json"),
            )
            reset_succeeded = True
            rollout_tokens = 0
            observed_turns = 0
            for _turn in range(DEFAULT_POLICY_DRIVEN_TOOL_ITERATIONS):
                policy_state = json.loads(rendered)["policy_state"]
                tool_name, arguments = _budget_probe_action(row, policy_state)
                rendered = getattr(environment, tool_name)(**arguments)
                rollout_tokens += _tool_result_suffix_tokens(
                    tokenizer,
                    tool_name=tool_name,
                    result=str(rendered),
                )
                observed_turns += 1
                if json.loads(rendered).get("done"):
                    break
            if rollout_tokens > max_tokens:
                max_tokens = rollout_tokens
                limiting_task_id = row.task.task_id
                limiting_environment = route
            max_observed_policy_turns = max(max_observed_policy_turns, observed_turns)
        finally:
            if reset_succeeded:
                # Finalize and close the per-rollout event-loop thread even if
                # the sampled transition is intentionally retryable.
                environment.get_reward()

    generated_reserve = _ACTION_TOKENS_PER_POLICY_TURN * DEFAULT_POLICY_DRIVEN_TOOL_ITERATIONS
    measured_minimum = max_tokens + generated_reserve
    return GRPOCompletionBudgetReport(
        sampled_tasks=len(selected),
        max_tool_result_tokens=max_tokens,
        generated_action_reserve_tokens=generated_reserve,
        minimum_completion_length=max(MIN_STATEFUL_COMPLETION_LENGTH, measured_minimum),
        max_observed_policy_turns=max_observed_policy_turns,
        limiting_task_id=limiting_task_id,
        limiting_environment=limiting_environment,
    )


def _budget_probe_action(
    row: GRPOCorpusRow,
    policy_state: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Choose one valid deterministic action solely for completion sizing."""
    allowed = list(policy_state.get("allowed_actions") or [])
    if not allowed:
        raise RuntimeError(f"budget probe has no allowed action for {row.task.task_id}")
    preferred = next(
        (
            action
            for action in (
                "capability_check",
                "get_weather",
                "accept_candidates",
                "search_pois",
                "get_poi_detail",
                "get_route_matrix",
                "solve_itinerary",
                "validate_itinerary",
                "accept_itinerary",
                "compose_draft",
                "finish",
                "ask_user",
                "propose_tradeoff",
                "abort",
            )
            if action in allowed
        ),
        allowed[0],
    )
    if preferred == "ask_user":
        missing = row.task.missing_slots[0] if row.task.missing_slots else "行程信息"
        return preferred, {"question": f"请补充{missing}。"}
    if preferred == "propose_tradeoff":
        reasons = list(row.task.feasibility_report.get("reasons") or [])
        return preferred, {
            "reason": str(reasons[0] if reasons else "当前约束不可行"),
            "options": [],
        }
    if preferred == "abort":
        return preferred, {"reason": "没有安全且可行的替代方案"}
    if preferred == "search_pois":
        return preferred, {"keywords": list(row.task.profile.get("interests") or [])}
    if preferred == "solve_itinerary":
        return preferred, {"strategy": "auto"}
    return preferred, {}


def _completion_budget_samples(
    rows: list[GRPOCorpusRow], *, max_samples: int
) -> list[GRPOCorpusRow]:
    selected: list[GRPOCorpusRow] = []
    seen: set[tuple[str, str | None]] = set()
    for row in rows:
        route = _environment_route(row.task, row.snapshot)
        first_error = None
        if route == "search":
            responses = row.snapshot.tool_responses.get("search_pois") or []
            first_error = responses[0].error_code if responses else None
        key = (route, first_error)
        if key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) >= max_samples:
            return selected
    for row in rows:
        if row not in selected:
            selected.append(row)
        if len(selected) >= max_samples:
            break
    return selected


def tool_result_suffix_ids(
    tokenizer: Any,
    *,
    tool_messages: list[dict[str, Any]],
    chat_template: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    """Return tool-result suffix IDs without assuming history re-tokenizes identically.

    Qwen3 conditionally inserts an empty thinking block when an assistant tool
    call is the final message.  Adding the subsequent tool result removes that
    block, so TRL's historical strict-prefix shortcut is not valid.  When that
    happens, align to the assistant end-of-turn token in the fully rendered
    conversation instead.
    """
    if not tool_messages:
        raise ValueError("at least one tool message is required")
    tool_name = str(tool_messages[0]["name"])
    tool_calls = [{"type": "function", "function": {"name": tool_name, "arguments": {}}}]
    prefix_messages = [
        {"role": "user", "content": "dummy"},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
    ]
    template_kwargs = dict(chat_template_kwargs or {})
    if chat_template is not None:
        template_kwargs["chat_template"] = chat_template
    prefix_ids = tokenizer.apply_chat_template(
        prefix_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
        **template_kwargs,
    )
    full_ids = tokenizer.apply_chat_template(
        [*prefix_messages, *tool_messages],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        **template_kwargs,
    )
    if prefix_ids and isinstance(prefix_ids[0], list):
        prefix_ids = prefix_ids[0]
    if full_ids and isinstance(full_ids[0], list):
        full_ids = full_ids[0]
    prefix_eos_positions = [
        index for index, token_id in enumerate(prefix_ids) if token_id == tokenizer.eos_token_id
    ]
    if prefix_eos_positions:
        trimmed_prefix = prefix_ids[: prefix_eos_positions[-1] + 1]
        if full_ids[: len(trimmed_prefix)] == trimmed_prefix:
            return list(full_ids[len(trimmed_prefix) :])

        full_eos_positions = [
            index for index, token_id in enumerate(full_ids) if token_id == tokenizer.eos_token_id
        ]
        assistant_eos_index = len(prefix_eos_positions) - 1
        if len(full_eos_positions) > assistant_eos_index:
            first_prefix_turn = prefix_ids[: prefix_eos_positions[0] + 1]
            first_full_turn = full_ids[: full_eos_positions[0] + 1]
            if first_prefix_turn == first_full_turn:
                boundary = full_eos_positions[assistant_eos_index] + 1
                return list(full_ids[boundary:])
    raise ValueError("tool suffix tokenization cannot locate a stable assistant boundary")


def _tool_result_suffix_tokens(tokenizer: Any, *, tool_name: str, result: str) -> int:
    suffix_ids = tool_result_suffix_ids(
        tokenizer,
        tool_messages=[{"role": "tool", "name": tool_name, "content": result}],
        chat_template_kwargs={"enable_thinking": False},
    )
    return len(suffix_ids)


def _environment_route(
    task: EnvironmentTask,
    snapshot: EnvironmentSnapshot | None = None,
) -> str:
    decision_state = (
        snapshot.hidden_test_facts.get("grpo_decision_state") if snapshot is not None else None
    )
    if isinstance(decision_state, dict):
        target_action = str(decision_state.get("target_action") or "")
        if target_action == "get_poi_detail":
            return "decision_get_poi_detail"
        raise ValueError(f"unsupported GRPO decision-state target: {target_action or 'missing'}")
    if task.missing_slots:
        return "clarification"
    if task.feasibility_report.get("feasible", True) is False:
        return "tradeoff"
    transport_modes = task.slots.get("transport_modes_requested") or []
    if transport_modes:
        return "search_transport"
    current_information = {
        "event",
        "opening_hours",
        "restaurant",
        "seasonal_activity",
        "closure",
        "general",
    }
    if current_information.intersection(task.slots.get("information_needs") or []):
        return "search_current"
    return "search"


def preflight_grpo_corpus(
    corpus_dir: Path,
    *,
    minimum_train_tasks: int = 1000,
    require_dependencies: bool = True,
) -> GRPOPreflightReport:
    train = load_grpo_corpus(corpus_dir / "train.jsonl")
    validation = load_grpo_corpus(corpus_dir / "validation.jsonl")
    errors: list[str] = []
    warnings: list[str] = []
    if len(train) < minimum_train_tasks:
        errors.append(f"TRAIN_TASKS_BELOW_MINIMUM:{len(train)}<{minimum_train_tasks}")
    if not validation:
        errors.append("VALIDATION_TASKS_EMPTY")

    train_ids = [row.task.task_id for row in train]
    validation_ids = [row.task.task_id for row in validation]
    if len(train_ids) != len(set(train_ids)):
        errors.append("DUPLICATE_TRAIN_TASK_ID")
    if len(validation_ids) != len(set(validation_ids)):
        errors.append("DUPLICATE_VALIDATION_TASK_ID")
    if set(train_ids) & set(validation_ids):
        errors.append("TASK_ID_SPLIT_OVERLAP")

    train_fingerprints = {environment_fingerprint(row.task, row.snapshot) for row in train}
    validation_fingerprints = {
        environment_fingerprint(row.task, row.snapshot) for row in validation
    }
    if train_fingerprints & validation_fingerprints:
        errors.append("INITIAL_STATE_SPLIT_OVERLAP")

    for split, rows in (("train", train), ("validation", validation)):
        for row in rows:
            prefix = f"{split}:{row.task.task_id}"
            payload = row.model_dump(mode="json")
            if redact_pii(payload) != payload:
                errors.append(f"PII_DETECTED:{prefix}")
            if _contains_unicode_replacement(payload):
                errors.append(f"TEXT_ENCODING_CORRUPT:{prefix}")
            if row.task.missing_slots:
                continue
            if row.task.feasibility_report.get("feasible", True) is False:
                feasibility = row.task.feasibility_report
                status = str(feasibility.get("status") or "")
                actionable = feasibility.get("actionable_alternatives")
                reasons = [
                    str(item).strip()
                    for item in feasibility.get("reasons") or []
                    if str(item).strip()
                ]
                alternatives = [
                    str(item).strip()
                    for item in feasibility.get("alternatives") or []
                    if str(item).strip()
                ]
                if status not in {"infeasible", "unsafe", "missing_tool"}:
                    errors.append(f"INFEASIBLE_STATUS_INVALID:{prefix}:{status or 'missing'}")
                if not isinstance(actionable, bool):
                    errors.append(f"INFEASIBLE_ACTIONABLE_FLAG_MISSING:{prefix}")
                if not reasons:
                    errors.append(f"INFEASIBLE_REASONS_EMPTY:{prefix}")
                if actionable is True and not alternatives:
                    errors.append(f"INFEASIBLE_ACTIONABLE_ALTERNATIVES_EMPTY:{prefix}")
                if actionable is False and alternatives:
                    errors.append(f"INFEASIBLE_NONACTIONABLE_ALTERNATIVES_PRESENT:{prefix}")
                if len(alternatives) != len(set(alternatives)):
                    errors.append(f"INFEASIBLE_ALTERNATIVES_DUPLICATED:{prefix}")
                continue
            available = set(row.snapshot.tool_responses)
            missing = sorted(_FULL_PLAN_TOOLS - available)
            if missing:
                errors.append(f"SNAPSHOT_TOOLS_MISSING:{prefix}:{','.join(missing)}")
            if any(not row.snapshot.tool_responses.get(name) for name in _FULL_PLAN_TOOLS):
                errors.append(f"SNAPSHOT_RESPONSES_EMPTY:{prefix}")
            if not row.snapshot.hidden_test_facts:
                warnings.append(f"HIDDEN_TEST_FACTS_EMPTY:{prefix}")

    dependencies = check_training_dependencies(extra_names=("jmespath",))
    missing_dependencies = [item.name for item in dependencies if not item.installed]
    if require_dependencies and missing_dependencies:
        errors.append("TRAINING_DEPENDENCIES_MISSING:" + ",".join(missing_dependencies))

    all_rows = [*train, *validation]
    return GRPOPreflightReport(
        ready=not errors,
        train_tasks=len(train),
        validation_tasks=len(validation),
        environment_versions=sorted({row.snapshot.environment_version for row in all_rows}),
        snapshot_versions=sorted({row.snapshot.snapshot_version for row in all_rows}),
        dependencies=dependencies,
        errors=sorted(set(errors)),
        warnings=sorted(set(warnings)),
    )


def _contains_unicode_replacement(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_unicode_replacement(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_unicode_replacement(item) for item in value)
    return isinstance(value, str) and "\ufffd" in value


__all__ = [
    "DEFAULT_POLICY_DRIVEN_TOOL_ITERATIONS",
    "GRPOCompletionBudgetReport",
    "GRPOCorpusRow",
    "GRPOPreflightReport",
    "MIN_POLICY_DRIVEN_TOOL_ITERATIONS",
    "MIN_STATEFUL_COMPLETION_LENGTH",
    "estimate_stateful_completion_budget",
    "episode_to_grpo_corpus_row",
    "load_grpo_corpus",
    "preflight_grpo_corpus",
    "tool_result_suffix_ids",
    "to_trl_environment_rows",
]
