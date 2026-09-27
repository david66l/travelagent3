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
    rollout_contracts: list[str]
    policy_turn_budget: int
    limiting_task_id: str | None = None
    limiting_environment: str | None = None


# TRL counts tool-result suffixes against max_completion_length. Full loops need
# enough room for every follow-up transition; verified one-decision replay does
# not. Keep separate conservative floors so malformed one-step completions
# cannot burn the entire multi-turn context budget.
MIN_STATEFUL_COMPLETION_LENGTH = 8192
MIN_DECISION_STATE_COMPLETION_LENGTH = 512
_ACTION_TOKENS_PER_POLICY_TURN = 96
# A nominal full production episode now contains eleven policy-visible
# transitions: search and verifier review each have an explicit accept/repair
# decision. Keep several extra turns for bounded recovery during GRPO.
MIN_POLICY_DRIVEN_TOOL_ITERATIONS = 11
DEFAULT_POLICY_DRIVEN_TOOL_ITERATIONS = 16
FRESH_LEDGER_ROLLOUT_CONTRACT = "fresh_ledger_no_teacher_prefix.v1"
VERIFIED_DECISION_STATE_REPLAY_CONTRACT = "verified_decision_state_replay.v1"
AUTHORITY_PAYLOAD_ENCODING = "canonical-json.v1"
VERIFIER_REPAIR_DECISION_SCHEMA_VERSION = "react-verifier-repair-decision.v5"


def encode_authority_payload(payload: dict[str, Any]) -> str:
    """Encode authority-bearing rollout state as an Arrow-safe scalar."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def decode_authority_payload(payload: dict[str, Any] | str, *, field: str) -> dict[str, Any]:
    """Decode a canonical JSON object, while preserving direct dict call sites."""
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        raise ValueError(f"{field} authority payload must be a dict or canonical JSON string")

    def reject_non_finite(constant: str) -> None:
        raise ValueError(f"non-finite JSON number {constant}")

    try:
        decoded = json.loads(payload, parse_constant=reject_non_finite)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{field} authority payload is not valid strict JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{field} authority payload must decode to a JSON object")
    if encode_authority_payload(decoded) != payload:
        raise ValueError(f"{field} authority payload is not canonical JSON")
    return decoded


def minimum_completion_length_floor(rows):
    return MIN_STATEFUL_COMPLETION_LENGTH


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


def load_grpo_corpus(path: Path, *, allow_blind_test: bool = False):
    payloads = load_jsonl(path)
    manifest_path = path.parent / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    is_blind = manifest.get("dataset_role") == "blind_test" or any(
        row.get("snapshot", {})
        .get("hidden_test_facts", {})
        .get("decision_loop_curriculum", {})
        .get("dataset_role")
        == "blind_test"
        for row in payloads
    )
    if is_blind and not allow_blind_test:
        raise ValueError("BLIND_TEST_CORPUS_NOT_ALLOWED_FOR_TRAINING")
    rows = [GRPOCorpusRow(**row) for row in payloads]
    for row in rows:
        _environment_route(row.task, row.snapshot)
    return rows


def to_trl_environment_rows(rows):
    converted = []
    for row in rows:
        converted.append(
            {
                "prompt": [
                    {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
                    {"role": "user", "content": row.task.user_request},
                ],
                "task": encode_authority_payload(row.task.model_dump(mode="json")),
                "snapshot": encode_authority_payload(row.snapshot.model_dump(mode="json")),
                "authority_payload_encoding": AUTHORITY_PAYLOAD_ENCODING,
                "environment": _environment_route(row.task, row.snapshot),
                "task_id": row.task.task_id,
                "difficulty": row.task.difficulty,
                "initial_state_fingerprint": environment_fingerprint(row.task, row.snapshot),
                "rollout_contract": FRESH_LEDGER_ROLLOUT_CONTRACT,
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
    static_floor = minimum_completion_length_floor(rows)
    decision_state_only = static_floor == MIN_DECISION_STATE_COMPLETION_LENGTH
    policy_turn_budget = 1 if decision_state_only else DEFAULT_POLICY_DRIVEN_TOOL_ITERATIONS
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
            for _turn in range(policy_turn_budget):
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

    generated_reserve = _ACTION_TOKENS_PER_POLICY_TURN * policy_turn_budget
    measured_minimum = max_tokens + generated_reserve
    rollout_contracts = sorted(
        {str(row["rollout_contract"]) for row in to_trl_environment_rows(rows)}
    )
    return GRPOCompletionBudgetReport(
        sampled_tasks=len(selected),
        max_tool_result_tokens=max_tokens,
        generated_action_reserve_tokens=generated_reserve,
        minimum_completion_length=max(static_floor, measured_minimum),
        max_observed_policy_turns=max_observed_policy_turns,
        rollout_contracts=rollout_contracts,
        policy_turn_budget=policy_turn_budget,
        limiting_task_id=limiting_task_id,
        limiting_environment=limiting_environment,
    )


def _budget_probe_action(row, context):
    """Deterministic sizing probe only; never used as an agent fallback or training label."""
    artifacts = {item.get("artifact_type"): item for item in context.get("relevant_artifacts", [])}
    if context.get("failure_summary"):
        return "ask_user", {"question": "Please clarify the unresolved travel requirement."}
    if context.get("missing_information"):
        return "ask_user", {"question": "Please provide the missing travel information."}
    if artifacts.get("validation_report", {}).get("hard_pass"):
        return "finish", {}
    if "solver_result" in artifacts:
        return "validate_itinerary", {}
    for kind, action in [
        ("city_knowledge", "retrieve_city_knowledge"),
        ("poi_candidate_set", "search_pois"),
        ("poi_detail_set", "get_poi_detail"),
        ("route_matrix", "get_route_matrix"),
    ]:
        if kind not in artifacts:
            return action, {}
    return "solve_itinerary", {"strategy": "cpsat"}


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


def _environment_route(task, snapshot):
    if snapshot.hidden_test_facts.get("grpo_decision_state"):
        raise ValueError(
            "Legacy teacher-forced decision replay is retired; create fresh full episodes"
        )
    return "travel"


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
            if row.snapshot.hidden_test_facts.get("grpo_decision_state"):
                errors.append(f"LEGACY_DECISION_REPLAY_RETIRED:{prefix}")
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
    "MIN_DECISION_STATE_COMPLETION_LENGTH",
    "MIN_STATEFUL_COMPLETION_LENGTH",
    "estimate_stateful_completion_budget",
    "episode_to_grpo_corpus_row",
    "load_grpo_corpus",
    "minimum_completion_length_floor",
    "preflight_grpo_corpus",
    "tool_result_suffix_ids",
    "to_trl_environment_rows",
]
