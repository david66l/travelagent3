"""Real TRL Agentic GRPO-B0 entrypoint for snapshot travel environments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import (  # noqa: E402
    DEFAULT_POLICY_DRIVEN_TOOL_ITERATIONS,
    MIN_POLICY_DRIVEN_TOOL_ITERATIONS,
    VERIFIER_REPAIR_ACTIONS_BY_ROUTE,
    VERIFIER_REPAIR_SCHEMA_CAPABILITY_BY_ROUTE,
    estimate_stateful_completion_budget,
    load_grpo_corpus,
    minimum_completion_length_floor,
    preflight_grpo_corpus,
    tool_result_suffix_ids,
    to_trl_environment_rows,
)
from agentic.training import require_git_commit  # noqa: E402
from agentic.chat_template_contract import (  # noqa: E402
    AGENT_CHAT_TEMPLATE_KWARGS,
    AGENT_CHAT_TEMPLATE_PATH,
    AGENT_CHAT_TEMPLATE_SHA256,
    AGENT_RENDER_PROTOCOL_VERSION,
    install_agent_chat_template,
    load_agent_chat_template,
    validate_agent_chat_template,
)
from agentic.reward import RewardConfig  # noqa: E402
from agentic.trl_environment import (  # noqa: E402
    VERIFIED_DECISION_STATE_REPLAY_CONTRACT,
    build_trl_environment_factories,
    canonical_trl_tool_schemas,
)


def _disable_unused_vllm_import() -> None:
    """Keep TRL's optional vLLM backend out of non-vLLM training imports.

    TRL imports its vLLM generation module whenever any vLLM distribution is
    installed, even when ``use_vllm=False``. Serving environments commonly pin
    an older vLLM release, so that eager optional import can prevent the native
    Transformers GRPO backend from starting. Patching TRL's availability probe
    is process-local and leaves the installed serving runtime untouched.
    """
    trl_import_utils = importlib.import_module("trl.import_utils")
    trl_import_utils.is_vllm_available = lambda min_version=None: False


def create_stable_tool_suffix_grpo_trainer_class(
    *,
    base_trainer_class,
    trl_version: str,
):
    """Bind the strict production schema bridge to the pinned TRL trainer."""

    class StableToolSuffixGRPOTrainer(base_trainer_class):
        """Keep Qwen suffix alignment and render production argument schemas."""

        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            if self.environment_factories is not None:
                if trl_version != "1.9.2":
                    raise RuntimeError(
                        "strict environment schema bridge is pinned to TRL 1.9.2; "
                        f"found {trl_version}"
                    )
                if not isinstance(getattr(self, "_env_tools", None), dict):
                    raise RuntimeError("TRL environment schema bridge structure changed")
                # TRL converts Python signatures to a permissive JSON schema
                # that omits Pydantic constraints such as additionalProperties
                # and minLength. Execution still uses the bound methods in
                # _sync_tool_dicts; only the model-visible prompt schemas are
                # replaced here with the same contract used in production and
                # the frozen checkpoint audit.
                self._env_tools = {
                    name: canonical_trl_tool_schemas(
                        tools,
                        action_order=VERIFIER_REPAIR_ACTIONS_BY_ROUTE.get(name),
                        capability=VERIFIER_REPAIR_SCHEMA_CAPABILITY_BY_ROUTE.get(name),
                    )
                    for name, tools in self._env_tools.items()
                }
                shared_template = load_agent_chat_template()
                self.processing_class.chat_template = shared_template
                tokenizer = getattr(self, "_tokenizer", None)
                if tokenizer is not None:
                    tokenizer.chat_template = shared_template
                # TRL has already installed its response parser from the native
                # Qwen template. From this point onward, use the explicit shared
                # prefix-preserving renderer rather than TRL's implicit copy.
                self.chat_template = None
                active_template = self.chat_template or self.processing_class.chat_template
                validate_agent_chat_template(active_template)
                if self.chat_template_kwargs != AGENT_CHAT_TEMPLATE_KWARGS:
                    raise RuntimeError(
                        "Trainer chat-template kwargs differ from the shared Agent contract"
                    )
                self.agent_render_protocol_version = AGENT_RENDER_PROTOCOL_VERSION
                self.agent_chat_template_sha256 = AGENT_CHAT_TEMPLATE_SHA256

        def _tool_call_loop(
            self,
            prompts,
            prompt_ids,
            completion_ids,
            completions,
            logprobs,
            images,
            multimodal_fields,
        ):
            for attribute in ("_sync_tool_dicts", "_async_tool_dicts"):
                tool_dicts = getattr(self, attribute, None)
                if not isinstance(tool_dicts, list) or any(
                    not isinstance(tool_dict, dict)
                    or any(not callable(tool) for tool in tool_dict.values())
                    for tool_dict in tool_dicts
                ):
                    raise RuntimeError(
                        "TRL execution bridge no longer contains bound environment methods"
                    )
            working_completions = copy.deepcopy(completions)
            preflight_call_count = 0
            preflight_failure_count = 0
            environments = list(self.environments or [])
            if environments and len(environments) != len(working_completions):
                raise RuntimeError("TRL environment/completion batch cardinality changed")
            for index, environment in enumerate(environments):
                if not getattr(environment, "_single_decision_tool_contract", False):
                    continue
                tool_calls = working_completions[index][0].get("tool_calls") or []
                if not tool_calls:
                    continue
                rejection_code = None
                if len(tool_calls) != 1:
                    rejection_code = "TOOL_CALL_CARDINALITY_INVALID"
                else:
                    tool_call = tool_calls[0]
                    function = tool_call.get("function") or {}
                    name = function.get("name")
                    arguments = function.get("arguments")
                    available = {
                        *self._sync_tool_dicts[index],
                        *self._async_tool_dicts[index],
                    }
                    if tool_call.get("type") != "function":
                        rejection_code = "TOOL_CALL_TYPE_INVALID"
                    elif not isinstance(name, str) or name not in available:
                        rejection_code = "TOOL_CALL_NAME_INVALID"
                    elif not isinstance(arguments, dict):
                        rejection_code = "TOOL_CALL_ARGUMENTS_INVALID"
                if rejection_code is not None:
                    environment._reject_policy_call_batch(
                        tool_calls,
                        rejection_code=rejection_code,
                    )
                    preflight_call_count += max(1, len(tool_calls))
                    preflight_failure_count += max(1, len(tool_calls))
                    working_completions[index][0]["tool_calls"] = []

            self._agent_tool_loop_environment_indices = [
                index
                for index, completion in enumerate(working_completions)
                if completion[0].get("tool_calls")
            ]
            self._agent_terminal_eos_environment_indices = set()
            self._agent_inside_tool_loop = True
            try:
                result = super()._tool_call_loop(
                    prompts,
                    prompt_ids,
                    completion_ids,
                    working_completions,
                    logprobs,
                    images,
                    multimodal_fields,
                )
                terminal_eos_indices = set(
                    self._agent_terminal_eos_environment_indices
                )
            finally:
                self._agent_inside_tool_loop = False
                self._agent_tool_loop_environment_indices = []
                self._agent_terminal_eos_environment_indices = set()
            (
                tool_mask,
                rendered_completions,
                rendered_completion_ids,
                rendered_logprobs,
                tool_call_count,
                tool_failure_count,
                tool_images,
            ) = result
            for index in terminal_eos_indices:
                if (
                    index >= len(rendered_completion_ids)
                    or not rendered_completion_ids[index]
                    or rendered_completion_ids[index][-1]
                    != self._tokenizer.eos_token_id
                ):
                    raise RuntimeError("terminal decision completion is missing its EOS")
                if index >= len(tool_mask) or not tool_mask[index]:
                    raise RuntimeError("terminal decision completion is missing its tool mask")
                # This EOS is inserted by the Trainer bridge after the terminal
                # tool result. It closes truncation accounting but is not a
                # model-authored token and must never receive policy gradient.
                tool_mask[index][-1] = 0
            return (
                tool_mask,
                rendered_completions,
                rendered_completion_ids,
                rendered_logprobs,
                tool_call_count + preflight_call_count,
                tool_failure_count + preflight_failure_count,
                tool_images,
            )

        def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
            if not getattr(self, "_agent_inside_tool_loop", False):
                return super()._generate_single_turn(prompt_ids, images, multimodal_fields)
            active_indices = list(self._agent_tool_loop_environment_indices)
            if len(active_indices) != len(prompt_ids):
                raise RuntimeError("TRL active tool-loop subset cardinality changed")
            terminal = [
                bool(self.environments[index]._trl_tool_loop_done())
                for index in active_indices
            ]
            for position, done in enumerate(terminal):
                if done:
                    self._agent_terminal_eos_environment_indices.add(
                        active_indices[position]
                    )
            live_positions = [index for index, done in enumerate(terminal) if not done]
            if not live_positions:
                terminal_logprobs = (
                    [[0.0] for _ in prompt_ids] if self.use_vllm else None
                )
                self._agent_tool_loop_environment_indices = []
                return (
                    [[self._tokenizer.eos_token_id] for _ in prompt_ids],
                    terminal_logprobs,
                )

            def select_rows(value):
                if value is None:
                    return None
                if isinstance(value, list):
                    return [value[index] for index in live_positions]
                try:
                    return value[live_positions]
                except (IndexError, TypeError):
                    return value

            live_prompt_ids = [prompt_ids[index] for index in live_positions]
            live_images = select_rows(images)
            live_multimodal_fields = {
                name: select_rows(value) for name, value in multimodal_fields.items()
            }
            live_ids, live_logprobs = super()._generate_single_turn(
                live_prompt_ids,
                live_images,
                live_multimodal_fields,
            )
            completion_ids = [
                [self._tokenizer.eos_token_id] if done else [] for done in terminal
            ]
            logprob_rows = (
                [[0.0] if done else [] for done in terminal]
                if live_logprobs is not None
                else None
            )
            for live_index, position in enumerate(live_positions):
                completion_ids[position] = live_ids[live_index]
                if logprob_rows is not None:
                    logprob_rows[position] = live_logprobs[live_index]

            from trl.chat_template_utils import parse_response

            next_active_indices = []
            for live_index, position in enumerate(live_positions):
                parsed = (
                    parse_response(
                        self._tokenizer,
                        live_ids[live_index],
                        prefix=live_prompt_ids[live_index],
                    )
                    if live_ids[live_index]
                    else {}
                )
                if parsed.get("tool_calls"):
                    next_active_indices.append(active_indices[position])
            self._agent_tool_loop_environment_indices = next_active_indices
            return completion_ids, logprob_rows

        def _get_tool_suffix_ids(self, tool_messages):
            return tool_result_suffix_ids(
                self.processing_class,
                tool_messages=tool_messages,
                chat_template=self.chat_template,
                chat_template_kwargs=self.chat_template_kwargs,
            )

    StableToolSuffixGRPOTrainer.__name__ = "StableToolSuffixGRPOTrainer"
    return StableToolSuffixGRPOTrainer


def _file_provenance(path: Path) -> dict[str, str | int]:
    """Return immutable evidence for one training input or output file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def run_rollout_only_audit(
    trainer,
    *,
    expected_group_size: int,
    source_adapter_path: Path,
) -> dict:
    """Exercise the real Trainer generation/reward path without backward or save."""
    import torch

    source_before = (
        _file_provenance(source_adapter_path) if source_adapter_path.is_file() else None
    )
    generation_batch = next(iter(trainer.get_train_dataloader()))
    if len(generation_batch) != expected_group_size:
        raise RuntimeError(
            "rollout-only generation batch does not match num_generations: "
            f"{len(generation_batch)}!={expected_group_size}"
        )
    optimizer_absent_before = trainer.optimizer is None
    with torch.no_grad():
        prepared = trainer._generate_and_score_completions(generation_batch)
    optimizer_absent_after = trainer.optimizer is None
    gradients_absent = all(parameter.grad is None for parameter in trainer.model.parameters())
    source_after = (
        _file_provenance(source_adapter_path) if source_adapter_path.is_file() else None
    )

    environments = list(trainer.environments or [])
    if len(environments) != expected_group_size:
        raise RuntimeError("rollout-only environment group cardinality changed")
    rewards = []
    decision_rows = []
    for environment in environments:
        rollout = environment.rollout_record
        if rollout is None:
            raise RuntimeError("rollout-only environment did not expose a scored record")
        reward = rollout.reward
        rewards.append(float(reward.episode_reward))
        credited_step_start = int(getattr(environment, "_decision_step_start", 0))
        policy_steps = [
            step
            for step in rollout.episode.steps[credited_step_start:]
            if step.action.decision_source != "controller"
        ]
        decision_rows.append(
            {
                "trajectory_id": rollout.episode.trajectory_id,
                "actions": [step.action.action for step in policy_steps],
                "policy_call_attempt_count": reward.audit_metrics.get(
                    "decision_policy_call_attempt_count"
                ),
                "policy_call_rejection_count": reward.audit_metrics.get(
                    "decision_policy_call_rejection_count"
                ),
                "policy_argument_rejection_count": reward.audit_metrics.get(
                    "decision_policy_argument_rejection_count"
                ),
                "single_policy_call": reward.audit_metrics.get(
                    "decision_single_policy_call"
                ),
                "no_policy_call_rejections": reward.audit_metrics.get(
                    "decision_no_policy_call_rejections"
                ),
                "decision_action_match": reward.audit_metrics.get(
                    "decision_action_match"
                ),
                "decision_step_valid": reward.audit_metrics.get("decision_step_valid"),
                "reward": float(reward.episode_reward),
            }
        )

    advantages = [float(value) for value in prepared["advantages"].detach().cpu().tolist()]
    metrics = {
        name: values[-1]
        for name, values in trainer._metrics["train"].items()
        if values
    }
    structural_valid = all(
        row["policy_call_attempt_count"] == 1
        and row["policy_call_rejection_count"] == 0
        and row["policy_argument_rejection_count"] == 0
        and row["single_policy_call"] is True
        and row["no_policy_call_rejections"] is True
        for row in decision_rows
    )
    reward_variance = len(set(rewards)) >= 2 and float(metrics.get("reward_std") or 0) > 0
    advantage_variance = any(value > 0 for value in advantages) and any(
        value < 0 for value in advantages
    )
    tool_metrics_valid = (
        float(metrics.get("tools/call_frequency") or 0) == 1.0
        and float(metrics.get("tools/failure_frequency") or 0) == 0.0
    )
    no_optimizer_activity = (
        optimizer_absent_before and optimizer_absent_after and gradients_absent
    )
    source_unchanged = source_before == source_after
    eligible = all(
        (
            structural_valid,
            reward_variance,
            advantage_variance,
            tool_metrics_valid,
            no_optimizer_activity,
            source_unchanged,
        )
    )
    return {
        "schema_version": "actual-trainer-rollout-only-audit.v1",
        "status": "eligible" if eligible else "ineligible",
        "optimizer_eligibility": eligible,
        "promotion_eligible": False,
        "deployment_eligible": False,
        "no_backward_or_optimizer": no_optimizer_activity,
        "source_adapter_unchanged": source_unchanged,
        "source_adapter": source_before,
        "group_size": expected_group_size,
        "rewards": rewards,
        "advantages": advantages,
        "decision_rows": decision_rows,
        "trainer_metrics": metrics,
        "gates": {
            "structural_valid": structural_valid,
            "reward_variance": reward_variance,
            "advantage_variance": advantage_variance,
            "tool_metrics_valid": tool_metrics_valid,
            "no_optimizer_activity": no_optimizer_activity,
            "source_unchanged": source_unchanged,
        },
        "render_protocol": {
            "version": getattr(trainer, "agent_render_protocol_version", None),
            "chat_template_sha256": getattr(
                trainer, "agent_chat_template_sha256", None
            ),
        },
    }


def validate_turn_credit_totals(totals: dict | None) -> list[str]:
    """Return hard R1-v2 evidence failures; an empty list is publishable."""
    if not totals:
        return ["TURN_CREDIT_TOTALS_MISSING"]
    errors: list[str] = []
    if int(totals.get("train_effective_nonzero_credited_turns") or 0) <= 0:
        errors.append("NO_EFFECTIVE_NONZERO_TRAIN_TURN_CREDIT")
    compared = int(totals.get("train_compared_turn_buckets") or 0)
    zero_variance = int(totals.get("train_zero_variance_turn_buckets") or 0)
    if compared <= 0:
        errors.append("NO_COMPARABLE_TURN_BUCKETS")
    elif zero_variance >= compared:
        errors.append("ALL_COMPARABLE_TURN_BUCKETS_ZERO_VARIANCE")
    if int(totals.get("train_invalid_action_positive_credit_count") or 0) > 0:
        errors.append("INVALID_ACTION_RECEIVED_POSITIVE_CREDIT")
    if int(totals.get("alignment_rejected_trajectories") or 0) > 0:
        errors.append("TURN_TO_TOKEN_ALIGNMENT_NOT_PROVEN")
    if int(totals.get("extra_unmatched_model_turns") or 0) > 0:
        errors.append("EXTRA_UNMATCHED_MODEL_TURNS")
    return errors


def latest_completed_eval_metrics(log_history: list[dict] | None) -> dict:
    """Return an epoch-end evaluation already completed inside ``train()``."""
    for item in reversed(log_history or []):
        if item.get("eval_runtime") is not None and item.get("eval_reward") is not None:
            return {key: value for key, value in item.items() if key.startswith("eval_")}
    return {}


def resolve_num_generations_eval(train_generations: int, eval_generations: int) -> int:
    """Resolve an optional eval-group override without weakening variance checks."""
    if train_generations < 4:
        raise ValueError(
            "GRPO num_generations must be at least 4 for meaningful group variance"
        )
    resolved = eval_generations or train_generations
    if resolved < 4:
        raise ValueError(
            "GRPO num_generations_eval must be zero or at least 4 for meaningful group variance"
        )
    return resolved


def validate_eval_corpus_size(validation_tasks: int, eval_generations: int) -> None:
    """Require at least one complete evaluation generation group."""
    if validation_tasks < eval_generations:
        raise ValueError(
            "validation corpus must contain at least num-generations-eval tasks "
            f"({validation_tasks}<{eval_generations})"
        )
    if validation_tasks % eval_generations:
        raise ValueError(
            "validation corpus size must be divisible by num-generations-eval "
            f"({validation_tasks}%{eval_generations})"
        )


def validate_routed_audit_protocol(
    manifest: dict,
    *,
    source_adapter_sha256: str | None,
    num_generations: int,
    temperature: float,
    execution_mode: str,
    audit_max_new_tokens: int,
    max_tool_calling_iterations: int,
    reward_config_version: str,
) -> list[str]:
    """Reject routed curricula whose audit contract differs from training."""
    if manifest.get("schema_version") != "routed-grpo-curriculum.v1":
        return []
    protocol = manifest.get("uniform_audit_protocol")
    if not isinstance(protocol, dict):
        return ["UNIFORM_AUDIT_PROTOCOL_MISSING"]
    expected = {
        "checkpoint_adapter_sha256": source_adapter_sha256,
        "execution_mode": execution_mode,
        "temperature": temperature,
        "decoding_mode": "native-unconstrained",
        "quantization": "nf4-double-quant",
        "group_size": num_generations,
        "max_new_tokens": audit_max_new_tokens,
        "max_tool_calling_iterations": max_tool_calling_iterations,
        "reward_config_versions": [reward_config_version],
    }
    return [
        f"AUDIT_TRAIN_PROTOCOL_MISMATCH:{field}"
        for field, expected_value in expected.items()
        if protocol.get(field) != expected_value
    ]


def main() -> int:
    run_started_at = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model", required=True, help="SFT checkpoint or merged policy model"
    )
    parser.add_argument(
        "--tokenizer",
        help=(
            "Optional tokenizer path/name. Use the base tokenizer when an archived "
            "adapter contains tokenizer metadata from an older Transformers release."
        ),
    )
    parser.add_argument("--minimum-train-tasks", type=int, default=1000)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument(
        "--num-generations-eval",
        type=int,
        default=0,
        help=(
            "Optional evaluation rollout-group size. Zero reuses --num-generations; "
            "set this explicitly when a wider exploration group is required for "
            "training but the frozen evaluation protocol uses a smaller group."
        ),
    )
    parser.add_argument(
        "--audit-max-new-tokens",
        type=int,
        default=0,
        help=(
            "Policy action token cap used by the routing audit. Routed curricula must "
            "declare the same value in their uniform audit protocol."
        ),
    )
    parser.add_argument(
        "--execution-mode",
        choices=("policy_driven", "controller_first", "react"),
        default="react",
        help=(
            "react matches production: the model owns research/recovery choices while "
            "the controller advances deterministic gates. controller_first is the old "
            "narrow baseline; policy_driven remains a full-DAG research stress mode."
        ),
    )
    parser.add_argument(
        "--max-tool-calling-iterations",
        type=int,
        default=DEFAULT_POLICY_DRIVEN_TOOL_ITERATIONS,
        help=(
            "Maximum policy tool rounds. Controller-first recovery usually needs "
            "two decisions; full-DAG policy_driven audits need the larger default."
        ),
    )
    parser.add_argument("--max-completion-length", type=int, default=16384)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("linear", "cosine", "constant", "constant_with_warmup"),
        default="linear",
        help=(
            "Optimizer learning-rate schedule. Short bounded GRPO runs should set "
            "this explicitly; the linear default otherwise decays almost to zero "
            "before a small curriculum has been covered."
        ),
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Fraction of optimizer steps used for LR warmup.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="On-policy rollout sampling temperature; must match the exploration audit.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="KL penalty against the frozen reference policy.",
    )
    parser.add_argument(
        "--credit-mode",
        choices=("trajectory_b0", "turn_r1"),
        default="trajectory_b0",
        help="B0 uses one group-relative trajectory advantage; R1 blends verified turn credit.",
    )
    parser.add_argument("--turn-credit-gamma", type=float, default=0.95)
    parser.add_argument("--turn-credit-blend", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Checkpoint interval; staged 20/40/60 audits should set this to 20.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Number of checkpoints to retain; keep at least 3 for 20/40/60 audits.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=5,
        help="Trainer metric interval; diagnostic one-step runs should set this to 1.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive values bound smoke/debug runs; formal training keeps -1.",
    )
    parser.add_argument(
        "--max-train-tasks",
        type=int,
        default=0,
        help="Optional deterministic prefix used only to keep smoke runs small.",
    )
    parser.add_argument(
        "--max-eval-tasks",
        type=int,
        default=0,
        help="Optional deterministic validation prefix used only for smoke runs.",
    )
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument(
        "--rollout-only",
        action="store_true",
        help="Run one actual Trainer generation/reward group without backward or save.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--allow-small-corpus",
        action="store_true",
        help="Smoke test only; never use this flag for a reported training run.",
    )
    parser.add_argument(
        "--allow-unknown-git-commit",
        action="store_true",
        help="Escape hatch for non-repo smoke sandboxes only; formal training runs "
        "must stay reproducible from a git commit.",
    )
    args = parser.parse_args()
    git_commit = require_git_commit(
        parser, REPO_ROOT, allow_unknown=args.allow_unknown_git_commit
    )

    if args.rollout_only and not args.allow_small_corpus:
        raise ValueError("rollout-only is an isolated diagnostic and requires --allow-small-corpus")

    eval_num_generations = resolve_num_generations_eval(
        args.num_generations, args.num_generations_eval
    )
    if args.beta < 0:
        raise ValueError("GRPO beta must be non-negative")
    if args.temperature <= 0:
        raise ValueError("GRPO temperature must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("GRPO warmup-ratio must be in [0, 1)")
    if args.save_steps <= 0:
        raise ValueError("GRPO save-steps must be positive")
    if args.save_total_limit <= 0:
        raise ValueError("GRPO save-total-limit must be positive")
    if args.logging_steps <= 0:
        raise ValueError("GRPO logging-steps must be positive")
    if not 0 < args.turn_credit_gamma <= 1:
        raise ValueError("turn-credit-gamma must be in (0, 1]")
    if not 0 <= args.turn_credit_blend <= 1:
        raise ValueError("turn-credit-blend must be in [0, 1]")
    if args.credit_mode == "turn_r1" and args.max_tool_calling_iterations < 2:
        raise ValueError("turn_r1 requires at least two tool-calling iterations")
    if (
        args.execution_mode == "policy_driven"
        and args.max_tool_calling_iterations < MIN_POLICY_DRIVEN_TOOL_ITERATIONS
    ):
        raise ValueError(
            "policy_driven GRPO requires at least "
            f"{MIN_POLICY_DRIVEN_TOOL_ITERATIONS} tool-calling iterations for the "
            "nominal production DAG"
        )
    effective_batch = args.batch_size * args.gradient_accumulation
    if effective_batch % args.num_generations:
        raise ValueError(
            "batch-size * gradient-accumulation must be divisible by num-generations "
            "on a single-GPU run"
        )
    manifest_path = args.corpus_dir / "manifest.json"
    corpus_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    source_adapter_path = Path(args.model) / "adapter_model.safetensors"
    source_adapter_sha256 = (
        str(_file_provenance(source_adapter_path)["sha256"])
        if source_adapter_path.is_file()
        else None
    )
    audit_protocol_errors = validate_routed_audit_protocol(
        corpus_manifest,
        source_adapter_sha256=source_adapter_sha256,
        num_generations=args.num_generations,
        temperature=args.temperature,
        execution_mode=args.execution_mode,
        audit_max_new_tokens=args.audit_max_new_tokens,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        reward_config_version=RewardConfig().config_version,
    )
    if audit_protocol_errors:
        raise ValueError(
            "routed GRPO audit/training protocol gate failed: "
            + ", ".join(audit_protocol_errors)
        )
    environment_factories = build_trl_environment_factories(args.execution_mode)
    minimum = 1 if args.allow_small_corpus else args.minimum_train_tasks
    report = preflight_grpo_corpus(
        args.corpus_dir,
        minimum_train_tasks=minimum,
        require_dependencies=not args.preflight_only,
    )
    print(report.model_dump_json(indent=2))
    if args.preflight_only:
        return (
            0
            if not [error for error in report.errors if "DEPENDENCIES" not in error]
            else 2
        )
    if not report.ready:
        return 2

    train_corpus = load_grpo_corpus(args.corpus_dir / "train.jsonl")
    validation_corpus = load_grpo_corpus(args.corpus_dir / "validation.jsonl")
    if args.max_train_tasks > 0:
        train_corpus = train_corpus[: args.max_train_tasks]
    if args.max_eval_tasks > 0:
        validation_corpus = validation_corpus[: args.max_eval_tasks]
    static_completion_floor = minimum_completion_length_floor(
        [*train_corpus, *validation_corpus]
    )
    if args.max_completion_length < static_completion_floor:
        raise ValueError(
            "Agentic GRPO max-completion-length is too small for the corpus contract: "
            f"{args.max_completion_length}<{static_completion_floor}. "
            "Full loops must retain tool-result state; verified one-decision replay "
            "uses a smaller bounded generation budget."
        )
    validate_eval_corpus_size(len(validation_corpus), eval_num_generations)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not args.use_vllm:
        _disable_unused_vllm_import()
    import trl
    from trl import GRPOConfig, GRPOTrainer
    StableToolSuffixGRPOTrainer = create_stable_tool_suffix_grpo_trainer_class(
        base_trainer_class=GRPOTrainer,
        trl_version=trl.__version__,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Agentic GRPO training requires a CUDA GPU")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer_source = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise RuntimeError(
            "policy checkpoint must provide a native tool-capable chat_template"
        )

    train_rows = to_trl_environment_rows(train_corpus)
    validation_rows = to_trl_environment_rows(validation_corpus)
    rollout_contracts = sorted({str(row["rollout_contract"]) for row in train_rows})
    native_chat_template = tokenizer.chat_template
    install_agent_chat_template(tokenizer)
    try:
        completion_budget = estimate_stateful_completion_budget(
            [*train_corpus, *validation_corpus],
            tokenizer,
            environment_factories,
        )
    finally:
        tokenizer.chat_template = native_chat_template
    print(completion_budget.model_dump_json(indent=2))
    if args.max_completion_length < completion_budget.minimum_completion_length:
        raise ValueError(
            "max-completion-length cannot hold the measured production tool result "
            "and a follow-up action: "
            f"{args.max_completion_length}<{completion_budget.minimum_completion_length} "
            f"(limiting task: {completion_budget.limiting_task_id})"
        )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    adapter_config = Path(args.model) / "adapter_config.json"
    if adapter_config.is_file():
        # Continue optimizing the existing SFT adapter. Passing an adapter path
        # as a plain model id makes Transformers look for full-model weights and
        # either fail or silently start from the wrong policy.
        peft_config = PeftConfig.from_pretrained(args.model)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=quantization,
            device_map="auto",
            dtype=compute_dtype,
            trust_remote_code=False,
        )
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True,
        )
        model = PeftModel.from_pretrained(base_model, args.model, is_trainable=True)
        trainer_quantization = None
        trainer_peft = None
        model_init_kwargs = None
        continued_from_adapter = True
    else:
        model = args.model
        trainer_quantization = quantization
        trainer_peft = lora
        model_init_kwargs = {"dtype": compute_dtype, "trust_remote_code": False}
        continued_from_adapter = False
    report_to = ["mlflow"] if os.environ.get("MLFLOW_TRACKING_URI") else []
    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        # TRL scores evaluation rewards in generation groups as well. A batch
        # smaller than the group cannot be reshaped into preference groups.
        per_device_eval_batch_size=eval_num_generations,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        num_generations=args.num_generations,
        num_generations_eval=eval_num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        temperature=args.temperature,
        beta=args.beta,
        # Current TRL recommends DAPO-style token aggregation. The historical
        # `grpo` loss is length-biased and rewards short positive trajectories.
        loss_type="dapo",
        scale_rewards="group",
        mask_truncated_completions=True,
        chat_template_kwargs=dict(AGENT_CHAT_TEMPLATE_KWARGS),
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        eval_strategy="no" if args.max_steps > 0 else "steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        seed=args.seed,
        report_to=report_to,
        run_name=(
            f"agent-policy-grpo-b0-{args.execution_mode}-{Path(args.model).name}"
        ),
        model_init_kwargs=model_init_kwargs,
    )
    trainer_class = StableToolSuffixGRPOTrainer
    trainer_extra: dict[str, float] = {}
    if args.credit_mode == "turn_r1":
        from ml.agentic.training.turn_credit_trainer import (
            create_turn_credit_trainer_class,
        )

        trainer_class = create_turn_credit_trainer_class(
            base_trainer_class=StableToolSuffixGRPOTrainer
        )
        trainer_extra = {
            "turn_credit_gamma": args.turn_credit_gamma,
            "turn_credit_blend": args.turn_credit_blend,
        }
    trainer = trainer_class(
        model=model,
        args=training_args,
        # Keep conversational messages as JSON objects without Arrow filling
        # heterogeneous message keys (for example ``tool_calls``) with None.
        train_dataset=Dataset.from_list(train_rows, on_mixed_types="use_json"),
        eval_dataset=Dataset.from_list(validation_rows, on_mixed_types="use_json"),
        processing_class=tokenizer,
        quantization_config=trainer_quantization,
        peft_config=trainer_peft,
        environment_factory=environment_factories,
        **trainer_extra,
    )
    if args.rollout_only:
        rollout_report = run_rollout_only_audit(
            trainer,
            expected_group_size=args.num_generations,
            source_adapter_path=source_adapter_path,
        )
        rollout_report.update(
            {
                "seed": args.seed,
                "temperature": args.temperature,
                "beta": args.beta,
                "configured_max_steps": args.max_steps,
                "executed_optimizer_steps": 0,
                "corpus": {
                    name: _file_provenance(args.corpus_dir / name)
                    for name in ("train.jsonl", "validation.jsonl", "manifest.json")
                    if (args.corpus_dir / name).is_file()
                },
            }
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "rollout_only_report.json").write_text(
            json.dumps(rollout_report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(rollout_report, ensure_ascii=False, indent=2, default=str))
        return 0 if rollout_report["optimizer_eligibility"] else 3
    train_result = trainer.train()
    # Preserve a completed optimization run before optional evaluation.
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    # Stateful grouped evaluation has a much higher peak-memory footprint than
    # a one-step smoke update.  Smoke checkpoints are evaluated afterwards by
    # the sequential native Agent Loop audit; formal runs retain Trainer eval.
    completed_eval_metrics = latest_completed_eval_metrics(trainer.state.log_history)
    if args.max_steps > 0:
        eval_metrics = {}
        eval_status = "skipped_for_smoke_external_agent_loop_audit"
    elif completed_eval_metrics:
        eval_metrics = completed_eval_metrics
        eval_status = "completed_during_train_epoch_end"
    else:
        eval_metrics = trainer.evaluate()
        eval_status = "completed_after_train"
    turn_credit_totals = getattr(trainer, "turn_credit_totals", None)
    turn_credit_gate_errors = (
        validate_turn_credit_totals(turn_credit_totals)
        if args.credit_mode == "turn_r1"
        else []
    )
    audit_path = os.environ.get("AGENTIC_GRPO_AUDIT_PATH")
    corpus_provenance = {
        name: _file_provenance(args.corpus_dir / name)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json")
        if (args.corpus_dir / name).is_file()
    }
    source_model_provenance = {
        name: _file_provenance(Path(args.model) / name)
        for name in ("adapter_config.json", "adapter_model.safetensors")
        if (Path(args.model) / name).is_file()
    }
    output_model_provenance = {
        name: _file_provenance(args.output_dir / name)
        for name in ("adapter_config.json", "adapter_model.safetensors")
        if (args.output_dir / name).is_file()
    }
    rollout_audit_provenance = (
        _file_provenance(Path(audit_path))
        if audit_path and Path(audit_path).is_file()
        else None
    )
    metadata = {
        "status": "rejected" if turn_credit_gate_errors else "trained",
        "run_scope": "smoke" if args.max_steps > 0 or args.allow_small_corpus else "formal",
        "method": (
            "group-relative-turn-credit-grpo-r1"
            if args.credit_mode == "turn_r1"
            else "trajectory-level-agentic-grpo-b0"
        ),
        "credit_mode": args.credit_mode,
        "execution_mode": args.execution_mode,
        "policy_decision_scope": {
            "policy_driven": "all_dag_actions",
            "controller_first": "legacy_narrow_delegated_actions",
            "react": "production_research_recovery_clarification_tradeoff_actions",
        }[args.execution_mode],
        "rollout_initialization_contract": (
            rollout_contracts[0] if len(rollout_contracts) == 1 else "mixed"
        ),
        "rollout_initialization_contracts": rollout_contracts,
        "teacher_trajectory_prefix": (
            VERIFIED_DECISION_STATE_REPLAY_CONTRACT in rollout_contracts
        ),
        "teacher_prefix_optimization_targets": False,
        "verified_replay_prefix_in_prompt": (
            VERIFIED_DECISION_STATE_REPLAY_CONTRACT in rollout_contracts
        ),
        "credit_assignment_claim": (
            "programmatic turn-relative research baseline"
            if args.credit_mode == "turn_r1"
            else "trajectory-level only"
        ),
        "turn_credit_gamma": (
            args.turn_credit_gamma if args.credit_mode == "turn_r1" else None
        ),
        "turn_credit_blend": (
            args.turn_credit_blend if args.credit_mode == "turn_r1" else None
        ),
        "turn_credit_totals": turn_credit_totals,
        "turn_credit_gate_errors": turn_credit_gate_errors,
        "source_model": args.model,
        "tokenizer": tokenizer_source,
        "render_protocol": {
            "version": AGENT_RENDER_PROTOCOL_VERSION,
            "chat_template": _file_provenance(AGENT_CHAT_TEMPLATE_PATH),
            "chat_template_sha256": AGENT_CHAT_TEMPLATE_SHA256,
            "chat_template_kwargs": AGENT_CHAT_TEMPLATE_KWARGS,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_vocab_sha256": _canonical_sha256(tokenizer.get_vocab()),
            "tokenizer_special_tokens_sha256": _canonical_sha256(
                tokenizer.special_tokens_map
            ),
        },
        "continued_from_sft_adapter": continued_from_adapter,
        "git_commit": git_commit,
        "provenance": {
            "run_started_at": run_started_at,
            "run_finished_at": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "training_script": _file_provenance(Path(__file__).resolve()),
            "corpus": corpus_provenance,
            "routed_audit_protocol": corpus_manifest.get("uniform_audit_protocol"),
            "source_model": source_model_provenance,
            "output_model": output_model_provenance,
            "rollout_audit": rollout_audit_provenance,
            "dependencies": [
                dependency.model_dump(mode="json") for dependency in report.dependencies
            ],
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            },
        },
        "seed": args.seed,
        "num_generations": args.num_generations,
        "num_generations_eval": eval_num_generations,
        "audit_max_new_tokens": args.audit_max_new_tokens,
        "temperature": args.temperature,
        "beta": args.beta,
        "optimization": {
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "logging_steps": args.logging_steps,
            "effective_completion_batch_size": effective_batch,
            "effective_prompt_batch_size": effective_batch // args.num_generations,
            "per_device_eval_batch_size": eval_num_generations,
            "loss_type": "dapo",
            "scale_rewards": "group",
            "mask_truncated_completions": True,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "load_in_4bit": True,
            "cuda_allocator_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "max_tool_calling_iterations": args.max_tool_calling_iterations,
        "max_completion_length": args.max_completion_length,
        "completion_budget": completion_budget.model_dump(mode="json"),
        "reward": RewardConfig().config_version,
        "environment_versions": report.environment_versions,
        "snapshot_versions": report.snapshot_versions,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "eval_status": eval_status,
        "rollout_audit_path": audit_path,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if turn_credit_gate_errors:
        raise RuntimeError(
            "turn_r1 training failed its evidence gate: "
            + ", ".join(turn_credit_gate_errors)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
