"""Thin TRL adapter for group-relative turn-to-token credit assignment."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from agentic.turn_credit import model_token_segments, project_turn_relative_advantages


_ALIGNMENT_AUDIT_FILENAME = "turn_credit_alignment_mismatches.jsonl"
_ALIGNMENT_AUDIT_LOCK = threading.Lock()
_MAX_AUDIT_SPAN_TOKENS = 64
_MAX_AUDIT_DECODED_CHARS = 512
_MAX_AUDIT_CREDIT_RECORDS = 16
_MAX_AUDIT_COLLECTION_ITEMS = 32
_MAX_AUDIT_STRING_CHARS = 512


def _bounded_json_value(value: Any, *, depth: int = 0) -> Any:
    """Convert an arbitrary credit record to a bounded JSON-safe value."""
    if depth >= 4:
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:_MAX_AUDIT_STRING_CHARS]
    if isinstance(value, dict):
        return {
            str(key)[:_MAX_AUDIT_STRING_CHARS]: _bounded_json_value(item, depth=depth + 1)
            for key, item in list(value.items())[:_MAX_AUDIT_COLLECTION_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded_json_value(item, depth=depth + 1)
            for item in value[:_MAX_AUDIT_COLLECTION_ITEMS]
        ]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _bounded_json_value(model_dump(mode="json"), depth=depth + 1)
    return None


def _row_as_ints(rows: Any, index: int) -> list[int] | None:
    """Read one tensor/list row without assuming a concrete tensor library."""
    if rows is None:
        return None
    try:
        row = rows[index]
        detach = getattr(row, "detach", None)
        if callable(detach):
            row = detach()
        cpu = getattr(row, "cpu", None)
        if callable(cpu):
            row = cpu()
        tolist = getattr(row, "tolist", None)
        if callable(tolist):
            row = tolist()
        return [int(item) for item in row]
    except (IndexError, TypeError, ValueError):
        return None


def _environment_identity(environment: Any) -> tuple[str | None, str | None]:
    """Return only identifiers explicitly exposed by the rollout environment."""
    task_id = getattr(environment, "_task_id", None)
    session = getattr(environment, "_session", None)
    recorder = getattr(session, "recorder", None) if session is not None else None
    episode = getattr(recorder, "episode", None) if recorder is not None else None
    trajectory_id = getattr(episode, "trajectory_id", None) if episode is not None else None
    if trajectory_id is None:
        reward = getattr(environment, "reward_record", None)
        if reward is None:
            reward = getattr(environment, "_reward", None)
        trajectory_id = (
            getattr(reward, "trajectory_id", None) if reward is not None else None
        )
    return (
        str(task_id) if task_id is not None else None,
        str(trajectory_id) if trajectory_id is not None else None,
    )


def create_turn_credit_trainer_class(base_trainer_class=None):
    """Import TRL lazily so dependency-light preflight remains usable."""
    if base_trainer_class is None:
        from trl import GRPOTrainer

        base_trainer_class = GRPOTrainer

    class TurnCreditGRPOTrainer(base_trainer_class):
        """Blend B0 trajectory advantage with relative per-turn RTG credit."""

        def __init__(
            self,
            *args: Any,
            turn_credit_gamma: float = 0.95,
            turn_credit_blend: float = 0.5,
            **kwargs: Any,
        ) -> None:
            if not 0 < turn_credit_gamma <= 1:
                raise ValueError("turn_credit_gamma must be in (0, 1]")
            if not 0 <= turn_credit_blend <= 1:
                raise ValueError("turn_credit_blend must be in [0, 1]")
            self.turn_credit_gamma = turn_credit_gamma
            self.turn_credit_blend = turn_credit_blend
            self.turn_credit_totals = {
                "batches": 0,
                "trajectories": 0,
                "eligible_multiturn_trajectories": 0,
                "model_turns": 0,
                "locally_credited_turns": 0,
                "effective_nonzero_credited_turns": 0,
                "zero_variance_turn_buckets": 0,
                "compared_turn_buckets": 0,
                "invalid_model_turns": 0,
                "external_failure_turns": 0,
                "unmatched_model_turns": 0,
                "alignment_rejected_trajectories": 0,
                "extra_unmatched_model_turns": 0,
                "invalid_action_positive_credit_count": 0,
                "train_locally_credited_turns": 0,
                "eval_locally_credited_turns": 0,
                "train_effective_nonzero_credited_turns": 0,
                "eval_effective_nonzero_credited_turns": 0,
                "train_compared_turn_buckets": 0,
                "eval_compared_turn_buckets": 0,
                "train_zero_variance_turn_buckets": 0,
                "eval_zero_variance_turn_buckets": 0,
                "train_invalid_action_positive_credit_count": 0,
                "eval_invalid_action_positive_credit_count": 0,
            }
            super().__init__(*args, **kwargs)
            output_dir = getattr(getattr(self, "args", None), "output_dir", None)
            self.turn_credit_alignment_audit_path = (
                Path(str(output_dir)) / _ALIGNMENT_AUDIT_FILENAME
                if output_dir is not None
                else None
            )

        def _decoded_model_spans(
            self,
            completion_ids: Any,
            row_index: int,
            spans: list[tuple[int, int]],
        ) -> list[dict[str, Any]]:
            token_row = _row_as_ints(completion_ids, row_index)
            decoder = getattr(self, "processing_class", None)
            decode = getattr(decoder, "decode", None)
            decoded: list[dict[str, Any]] = []
            for start, end in spans:
                text: str | None = None
                decode_error: str | None = None
                token_truncated = end - start > _MAX_AUDIT_SPAN_TOKENS
                if token_row is not None:
                    token_ids = token_row[start : min(end, start + _MAX_AUDIT_SPAN_TOKENS)]
                    if callable(decode):
                        try:
                            text = str(decode(token_ids, skip_special_tokens=False))
                        except (RuntimeError, TypeError, ValueError) as error:
                            decode_error = type(error).__name__
                text_truncated = text is not None and len(text) > _MAX_AUDIT_DECODED_CHARS
                decoded.append(
                    {
                        "range": [start, end],
                        "token_count": end - start,
                        "decoded_text": (
                            text[:_MAX_AUDIT_DECODED_CHARS] if text is not None else None
                        ),
                        "truncated": token_truncated or text_truncated,
                        "decode_error": decode_error,
                    }
                )
            return decoded

        def _write_alignment_mismatch_audit(
            self,
            *,
            environment: Any,
            batch_index: int,
            row_index: int,
            spans: list[tuple[int, int]],
            records: list[dict[str, Any]] | None,
            credit_count: int,
            completion_ids: Any,
        ) -> None:
            path = self.turn_credit_alignment_audit_path
            if path is None:
                raise RuntimeError(
                    "turn-credit alignment mismatch cannot be audited: output_dir is unavailable"
                )
            task_id, trajectory_id = _environment_identity(environment)
            extra_spans = max(0, len(spans) - credit_count - 1)
            missing_spans = max(0, credit_count - len(spans))
            payload = {
                "schema_version": "turn-credit-alignment-mismatch.v1",
                "task_id": task_id,
                "trajectory_id": trajectory_id,
                "trainer_batch_index": batch_index,
                "batch_row_index": row_index,
                "model_span_count": len(spans),
                "model_span_ranges": [[start, end] for start, end in spans],
                "credit_record_count": credit_count,
                "credit_records": (
                    [
                        _bounded_json_value(item)
                        for item in records[:_MAX_AUDIT_CREDIT_RECORDS]
                    ]
                    if records is not None
                    else None
                ),
                "credit_records_truncated": (
                    len(records) > _MAX_AUDIT_CREDIT_RECORDS
                    if records is not None
                    else None
                ),
                "extra_unmatched_model_spans": extra_spans,
                "missing_model_spans": missing_spans,
                "mismatch_type": (
                    "extra_model_spans" if extra_spans else "missing_model_spans"
                ),
                "decoded_model_spans": self._decoded_model_spans(
                    completion_ids, row_index, spans
                ),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            try:
                with _ALIGNMENT_AUDIT_LOCK, path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            except OSError as error:
                raise RuntimeError(
                    f"turn-credit alignment mismatch audit write failed: {path}"
                ) from error

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            environments = list(self.environments or [])
            masks = output["completion_mask"]
            if "tool_mask" in output:
                masks = masks * output["tool_mask"]
            credits: list[list[float]] = []
            validities: list[list[str]] = []
            credit_records: list[list[dict[str, Any]] | None] = []
            for environment in environments:
                credit_fn = getattr(environment, "_policy_turn_credits", None)
                record_fn = getattr(environment, "_policy_turn_credit_records", None)
                if record_fn is not None:
                    records = record_fn(self.turn_credit_gamma)
                    credits.append([float(item["credit"]) for item in records])
                    validities.append([str(item["validity"]) for item in records])
                    credit_records.append(records)
                else:
                    row = credit_fn(self.turn_credit_gamma) if credit_fn is not None else []
                    credits.append(row)
                    validities.append(["valid"] * len(row))
                    credit_records.append(None)
            if len(credits) != int(output["advantages"].shape[0]):
                raise RuntimeError(
                    "turn-credit environments must align with local rollout advantages"
                )
            mask_rows = masks.detach().int().cpu().tolist()
            segments = [model_token_segments(mask) for mask in mask_rows]
            batch_index = int(self.turn_credit_totals["batches"])
            for row_index, (environment, spans, records, row_credits) in enumerate(
                zip(environments, segments, credit_records, credits, strict=True)
            ):
                aligned = len(spans) >= len(row_credits) and len(spans) - len(row_credits) <= 1
                if not aligned:
                    self._write_alignment_mismatch_audit(
                        environment=environment,
                        batch_index=batch_index,
                        row_index=row_index,
                        spans=spans,
                        records=records,
                        credit_count=len(row_credits),
                        completion_ids=output.get("completion_ids"),
                    )
            projected, report = project_turn_relative_advantages(
                output["advantages"].detach().float().cpu().tolist(),
                credits,
                mask_rows,
                group_size=(
                    self.num_generations
                    if self.model.training
                    else self.num_generations_eval
                ),
                blend_weight=self.turn_credit_blend,
                policy_turn_validities=validities,
            )
            import torch

            output["advantages"] = torch.tensor(
                projected,
                dtype=output["advantages"].dtype,
                device=output["advantages"].device,
            )
            self.turn_credit_totals["batches"] += 1
            for name in (
                "trajectories",
                "eligible_multiturn_trajectories",
                "model_turns",
                "locally_credited_turns",
                "effective_nonzero_credited_turns",
                "zero_variance_turn_buckets",
                "compared_turn_buckets",
                "invalid_model_turns",
                "external_failure_turns",
                "unmatched_model_turns",
                "alignment_rejected_trajectories",
                "extra_unmatched_model_turns",
                "invalid_action_positive_credit_count",
            ):
                self.turn_credit_totals[name] += int(getattr(report, name))
            mode = "train" if self.model.training else "eval"
            self.turn_credit_totals[f"{mode}_locally_credited_turns"] += int(
                report.locally_credited_turns
            )
            self.turn_credit_totals[
                f"{mode}_effective_nonzero_credited_turns"
            ] += int(report.effective_nonzero_credited_turns)
            self.turn_credit_totals[f"{mode}_compared_turn_buckets"] += int(
                report.compared_turn_buckets
            )
            self.turn_credit_totals[f"{mode}_zero_variance_turn_buckets"] += int(
                report.zero_variance_turn_buckets
            )
            self.turn_credit_totals[
                f"{mode}_invalid_action_positive_credit_count"
            ] += int(report.invalid_action_positive_credit_count)
            ratio = (
                report.locally_credited_turns / report.model_turns
                if report.model_turns
                else 0.0
            )
            self._log_metric("turn_credit/credited_turn_ratio", ratio)
            self._log_metric(
                "turn_credit/eligible_trajectories",
                float(report.eligible_multiturn_trajectories),
            )
            self._log_metric(
                "turn_credit/zero_advantage_group_ratio",
                report.zero_advantage_group_ratio,
            )
            self._log_metric(
                "turn_credit/invalid_action_positive_credit_rate",
                report.invalid_action_positive_credit_rate,
            )
            self._log_metric(
                "turn_credit/effective_nonzero_credited_turns",
                float(report.effective_nonzero_credited_turns),
            )
            return output

    return TurnCreditGRPOTrainer


__all__ = ["create_turn_credit_trainer_class"]
