"""Version-bound approval tokens for resumable itinerary drafts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4


APPROVAL_SCHEMA_VERSION = "itinerary-approval.v1"
DEFAULT_APPROVAL_TTL = timedelta(minutes=30)


class ApprovalValidationError(ValueError):
    """A control action does not target the currently pending draft."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def itinerary_hash(itinerary: Any) -> str:
    canonical = json.dumps(
        itinerary or [],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def issue_pending_approval(
    state: dict[str, Any],
    *,
    now: datetime | None = None,
    ttl: timedelta = DEFAULT_APPROVAL_TTL,
) -> dict[str, Any]:
    issued_at = now or datetime.now(UTC)
    ledger = state.get("agent_ledger") or {}
    goal = ledger.get("goal") or {}
    task_graph = ledger.get("task_graph") or {}
    return {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "approval_id": str(uuid4()),
        "user_id": str(state.get("user_id") or ""),
        "goal_version": int(goal.get("goal_version") or 1),
        "plan_version": int(task_graph.get("plan_version") or 1),
        "itinerary_hash": itinerary_hash(state.get("itinerary")),
        "issued_at": issued_at.isoformat(),
        "expires_at": (issued_at + ttl).isoformat(),
        "action_scope": ["confirm", "modify", "reject"],
    }


def public_approval(approval: dict[str, Any] | None) -> dict[str, Any] | None:
    if not approval:
        return None
    return {
        key: approval.get(key)
        for key in (
            "schema_version",
            "approval_id",
            "goal_version",
            "plan_version",
            "itinerary_hash",
            "issued_at",
            "expires_at",
            "action_scope",
        )
    }


def validate_pending_approval(
    state: dict[str, Any],
    provided: dict[str, Any] | None,
    *,
    action: str,
    user_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    pending = state.get("pending_approval")
    if not isinstance(pending, dict):
        raise ApprovalValidationError(
            "APPROVAL_NOT_ISSUED",
            "当前草案没有有效确认凭证，请重新生成或刷新草案。",
        )
    if not isinstance(provided, dict):
        raise ApprovalValidationError(
            "APPROVAL_REQUIRED",
            "确认请求缺少草案凭证，请刷新页面后重试。",
        )
    if pending.get("schema_version") != APPROVAL_SCHEMA_VERSION:
        raise ApprovalValidationError(
            "APPROVAL_VERSION_MISMATCH",
            "草案由旧版本系统生成，请重新生成后确认。",
        )
    if str(pending.get("user_id") or "") != str(user_id):
        raise ApprovalValidationError(
            "APPROVAL_USER_MISMATCH",
            "该确认凭证不属于当前用户。",
        )
    for key in ("approval_id", "goal_version", "plan_version", "itinerary_hash"):
        if provided.get(key) != pending.get(key):
            raise ApprovalValidationError(
                "APPROVAL_STALE",
                "行程草案已发生变化，请查看最新方案后重新确认。",
            )
    if action not in set(pending.get("action_scope") or []):
        raise ApprovalValidationError(
            "APPROVAL_SCOPE_MISMATCH",
            "该凭证不能执行当前操作。",
        )
    try:
        expires_at = datetime.fromisoformat(str(pending.get("expires_at")))
    except (TypeError, ValueError) as exc:
        raise ApprovalValidationError(
            "APPROVAL_INVALID",
            "确认凭证格式无效，请重新生成草案。",
        ) from exc
    current_time = now or datetime.now(UTC)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at <= current_time:
        raise ApprovalValidationError(
            "APPROVAL_EXPIRED",
            "行程草案确认已过期，请刷新实时信息后重新确认。",
        )
    if itinerary_hash(state.get("itinerary")) != pending.get("itinerary_hash"):
        raise ApprovalValidationError(
            "APPROVAL_STALE",
            "行程内容已发生变化，请查看最新方案后重新确认。",
        )
    return pending
