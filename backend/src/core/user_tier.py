"""User tier mapping for quotas and cost controls (PRD §4.10.7)."""

from __future__ import annotations

from dataclasses import dataclass

from core.settings import settings


@dataclass(frozen=True)
class TierLimits:
    name: str
    daily_tokens: int
    daily_itineraries: int
    daily_external_api_calls: int
    allow_large_model: bool


def resolve_tier(role: str) -> str:
    """Map application role to quota tier."""
    if role == "guest":
        return "guest"
    if role == "admin":
        return "admin"
    if role in ("member", "premium"):
        return role
    return "free"


def tier_limits(role: str) -> TierLimits:
    tier = resolve_tier(role)
    if tier == "guest":
        return TierLimits(
            name="guest",
            daily_tokens=settings.llm_quota_guest_daily,
            daily_itineraries=settings.guest_max_completed_itineraries,
            daily_external_api_calls=settings.external_api_quota_guest_daily,
            allow_large_model=False,
        )
    if tier == "member":
        return TierLimits(
            name="member",
            daily_tokens=settings.llm_quota_member_daily,
            daily_itineraries=settings.member_max_completed_itineraries,
            daily_external_api_calls=settings.external_api_quota_member_daily,
            allow_large_model=True,
        )
    if tier == "premium":
        return TierLimits(
            name="premium",
            daily_tokens=settings.llm_quota_premium_daily,
            daily_itineraries=settings.premium_max_completed_itineraries,
            daily_external_api_calls=settings.external_api_quota_premium_daily,
            allow_large_model=True,
        )
    if tier == "admin":
        return TierLimits(
            name="admin",
            daily_tokens=10_000_000_000,
            daily_itineraries=10_000_000,
            daily_external_api_calls=10_000_000,
            allow_large_model=True,
        )
    return TierLimits(
        name="free",
        daily_tokens=settings.llm_quota_user_daily,
        daily_itineraries=settings.free_max_completed_itineraries,
        daily_external_api_calls=settings.external_api_quota_free_daily,
        allow_large_model=settings.free_user_allow_large_model,
    )
