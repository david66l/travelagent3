"""Unit tests for user tier quotas."""

from core.user_tier import resolve_tier, tier_limits
from core.settings import settings


def test_resolve_tier_mapping():
    assert resolve_tier("guest") == "guest"
    assert resolve_tier("user") == "free"
    assert resolve_tier("member") == "member"
    assert resolve_tier("admin") == "admin"


def test_guest_limits():
    limits = tier_limits("guest")
    assert limits.daily_tokens == settings.llm_quota_guest_daily
    assert limits.allow_large_model is False


def test_free_user_limits():
    limits = tier_limits("user")
    assert limits.daily_tokens == settings.llm_quota_user_daily
    assert limits.name == "free"
