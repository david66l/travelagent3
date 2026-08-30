"""Settings safety and Redis URL helpers."""

import pytest

from core.settings import Settings, _prepare_runtime_secrets, settings


def test_redis_url_for_db_replaces_suffix():
    url = settings.redis_url_for_db(2)
    assert url.endswith("/2")


def test_redis_cache_and_state_urls_use_configured_db():
    assert settings.redis_cache_url.endswith(f"/{settings.redis_db_cache}")
    assert settings.redis_state_url.endswith(f"/{settings.redis_db_state}")


def test_redlock_urls_default_to_state():
    assert settings.redis_redlock_url_list == [settings.redis_state_url]


@pytest.mark.parametrize(
    "secret",
    ["short", "change-me-in-production", "your-secret-key-change-in-production"],
)
def test_runtime_secrets_reject_weak_jwt(secret):
    config = Settings(
        _env_file=None,
        app_env="production",
        jwt_secret=secret,
        privacy_encryption_key="p" * 32,
    )

    with pytest.raises(ValueError, match="JWT_SECRET"):
        _prepare_runtime_secrets(config)


def test_runtime_secrets_require_both_values_in_production():
    config = Settings(
        _env_file=None,
        app_env="production",
        jwt_secret="j" * 32,
        privacy_encryption_key="",
    )

    with pytest.raises(ValueError, match="PRIVACY_ENCRYPTION_KEY"):
        _prepare_runtime_secrets(config)


def test_runtime_secrets_accept_strong_production_values():
    config = Settings(
        _env_file=None,
        app_env="production",
        jwt_secret="j" * 32,
        privacy_encryption_key="p" * 32,
    )

    _prepare_runtime_secrets(config)

    assert len(config.jwt_secret) == 32
    assert len(config.privacy_encryption_key) == 32
