"""Redis URL helpers for logical DB split (M2)."""

from core.settings import settings


def test_redis_url_for_db_replaces_suffix():
    url = settings.redis_url_for_db(2)
    assert url.endswith("/2")


def test_redis_cache_and_state_urls_use_configured_db():
    assert settings.redis_cache_url.endswith(f"/{settings.redis_db_cache}")
    assert settings.redis_state_url.endswith(f"/{settings.redis_db_state}")


def test_redlock_urls_default_to_state():
    assert settings.redis_redlock_url_list == [settings.redis_state_url]
