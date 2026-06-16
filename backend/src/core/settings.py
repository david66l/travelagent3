import os
import secrets
import warnings
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

# Locate .env relative to this file: backend/src/core/settings.py -> project root
# settings.py -> core -> src -> backend -> project_root
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database (default is local dev only; production must override via DATABASE_URL)
    database_url: str = (
        "postgresql+asyncpg://travelagent:travelagent123@localhost:5432/travel_agent"
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_cluster: bool = False
    redis_db_cache: int = 0
    redis_db_queue: int = 1
    redis_db_state: int = 2

    # LLM / vLLM
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    llm_timeout: int = 60
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "not-needed"
    default_model: str = "travel-plan-v1"
    small_model: str = "travel-chat-v1"

    # Search (Tavily - https://tavily.com, free 1000 calls/month)
    tavily_api_key: str = ""
    search_engine: Literal["tavily", "duckduckgo"] = "tavily"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Crawler
    crawl_rate_limit: float = 1.0
    crawl_max_retries: int = 3
    crawl_timeout: int = 30

    # Seed Data
    seed_data_dir: str = "backend/seed_data"
    seed_cities: str = (
        "北京,上海,广州,深圳,成都,杭州,西安,重庆,苏州,南京,厦门,青岛,大理,丽江,三亚,长沙,武汉,昆明,桂林,拉萨"
    )

    # Authentication
    # If JWT_SECRET is not provided, a one-time random secret is generated at
    # startup so the app can run without hard-coding a production credential.
    # Sessions will not survive restarts until JWT_SECRET is explicitly set.
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    guest_token_expire_hours: int = 24

    # Rate limiting
    rate_limit_ip_per_minute: int = 60
    rate_limit_user_per_minute: int = 30
    rate_limit_guest_per_minute: int = 10
    rate_limit_max_concurrent_sse: int = 3

    # Cache TTLs (seconds)
    cache_ttl_poi: int = 21600  # 6 hours
    cache_ttl_weather: int = 3600  # 1 hour
    cache_ttl_price: int = 1800  # 30 minutes
    cache_ttl_itinerary: int = 43200  # 12 hours
    cache_ttl_route: int = 86400  # 24 hours
    cache_ttl_session: int = 1800  # 30 minutes
    cache_ttl_rate_limit: int = 60  # 1 minute
    cache_ttl_jitter_ratio: float = 0.1
    cache_warm_top_n_cities: int = 20
    local_cache_ttl_seconds: int = 300

    # Celery / planning task retry (PRD §4.7.3)
    planning_task_max_retries: int = 3
    planning_task_retry_initial_seconds: float = 2.0
    planning_task_retry_max_seconds: float = 30.0
    task_retry_counter_ttl_seconds: int = 604800  # 7 days
    dead_letter_archive_days: int = 7

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_default_queue: str = "default"
    celery_planning_queue: str = "planning"
    celery_dead_letter_queue: str = "planning_dead_letter"
    celery_worker_prefetch_multiplier: int = 1
    # celery: dispatch to Celery worker queue; embedded: in-process PlanningWorker poll loop
    planning_executor: str = "celery"

    # External APIs
    amap_key: str = ""
    weather_key: str = ""

    # Tool / circuit breaker
    tool_timeout_seconds: float = 3.0
    tool_max_retries: int = 3
    circuit_breaker_failure_threshold: float = 0.5
    circuit_breaker_min_failures: int = 20
    circuit_breaker_recovery_seconds: int = 30
    circuit_breaker_window_seconds: int = 10

    # Cost control
    cost_circuit_breaker_daily_tokens: int = 50_000_000
    cost_circuit_breaker_daily_api_cost_cny: float = 1000.0
    cost_circuit_breaker_hourly_gpu_cost_cny: float = 500.0

    @property
    def seed_cities_list(self) -> list[str]:
        return [c.strip() for c in self.seed_cities.split(",") if c.strip()]

    @property
    def database_url_sync(self) -> str:
        """Synchronous database URL for Alembic and admin tools."""
        return self.database_url.replace("+asyncpg", "")


settings = Settings()

# Reject the unsafe placeholder explicitly.
if settings.jwt_secret == "your-secret-key-change-in-production":
    raise ValueError(
        "JWT_SECRET is using the unsafe placeholder. "
        "Please set a strong JWT_SECRET in your .env file."
    )

# Dev-only fallback: generate a random secret if none was configured.
if not settings.jwt_secret:
    _generated_jwt_secret = secrets.token_urlsafe(32)
    settings.jwt_secret = _generated_jwt_secret
    warnings.warn(
        "JWT_SECRET is not configured. A one-time random secret has been generated "
        "for this process only. Set JWT_SECRET in your .env file for persistent "
        "sessions across restarts.",
        RuntimeWarning,
        stacklevel=2,
    )
