import hashlib
import os
import re
import secrets
import warnings
from pydantic import AliasChoices, Field
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
        populate_by_name=True,
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
    # Comma-separated Redis URLs for Redlock (multi-master); empty → state DB only
    redis_redlock_urls: str = ""

    # LLM — 非意图任务走 DeepSeek，意图识别走本地 Qwen
    openai_api_key: str = ""  # 备用 (OpenAI Key)
    deepseek_api_key: str = ""  # DeepSeek API Key
    openai_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"  # DeepSeek-V3 (最新旗舰)
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    llm_timeout: int = 60
    vllm_base_url: str = "http://vllm:8000/v1"
    vllm_api_key: str = "not-needed"
    vllm_enabled: bool = False
    vllm_max_retries: int = 2
    default_model: str = "travel-plan-v1"
    small_model: str = "travel-chat-v1"
    repair_model: str = "travel-repair-v1"
    # 本地 llama.cpp 推理（意图识别用小模型）
    local_llm_url: str = "http://localhost:8081/v1"
    local_llm_model: str = "qwen2.5-7b-instruct"
    local_llm_enabled: bool = True

    # Search (Tavily - https://tavily.com, free 1000 calls/month)
    tavily_api_key: str = ""
    search_engine: Literal["tavily", "duckduckgo"] = "tavily"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    public_app_url: str | None = None  # external URL for generated download links
    debug: bool = False
    # Output formatting — when True, a streaming LLM pass writes the final itinerary
    # Markdown token-by-token (true real-time streaming: the model emits, the page
    # renders live). When False, the writer's pre-built Markdown is streamed back as
    # a fast replay (lower latency, no second LLM call). Default True for the live
    # streaming experience; set OUTPUT_POLISH_ENABLED=false to prioritise speed.
    output_polish_enabled: bool = True
    # Server-side PDF/Excel export. The frontend already exports client-side
    # (ExportCenter via jsPDF / write-excel-file), so the WeasyPrint/openpyxl
    # generation here is redundant and burns CPU + needs native libs. Off by
    # default; set SERVER_SIDE_EXPORT_ENABLED=true only if a client needs the URLs.
    server_side_export_enabled: bool = False
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

    # Vector embeddings are optional. The structured + lexical RAG path works
    # without a local model; enable "local" only in an image/environment that
    # installs the ``local-embedding`` dependency group and preloads BGE.
    embedding_provider: Literal["disabled", "local"] = "disabled"

    # Authentication
    # If JWT_SECRET is not provided, a one-time random secret is generated at
    # startup so the app can run without hard-coding a production credential.
    # Sessions will not survive restarts until JWT_SECRET is explicitly set.
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    guest_token_expire_hours: int = 24

    # Privacy
    # If PRIVACY_ENCRYPTION_KEY is not provided, a deterministic key is derived
    # from JWT_SECRET at startup.  This is process-only and will not survive
    # restarts until PRIVACY_ENCRYPTION_KEY is explicitly set.
    privacy_encryption_key: str = ""

    # Rate limiting
    rate_limit_ip_per_minute: int = 60
    rate_limit_user_per_minute: int = 30
    rate_limit_guest_per_minute: int = 10
    rate_limit_max_concurrent_sse: int = 3

    # Guest / LLM quotas (PRD §4.1 / §4.3 / §4.10.7)
    guest_max_completed_itineraries: int = 1
    free_max_completed_itineraries: int = 5
    member_max_completed_itineraries: int = 20
    premium_max_completed_itineraries: int = 100
    llm_quota_guest_daily: int = 10_000
    llm_quota_user_daily: int = 100_000
    llm_quota_member_daily: int = 500_000
    llm_quota_premium_daily: int = 2_000_000
    external_api_quota_guest_daily: int = 20
    external_api_quota_free_daily: int = 200
    external_api_quota_member_daily: int = 1000
    external_api_quota_premium_daily: int = 5000
    free_user_allow_large_model: bool = True
    external_api_default_cost_cny: float = 0.01

    # Prompt compression (M6 §4.10.1)
    prompt_compress_max_messages: int = 12
    prompt_compress_max_chars: int = 2000

    # Alerting (PRD §4.7.4 dead-letter webhook)
    alert_webhook_url: str = ""

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
    celery_memory_queue: str = "memory"
    celery_dead_letter_queue: str = "planning_dead_letter"
    celery_worker_prefetch_multiplier: int = 1
    # Redis broker re-delivers a message if not acked within this window. Must
    # comfortably exceed the longest task / countdown (now ~30s) so a long task
    # is never duplicated. Kept high since no countdown approaches it anymore.
    celery_visibility_timeout_seconds: int = 7200
    # celery: dispatch to Celery worker queue; embedded: in-process PlanningWorker poll loop
    planning_executor: str = "celery"

    # External APIs
    amap_key: str = Field(
        default="",
        validation_alias=AliasChoices("AMAP_KEY", "GAODE_API_KEY"),
    )
    weather_key: str = ""

    # VRP solver microservice
    vrp_solver_url: str = "http://localhost:8001"

    # Tool / circuit breaker
    tool_timeout_seconds: float = 3.0
    tool_max_retries: int = 3
    circuit_breaker_failure_threshold: float = 0.5
    circuit_breaker_min_failures: int = 20
    circuit_breaker_recovery_seconds: int = 30
    circuit_breaker_window_seconds: int = 10

    # Cost control (M6 §4.10.6)
    cost_circuit_breaker_enabled: bool = True
    cost_circuit_breaker_daily_tokens: int = 50_000_000
    cost_circuit_breaker_daily_api_cost_cny: float = 1000.0
    cost_circuit_breaker_hourly_gpu_cost_cny: float = 500.0

    # Observability (M5)
    otel_enabled: bool = False
    otel_exporter_endpoint: str = "http://localhost:4318/v1/traces"
    otel_service_name: str = "travel-agent-backend"
    metrics_path: str = "/api/v1/metrics"

    # MLflow (M5)
    mlflow_tracking_uri: str = ""
    mlflow_registry_uri: str = ""

    # LangSmith tracing (M5 §4.10.5)
    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str = ""
    langsmith_project: str = "TravelAgent"

    # AI safety (M5 §4.9)
    input_safety_enabled: bool = True
    output_safety_enabled: bool = True

    @property
    def seed_cities_list(self) -> list[str]:
        return [c.strip() for c in self.seed_cities.split(",") if c.strip()]

    def redis_url_for_db(self, db: int) -> str:
        """Build a Redis URL for a logical database index (PRD cache/queue/state split)."""
        base = self.redis_url.strip()
        if re.search(r"/\d+$", base):
            return re.sub(r"/\d+$", f"/{db}", base)
        return f"{base.rstrip('/')}/{db}"

    @property
    def redis_cache_url(self) -> str:
        return self.redis_url_for_db(self.redis_db_cache)

    @property
    def redis_state_url(self) -> str:
        return self.redis_url_for_db(self.redis_db_state)

    @property
    def redis_redlock_url_list(self) -> list[str]:
        if self.redis_redlock_urls.strip():
            return [u.strip() for u in self.redis_redlock_urls.split(",") if u.strip()]
        return [self.redis_state_url]

    @property
    def database_url_sync(self) -> str:
        """Synchronous database URL for Alembic and admin tools."""
        return self.database_url.replace("+asyncpg", "")


settings = Settings()


def _apply_langsmith_env(cfg: Settings) -> None:
    """Sync LangSmith settings into os.environ for SDK + @traceable decorators."""
    if not cfg.langsmith_api_key:
        return
    os.environ["LANGSMITH_TRACING"] = "true" if cfg.langsmith_tracing else "false"
    os.environ["LANGSMITH_API_KEY"] = cfg.langsmith_api_key
    os.environ["LANGSMITH_ENDPOINT"] = cfg.langsmith_endpoint
    os.environ["LANGSMITH_PROJECT"] = cfg.langsmith_project


_apply_langsmith_env(settings)

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

# Dev-only fallback: derive a deterministic privacy key from JWT_SECRET if none
# was configured.  This is process-only when JWT_SECRET is also process-only.
if not settings.privacy_encryption_key:
    settings.privacy_encryption_key = hashlib.sha256(
        settings.jwt_secret.encode("utf-8")
    ).hexdigest()
    warnings.warn(
        "PRIVACY_ENCRYPTION_KEY is not configured. A deterministic key has been "
        "derived from JWT_SECRET for this process only. Set PRIVACY_ENCRYPTION_KEY "
        "in your .env file for persistent encryption across restarts.",
        RuntimeWarning,
        stacklevel=2,
    )
