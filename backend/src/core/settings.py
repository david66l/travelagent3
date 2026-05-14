import os
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

    # Database
    database_url: str = (
        "postgresql+asyncpg://travelagent:travelagent123@localhost:5432/travel_agent"
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # LLM
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096

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
    seed_cities: str = "北京,上海,广州,深圳,成都,杭州,西安,重庆,苏州,南京,厦门,青岛,大理,丽江,三亚,长沙,武汉,昆明,桂林,拉萨"

    @property
    def seed_cities_list(self) -> list[str]:
        return [c.strip() for c in self.seed_cities.split(",") if c.strip()]


settings = Settings()
