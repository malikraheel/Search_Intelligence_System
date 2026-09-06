"""Application settings (pydantic-settings). Every knob documented in .env.example."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Must be set before `litellm` is imported anywhere: otherwise litellm fetches its model
# cost map over the network at import time (slow, and hangs offline).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── application ──
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    database_url: str = "sqlite:///./search_intel.db"

    # ── AI gateway ──
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None

    gateway_planner_model: str = "openai/gpt-4o-mini"
    gateway_planner_fallbacks: str = "openai/gpt-4o"
    gateway_retrieval_model: str = "openai/gpt-4o-mini"
    gateway_retrieval_fallbacks: str = ""
    gateway_analyst_model: str = "openai/gpt-4o"
    gateway_analyst_fallbacks: str = "openai/gpt-4o-mini"
    gateway_reporter_model: str = "openai/gpt-4o-mini"
    gateway_reporter_fallbacks: str = ""
    gateway_timeout_s: float = 30.0
    gateway_max_retries: int = Field(default=2, ge=0)
    gateway_retry_base_delay_s: float = Field(default=0.5, ge=0)
    gateway_cache_enabled: bool = True
    gateway_cache_ttl_s: int = 600
    gateway_breaker_failure_threshold: int = Field(default=5, ge=1)
    gateway_breaker_reset_s: float = 30.0

    # ── DataForSEO ──
    dataforseo_mode: Literal["mock", "live", "chaos"] = "mock"
    dataforseo_login: str | None = None
    dataforseo_password: str | None = None
    dataforseo_base_url: str = "https://api.dataforseo.com"
    dataforseo_connect_timeout_s: float = 5.0
    dataforseo_read_timeout_s: float = 30.0
    dataforseo_max_attempts: int = Field(default=4, ge=1)
    dataforseo_retry_base_delay_s: float = Field(default=0.5, ge=0)
    dataforseo_retry_max_delay_s: float = Field(default=8.0, ge=0)
    dataforseo_breaker_failure_threshold: int = Field(default=5, ge=1)
    dataforseo_breaker_reset_s: float = 30.0
    dataforseo_chaos_script: str = "serp_organic:429,429,ok;llm_responses:500,500,500,500;keyword_volume:timeout,ok"

    # ── pipeline ──
    pipeline_max_concurrency: int = Field(default=4, ge=1)
    pipeline_recursion_limit: int = Field(default=60, ge=10)
    pipeline_default_location_code: int = 2840
    pipeline_default_language_code: str = "en"

    # ── derived helpers ──
    def gateway_fallbacks(self, role: str) -> list[str]:
        return _split_csv(getattr(self, f"gateway_{role}_fallbacks", ""))

    def gateway_model(self, role: str) -> str:
        return getattr(self, f"gateway_{role}_model")

    def provider_keys(self) -> dict[str, str | None]:
        return {"openai": self.openai_api_key, "anthropic": self.anthropic_api_key, "gemini": self.gemini_api_key}


@lru_cache
def get_settings() -> Settings:
    return Settings()
