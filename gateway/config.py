from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Gateway settings. Phase 0: a single Groq backend, read from the environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str
    groq_base_url: str = "https://api.groq.com/openai/v1"
    request_timeout_s: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
