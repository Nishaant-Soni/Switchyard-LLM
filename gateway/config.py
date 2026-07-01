from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Populate os.environ from .env so the registry can resolve per-provider api_key_env names.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    providers_config: str = "config/providers.yaml"
    models_config: str = "config/models.yaml"
    request_timeout_s: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
