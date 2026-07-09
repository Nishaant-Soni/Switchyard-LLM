from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Populate os.environ from .env so the registry can resolve per-provider api_key_env names.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    providers_config: str = "config/providers.yaml"
    models_config: str = "config/models.yaml"
    tenants_config: str = "config/tenants.yaml"
    pricing_config: str = "config/pricing.yaml"
    request_timeout_s: float = 60.0

    redis_url: str = "redis://localhost:6379/0"
    rate_limit_window_s: int = 60

    # Semantic cache (Phase 4). Enabled by default; the MiniLM model loads lazily on the first
    # cache lookup (first request downloads ~80MB). Set cache_enabled=false for routing-only.
    cache_enabled: bool = True
    # Tuned via the Phase 7 sweep: 0.85 was too loose (~45% precision on MiniLM); 0.90 gives
    # ~91-97% precision at ~89% recall. See docs/iteration_18.md.
    cache_similarity_threshold: float = 0.90
    cache_ttl_s: float = 3600.0
    cache_max_entries: int = 10_000
    cache_per_tenant: bool = False  # shared across tenants by default (max hit rate)


@lru_cache
def get_settings() -> Settings:
    return Settings()
