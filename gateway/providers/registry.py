"""Builds provider adapters from config/providers.yaml. Providers whose API key is unset
are skipped (logged), so the gateway runs with whatever subset of the fleet is configured."""

import logging
import os
import re

import httpx
import yaml

from gateway.providers.base import ProviderAdapter
from gateway.providers.gemini import GeminiAdapter
from gateway.providers.openai_compat import OpenAICompatAdapter

logger = logging.getLogger("gateway.registry")

# ${VAR} / ${VAR:-default} substitution in config strings — lets one providers.yaml serve both a
# local run (localhost defaults) and Docker Compose (service-name URLs via env), e.g. Ollama.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: str) -> str:
    return _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1)) or (m.group(2) or ""), value)


ADAPTER_TYPES: dict[str, type[OpenAICompatAdapter]] = {
    "openai_compat": OpenAICompatAdapter,
    "gemini": GeminiAdapter,
}


class ProviderRegistry:
    def __init__(self, adapters: dict[str, ProviderAdapter]):
        self._adapters = adapters

    @classmethod
    def from_config(cls, path: str, client: httpx.AsyncClient) -> "ProviderRegistry":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        adapters: dict[str, ProviderAdapter] = {}
        for name, pcfg in (cfg.get("providers") or {}).items():
            auth = pcfg.get("auth", "bearer")
            api_key = ""
            if auth != "none":
                env_var = pcfg.get("api_key_env")
                api_key = os.getenv(env_var, "") if env_var else ""
                if not api_key:
                    logger.warning("provider %r skipped: env %s not set", name, env_var)
                    continue
            adapter_cls = ADAPTER_TYPES[pcfg.get("type", "openai_compat")]
            base_url = _expand_env(pcfg["base_url"])
            adapters[name] = adapter_cls(
                name=name,
                base_url=base_url,
                api_key=api_key,
                client=client,
            )
            logger.info("provider %r ready -> %s", name, base_url)
        return cls(adapters)

    def get(self, name: str) -> ProviderAdapter | None:
        return self._adapters.get(name)

    def available(self) -> list[str]:
        return list(self._adapters)
