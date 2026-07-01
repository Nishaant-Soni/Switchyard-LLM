"""API key -> tenant resolution. Tenants + their per-minute limits come from config/tenants.yaml.

If no tenants are configured, the registry is *disabled* (auth + rate limiting off) so the gateway
stays usable with no Redis and no keys — matching the pre-Phase-3 behavior.
"""

from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class Tenant:
    id: str
    requests_per_min: int
    tokens_per_min: int


class TenantRegistry:
    def __init__(self, by_key: dict[str, Tenant]):
        self._by_key = by_key

    @classmethod
    def from_config(cls, path: str) -> "TenantRegistry":
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            cfg = {}
        by_key = {
            key: Tenant(
                id=t["tenant"],
                requests_per_min=int(t["requests_per_min"]),
                tokens_per_min=int(t["tokens_per_min"]),
            )
            for key, t in (cfg.get("tenants") or {}).items()
        }
        return cls(by_key)

    @property
    def enabled(self) -> bool:
        """Whether any tenants are configured (i.e. auth + rate limiting are active)."""
        return bool(self._by_key)

    def resolve(self, api_key: str | None) -> Tenant | None:
        """Return the Tenant for a key, or None if the key is unknown/missing."""
        if api_key is None:
            return None
        return self._by_key.get(api_key)


def extract_bearer(authorization: str | None) -> str | None:
    """Pull the key out of an `Authorization: Bearer <key>` header (tolerates a raw key)."""
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return authorization
