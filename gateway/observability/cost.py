"""Counterfactual cost attribution: usage × list prices from config/pricing.yaml.

The gateway runs on free/local tiers, so this estimates what the same traffic *would* cost at the
providers' paid list prices — the number behind per-tenant cost attribution (Phase 6). Models absent
from the price book (and local/free models priced at 0) cost nothing.
"""

import yaml


class PriceBook:
    """Per-model input/output list prices in USD per 1M tokens."""

    def __init__(self, prices: dict[str, dict[str, float]]):
        self._prices = prices

    @classmethod
    def from_config(cls, path: str) -> "PriceBook":
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            cfg = {}
        prices = {
            model: {
                "input": float(p.get("input", 0.0)),
                "output": float(p.get("output", 0.0)),
            }
            for model, p in (cfg.get("models") or {}).items()
        }
        return cls(prices)

    def cost_usd(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Counterfactual USD cost for one call. Unknown model -> 0.0 (free/local)."""
        p = self._prices.get(model)
        if p is None:
            return 0.0
        return (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1_000_000
