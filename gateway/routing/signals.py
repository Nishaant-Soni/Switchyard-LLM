"""In-process live signals that feed latency-/cost-aware routing (Phase 6 Group 2).

The resilient executor records the served provider's call latency after each success; routing reads
a smoothed (EWMA) latency per provider at resolve time to order targets. Cost is read from the price
book (static list prices) as a per-model ordering proxy.

This is the deliberate design choice behind cost-/latency-aware routing: the router consumes these
in-process signals **directly**, rather than scraping the gateway's own Prometheus endpoint — that
would be circular and lag by a whole scrape interval. Prometheus/Grafana reflect the same underlying
numbers, but for humans.
"""


class RoutingSignals:
    def __init__(self, pricebook, *, alpha: float = 0.3):
        # `pricebook` is a PriceBook; duck-typed here to avoid a routing -> observability import.
        self._pricebook = pricebook
        self._alpha = alpha
        self._latency_ms: dict[str, float] = {}

    def record_latency(self, provider: str, seconds: float) -> None:
        """Fold one observed call latency into the provider's EWMA (first sample seeds it)."""
        ms = seconds * 1000.0
        prev = self._latency_ms.get(provider)
        if prev is None:
            self._latency_ms[provider] = ms
        else:
            self._latency_ms[provider] = self._alpha * ms + (1 - self._alpha) * prev

    def latency_ms(self, provider: str) -> float | None:
        """Smoothed latency for a provider, or None if it has no samples yet (cold start)."""
        return self._latency_ms.get(provider)

    def unit_cost(self, model: str) -> float:
        """Per-token list-price proxy (1 input + 1 output token) for cost-aware ordering."""
        return self._pricebook.cost_usd(model, 1, 1)
