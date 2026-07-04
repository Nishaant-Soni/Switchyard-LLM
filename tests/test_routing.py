import random

import pytest

from gateway.observability.cost import PriceBook
from gateway.routing.policies import Target, cost_aware, latency_aware, priority, weighted
from gateway.routing.router import Router
from gateway.routing.signals import RoutingSignals


def _signals(prices=None):
    return RoutingSignals(PriceBook(prices or {}))


def _targets():
    return [
        Target("groq", "a", weight=2.0),
        Target("ollama", "b", weight=1.0),
    ]


def test_priority_preserves_order():
    ts = _targets()
    assert priority(ts, random.Random(0)) == ts


def test_weighted_returns_full_permutation():
    ts = _targets()
    out = weighted(ts, random.Random(0))
    assert sorted(out, key=lambda t: t.model) == sorted(ts, key=lambda t: t.model)


def test_weighted_distribution_matches_weights():
    ts = _targets()  # 2:1 weight
    rng = random.Random(42)
    picks = [weighted(ts, rng)[0].provider for _ in range(3000)]
    groq_share = picks.count("groq") / len(picks)
    assert 0.6 < groq_share < 0.73  # ~0.667


def test_router_resolves_alias_from_config():
    router = Router.from_config("config/models.yaml", rng=random.Random(0))
    targets = router.resolve("fast")
    assert targets[0].provider == "groq"
    assert targets[0].model == "llama-3.3-70b-versatile"


def test_router_unknown_alias_raises():
    router = Router.from_config("config/models.yaml")
    with pytest.raises(KeyError):
        router.resolve("does-not-exist")


def test_router_lists_configured_aliases():
    router = Router.from_config("config/models.yaml")
    assert {"fast", "smart", "cheap", "balanced"} <= set(router.aliases())


# --- latency-aware / cost-aware policies (Phase 6 Group 2) --------------------------------
def test_latency_aware_orders_by_measured_latency():
    ts = [Target("groq", "a"), Target("ollama", "b")]
    sig = _signals()
    sig.record_latency("groq", 0.5)  # 500 ms
    sig.record_latency("ollama", 0.1)  # 100 ms
    out = latency_aware(ts, random.Random(0), sig)
    assert [t.provider for t in out] == ["ollama", "groq"]  # fastest first


def test_latency_aware_cold_start_keeps_config_order():
    ts = [Target("groq", "a"), Target("ollama", "b")]
    assert latency_aware(ts, random.Random(0), _signals()) == ts  # no samples -> config order


def test_latency_aware_sampled_before_unsampled():
    ts = [Target("groq", "a"), Target("ollama", "b")]
    sig = _signals()
    sig.record_latency("ollama", 0.3)  # only ollama has a sample
    out = latency_aware(ts, random.Random(0), sig)
    assert [t.provider for t in out] == ["ollama", "groq"]  # sampled first; cold groq appended


def test_latency_aware_none_signals_falls_back():
    ts = [Target("groq", "a"), Target("ollama", "b")]
    assert latency_aware(ts, random.Random(0), None) == ts


def test_cost_aware_orders_cheapest_first():
    ts = [Target("groq", "expensive"), Target("ollama", "free")]
    sig = _signals(
        {"expensive": {"input": 1.0, "output": 1.0}, "free": {"input": 0.0, "output": 0.0}}
    )
    out = cost_aware(ts, random.Random(0), sig)
    assert [t.provider for t in out] == ["ollama", "groq"]  # free/local first


def test_router_applies_latency_aware_policy_from_signals():
    ts = [Target("groq", "a"), Target("ollama", "b")]
    sig = _signals()
    sig.record_latency("groq", 0.05)  # 50 ms
    sig.record_latency("ollama", 0.4)  # 400 ms
    router = Router({"fast": ("latency-aware", ts)}, signals=sig)
    assert router.resolve("fast")[0].provider == "groq"  # measured-fastest routed first


def test_ewma_smooths_successive_latencies():
    sig = _signals()
    sig.record_latency("groq", 0.1)  # seeds at 100 ms
    sig.record_latency("groq", 0.2)  # 0.3*200 + 0.7*100 = 130 ms (alpha=0.3)
    assert abs(sig.latency_ms("groq") - 130.0) < 1e-9


def test_config_fast_latency_aware_shifts_to_faster_provider():
    # `fast` ships as latency-aware: config order is Groq-first, but once measured the faster
    # backend is promoted — a genuine reorder, proving the shipped policy is live (cold start is
    # covered by test_router_resolves_alias_from_config, which resolves Groq-first with no signals).
    sig = _signals()
    sig.record_latency("groq", 0.5)  # 500 ms
    sig.record_latency("ollama", 0.05)  # 50 ms
    router = Router.from_config("config/models.yaml", signals=sig)
    assert router.resolve("fast")[0].provider == "ollama"


def test_config_cheap_cost_aware_prefers_free_model():
    # `cheap` ships as cost-aware, reading the real config/pricing.yaml: the free OpenRouter model
    # sorts ahead of the paid Groq fallback.
    sig = RoutingSignals(PriceBook.from_config("config/pricing.yaml"))
    router = Router.from_config("config/models.yaml", signals=sig)
    assert [t.provider for t in router.resolve("cheap")] == ["openrouter", "groq"]
