import random

import pytest

from gateway.routing.policies import Target, priority, weighted
from gateway.routing.router import Router


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
