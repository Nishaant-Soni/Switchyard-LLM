"""Routing policies. A policy maps an alias's targets to an *ordered* list — targets[0]
is the chosen backend; the rest are the fallback order that Phase 2 consumes.

Keeping every policy return an ordered list (not just a single pick) means the resilience
layer wraps the same interface with no rework. Every policy takes the same 3 args
(targets, rng, signals) so they're interchangeable; static policies just ignore `signals`, and the
live-aware ones fall back to the configured order when no signal is available yet (cold start)."""

import random
from collections.abc import Callable
from dataclasses import dataclass

from gateway.routing.signals import RoutingSignals


@dataclass(frozen=True)
class Target:
    provider: str
    model: str
    weight: float = 1.0


def priority(
    targets: list[Target], rng: random.Random, signals: RoutingSignals | None = None
) -> list[Target]:
    """Failover order = configured order."""
    return list(targets)


def weighted(
    targets: list[Target], rng: random.Random, signals: RoutingSignals | None = None
) -> list[Target]:
    """Weighted-random ordering without replacement; targets[0] is the weighted pick."""
    remaining = list(targets)
    ordered: list[Target] = []
    while remaining:
        total = sum(t.weight for t in remaining)
        r = rng.uniform(0, total)
        cumulative = 0.0
        for t in remaining:
            cumulative += t.weight
            if r <= cumulative:
                ordered.append(t)
                remaining.remove(t)
                break
    return ordered


def latency_aware(
    targets: list[Target], rng: random.Random, signals: RoutingSignals | None = None
) -> list[Target]:
    """Order by ascending smoothed (EWMA) latency. Providers with samples come first (fastest
    first); unsampled providers keep their configured order, appended after — so a cold provider is
    still a fallback but is never chosen over a measured-fast one. Falls back to the configured
    order entirely when no signals are wired."""
    if signals is None:
        return list(targets)

    def key(item: tuple[int, Target]):
        idx, t = item
        lat = signals.latency_ms(t.provider)
        # (sampled-flag, latency, config-index): sampled providers sort ahead by latency; unsampled
        # keep their relative order via the index. Stable and total.
        return (0, lat, idx) if lat is not None else (1, 0.0, idx)

    return [t for _, t in sorted(enumerate(targets), key=key)]


def cost_aware(
    targets: list[Target], rng: random.Random, signals: RoutingSignals | None = None
) -> list[Target]:
    """Order by ascending per-token list price (free/local providers first). Ties keep configured
    order. Falls back to the configured order when no price signal is wired."""
    if signals is None:
        return list(targets)
    return [
        t
        for _, t in sorted(
            enumerate(targets), key=lambda item: (signals.unit_cost(item[1].model), item[0])
        )
    ]


Policy = Callable[[list[Target], random.Random, RoutingSignals | None], list[Target]]
POLICIES: dict[str, Policy] = {
    "priority": priority,
    "weighted": weighted,
    "latency-aware": latency_aware,
    "cost-aware": cost_aware,
}
