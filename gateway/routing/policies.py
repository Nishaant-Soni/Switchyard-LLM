"""Routing policies. A policy maps an alias's targets to an *ordered* list — targets[0]
is the chosen backend; the rest are the fallback order that Phase 2 will consume.

Keeping every policy return an ordered list (not just a single pick) means the resilience
layer wraps the same interface with no rework."""

import random
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    provider: str
    model: str
    weight: float = 1.0


def priority(targets: list[Target], rng: random.Random) -> list[Target]:
    """Failover order = configured order."""
    return list(targets)


def weighted(targets: list[Target], rng: random.Random) -> list[Target]:
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


Policy = Callable[[list[Target], random.Random], list[Target]]
POLICIES: dict[str, Policy] = {"priority": priority, "weighted": weighted}
