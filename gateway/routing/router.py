"""Alias -> ordered targets, per the configured policy. The control-plane half of the gateway."""

import random

import yaml

from gateway.routing.policies import POLICIES, Target


class Router:
    def __init__(
        self,
        aliases: dict[str, tuple[str, list[Target]]],
        rng: random.Random | None = None,
    ):
        self._aliases = aliases
        self._rng = rng or random.Random()

    @classmethod
    def from_config(cls, path: str, rng: random.Random | None = None) -> "Router":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        aliases: dict[str, tuple[str, list[Target]]] = {}
        for name, acfg in (cfg.get("aliases") or {}).items():
            policy = acfg.get("policy", "priority")
            if policy not in POLICIES:
                raise ValueError(f"alias {name!r}: unknown policy {policy!r}")
            targets = [
                Target(
                    provider=t["provider"],
                    model=t["model"],
                    weight=float(t.get("weight", 1.0)),
                )
                for t in acfg["targets"]
            ]
            if not targets:
                raise ValueError(f"alias {name!r} has no targets")
            aliases[name] = (policy, targets)
        return cls(aliases, rng=rng)

    def resolve(self, alias: str) -> list[Target]:
        """Return the policy-ordered targets for an alias. Raises KeyError if unknown."""
        entry = self._aliases.get(alias)
        if entry is None:
            raise KeyError(alias)
        policy, targets = entry
        return POLICIES[policy](targets, self._rng)

    def aliases(self) -> list[str]:
        return list(self._aliases)
