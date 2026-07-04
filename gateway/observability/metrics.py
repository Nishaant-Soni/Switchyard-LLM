"""Prometheus metrics for the gateway (Phase 6 Group 1).

All series live in a dedicated `CollectorRegistry` (not the process-global default) so the
exposition is self-contained and tests can read samples deterministically. Recording is side-effect
only and never alters request behavior — instrumentation is purely additive.

Label cardinality is deliberately bounded: `alias`/`provider` come from config (client-supplied bad
aliases collapse to `<unknown>`), `error` types are a fixed vocabulary, so no client input can blow
up the series count.

Note: `prometheus_client` appends `_total` to counter names automatically, so the counters are
declared without that suffix (the exposition + `get_sample_value` still use `..._total`).
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

# LLM latencies span sub-second to tens of seconds; bucket accordingly.
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)

REQUESTS = Counter(
    "switchyard_requests",
    "Admitted routing attempts (past auth + alias validation), by outcome.",
    ["alias", "provider", "outcome", "stream"],  # outcome: success | error | throttled
    registry=REGISTRY,
)
LATENCY = Histogram(
    "switchyard_request_latency_seconds",
    "Upstream call latency (time-to-first-byte for streams), by provider.",
    ["provider", "stream"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
CACHE_EVENTS = Counter(
    "switchyard_cache_events",
    "Semantic-cache outcomes.",
    ["event"],  # hit | miss | bypass
    registry=REGISTRY,
)
ERRORS = Counter(
    "switchyard_errors",
    "Error responses by type.",
    ["type"],
    registry=REGISTRY,
)
TOKENS = Counter(
    "switchyard_tokens",
    "Tokens processed, by provider and direction.",
    ["provider", "direction"],  # prompt | completion
    registry=REGISTRY,
)
COST = Counter(
    "switchyard_cost_usd",
    "Counterfactual cost (usage x list price) attributed per tenant and provider.",
    ["tenant", "provider"],
    registry=REGISTRY,
)


def _stream_label(stream: bool) -> str:
    return "true" if stream else "false"


def record_request(alias: str, provider: str, outcome: str, stream: bool) -> None:
    REQUESTS.labels(
        alias=alias, provider=provider, outcome=outcome, stream=_stream_label(stream)
    ).inc()


def observe_latency(provider: str, stream: bool, seconds: float) -> None:
    LATENCY.labels(provider=provider, stream=_stream_label(stream)).observe(seconds)


def record_cache(event: str) -> None:
    CACHE_EVENTS.labels(event=event).inc()


def record_error(error_type: str) -> None:
    ERRORS.labels(type=error_type).inc()


def record_usage(
    tenant: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
) -> None:
    """Record token throughput and counterfactual cost for one successful call."""
    if prompt_tokens:
        TOKENS.labels(provider=provider, direction="prompt").inc(prompt_tokens)
    if completion_tokens:
        TOKENS.labels(provider=provider, direction="completion").inc(completion_tokens)
    if cost_usd:
        COST.labels(tenant=tenant, provider=provider).inc(cost_usd)


def render() -> tuple[bytes, str]:
    """Return (exposition bytes, content-type) for the /metrics endpoint."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
