# Architecture

How Switchyard is put together and *why*. The README is the "what it is / how to run it"; this is the
"how it works and the tradeoffs behind it."

---

## 1. Overview

Switchyard is a reverse proxy that exposes one **OpenAI-compatible** surface
(`POST /v1/chat/completions`) and routes each request to one of several heterogeneous providers
(Groq, Google Gemini, OpenRouter, local Ollama). A client sends a **logical alias** (`fast`, `smart`,
`cheap`, `balanced`) as the `model`; the gateway resolves that alias to an ordered list of real
`(provider, model)` targets and serves the request through a pipeline of cross-cutting concerns —
auth, rate limiting, caching, resilience, observability — that would otherwise be duplicated in every
client.

```
                         ┌──────────────────────────────────────────────┐
   OpenAI SDK client ──▶ │                 LLM GATEWAY                    │
   (base_url=gateway)    │                                                │
                         │  1. Auth / tenant resolution (API key)         │
                         │  2. Alias validation (router.resolve)          │
                         │  3. Rate limit admission (Redis: req + tokens) │
                         │  4. Semantic cache lookup (embed → ANN)        │──hit──▶ cached response
                         │  5. Resilience: breaker + retry + fallback     │
                         │  6. Provider adapter → upstream call (httpx)   │──────▶ Groq / Gemini /
                         │  7. (stream) SSE passthrough w/ backpressure   │        OpenRouter / Ollama
                         │  8. Token reconcile + cost attribution         │◀──────
                         │  9. Cache write + metrics emit (Prometheus)    │
                         └──────────────────────────────────────────────┘
                                    │                         │
                              ┌─────▼─────┐            ┌───────▼────────┐
                              │   Redis   │            │ Prometheus +   │
                              │  (limits) │            │ Grafana        │
                              └───────────┘            └────────────────┘
```

## 2. Module layout

| Package | Responsibility |
|---|---|
| `gateway/main.py` | FastAPI app + the request pipeline (`chat_completions`, `_streaming_completion`); lifespan wiring of all components into `app.state`. |
| `gateway/schemas.py` | Canonical OpenAI Chat Completions models (`extra="allow"` so unknown fields pass through). |
| `gateway/config.py` | `Settings` (env/`.env`-driven config paths + tunables). |
| `gateway/providers/` | `base.py` (adapter ABC + `UpstreamError`), `openai_compat.py`, `gemini.py`, `registry.py` (build adapters from `providers.yaml`, env-expandable `base_url`). |
| `gateway/routing/` | `policies.py` (priority / weighted / latency-aware / cost-aware), `router.py` (alias → ordered targets), `signals.py` (in-process live latency/cost signals). |
| `gateway/resilience/` | `circuit_breaker.py` (per-provider breaker + registry), `retry.py` (`ResilientExecutor`: breaker-guarded, jittered-backoff, cross-provider fallback; `execute` + `execute_stream`). |
| `gateway/ratelimit/` | `limiter.py` (2-D Redis token bucket via atomic Lua), `estimate.py` (tiktoken pre-call estimate). |
| `gateway/tenancy/` | `auth.py` (API key → tenant from `tenants.yaml`). |
| `gateway/cache/` | `embedder.py` (MiniLM), `semantic_cache.py` (per-scope FAISS index, TTL, LRU). |
| `gateway/observability/` | `metrics.py` (Prometheus registry + recorders), `cost.py` (`PriceBook`, counterfactual cost). |
| `gateway/streaming/` | `sse.py` (re-emit chunks + `[DONE]`, tap usage, run post-stream accounting). |
| `bench/` | Benchmark harness (`workload.py`, `run_benchmark.py`) + resilience demo (`mock_upstream.py`, `demo_resilience.py`). |

## 3. Request lifecycle (non-streaming)

`gateway/main.py · chat_completions`:

1. **Auth** — if tenants are configured, resolve the bearer key → `Tenant`; unknown/missing → `401`.
2. **Alias validation** — `Router.resolve(model)` → ordered targets. Unknown alias → `400`. This runs
   **before** rate-limit admission on purpose: resolving an alias is cheap and side-effect-free, so a
   bad alias costs no quota (a clean `400`, not charge-then-refund).
3. **Rate-limit admission** — `estimate_tokens` (tiktoken) then `RateLimiter.check`; over budget →
   `429 + Retry-After`. This charges the estimate against both buckets.
4. **Cache read** — embed once (off the event loop), FAISS lookup within the request's scope; a hit
   refunds the token estimate and returns immediately (`x-switchyard-cache: hit`).
5. **Resilient execution** — `ResilientExecutor.execute` walks the targets through breakers + retry +
   fallback, returns the served response and target.
6. **Reconcile + attribute** — adjust the token bucket to the real `usage.total_tokens`; record
   tokens + counterfactual cost.
7. **Cache write** (on a miss) + response headers (`x-switchyard-provider/model/cache`).

The **streaming** path (`_streaming_completion`) branches after admission (step 3), bypasses the
cache, and opens the stream with fallback *up to first byte*.

## 4. Control plane vs. data plane

- **Control plane** = configuration and routing decisions: `providers.yaml`, `models.yaml`,
  `tenants.yaml`, `pricing.yaml`, loaded at startup into the router/registry/limiter/pricebook.
- **Data plane** = the per-request path above.
- The two meet at the **routing signal registry** (`routing/signals.py`): the data plane (executor)
  *writes* per-provider latency after each call; the control plane (router) *reads* it at resolve time
  for latency-aware routing. This is the observability→routing control loop (§8).

## 5. Provider adapters (Strategy pattern)

`ProviderAdapter` (ABC) declares `chat_completion` and `stream_chat_completion`. `OpenAICompatAdapter`
implements both against any OpenAI-compatible `base_url`; `GeminiAdapter` subclasses it to normalize
Gemini's quirks. The registry builds one adapter per configured provider from `providers.yaml`,
skipping providers whose API key is unset. New backend that speaks OpenAI? Add a YAML block — no code.
Backend with quirks? Subclass the adapter. This keeps the pipeline provider-agnostic.

## 6. Resilience

`ResilientExecutor` iterates the router's ordered targets, each guarded by a per-provider
`CircuitBreaker`:

- **Retryable** failures (`429`, `5xx`, timeout, transport error) → record a breaker failure and
  advance to the **next** target (never retry the same struggling backend). Non-retryable `4xx` →
  the provider is healthy and the request is the problem, so surface it immediately.
- The breaker trips `closed → open` on a failure rate over a rolling window (default `min_calls=5`,
  `failure_threshold=0.5`, `window=20`), probes `half-open` after a reset timeout, and closes on a
  successful probe. An open breaker is skipped without an attempt.
- On streaming, fallback is possible only **up to the first byte** — once bytes are on the wire the
  choice is committed and a later mid-stream error surfaces (it can't be retried).

## 7. Rate limiting

Two buckets per tenant — **request rate** and **token rate** — refilled and checked in one atomic
Redis Lua script, so a request is admitted only if *both* have capacity (LLM cost is
token-denominated, so the token limit usually binds first). Admission charges a tiktoken estimate;
after the call the token bucket is **reconciled** to the provider's real `usage`. The governing rule
on non-success paths: *charge a request slot iff a real upstream attempt was made* — a bad alias or a
blocked `429` charges nothing; a genuine upstream failure refunds the tokens but keeps the request
slot (so retrying a dead route is still throttled).

## 8. Observability and the control loop

`prometheus_client` in a dedicated registry exposes latency histograms, request/cache/error counters,
per-provider failure counts, token throughput, and per-tenant **counterfactual** cost (`usage ×
pricing.yaml`). Grafana auto-provisions a dashboard over these.

The routing control loop: the executor records each served provider's latency into an **in-process**
EWMA registry; the latency-aware policy reads it directly at resolve time. The router does **not**
scrape its own `/metrics` — that would be circular and lag by a scrape interval. Prometheus/Grafana
reflect the same underlying numbers for humans.

## 9. Provider quirks (known gotchas)

- **Gemini streaming usage:** Gemini's OpenAI-compatible streaming emits `usage` in *every* chunk
  (the spec puts it only in the last). `GeminiAdapter` keeps it only on the final chunk so token
  accounting isn't inflated. (Groq was also observed emitting usage in >1 chunk → "take the last".)
- **Free-tier `429`s are normal:** handled by fallback, not treated as a hard error; live conformance
  tests skip on `429`/`5xx`.
- **`tiktoken` is an approximation** for non-OpenAI tokenizers — used only for the pre-call estimate,
  then reconciled against the provider's real `usage`.
- **faiss + torch OpenMP clash on macOS:** `faiss-cpu` and `torch` each bundle an OpenMP runtime and
  can abort the process; `gateway/cache/__init__.py` sets `OMP_NUM_THREADS=1` +
  `KMP_DUPLICATE_LIB_OK=TRUE` before importing either.

## Design decisions & tradeoffs

One paragraph per defensible decision — the interview talking points.

- **Alias resolution before rate-limit admission.** Resolving an alias is cheap and side-effect-free,
  so validating it first means an unknown alias is a clean `400` that charges no quota, instead of a
  charge-then-refund. It also keeps the `alias` metric label bounded (only configured aliases ever
  reach a counter), which matters because the label is client-supplied.

- **Counterfactual cost, not real spend.** The gateway runs on free tiers, so "cost" is
  `usage × published list prices` — what the same traffic *would* cost on paid tiers. This makes the
  cache's cost-savings number concrete and honest without spending money; the caveat (list prices,
  free tiers) is stated wherever the number appears.

- **In-process routing signals, not self-scraping Prometheus.** Latency-aware routing reads a live
  EWMA the executor writes in-process. Scraping the gateway's own `/metrics` to route would be
  circular and lag by a whole scrape interval (seconds ≫ a request). Prometheus is for humans;
  routing consumes the same signal directly and instantly.

- **Cross-provider fallback only "up to first byte" on streams.** Once the first SSE chunk has been
  sent, the HTTP response is committed (status + headers are on the wire), so a mid-stream failure
  can't be retried against another provider. The executor peeks the first chunk so a *failed open*
  still falls back with a proper error status, but a committed stream surfaces later errors as-is.

- **Cache similarity threshold = precision/recall tradeoff.** Too loose serves wrong answers (a
  paraphrase of a *different* intent collides); too tight collapses the hit rate. The benchmark sweep
  measured this on MiniLM and tuned the default from 0.85 (~45% precision) to **0.90** (~91–97%
  precision at ~89% recall).

- **Retry the next provider, never the same one.** Retrying a backend that just failed (429/5xx) only
  amplifies its load; advancing to a different provider is both faster to recover and kinder to the
  struggling one. The breaker then stops sending traffic to it entirely until a reset probe succeeds.

- **Token reconciliation over trusting the estimate.** tiktoken can't exactly tokenize non-OpenAI
  models (and misses reasoning tokens), so admission charges an estimate and the bucket is corrected
  to the provider's real `usage` afterward — accurate accounting without a per-provider tokenizer.

- **Config over code for the fleet.** Providers, aliases, policies, tenants, and prices are YAML
  loaded at startup; adding a backend or changing a routing policy needs no code change. The adapter
  Strategy layer keeps the pipeline provider-agnostic.
