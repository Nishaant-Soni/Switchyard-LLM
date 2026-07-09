# Switchyard — LLM Gateway / Inference Router

A provider-agnostic **LLM gateway**: a reverse proxy that exposes one stable, **OpenAI-compatible**
API and routes each request across a fleet of heterogeneous providers. Putting a gateway in the
middle turns failover, cost control, rate limiting, caching, and observability into **centralized,
swappable concerns** instead of per-client duplication.

Runs end-to-end at **zero cost** on a laptop using free provider tiers plus local components. Built
from first principles (the LiteLLM / Portkey / Helicone problem space) to demonstrate the design and
its tradeoffs — see [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## Features

- **One endpoint, many backends.** `POST /v1/chat/completions` — existing OpenAI SDK clients work by
  changing only `base_url`. Send a logical **alias** (`fast`, `smart`, `cheap`) as the model.
- **Policy-based routing.** An alias maps to an ordered list of `(provider, model)` targets under a
  runtime-swappable policy: `priority`, `weighted`, `latency-aware`, `cost-aware`.
- **Resilience.** Per-provider circuit breaker (closed → open → half-open), jittered-backoff retry,
  and **cross-provider fallback** so a dead backend never reaches the client as an error — including
  fallback *up to first byte* on streaming responses.
- **Semantic cache.** Near-duplicate prompts are served from an in-process cache (local embeddings +
  FAISS ANN search), cutting cost and tail latency at a measured hit rate.
- **Per-tenant rate limiting.** Two-dimensional — **request rate and token rate** — in Redis, with a
  `tiktoken` pre-call estimate reconciled against the provider's real `usage`.
- **SSE streaming** end-to-end with natural backpressure.
- **Observability.** Prometheus metrics (latency percentiles, cache hit rate, per-provider failures,
  per-tenant counterfactual cost) in Grafana — and fed back into routing.

## The fleet (free-tier / local providers)

All four expose an OpenAI-compatible `chat/completions` surface, keeping the adapter layer thin.

| Provider | Role | Notes |
|---|---|---|
| **Groq** | Fast path (low latency) | LPU-served open models; high tokens/sec. Request- and token-capped/day. |
| **Google Gemini** | Quality / multimodal | Flash-class, 1M context, vision. Has a streaming-usage quirk (normalized in the adapter). |
| **OpenRouter** | Breadth path | One key, many `:free` models; wide fallback pool. |
| **Ollama (local)** | Offline fallback | Runs on the laptop; no network, no quota — guarantees the fallback chain always terminates. |

> Free-tier limits change frequently and vary by account/region. The gateway treats limits as
> **config, not hardcoded constants** — verify current numbers in each provider's dashboard.

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
   OpenAI SDK client ──▶ │                 LLM GATEWAY                    │
   (base_url=gateway)    │                                                │
                         │  1. Auth / tenant resolution (API key)         │
                         │  2. Alias validation (router.resolve)          │
                         │  3. Rate limit check (Redis: req + tokens)     │
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

**Control plane vs. data plane:** routing policies and provider/model config live in YAML loaded at
startup; live signals (per-provider EWMA latency) computed on the request path feed the
latency-/cost-aware policies — so observability is an *input* to routing, not a bolt-on. Full
lifecycle, module map, and design-decision rationale: [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## Quickstart

```bash
# 1. Provider key(s) — Groq is enough to start; the others are optional.
cp .env.example .env        # then set GROQ_API_KEY=gsk_...  (https://console.groq.com/keys)

# 2a. Run locally
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn gateway.main:app --port 8000

# 2b. …or bring up the whole stack (gateway + Redis + Prometheus + Grafana + Ollama)
docker compose up --build
```

Call it with any OpenAI SDK client — send a logical **alias** as the model:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")  # any key works until tenants are configured
print(
    client.chat.completions.create(
        model="fast",  # alias, not a real model name → resolved by the routing policy
        messages=[{"role": "user", "content": "Hello from the gateway"}],
    ).choices[0].message.content
)
```

Or with `curl` (streaming):

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "fast", "stream": true, "messages": [{"role": "user", "content": "hi"}]}'
```

**Endpoints:** `POST /v1/chat/completions` (streaming + non-streaming), `GET /healthz` (breaker
snapshot), `GET /metrics` (Prometheus exposition). Send `"stream": true` for token-by-token SSE.
Unknown aliases return `400`.

---

## Configuration

Four YAML files under `config/` drive everything (config, not code):

- **`providers.yaml`** — each provider's `base_url`, `auth`, `api_key_env`. A provider whose key is
  unset is skipped at startup. `base_url` supports `${VAR:-default}` env substitution (so one file
  serves both local and Docker — e.g. Ollama's URL).
- **`models.yaml`** — each alias → an ordered list of `(provider, model)` targets + a routing `policy`
  (`priority`, `weighted`, `latency-aware`, `cost-aware`).
- **`tenants.yaml`** — API key → tenant with per-minute request/token limits. Empty ⇒ auth + rate
  limiting are **off** (no Redis needed).
- **`pricing.yaml`** — per-model list prices (USD / 1M tokens) for counterfactual cost attribution.

Shipped aliases: `fast` (Groq→Ollama, **latency-aware**), `smart` (Gemini→Groq, priority), `cheap`
(OpenRouter→Groq, **cost-aware**), `balanced` (weighted 2:1 across two Groq models).

### Routing

- `priority` — try targets in listed order (static failover order).
- `weighted` — split traffic across targets by `weight`.
- `latency-aware` — order by live per-provider EWMA latency the executor records after each call
  (fastest measured first); unmeasured providers keep configured order (never chosen over a
  measured-fast one), so it behaves like `priority` until samples arrive.
- `cost-aware` — order by per-token list price (free/local first).

The router reads the live signal **in-process** (not by scraping its own `/metrics`), so routing
reacts within one request rather than one scrape interval.

### Resilience

The router's ordered targets execute through per-provider **circuit breakers** with jittered-backoff
retry and cross-provider fallback:

- `429` / `5xx` / timeouts / transport errors → advance to the next target. Other `4xx` (client
  errors) surface immediately.
- Each breaker trips `closed → open` on a failure-rate threshold over a rolling window, probes
  `half-open` after a reset timeout, and closes again on a successful probe.
- The served target is reported in `x-switchyard-provider` / `x-switchyard-model`; live breaker states
  are at `GET /healthz`. A primary pointed at a dead URL yields a normal `200` served by the next
  provider — no client-visible error.

### Rate limiting (per tenant)

```yaml
# config/tenants.yaml
tenants:
  sk-free-demo:
    tenant: free
    requests_per_min: 20
    tokens_per_min: 2000
```

- **Disabled by default:** with no tenants, auth + rate limiting are off and any key works. Add a
  tenant to enforce; an unknown key then → `401`.
- **Two-dimensional token bucket in Redis** (atomic Lua): admitted only if both the request-rate and
  token-rate buckets have capacity. Token limits usually bind first (LLM cost is token-denominated).
- Over-budget → `429` + `Retry-After`. Counters live in Redis, surviving restarts / shared across
  instances.
- Admission charges a **`tiktoken`** estimate, then **reconciles** the token bucket against the real
  `usage.total_tokens`. Accounting rule on non-success paths: *charge a request slot iff a real
  upstream attempt was made* — a bad alias / blocked `429` charges nothing; a genuine upstream failure
  refunds the tokens but keeps the request slot.

### Semantic cache

- **Flow:** embed the prompt locally (`sentence-transformers` MiniLM-L6-v2, CPU, 384-dim, L2-norm) →
  FAISS `IndexFlatIP` search within the request's **scope** → cosine ≥ threshold serves the cached
  response, else miss → upstream → write. Embedded **once per request, off the event loop**, and the
  vector is reused for read and write (a miss never embeds twice).
- **Scope** = `(alias, output-affecting params: temperature/top_p/max_tokens/stop/n/seed)`, so a
  response is never shared across params that change the answer. Shared across tenants by default
  (`cache_per_tenant=true` isolates).
- A **cache hit refunds the token estimate** (zero upstream tokens) while keeping the request slot, so
  caching visibly cuts a tenant's token cost.
- **Tunables** (env/`config.py`): `cache_enabled`, `cache_similarity_threshold` (default **0.90**,
  tuned via the benchmark sweep), `cache_ttl_s`, `cache_max_entries` (LRU). Streaming bypasses the
  cache. Served source is reported via `x-switchyard-cache: hit|miss`.
- The MiniLM model loads lazily on the first lookup (first request downloads ~80 MB); set
  `CACHE_ENABLED=false` for routing-only.

### Streaming (SSE)

Send `"stream": true` for a `text/event-stream` response (backpressure is natural — a slow client
throttles the upstream read):

- The gateway **parses, normalizes, and re-emits** every chunk (spec-conformant regardless of backend
  quirks — e.g. Gemini emits `usage` in every chunk; the gateway keeps it only on the last).
- **Post-stream accounting:** `stream_options.include_usage` is requested upstream; token
  reconciliation runs when the stream finishes (refund on a mid-stream error/disconnect).
- A failed *open* returns a proper error status (the first chunk is peeked before committing to `200`)
  and **falls back cross-provider up to first byte**; once the first byte is sent the choice is
  committed.

### Observability

`GET /metrics` serves a live Prometheus exposition. Instrumentation is additive (never changes request
behavior) and label cardinality is bounded (aliases/providers come from config; a bad alias is `400`
before any metric, so it never becomes a label). Series:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `switchyard_requests_total` | counter | `alias, provider, outcome, stream` | admitted routing attempts; a cache hit is `provider="cache"` |
| `switchyard_request_latency_seconds` | histogram | `provider, stream` | upstream latency; time-to-first-byte for streams |
| `switchyard_cache_events_total` | counter | `event` | `hit` / `miss` / `bypass` |
| `switchyard_errors_total` | counter | `type` | error responses by type |
| `switchyard_provider_failures_total` | counter | `provider, reason` | per-provider attempt failures recovered via fallback / breaker skips |
| `switchyard_tokens_total` | counter | `provider, direction` | prompt / completion tokens |
| `switchyard_cost_usd_total` | counter | `tenant, provider` | counterfactual cost (`usage × pricing.yaml`, free/local = 0) |

`docker compose up` brings up Prometheus (`:9090`) + Grafana (`:3000`, anonymous admin) alongside the
gateway; Grafana auto-provisions the datasource and a 7-panel dashboard
([`dashboards/grafana.json`](./dashboards/grafana.json)).

---

## Benchmark

A reproducible harness (`bench/`) runs a synthetic workload with a tunable semantic-duplicate rate `d`
through the **real** MiniLM + FAISS cache. Cost is **counterfactual** (`usage × pricing.yaml`, since
providers are free); miss (upstream) latency is a seeded lognormal model, so results reproduce (hit
rate and cost exactly, p95 within ~0.2%).

```bash
python -m bench.run_benchmark            # baseline vs cache-on + threshold sweep, at d = 20% and 40%
```

Results (seed 42, n = 400, `all-MiniLM-L6-v2`, threshold **0.90**, miss latency lognormal(median 0.6 s, σ 0.5)):

| duplicate rate d | hit rate | cost ↓ | p50 ↓ | p95 ↓ | p99 ↓ |
|---|---|---|---|---|---|
| 20% | 17.5% | 18.2% | 9.4% | 3.7% | 3.7% |
| 40% | 35.2% | 34.7% | 32.0% | 15.7% | 13.0% |

The similarity-threshold sweep (the precision/recall tradeoff) is why the default is **0.90** — at
0.85 MiniLM collides distinct prompts and precision drops to ~45%; 0.90 gives ~91–97% precision at
~89% recall.

### Resilience demo

```bash
python -m bench.demo_resilience          # self-contained; no keys, no Docker
```

Launches a mock upstream (two backends) + a real gateway (temp config; real `config/` untouched),
drives traffic in two phases, and takes the primary down mid-run: Phase 1 is served by the primary;
in Phase 2 the primary returns `503`, traffic **transparently falls back** to the secondary, the
primary's circuit trips **open**, and **every client request still returns 200**. It prints the
per-phase provider, the `/healthz` breaker snapshot, and the `/metrics` counters — the same signals
that animate Grafana's *Provider failures* and *Request rate by provider* panels.

---

## Development & tests

```bash
.venv/bin/pip install -r requirements-dev.txt
ruff check .            # lint
ruff format --check .   # formatting
pytest -q               # hermetic unit tests + live per-provider conformance (skips without keys)
```

Conformance tests make real upstream calls and **skip** providers that are unconfigured, unreachable,
or returning a transient `429`/`5xx` — so the suite is green with just a Groq key (and fully hermetic
with none). Run everything from the project root. CI (GitHub Actions) runs lint + format + tests on
Python 3.11 and 3.12.

### Docker stack

```bash
docker compose up --build                        # gateway :8000, Redis :6379, Prometheus :9090, Grafana :3000, Ollama :11434
docker compose exec ollama ollama pull llama3.2  # one-time: activate the offline fallback model
```

---

## Tech stack

- **Service:** Python 3.11, FastAPI, `httpx.AsyncClient`
- **State:** Redis (rate-limit counters)
- **Cache:** `sentence-transformers` (`all-MiniLM-L6-v2`, CPU) + FAISS (in-process)
- **Token estimate:** `tiktoken` (reconciled against provider `usage`)
- **Observability:** `prometheus_client`, Prometheus, Grafana
- **Local model:** Ollama
- **Packaging:** Docker + `docker compose`

## Documentation

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — request lifecycle, module map, control/data-plane split, adapter pattern, provider quirks, and design decisions.
- [`PRD.md`](./PRD.md) — problem, goals/non-goals, component specs, risks.

## Non-goals

Not running model inference (no GPU serving), not a production/SLA/multi-region service, not a UI
product (Grafana is the only "frontend"), and not every OpenAI endpoint — core scope is
`chat/completions` (streaming + non-streaming) plus `/healthz` and `/metrics`.
