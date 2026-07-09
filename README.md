# Switchyard — LLM Gateway / Inference Router

> **Status:** Phases 0–7 complete — resilient multi-provider routing, per-tenant rate limiting, a **semantic cache**, **SSE streaming** with post-stream usage accounting *and* cross-provider fallback up to first byte, **Prometheus metrics + per-tenant counterfactual cost attribution**, a **Grafana dashboard**, **latency-/cost-aware routing** that consumes live in-process signals (the observability→routing control loop), and a **reproducible benchmark harness + scripted resilience demo** producing the headline cache numbers. Design is locked in [`PRD.md`](./PRD.md); code is being built phase by phase (see [Roadmap](#roadmap)).

A provider-agnostic **LLM gateway**: a reverse proxy that exposes a single, stable,
**OpenAI-compatible** API on the front and routes to many heterogeneous providers on the
back. Putting a gateway in the middle turns failover, cost control, rate limiting, caching,
and observability into **centralized, swappable concerns** instead of per-client duplication.

This is built from first principles (the LiteLLM / Portkey / Helicone problem space) to
demonstrate the *design and its tradeoffs* — not to ship a feature checklist. It runs
end-to-end at **zero cost** on a laptop using free provider tiers plus local components.

---

## What it does

- **One endpoint, many backends.** `POST /v1/chat/completions` — existing OpenAI SDK clients
  work by changing only `base_url`.
- **Policy-based routing.** A logical alias (`fast`, `smart`, `cheap`) maps to an ordered list
  of real providers under a runtime-swappable policy (priority, weighted, latency-aware, cost-aware).
- **Resilience.** Per-provider circuit breaker (closed → open → half-open), retry with jittered
  backoff, and **cross-provider** fallback so a dead backend never reaches the client as an error.
- **Semantic cache (headline feature).** Near-duplicate prompts are served from cache via local
  embeddings + ANN search, cutting cost and tail latency at a measured hit rate.
- **Per-tenant rate limiting.** Two-dimensional — **request rate and token rate** — enforced in
  Redis, with pre-call token estimation reconciled against the provider's actual `usage`.
- **SSE streaming** end-to-end with natural backpressure.
- **Full observability.** Prometheus latency percentiles, cache hit rate, error rates, and
  per-tenant cost attribution, rendered in Grafana — and fed back into routing.

## The fleet (free-tier / local providers)

All four expose an OpenAI-compatible `chat/completions` surface, keeping the adapter layer thin.

| Provider | Role | Notes |
|---|---|---|
| **Groq** | Fast path (low latency) | LPU-served open models; high tokens/sec. Request- and token-capped/day. |
| **Google Gemini** | Quality / multimodal | Frontier-class Flash, 1M context, vision. Has a streaming-usage quirk (normalized in the adapter). |
| **OpenRouter** | Breadth path | One key, many `:free` models; wide fallback pool. |
| **Ollama (local)** | Offline fallback | Runs on the laptop; no network, no quota — guarantees the fallback chain always terminates. |

> Free-tier limits change frequently and vary by account/region. The gateway treats limits as
> **config, not hardcoded constants** — verify current numbers in each provider's dashboard before benchmarking.

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
   OpenAI SDK client ──▶ │                 LLM GATEWAY                    │
   (base_url=gateway)    │                                                │
                         │  1. Auth / tenant resolution (API key)         │
                         │  2. Rate limit check (Redis: req + tokens)     │
                         │  3. Semantic cache lookup (embed → ANN)        │──hit──▶ cached response
                         │  4. Router: alias → ordered provider targets   │
                         │  5. Resilience: breaker + retry + fallback     │
                         │  6. Provider adapter → upstream call (httpx)   │──────▶ Groq / Gemini /
                         │  7. (stream) SSE passthrough w/ backpressure   │        OpenRouter / Ollama
                         │  8. Token reconcile + cost attribution         │◀──────
                         │  9. Cache write + metrics emit (Prometheus)    │
                         └──────────────────────────────────────────────┘
                                    │                         │
                              ┌─────▼─────┐            ┌───────▼────────┐
                              │   Redis   │            │ Prometheus +   │
                              │ (limits,  │            │ Grafana        │
                              │  cache    │            │ (dashboards)   │
                              │  vectors) │            └────────────────┘
                              └───────────┘
```

The numbered steps are the target lifecycle. One refinement in the current code: **alias validation
runs before rate-limit admission** — resolving an alias to its targets is a cheap, side-effect-free
step, so an unknown alias returns `400` before any quota is charged (rather than charge-then-refund).
The rest of the ordering is as shown.

**Control plane vs. data plane:** routing policies and provider/model config live in YAML loaded
at startup. Live signals (per-provider p95 latency, breaker state) computed by the observability
layer feed the latency-/cost-aware policies — so observability is an *input* to routing, not a bolt-on.

---

## Quickstart

**Works today** — OpenAI-compatible gateway routing logical aliases across the provider fleet, non-streaming.

```bash
# 1. Provider key(s) — Groq is enough to start; others are optional
cp .env.example .env        # then set GROQ_API_KEY=gsk_...  (https://console.groq.com/keys)

# 2a. Run locally
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn gateway.main:app --port 8000

# 2b. …or via Docker
docker compose up --build

# 3. Point any OpenAI SDK client at the gateway — send a logical ALIAS as the model
python - <<'PY'
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused-until-phase3")
print(client.chat.completions.create(
    model="fast",     # alias, not a real model name → resolved by routing policy
    messages=[{"role": "user", "content": "Hello from the gateway"}],
).choices[0].message.content)
PY
```

Endpoints: `POST /v1/chat/completions` (streaming + non-streaming), `GET /healthz`, `GET /metrics`
(live Prometheus exposition — see [Observability](#observability)). Send `"stream": true` for
token-by-token SSE. Unknown aliases return `400`.

### Configuring the fleet & routing

Two YAML files drive everything (config, not code):

- **`config/providers.yaml`** — each provider's `base_url`, `auth`, and `api_key_env`. A provider
  whose key is unset is skipped at startup (the gateway still runs on whatever's configured).
- **`config/models.yaml`** — each alias → an ordered list of `(provider, model)` targets + a
  routing `policy`:
  - `priority` — try targets in listed order (static failover order).
  - `weighted` — split traffic across targets by `weight`.
  - `latency-aware` — order by live per-provider EWMA latency (fastest measured first).
  - `cost-aware` — order by per-token list price (free/local first).

Shipped aliases: `fast` (Groq→Ollama, **latency-aware**), `smart` (Gemini→Groq, priority), `cheap`
(OpenRouter→Groq, **cost-aware** — prefers the free model), and `balanced` (weighted 2:1 across two
Groq models — demonstrates weighted routing with only a Groq key). See
[Observability → live-aware routing](#observability) for how the live-aware policies close the loop.

### Resilience

The router's ordered targets are executed through per-provider **circuit breakers** with
**jittered-backoff retry** and **cross-provider fallback**:

- `429` / `5xx` / timeouts / transport errors → advance to the next target (retrying a struggling
  backend just amplifies its load). Other `4xx` (client errors) surface immediately.
- Each breaker trips `closed → open` on a failure-rate threshold over a rolling window, probes
  `half-open` after a reset timeout, and closes again on a successful probe.
- The served target is reported in the `x-switchyard-provider` / `x-switchyard-model` response
  headers; live breaker states are exposed at `GET /healthz` (`{"circuits": {...}}`).

So pointing a primary target at a dead URL yields a normal `200` served by the next provider —
no client-visible error.

### Rate limiting (per tenant)

`config/tenants.yaml` maps an API key → a tenant with **per-minute request and token limits**:

```yaml
tenants:
  sk-free-demo:
    tenant: free
    requests_per_min: 20
    tokens_per_min: 2000
```

- **Disabled by default:** if `tenants.yaml` has no tenants, auth + rate limiting are off (no Redis
  needed) and any `api_key` works. Add a tenant to turn enforcement on; an unknown key then → `401`.
- **Two-dimensional token-bucket in Redis** (atomic via a Lua script): a request is admitted only if
  both the request-rate and token-rate buckets have capacity. Because LLM cost is token-denominated,
  the **token limit usually binds before the request limit**.
- Over-budget → `429` with a `Retry-After` header. Counters live in Redis, so they **survive gateway
  restarts** and are shared across instances.
- Admission charges a **`tiktoken`** pre-call estimate; after the response, the token bucket is
  **reconciled** against the provider's real `usage.total_tokens` (refund if over-estimated, extra
  charge if under). tiktoken is an approximation for non-OpenAI tokenizers — reconciliation makes it
  exact. (First rate-limited request lazily downloads the tiktoken vocab; it falls back to a
  heuristic if offline.)
- **Accounting on non-success paths** follows one rule — *charge a request slot iff a real upstream
  attempt was made:*
  - **Unknown alias** (`400`) is validated *before* admission, and a rejected **streaming** request
    (`501`) is rejected before admission too — both charge **nothing** (auth still runs first, so a
    bad key is `401` either way).
  - When an upstream attempt *was* made but produced nothing (every target fails, or a non-retryable
    provider `4xx`), the token estimate is **refunded** (actual usage was 0) but the **request slot
    stays charged** — so retrying a dead route is still throttled by the request-rate limit.

### Semantic cache (the headline feature)

Near-duplicate prompts are served from an in-process cache instead of re-hitting a provider:

- **Flow:** embed the prompt locally (`sentence-transformers` MiniLM-L6-v2, CPU, 384-dim,
  L2-normalized) → FAISS `IndexFlatIP` nearest-neighbor search within the request's **scope** → if
  cosine ≥ threshold, serve the cached response; else miss → call upstream → write `(embedding, response)`.
  The prompt is embedded **once per request, off the event loop** (`run_in_threadpool`), and the vector
  is reused for the read and the write — the CPU-bound embed never blocks the async loop, and a miss
  never embeds twice.
- **Scope** = `(model alias, output-affecting params: temperature/top_p/max_tokens/stop/n/seed)`, so a
  response is never shared across params that would change the answer. Shared across tenants by default
  for hit rate (`cache_per_tenant=true` isolates per tenant).
- **Read sits after rate-limit admission** (per the PRD lifecycle). A **cache hit refunds the token
  estimate** (zero upstream tokens) while keeping the request slot — so caching visibly cuts the
  tenant's token cost.
- **Tunables** (config/env): `cache_enabled`, `cache_similarity_threshold` (default `0.85`, tuned in
  Phase 7), `cache_ttl_s`, `cache_max_entries` (LRU eviction). Streaming bypasses the cache.
- Served source is reported via `x-switchyard-cache: hit|miss`. The MiniLM model loads lazily on the
  first lookup (first request downloads ~80MB); set `cache_enabled=false` for routing-only.

> **Live example:** a prompt then its paraphrase → the paraphrase is served from cache in **~9 ms**
> vs. **~1.3 s** for a live provider call (~140× faster), with no false-hit on an unrelated prompt.

> **macOS note:** `faiss-cpu` and `torch` each bundle their own OpenMP runtime and clash in one
> process; the cache package sets `OMP_NUM_THREADS=1` + `KMP_DUPLICATE_LIB_OK=TRUE` before importing
> either to avoid it.

Bring up Redis with the stack via `docker compose up`, or run it standalone
(`docker run -p 6379:6379 redis:7-alpine`) for local dev.

### Streaming (SSE)

Send `"stream": true` for a token-by-token `text/event-stream` response (backpressure is natural —
a slow client throttles the upstream read):

- The gateway **parses, normalizes, and re-emits** every chunk, so output is spec-conformant
  regardless of backend quirks — e.g. Gemini emits `usage` in *every* chunk; the gateway keeps it
  only on the final one.
- **Post-stream accounting:** `stream_options.include_usage` is requested upstream; token
  reconciliation runs when the stream finishes (refund on a mid-stream error/disconnect). The
  upstream generator is closed on the way out, so its httpx connection is released promptly when a
  client disconnects mid-stream.
- Streaming **bypasses the cache** but still passes auth + rate-limit admission.
- A stream that fails to open returns a proper error status (the first chunk is peeked before
  committing to `200`), and it **falls back cross-provider up to first byte** — a dead primary is
  retried against the next target, so the client gets a normal streamed `200` from the fallback. Once
  the first byte is sent the choice is committed (a later mid-stream error surfaces, not retried).

### Observability

`GET /metrics` serves a live Prometheus exposition (`text/plain; version=0.0.4`). Instrumentation is
purely additive — it never changes request behavior — and label cardinality is bounded (aliases and
providers come from config; an unknown alias is rejected with 400 before any request metric is
recorded, so it never becomes a label, and error types are a fixed set), so no client input can blow
up the series count. The series:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `switchyard_requests_total` | counter | `alias, provider, outcome, stream` | admitted routing attempts (`success`/`error`/`throttled`); a cache hit is `provider="cache"` |
| `switchyard_request_latency_seconds` | histogram | `provider, stream` | upstream latency; **time-to-first-byte** for streams |
| `switchyard_cache_events_total` | counter | `event` | `hit` / `miss` / `bypass` (streaming bypasses the cache) |
| `switchyard_errors_total` | counter | `type` | error responses by type (`invalid_api_key`, `rate_limit_exceeded`, `upstream_<status>`, `stream_incomplete`, …) |
| `switchyard_provider_failures_total` | counter | `provider, reason` | per-provider upstream attempt failures the executor recovered from via fallback (or breaker-driven skips) — visible even when the request ultimately succeeded elsewhere |
| `switchyard_tokens_total` | counter | `provider, direction` | prompt / completion tokens processed |
| `switchyard_cost_usd_total` | counter | `tenant, provider` | **counterfactual cost** — `usage × config/pricing.yaml` list prices (free/local = 0) |

Cost attribution is *counterfactual*: the gateway runs on free/local tiers, so it prices the same
traffic at the providers' paid list prices ([`config/pricing.yaml`](./config/pricing.yaml)) to answer
"what would this have cost per tenant?". For streams, tokens + cost are recorded once the final-chunk
`usage` arrives (in the same post-stream hook that reconciles the rate-limit bucket).

**Dashboards.** `docker compose up` brings up Prometheus (`:9090`) + Grafana (`:3000`, anonymous
admin, no login) alongside the gateway. Grafana auto-provisions the datasource and a 7-panel dashboard
([`dashboards/grafana.json`](./dashboards/grafana.json)): p95 latency by provider, request rate, cache
hit rate, token throughput, per-tenant cost, errors by type, and provider failures / fallbacks.

**Live-aware routing (the control loop).** Two policies feed off the same signals, and ship live on
the default aliases — `fast` is `latency-aware` and `cheap` is `cost-aware` in
[`config/models.yaml`](./config/models.yaml):

- `latency-aware` orders targets by a **live per-provider EWMA latency** the executor records after
  each successful call, so traffic shifts to the faster backend as it's measured. Unmeasured providers
  (cold start) keep their configured order and are never chosen over a measured-fast one — so `fast`
  behaves like priority until both Groq and Ollama have been sampled.
- `cost-aware` orders by per-token list price (free/local backends first) — so `cheap` prefers the
  free OpenRouter model and falls back to the paid Groq model.

Crucially the router reads these signals **in-process**, directly from the registry the executor
writes — it does *not* scrape the gateway's own `/metrics` (that would be circular and lag by a scrape
interval). Prometheus/Grafana reflect the same underlying numbers, but for humans.

## Development & tests

```bash
pip install -r requirements-dev.txt

ruff check .            # lint
ruff format .           # auto-format (use --check in CI)
pytest -q               # routing unit tests + live per-provider conformance
```

Conformance tests make real upstream calls and **skip** providers that are unconfigured,
unreachable, or returning a transient upstream error (`429` quota / `5xx` overload) — so the suite
is green with just a Groq key (and fully hermetic with none). Genuine problems (`400`/`401`/`404`)
still fail. Run everything from the project root.

> **Target acceptance bar (later phases):** one-command `docker compose up` brings up the whole
> stack (gateway + Redis + Prometheus + Grafana + Ollama), with an OpenAI SDK client against
> `localhost` getting correct streamed and non-streamed completions via logical aliases.
>
> **First-run note (from Phase 4 on):** `sentence-transformers` downloads the MiniLM weights
> (~80 MB) on first use — the first request may pause while that completes; it is not a hang.

---

## Roadmap

Each phase is independently demoable and maps to a clause of the target resume bullet.

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Walking skeleton — OpenAI SDK → gateway → one provider → response | ✅ |
| 1 | Multi-provider registry + priority/weighted routing | ✅ |
| 2 | Resilience — circuit breaker + retry + cross-provider fallback | ✅ |
| 3 | Rate limiting — Redis, per-tenant, request + token aware | ✅ |
| 4 | Semantic cache (the headline) | ✅ |
| 5 | SSE streaming passthrough (+ Gemini usage normalization) | ✅ |
| 6 | Observability + cost attribution; latency-/cost-aware routing | ✅ |
| 7 | Benchmark harness — reproducible X / Y / Z numbers | ✅ |
| 8 | Polish & packaging (one-command stack, ARCHITECTURE.md, CI) | ☐ |
| 9 | *(optional)* Responses-API translating front door | ☐ |

**Critical path for a demoable repo:** Phases 0 → 1 → 2 give a routing + resilient gateway.
Add Phase 4 (cache, for the number) and Phase 6 (dashboard) for a complete submission; 3/5/7/8 deepen it.

---

## Benchmark (headline numbers)

Produced by a reproducible harness (`bench/`) — a synthetic workload with a tunable semantic-duplicate
rate `d` run through the **real** MiniLM + FAISS cache. Cost is a **counterfactual** (`usage ×
config/pricing.yaml`, since all providers are free); miss (upstream) latency is a seeded lognormal
model, so the numbers reproduce (hit rate and cost exactly; latency within ~0.2%).

```bash
python -m bench.run_benchmark            # baseline vs cache-on + threshold sweep, at d = 20% and 40%
```

**Results** (seed 42, n = 400, `all-MiniLM-L6-v2`, threshold **0.90**, modeled miss latency
lognormal(median 0.6 s, σ 0.5)):

| duplicate rate d | hit rate (Z) | cost ↓ (X) | p50 ↓ | p95 ↓ (Y) | p99 ↓ |
|---|---|---|---|---|---|
| 20% | 17.5% | 18.2% | 9.4% | 3.7% | 3.7% |
| 40% | 35.2% | 34.7% | 32.0% | 15.7% | 13.0% |

**Threshold tuning (the point of the sweep).** MiniLM places many *distinct* prompts close together,
so a loose threshold serves wrong answers. The sweep (at d = 20%) shows the precision/recall tradeoff
and is why the default `cache_similarity_threshold` was tuned from 0.85 to **0.90**:

| threshold | hit rate | false-hit rate | precision | recall |
|---|---|---|---|---|
| 0.85 | 33.5% | 18.5% | 44.8% | 83.3% |
| 0.88 | 22.5% | 5.8% | 74.4% | 93.1% |
| **0.90** | **17.5%** | **1.5%** | **91.4%** | **88.9%** |
| 0.95 | 5.2% | 0.0% | 100.0% | 29.2% |

> Numbers are machine- and workload-dependent; the scoping (`d`, threshold, embedding model, modeled
> miss latency) is stated so they're honest and reproducible.

In résumé terms: *at a 40% workload duplicate rate, the semantic cache achieved a **35% hit rate**,
cutting counterfactual request cost by **~35%** and p95 latency by **~16%*** (`all-MiniLM-L6-v2`,
threshold 0.90).

### Resilience demo

```bash
python -m bench.demo_resilience          # self-contained; no keys, no Docker
```

Launches a mock upstream (two backends) + a real gateway (temp config; real `config/` untouched),
drives traffic in two phases, and takes the primary down mid-run:

- **Phase 1 (healthy):** every request served by the primary (`mock_a`).
- **Phase 2 (primary 503s):** traffic **transparently falls back** to `mock_b`; after the failure
  rate crosses the breaker's threshold, `mock_a`'s circuit trips **open** and it's skipped entirely —
  yet **every client request still returns 200**.

The script prints the per-phase serving provider, the breaker snapshot (`/healthz`), and the
`switchyard_provider_failures_total` / `switchyard_requests_total` counters (`/metrics`). Under
`docker compose up`, the same signals animate Grafana's *Provider failures / fallbacks* and *Request
rate by provider* panels (a recorded GIF is Phase 8 polish).

---

## Tech stack (all free / local)

- **Service:** Python 3.11, FastAPI, `httpx.AsyncClient` (I/O-bound on upstream calls)
- **State:** Redis (rate-limit counters, optional shared cache vectors)
- **Cache:** `sentence-transformers` (`all-MiniLM-L6-v2`, CPU) + FAISS (in-process) / Redis-vector
- **Token estimate:** `tiktoken` (approximation, reconciled against provider `usage`)
- **Observability:** `prometheus_client`, Prometheus, Grafana
- **Local model:** Ollama
- **Packaging:** Docker + `docker-compose`

---

## Documents

- [`PRD.md`](./PRD.md) — problem, goals/non-goals, component specs, provider quirks, risks

## Non-goals

Not running model inference (no GPU serving), not a production/SLA/multi-region service, not a UI
product (Grafana is the only "frontend"), and not every OpenAI endpoint — core scope is
`chat/completions` (streaming + non-streaming) plus `/healthz` and `/metrics`.
