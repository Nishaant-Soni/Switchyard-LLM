# PRD — LLM Gateway / Inference Router

---

## 1. Problem statement

Any application that talks to LLMs directly couples the caller to a single provider and smears cross-cutting concerns — failover, cost control, rate limiting, caching, observability — across every client. As soon as there is more than one model, more than one provider, or more than one tenant, this stops scaling.

The standard senior-level answer to *"design an LLM platform / chatbot backend at scale"* is to put a **gateway** in the middle: a reverse proxy that exposes a single, stable, OpenAI-compatible API on its front and routes to many heterogeneous providers on its back. This decouples the caller from the backend, which is the architectural fact that turns routing, resilience, cost, caching, and observability into **centralized, swappable concerns** instead of per-client duplication.

This project builds that gateway from first principles (the LiteLLM / Portkey / Helicone problem space) rather than consuming an existing library, in order to demonstrate the *design* — the tradeoffs, not the feature checklist.

## 2. Goals

1. Expose a single OpenAI-compatible `POST /v1/chat/completions` endpoint that existing OpenAI SDK clients can use by changing only `base_url`.
2. Route a logical model alias to one of several real backend providers under a configurable, runtime-swappable policy.
3. Survive provider failure gracefully via circuit breaking, retry-with-backoff, and cross-provider fallback.
4. Reduce cost and tail latency via a semantic cache, and produce a defensible, reproducible benchmark number for it.
5. Enforce per-tenant rate limits on both request rate and token rate.
6. Stream responses (SSE) end-to-end with correct backpressure.
7. Expose full observability: per-provider latency percentiles, cost attribution, cache hit rate, error rates.
8. Run end-to-end at **zero cost** on a laptop using free provider tiers + local components.

## 3. Non-goals

- Not running model inference itself (no GPU serving — that is the separate fine-tuning/serving project on the ML track).
- Not a production, SLA-backed, multi-region service. Free tiers are explicitly *not* production-grade; this is a portfolio-grade demonstration of the design.
- Not a UI product. A minimal Grafana dashboard is the only "frontend"; the deliverable is the gateway + its benchmark + its docs.
- Not implementing every OpenAI endpoint. Core scope is `chat/completions` (streaming and non-streaming) plus `/healthz` and `/metrics`. A `/v1/responses` front door is an optional advanced feature (see §7.1 and Phase 9) that translates to the canonical Chat Completions representation rather than reimplementing the full Responses surface. `embeddings` is internal-only (used by the cache).

## 4. Target users

- **API clients** (any OpenAI-SDK app) that want provider-agnostic, resilient, cheaper LLM access through one endpoint and one key.
- **Operators** (you, in the demo) who configure routing policy, watch the dashboard, and set per-tenant limits.

## 5. Backend providers (the "fleet")

The gateway needs **genuinely heterogeneous** backends so that routing and failover do real work rather than being simulated. The free-tier mix below gives four providers with different latency / quality / limit / availability profiles. **All four expose an OpenAI-compatible `chat/completions` surface**, which keeps the adapter layer thin.

| Provider | Role in the fleet | Surface | Notes |
|---|---|---|---|
| **Groq** | Fast path (low latency) | OpenAI-compatible | LPU-served open models (Llama, Qwen, etc.); high tokens/sec. Free tier is request- *and* token-capped per day. |
| **Google Gemini** | Quality / multimodal path | OpenAI-compatible at `https://generativelanguage.googleapis.com/v1beta/openai/` | Frontier-class Flash, 1M context, native vision. Has a streaming quirk (see §10). |
| **OpenRouter** | Breadth path | OpenAI-compatible | Single key, many `:free` models; useful as a wide fallback pool. |
| **Ollama (local)** | Always-available offline fallback | OpenAI-compatible (`/v1`) | Runs on the laptop; no network, no quota. Guarantees the fallback chain always terminates in *something*. |

> Free-tier rate limits change frequently and vary by account/region — the gateway treats limits as **config**, not hardcoded constants, so they can be updated without code changes. Verify current numbers in each provider's dashboard before benchmarking.

## 6. System architecture

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

**Request lifecycle (non-streaming):** auth → rate-limit → cache read → route → (breaker-guarded) upstream call with retry/fallback → token reconcile → cost attribution → cache write → metrics → respond.

> **Note — alias validation precedes admission.** Resolving the logical alias to its ordered targets is a cheap, side-effect-free in-memory step, so it runs *before* rate-limit admission: an unknown alias is a client-side `400` that touches no provider and does no billable work, and must not consume any quota. The rest of the lifecycle ordering above is unchanged; only alias *validation* is hoisted ahead of the rate-limit check.

**Control plane vs. data plane:** routing policies and provider/model config live in YAML loaded at startup (and hot-reloadable as a stretch goal). Live signals consumed by the control plane (per-provider p95 latency, breaker state) are computed from the observability layer — closing the loop so observability is an *input* to routing, not a bolt-on.

## 7. Component specifications

### 7.1 API surface + provider adapters
- **Canonical internal representation = the OpenAI `chat/completions` schema** (Pydantic models). This is a deliberate choice: Chat Completions is *not* deprecated (OpenAI has stated it remains supported indefinitely as an industry standard) and it is the **cross-provider common denominator** — all four backends (Groq, Gemini's compat layer, OpenRouter, Ollama) speak it, whereas OpenAI's newer Responses API is still largely OpenAI-specific (Gemini's compat endpoint 404s on `/responses`; Ollama support was still pending at time of writing). A multi-provider gateway would have to translate down to Chat Completions for every backend regardless, so making it the canonical format is correct.
- **Front door — Chat Completions:** `POST /v1/chat/completions` is the primary, always-on surface. Existing OpenAI SDK clients work by changing only `base_url`.
- **Front door — Responses (optional, Phase 9):** OpenAI now *recommends* the item-centered Responses API (`/v1/responses`) for new projects. Rather than reimplement it, the gateway can expose a `/v1/responses` endpoint that **translates Responses ↔ Chat Completions** at the boundary (decode the typed-item `input`/`instructions` into canonical messages; re-encode `choices` into typed `output` items; map streaming event types both ways). This turns the OpenAI API divergence into a demonstration of the gateway's core purpose — absorbing protocol heterogeneity — and mirrors real ecosystem bridges (e.g. `wire_api`-style adapters).
- **Adapter pattern (Strategy):** each provider implements a `ProviderAdapter` interface. Because all four targets are OpenAI-compatible (Chat Completions), the base adapter is near pass-through (swap `base_url`, auth header, and model name); per-provider subclasses handle quirks (auth scheme, streaming-usage normalization, unsupported params).
- **Why it matters (interview):** decoupling caller from backend is the whole point of a gateway; the adapter layer is where that decoupling lives, and the optional Responses front door shows the same principle applied at the *inbound* protocol boundary, not just the outbound one.

### 7.2 Routing / control plane
- Input: logical model alias (e.g. `fast`, `smart`, `cheap`) → ordered list of concrete provider targets + a policy.
- Policies (pluggable):
  - **Priority / failover** — try targets in order; first healthy one wins.
  - **Weighted** — split traffic by weight (canary / A-B).
  - **Latency-aware** — pick the target with the best current p95 (consumes observability).
  - **Cost-aware** — pick the cheapest target meeting constraints (consumes pricing table).
- **Why it matters:** identical in shape to load-balancer policy design; "routing policy is a swappable strategy, not hardcoded" is the defensible statement.

### 7.3 Semantic cache (headline feature)
- **Flow:** embed incoming prompt (local `sentence-transformers`, e.g. `all-MiniLM-L6-v2`, 384-dim, CPU) → ANN nearest-neighbor search over cached request embeddings → if cosine similarity ≥ threshold, serve the cached response; else miss → call upstream → write `(embedding, response)` to cache.
- **Scoping / correctness:** cache key is scoped by `(model alias, normalized params that affect output)`; responses are not shared across scopes that would change the answer. Optional per-tenant scoping (correctness/privacy vs. hit-rate tradeoff).
- **Tunables:** similarity threshold (precision/recall tradeoff — too loose serves wrong answers, too tight collapses hit rate), TTL, max entries (eviction).
- **Index:** FAISS (in-process) for the benchmark; Redis-vector as the shared-state alternative for multi-instance.
- **Streaming:** streaming requests bypass the cache *read* path; on a streaming miss, the assembled full response may be written to cache for future non-streaming hits (stretch).
- **Why it matters:** this produces the resume number (cost ↓, p95 ↓ at a measured hit rate). Owning the threshold-tuning story carries an entire interview.

### 7.4 Rate limiting
- **Per API key / tenant**, enforced in Redis so counters survive horizontal scaling.
- Algorithm: token-bucket *or* sliding-window-counter (document the choice and tradeoff).
- **Two-dimensional:** limit on request rate *and* on token rate, because LLM cost is token-denominated and **token limits often bind before request limits** (e.g. a provider allowing ~1,000 req/day but only ~100K tokens/day caps you at ~100 calls/day for 1K-token calls).
- **Token accounting:** estimate tokens pre-call (heuristic / `tiktoken` approximation) for the admission check, then **reconcile** against the provider's actual `usage` in the response. Pre/post reconciliation is a deliberate depth point.
- **Why it matters:** token-bucket vs. sliding-window is a classic distributed-systems question; doing it per-tenant in shared state is the realistic version.

### 7.5 Resilience
- **Circuit breaker per provider:** closed → open (after failure-rate threshold) → half-open (probe after reset timeout) → closed. Stops hammering a dead backend and lets it recover.
- **Retry:** exponential backoff **with jitter**; retries go to the *next provider in the fallback chain*, not the same one (cross-provider fallback), because retrying a struggling backend amplifies its load.
- **Timeout budget:** per-attempt and per-request deadlines so a slow provider can't stall the whole request.
- **Why it matters:** this is the resilience vocabulary the current resume lacks. Being able to explain the half-open state and why fallback is cross-provider is staff-level signal.

### 7.6 SSE streaming passthrough
- Upstream streamed with `httpx` streaming; forwarded to the client via FastAPI `StreamingResponse` as an async generator, chunk by chunk.
- **Backpressure** is handled naturally by async iteration (a slow client throttles the upstream read).
- Streaming complicates caching (bypass read) and token accounting (usage arrives at the end via `stream_options: {include_usage: true}`), so it is treated as a distinct path.
- **Why it matters:** demonstrating *why* streaming complicates the rest of the system is the signal.

### 7.7 Observability
- `prometheus_client` exposes: latency histograms (per provider, p50/p95/p99), counters (requests, cache hits/misses, errors by type, tokens in/out), and cost gauges per tenant.
- **Cost attribution:** computed from `usage` × a per-model **list-price** table. Since the providers are free, this is a *counterfactual* ("what this would cost on paid tiers") — which is honest and still makes the caching cost-savings number concrete and demonstrable.
- Grafana dashboard renders it; latency-aware and cost-aware routing consume these same signals.
- **Why it matters:** you can't operate a multi-provider system you can't see, and the metrics feed routing — observability is part of the control loop.

### 7.8 Multi-tenancy
- API keys map to tenants. Each tenant has its own rate-limit budget, cost attribution, and optional cache scope.

## 8. The resume bullet (target)

Drafted up front so every phase maps to part of it:

> *Built a provider-agnostic LLM gateway (FastAPI/async) fronting 4 model providers with an OpenAI-compatible API, implementing policy-based routing, circuit-breaker + cross-provider failover, per-tenant token-aware rate limiting, and a semantic cache that cut request cost by **~X%** and p95 latency by **~Y%** at a **Z%** hit rate; full Prometheus/Grafana observability with per-tenant cost attribution.*

`X`, `Y`, `Z` come from the benchmark (§9), not from guessing.

## 9. Success metrics & benchmark methodology

The headline number must be **reproducible and honestly scoped**, because the cache's effect depends on the workload's duplicate rate.

1. **Workload:** a synthetic request set with a **tunable semantic-duplicate rate** `d` (fraction of requests that are near-paraphrases of earlier ones), drawn from a realistic prompt distribution. Report results at a couple of values of `d` (e.g. 20% and 40%).
2. **Baseline run:** cache disabled → record total notional cost and full latency distribution (p50/p95/p99).
3. **Cache run:** cache enabled at the tuned threshold → record hit rate, notional cost, latency distribution.
4. **Report:** *"At a workload duplicate rate of d%, the semantic cache achieved a Z% hit rate, reducing notional cost by X% and p95 latency by Y%."* State `d`, the threshold, and the embedding model — that scoping is the defensible part.
5. **Resilience check:** kill a provider mid-run; show the breaker tripping and traffic shifting via the dashboard with zero client-visible errors.

**Acceptance:** one-command `docker compose up` brings up gateway + Redis + Prometheus + Grafana + Ollama; a client using the OpenAI SDK against `base_url=localhost` gets correct streamed and non-streamed completions; the benchmark script reproduces the headline numbers; the resilience demo is recordable.

## 10. Known provider quirks (normalization requirements)

A real gateway earns its keep by normalizing provider differences. Documented quirks to handle:

- **Gemini streaming usage:** Gemini's OpenAI-compatible streaming has been observed to return `usage` in **every** chunk rather than only the final chunk (which violates the OpenAI spec where usage appears optionally only in the last chunk). Naive OpenAI-compliant accounting double-counts and can massively inflate reported tokens. The Gemini adapter must take usage only from the final chunk (or de-duplicate) so cost attribution stays correct.
- **Auth schemes differ:** Bearer token vs. key param vs. header name — normalized in the adapter.
- **Param support differs:** some providers ignore or reject certain OpenAI params (`n`, `logprobs`, reasoning controls); the adapter strips/maps unsupported params and records when it does.
- **Free-tier 429s are normal, not exceptional:** the fallback chain is the intended handler for quota exhaustion, not an error path.
- **Responses ↔ Chat Completions translation boundary (if Phase 9 is built):** the two formats differ structurally — Responses is item-centered (typed `input`/`output` items, `instructions`, hosted-tool items, reasoning items) vs. Chat Completions' message/`choices` shape; structured outputs use `text.format` vs. `response_format`; function-calling shapes differ; and streaming uses different SSE event types (`response.output_text.delta` etc. vs. `chat.completion.chunk`). The hard parts are tool-call id/index stability across streaming deltas and reasoning metadata. Note also that `previous_response_id`/server-side statefulness has no equivalent on a stateless Chat Completions backend, so a translating front door must either store conversation state in the gateway or require clients to send full context (stateless mode).

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Free-tier limits change / shrink | Limits are config, not code; Ollama local fallback guarantees the chain always terminates. |
| Provider quirks break OpenAI-compatibility assumptions | Per-provider adapters + a conformance test per provider. |
| Semantic cache serves a subtly wrong answer | Conservative default threshold, scoped keys, threshold sweep in the benchmark to show the precision/recall curve. |
| "Zero cost" undermined by always-on hosting | Run locally; deliverable is repo + README + recorded demo + benchmark, not 24/7 uptime. |
| Scope creep (turning it into a product) | Non-goals (§3) are firm; UI is Grafana only. |

## 12. Tech stack (all free / local)

- **Service:** Python 3.11, FastAPI, `httpx.AsyncClient` (async I/O — gateway is I/O-bound on upstream calls).
- **State:** Redis (rate-limit counters, optional shared cache vectors).
- **Cache embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`), CPU.
- **ANN index:** FAISS (in-process) and/or Redis-vector.
- **Token estimate:** `tiktoken` (approximation for non-OpenAI tokenizers; reconciled against provider `usage`).
- **Observability:** `prometheus_client`, Prometheus, Grafana.
- **Local model:** Ollama.
- **Packaging:** Docker + `docker-compose`.
- **Providers:** Groq, Gemini, OpenRouter (free tiers), Ollama (local).
