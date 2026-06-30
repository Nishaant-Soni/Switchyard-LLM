# Switchyard — LLM Gateway / Inference Router

> **Status:** Planning / scaffolding. Design is locked in [`PRD.md`](./PRD.md) and
> [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md); code is being built phase by phase (see [Roadmap](#roadmap)).
>
> **Codename:** `llm-gateway` · **Track:** AI Engineer (system-design depth)

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

**Control plane vs. data plane:** routing policies and provider/model config live in YAML loaded
at startup. Live signals (per-provider p95 latency, breaker state) computed by the observability
layer feed the latency-/cost-aware policies — so observability is an *input* to routing, not a bolt-on.

See [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) (Phase 8) for the full request lifecycle and design-decision rationale.

---

## Quickstart

> Not yet runnable — these are the **target** commands once Phase 0 lands. The acceptance bar is:
> one-command `docker compose up`, and an OpenAI SDK client against `localhost` getting correct
> streamed and non-streamed completions.

```bash
# 1. Provider keys
cp .env.example .env        # then add free-tier keys for Groq / Gemini / OpenRouter

# 2. Bring up the whole stack (gateway + Redis + Prometheus + Grafana + Ollama)
docker compose up

# 3. Point any OpenAI SDK client at the gateway
python - <<'PY'
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="<tenant-key>")
print(client.chat.completions.create(
    model="fast",                         # logical alias, not a real model name
    messages=[{"role": "user", "content": "Hello from the gateway"}],
).choices[0].message.content)
PY
```

> **First run note:** `sentence-transformers` downloads the MiniLM weights (~80 MB) on first use —
> the first request may pause while that completes; it is not a hang.

---

## Roadmap

Each phase is independently demoable and maps to a clause of the target resume bullet.

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Walking skeleton — OpenAI SDK → gateway → one provider → response | ☐ |
| 1 | Multi-provider registry + priority/weighted routing | ☐ |
| 2 | Resilience — circuit breaker + retry + cross-provider fallback | ☐ |
| 3 | Rate limiting — Redis, per-tenant, request + token aware | ☐ |
| 4 | Semantic cache (the headline) | ☐ |
| 5 | SSE streaming passthrough (+ Gemini usage normalization) | ☐ |
| 6 | Observability + cost attribution; latency-/cost-aware routing | ☐ |
| 7 | Benchmark harness — reproducible X / Y / Z numbers | ☐ |
| 8 | Polish & packaging (one-command stack, ARCHITECTURE.md, CI) | ☐ |
| 9 | *(optional)* Responses-API translating front door | ☐ |

**Critical path for a demoable repo:** Phases 0 → 1 → 2 give a routing + resilient gateway.
Add Phase 4 (cache, for the number) and Phase 6 (dashboard) for a complete submission; 3/5/7/8 deepen it.

---

## Benchmark (headline numbers)

Filled in after Phase 7. The number is reproducible and honestly scoped — its value depends on the
workload's duplicate rate, which is stated explicitly.

> *At a workload duplicate rate of **d%**, the semantic cache achieved a **Z%** hit rate, reducing
> notional cost by **X%** and p95 latency by **Y%**.* (embedding model: `all-MiniLM-L6-v2`; similarity threshold: `T`)

Run with:

```bash
python bench/run_benchmark.py     # baseline (cache off) vs. cache on; threshold sweep; d = 20% and 40%
```

Cost is a **counterfactual** ("what this would cost on paid tiers"), since all providers are free.

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
- [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) — phase-by-phase build plan + target repo layout

## Non-goals

Not running model inference (no GPU serving), not a production/SLA/multi-region service, not a UI
product (Grafana is the only "frontend"), and not every OpenAI endpoint — core scope is
`chat/completions` (streaming + non-streaming) plus `/healthz` and `/metrics`.
