# Switchyard — LLM Gateway / Inference Router

> **Status:** Phase 2 complete — config-driven multi-provider routing **with resilience** (per-provider circuit breakers + jittered-backoff retry + cross-provider fallback), non-streaming. Design is locked in [`PRD.md`](./PRD.md); code is being built phase by phase (see [Roadmap](#roadmap)).

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

Endpoints: `POST /v1/chat/completions`, `GET /healthz`, `GET /metrics` (stub until Phase 6).
Streaming returns `501` until Phase 5. Unknown aliases return `400`.

### Configuring the fleet & routing

Two YAML files drive everything (config, not code):

- **`config/providers.yaml`** — each provider's `base_url`, `auth`, and `api_key_env`. A provider
  whose key is unset is skipped at startup (the gateway still runs on whatever's configured).
- **`config/models.yaml`** — each alias → an ordered list of `(provider, model)` targets + a
  routing `policy`:
  - `priority` — try targets in listed order (failover *wiring* lands in Phase 2).
  - `weighted` — split traffic across targets by `weight`.

Shipped aliases: `fast` (Groq→Ollama), `smart` (Gemini→Groq), `cheap` (OpenRouter→Groq), and
`balanced` (weighted 2:1 across two Groq models — demonstrates weighted routing with only a Groq key).

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

## Development & tests

```bash
pip install -r requirements-dev.txt

ruff check .            # lint
ruff format .           # auto-format (use --check in CI)
pytest -q               # routing unit tests + live per-provider conformance
```

Conformance tests make real upstream calls and **skip** providers that are unconfigured,
rate-limited (429), or unreachable — so the suite is green with just a Groq key (and fully
hermetic with none). Run everything from the project root.

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

## Non-goals

Not running model inference (no GPU serving), not a production/SLA/multi-region service, not a UI
product (Grafana is the only "frontend"), and not every OpenAI endpoint — core scope is
`chat/completions` (streaming + non-streaming) plus `/healthz` and `/metrics`.
