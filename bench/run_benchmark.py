"""Semantic-cache benchmark harness (Phase 7 Group 1) — the reproducible headline numbers.

Runs a synthetic workload (see `workload.py`) through the **real** `SemanticCache` twice — baseline
(cache off) and cache-on — and reports, at each duplicate rate `d`:

  Z = hit rate            (deterministic given workload + threshold + embedder)
  X = cost reduction      = 1 - cost_on / cost_off  (counterfactual: usage x pricing.yaml, hits = 0)
  Y = p95 latency drop    = 1 - p95_on / p95_off

plus a **similarity-threshold sweep** (hit rate / false-hit rate / precision / recall) — the
precision-vs-recall tradeoff that is the cache's threshold-tuning story.

Modeling choices (locked in IMPLEMENTATION_PLAN.md Phase 7):
  - Hit/miss decisions + hit latency come from the real MiniLM + FAISS cache (the thing under test).
  - Cost is counterfactual list-price cost; a hit costs 0 (served with no upstream call).
  - Miss (upstream) latency is *modeled* — a seeded lognormal — because a live free-tier provider's
    noisy, rate-limited latency can't be reproduced. Each request's modeled miss latency is fixed
    and shared by both runs, so the only difference between the runs is the cache itself.

Z and X are exactly reproducible; Y depends on the modeled miss distribution (seeded) plus the
small, machine-dependent measured cache overhead (embed + FAISS search).
Run: `python -m bench.run_benchmark`.
"""

import argparse
import json
import math
import random
import time

import numpy as np

from bench.workload import WorkloadItem, distinct_base_count, generate_workload
from gateway.cache.embedder import Embedder, SentenceTransformerEmbedder
from gateway.cache.semantic_cache import (
    SemanticCache,
    build_query_text,
    build_scope_key,
)
from gateway.observability.cost import PriceBook
from gateway.schemas import ChatCompletionResponse, Choice, Message, Usage

_EMBED_MODEL = "all-MiniLM-L6-v2"


def _prompt_tokens(text: str) -> int:
    """Cheap, deterministic prompt-token estimate (~4 chars/token) — reproducible, no tiktoken."""
    return max(1, len(text) // 4)


def _response(model: str, base_id: str, prompt_tokens: int, completion_tokens: int):
    # `id = base_id` lets a cache hit report which base it served (for precision/recall).
    return ChatCompletionResponse(
        id=base_id,
        object="chat.completion",
        created=0,
        model=model,
        choices=[Choice(index=0, message=Message(role="assistant", content=f"answer::{base_id}"))],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _sample_miss_latencies(
    n: int, rng: random.Random, median_s: float, sigma: float
) -> list[float]:
    """One modeled upstream latency per request (seconds), lognormal for a realistic right tail.
    median = exp(mu) => mu = ln(median)."""
    mu = math.log(median_s)
    return [rng.lognormvariate(mu, sigma) for _ in range(n)]


def _prepare(items: list[WorkloadItem], embedder: Embedder) -> list[tuple[np.ndarray, float]]:
    """Embed each request once (timed) so the threshold sweep reuses vectors, not re-embeds.
    Returns (vector, embed_seconds) per item."""
    prepared = []
    for item in items:
        text = build_query_text(item.request)
        t0 = time.perf_counter()
        vec = np.asarray(embedder.embed([text]), dtype="float32")
        prepared.append((vec, time.perf_counter() - t0))
    return prepared


def _run(
    items: list[WorkloadItem],
    prepared: list[tuple[np.ndarray, float]],
    threshold: float,
    miss_latencies: list[float],
    pricebook: PriceBook,
    embedder: Embedder,
    *,
    cache_on: bool,
) -> dict:
    """One pass over the workload. Returns aggregate counts, total counterfactual cost, and the
    per-request latencies (ms). Latency: baseline = modeled miss latency; cache-on hit =
    embed+search; cache-on miss = embed+search+set + modeled miss latency."""
    cache = SemanticCache(embedder, threshold=threshold) if cache_on else None
    latencies_ms: list[float] = []
    total_cost = 0.0
    hits = correct_hits = wrong_hits = should_hit_total = 0

    for i, item in enumerate(items):
        req = item.request
        prompt_tokens = _prompt_tokens(build_query_text(req))
        if item.should_hit:
            should_hit_total += 1

        if not cache_on:
            total_cost += pricebook.cost_usd(req.model, prompt_tokens, item.completion_tokens)
            latencies_ms.append(miss_latencies[i] * 1000.0)
            continue

        vec, embed_s = prepared[i]
        scope = build_scope_key(req)
        t0 = time.perf_counter()
        cached = cache.get(scope, vec)
        search_s = time.perf_counter() - t0

        if cached is not None:
            hits += 1
            if cached.id == item.base_id:
                correct_hits += 1
            else:
                wrong_hits += 1
            latencies_ms.append((embed_s + search_s) * 1000.0)
            continue

        resp = _response(req.model, item.base_id, prompt_tokens, item.completion_tokens)
        t1 = time.perf_counter()
        cache.set(scope, vec, resp)
        set_s = time.perf_counter() - t1
        total_cost += pricebook.cost_usd(req.model, prompt_tokens, item.completion_tokens)
        latencies_ms.append((embed_s + search_s + set_s) * 1000.0 + miss_latencies[i] * 1000.0)

    return {
        "n": len(items),
        "hits": hits,
        "correct_hits": correct_hits,
        "wrong_hits": wrong_hits,
        "should_hit_total": should_hit_total,
        "cost": total_cost,
        "latencies_ms": latencies_ms,
    }


def _pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(p / 100.0 * len(s)) - 1))
    return s[k]


def _drop(baseline: float, cache_on: float) -> float:
    return 1.0 - cache_on / baseline if baseline else 0.0


def benchmark(
    n: int,
    d_values: list[float],
    threshold: float,
    sweep: list[float],
    seed: int,
    miss_median: float,
    miss_sigma: float,
    model: str,
    pricebook: PriceBook,
    embedder: Embedder,
) -> dict:
    """Run the full benchmark and return a JSON-able results dict."""
    runs = []
    for d in d_values:
        items = generate_workload(n, d, seed, model=model)
        miss_latencies = _sample_miss_latencies(n, random.Random(seed + 1), miss_median, miss_sigma)
        prepared = _prepare(items, embedder)

        base = _run(items, prepared, threshold, miss_latencies, pricebook, embedder, cache_on=False)
        on = _run(items, prepared, threshold, miss_latencies, pricebook, embedder, cache_on=True)

        headline = {
            "d": d,
            "hit_rate": on["hits"] / n,
            "cost_reduction": _drop(base["cost"], on["cost"]),
            "p50_drop": _drop(_pct(base["latencies_ms"], 50), _pct(on["latencies_ms"], 50)),
            "p95_drop": _drop(_pct(base["latencies_ms"], 95), _pct(on["latencies_ms"], 95)),
            "p99_drop": _drop(_pct(base["latencies_ms"], 99), _pct(on["latencies_ms"], 99)),
            "baseline_p95_ms": _pct(base["latencies_ms"], 95),
            "cache_p95_ms": _pct(on["latencies_ms"], 95),
            "baseline_cost_usd": base["cost"],
            "cache_cost_usd": on["cost"],
        }

        sweep_rows = []
        for tau in sweep:
            r = _run(items, prepared, tau, miss_latencies, pricebook, embedder, cache_on=True)
            hits = r["hits"]
            sweep_rows.append(
                {
                    "threshold": tau,
                    "hit_rate": hits / n,
                    "false_hit_rate": r["wrong_hits"] / n,
                    "precision": (r["correct_hits"] / hits) if hits else None,
                    "recall": (
                        r["correct_hits"] / r["should_hit_total"] if r["should_hit_total"] else None
                    ),
                }
            )

        runs.append({"headline": headline, "sweep": sweep_rows})

    return {
        "config": {
            "n": n,
            "d_values": d_values,
            "threshold": threshold,
            "seed": seed,
            "model": model,
            "embedding_model": _EMBED_MODEL,
            "miss_latency_median_s": miss_median,
            "miss_latency_sigma": miss_sigma,
            "distinct_bases": distinct_base_count(),
        },
        "runs": runs,
    }


def _pctf(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def render_markdown(results: dict) -> str:
    cfg = results["config"]
    lines = [
        "## Semantic cache benchmark",
        "",
        f"**Scoping:** embedding=`{cfg['embedding_model']}`, model=`{cfg['model']}`, "
        f"threshold={cfg['threshold']}, n={cfg['n']}, seed={cfg['seed']}, "
        f"modeled miss latency = lognormal(median={cfg['miss_latency_median_s']}s, "
        f"sigma={cfg['miss_latency_sigma']}).",
        "",
        "| duplicate rate d | hit rate (Z) | cost ↓ (X) | p50 ↓ | p95 ↓ (Y) | p99 ↓ |",
        "|---|---|---|---|---|---|",
    ]
    for run in results["runs"]:
        h = run["headline"]
        lines.append(
            f"| {_pctf(h['d'])} | {_pctf(h['hit_rate'])} | {_pctf(h['cost_reduction'])} | "
            f"{_pctf(h['p50_drop'])} | {_pctf(h['p95_drop'])} | {_pctf(h['p99_drop'])} |"
        )

    for run in results["runs"]:
        h = run["headline"]
        lines += [
            "",
            f"### Threshold sweep (d = {_pctf(h['d'])})",
            "",
            "| threshold | hit rate | false-hit rate | precision | recall |",
            "|---|---|---|---|---|",
        ]
        for s in run["sweep"]:
            lines.append(
                f"| {s['threshold']} | {_pctf(s['hit_rate'])} | {_pctf(s['false_hit_rate'])} | "
                f"{_pctf(s['precision'])} | {_pctf(s['recall'])} |"
            )
    return "\n".join(lines)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Semantic-cache benchmark harness")
    p.add_argument("--n", type=int, default=400, help="requests per duplicate-rate run")
    p.add_argument("--d", type=float, nargs="+", default=[0.2, 0.4], help="duplicate rates")
    p.add_argument("--threshold", type=float, default=0.85, help="headline similarity threshold")
    p.add_argument(
        "--sweep",
        type=float,
        nargs="+",
        default=[0.70, 0.75, 0.80, 0.83, 0.85, 0.88, 0.90, 0.95],
        help="thresholds for the precision/recall sweep",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--miss-median", type=float, default=0.6, help="modeled miss latency median (s)")
    p.add_argument("--miss-sigma", type=float, default=0.5, help="modeled miss lognormal sigma")
    p.add_argument("--model", type=str, default="llama-3.3-70b-versatile")
    p.add_argument("--pricing", type=str, default="config/pricing.yaml")
    p.add_argument("--json-out", type=str, default=None, help="also write raw results JSON here")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    pricebook = PriceBook.from_config(args.pricing)
    embedder = SentenceTransformerEmbedder()
    results = benchmark(
        n=args.n,
        d_values=args.d,
        threshold=args.threshold,
        sweep=args.sweep,
        seed=args.seed,
        miss_median=args.miss_median,
        miss_sigma=args.miss_sigma,
        model=args.model,
        pricebook=pricebook,
        embedder=embedder,
    )
    print(render_markdown(results))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n(raw results written to {args.json_out})")


if __name__ == "__main__":
    main()
