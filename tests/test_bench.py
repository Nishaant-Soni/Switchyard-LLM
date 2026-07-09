"""Phase 7 Group 1 — benchmark harness tests (hermetic; no MiniLM).

The workload generator is checked directly; the benchmark engine is checked with a fake embedder
that maps a text's "group" to a one-hot vector, so hit/miss (and wrong-hit) behavior is fully
controlled and the cost / hit-rate / precision / recall arithmetic can be asserted exactly. The
real-MiniLM run is the `python -m bench.run_benchmark` script, not a unit test.
"""

import numpy as np

from bench import run_benchmark as rb
from bench.workload import WorkloadItem, distinct_base_count, generate_workload
from gateway.observability.cost import PriceBook
from gateway.schemas import ChatCompletionRequest, Message


# --- workload -----------------------------------------------------------------------------
def test_workload_duplicate_rate_and_labels():
    # n kept below the base-pool ceiling (distinct_base_count / (1 - d)) so novels stay unique and
    # the target duplicate rate is honoured (above it, novel slots fall back to duplicates).
    items = generate_workload(n=600, d=0.3, seed=7)
    assert len(items) == 600
    assert items[0].should_hit is False  # the first request can't be a duplicate
    dup_frac = sum(1 for it in items if it.should_hit) / len(items)
    assert 0.25 < dup_frac < 0.35  # ~= d (0.30)
    # every duplicate references a base that was introduced earlier in the stream
    seen: set[str] = set()
    for it in items:
        if it.should_hit:
            assert it.base_id in seen
        seen.add(it.base_id)


def test_workload_is_deterministic_for_seed():
    a = generate_workload(200, 0.2, seed=1)
    b = generate_workload(200, 0.2, seed=1)
    assert [x.base_id for x in a] == [x.base_id for x in b]
    assert [x.request.messages[0].content for x in a] == [x.request.messages[0].content for x in b]


def test_workload_stays_within_base_pool():
    # Below the base-pool ceiling, novels stay unique so the target duplicate rate is honoured.
    assert distinct_base_count() > 400
    items = generate_workload(300, 0.2, seed=3)
    novel_bases = [it.base_id for it in items if not it.should_hit]
    assert len(novel_bases) == len(set(novel_bases))  # every novel introduced a distinct base


# --- benchmark engine (fake embedder) -----------------------------------------------------
class _GroupEmbedder:
    """One-hot vector by the integer group parsed from a 'g<id>|' marker in the text. Same group =>
    cosine 1.0 (hit at any sane threshold); different group => orthogonal (never hits)."""

    dim = 8

    def embed(self, texts):
        rows = []
        for t in texts:
            gid = int(t.split("g", 1)[1].split("|", 1)[0])
            vec = np.zeros(self.dim, dtype="float32")
            vec[gid % self.dim] = 1.0
            rows.append(vec)
        return np.array(rows, dtype="float32")


def _item(gid: int, base_id: str, should_hit: bool) -> WorkloadItem:
    # Equal-length content so every request has identical prompt tokens (clean cost arithmetic).
    content = f"g{gid}|padding__"
    req = ChatCompletionRequest(model="m", messages=[Message(role="user", content=content)])
    return WorkloadItem(request=req, base_id=base_id, should_hit=should_hit, completion_tokens=100)


_PRICES = PriceBook({"m": {"input": 1.0, "output": 1.0}})


def _run_pair(items, threshold=0.9):
    embedder = _GroupEmbedder()
    prepared = rb._prepare(items, embedder)
    miss = [0.5] * len(items)
    base = rb._run(items, prepared, threshold, miss, _PRICES, embedder, cache_on=False)
    on = rb._run(items, prepared, threshold, miss, _PRICES, embedder, cache_on=True)
    return base, on


def test_benchmark_hitrate_cost_precision_recall():
    # 5 requests: two are repeats of an earlier group -> two correct hits, three misses.
    items = [
        _item(0, "A", should_hit=False),
        _item(1, "B", should_hit=False),
        _item(0, "A", should_hit=True),
        _item(2, "C", should_hit=False),
        _item(1, "B", should_hit=True),
    ]
    base, on = _run_pair(items)

    assert base["hits"] == 0  # cache off never hits
    assert on["hits"] == 2
    assert on["correct_hits"] == 2 and on["wrong_hits"] == 0
    assert on["should_hit_total"] == 2
    assert on["hits"] / on["n"] == 0.4  # Z

    # equal per-request cost => X = 1 - misses/total = 1 - 3/5
    assert abs((1 - on["cost"] / base["cost"]) - 0.4) < 1e-9
    # precision = correct/hits, recall = correct/should-hits
    assert on["correct_hits"] / on["hits"] == 1.0
    assert on["correct_hits"] / on["should_hit_total"] == 1.0


def test_benchmark_counts_wrong_hit():
    # Same embedding group but distinct base ids: the second request hits the first entry and is
    # served the WRONG answer -> a false hit (precision < 1), and it should not count as correct.
    items = [
        _item(0, "A", should_hit=False),
        _item(0, "B", should_hit=False),  # novel base, but collides in vector space with A
    ]
    _, on = _run_pair(items)
    assert on["hits"] == 1
    assert on["wrong_hits"] == 1 and on["correct_hits"] == 0
    assert on["should_hit_total"] == 0  # neither was a genuine repeat


def test_benchmark_latency_drops_with_hits():
    # A workload with repeats should lower cache-on p50 vs baseline (hits skip the upstream).
    items = [_item(i % 2, f"base{i % 2}", should_hit=(i >= 2)) for i in range(6)]
    base, on = _run_pair(items)
    assert rb._pct(on["latencies_ms"], 50) < rb._pct(base["latencies_ms"], 50)


# --- reproducibility guard ----------------------------------------------------------------
def test_harness_default_threshold_matches_gateway_default():
    # The headline table must reproduce with the gateway's *tuned* default threshold. A stale
    # harness default (was 0.85) prints numbers the sweep rejected as too loose (~45% precision),
    # so `python -m bench.run_benchmark` would not reproduce the README's 0.90 headline.
    from gateway.config import Settings

    harness_default = rb._parse_args([]).threshold
    gateway_default = Settings.model_fields["cache_similarity_threshold"].default
    assert harness_default == gateway_default == 0.90
