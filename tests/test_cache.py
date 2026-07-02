import os

import numpy as np
import pytest

from gateway.cache.semantic_cache import SemanticCache, build_query_text, build_scope_key
from gateway.schemas import ChatCompletionRequest, ChatCompletionResponse, Choice, Message, Usage

THRESHOLD = 0.9


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeEmbedder:
    """Maps each text to a pre-registered raw vector, then L2-normalizes — so tests control cosine
    similarity exactly (cosine([1,0], [cosθ, sinθ]) == cosθ)."""

    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, texts):
        arr = np.array([self.vectors[t] for t in texts], dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return (arr / norms).astype("float32")


def _resp(rid="r"):
    return ChatCompletionResponse(
        id=rid,
        object="chat.completion",
        created=0,
        model="m",
        choices=[Choice(index=0, message=Message(role="assistant", content="hi"))],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _unit_at_cosine(c):
    return [c, float(np.sqrt(1 - c * c))]  # unit-ish vector whose cosine with [1,0] is c


# get/set take a precomputed vector; these embed the text first (mirrors the pipeline, which
# embeds once via cache.embed_query and reuses the vector across the read and write).
def _set(cache, scope, text, response):
    cache.set(scope, cache.embed_query(text), response)


def _get(cache, scope, text):
    return cache.get(scope, cache.embed_query(text))


# --- store: hits / misses ----------------------------------------------------------------
def test_exact_and_paraphrase_hit():
    emb = FakeEmbedder({"cat": [1.0, 0.0], "cat-para": _unit_at_cosine(0.95)})
    cache = SemanticCache(emb, threshold=THRESHOLD)
    _set(cache, "s", "cat", _resp("cat"))
    assert _get(cache, "s", "cat").id == "cat"  # exact, cosine 1.0
    assert _get(cache, "s", "cat-para").id == "cat"  # paraphrase, cosine 0.95 >= 0.9


def test_below_threshold_miss():
    emb = FakeEmbedder({"cat": [1.0, 0.0], "far": _unit_at_cosine(0.85)})
    cache = SemanticCache(emb, threshold=THRESHOLD)
    _set(cache, "s", "cat", _resp())
    assert _get(cache, "s", "far") is None  # cosine 0.85 < 0.9


def test_scope_isolation():
    emb = FakeEmbedder({"cat": [1.0, 0.0]})
    cache = SemanticCache(emb, threshold=THRESHOLD)
    _set(cache, "scope-a", "cat", _resp())
    assert _get(cache, "scope-a", "cat") is not None
    assert _get(cache, "scope-b", "cat") is None  # same prompt, different scope => miss


def test_empty_scope_miss():
    cache = SemanticCache(FakeEmbedder({"x": [1.0, 0.0]}), threshold=THRESHOLD)
    assert _get(cache, "never-written", "x") is None


# --- store: TTL + eviction ---------------------------------------------------------------
def test_ttl_expiry():
    clock = FakeClock(1000.0)
    cache = SemanticCache(
        FakeEmbedder({"cat": [1.0, 0.0]}), threshold=THRESHOLD, ttl_s=30, now=clock
    )
    _set(cache, "s", "cat", _resp())
    assert _get(cache, "s", "cat") is not None
    clock.advance(31)
    assert _get(cache, "s", "cat") is None  # expired


def test_lru_eviction_respects_recency():
    emb = FakeEmbedder({"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0], "c": [0.0, 0.0, 1.0]})
    cache = SemanticCache(emb, threshold=THRESHOLD, max_entries=2)
    _set(cache, "s", "a", _resp("a"))
    _set(cache, "s", "b", _resp("b"))
    assert _get(cache, "s", "a").id == "a"  # touch 'a' -> 'b' becomes least-recently-used
    _set(cache, "s", "c", _resp("c"))  # over capacity -> evict LRU ('b')
    assert _get(cache, "s", "b") is None
    assert _get(cache, "s", "a").id == "a"
    assert _get(cache, "s", "c").id == "c"


# --- key builders ------------------------------------------------------------------------
def _req(**kw):
    base = {"model": "fast", "messages": [Message(role="user", content="hi")]}
    base.update(kw)
    return ChatCompletionRequest(**base)


def test_scope_key_stable_for_same_request():
    assert build_scope_key(_req(temperature=0.5)) == build_scope_key(_req(temperature=0.5))


def test_scope_key_differs_on_output_params():
    assert build_scope_key(_req(max_tokens=10)) != build_scope_key(_req(max_tokens=20))
    assert build_scope_key(_req(model="fast")) != build_scope_key(_req(model="smart"))


def test_scope_key_ignores_non_output_params():
    # `stream` doesn't change the completion content -> not part of the scope.
    assert build_scope_key(_req(stream=True)) == build_scope_key(_req(stream=False))


def test_query_text_reflects_full_messages():
    a = build_query_text(_req(messages=[Message(role="user", content="hello")]))
    b = build_query_text(_req(messages=[Message(role="user", content="world")]))
    assert a != b and "hello" in a


# --- real embedder smoke (opt-in; downloads MiniLM weights, so off in CI) ----------------
@pytest.mark.skipif(
    not os.getenv("SWITCHYARD_MODEL_TESTS"),
    reason="set SWITCHYARD_MODEL_TESTS=1 to run the real-MiniLM smoke test (downloads weights)",
)
def test_real_minilm_paraphrase_hit():
    try:
        from gateway.cache.embedder import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
        embedder.embed(["warmup"])  # triggers model load / download
    except Exception as exc:  # torch / sentence-transformers / network unavailable
        pytest.skip(f"real MiniLM unavailable: {exc}")

    cache = SemanticCache(embedder, threshold=0.6)
    _set(cache, "s", "How do I reset my password?", _resp("pw"))
    assert _get(cache, "s", "What's the way to reset my password?").id == "pw"  # paraphrase
    assert _get(cache, "s", "What is the capital of France?") is None  # unrelated
