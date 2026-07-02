"""In-process semantic cache: embed the prompt, ANN-search a FAISS index, serve the stored
response when cosine similarity clears a threshold.

Design:
  - One FAISS `IndexFlatIP` per **scope** (scope = model alias + output-affecting params), wrapped
    in `IndexIDMap2` so entries can be removed by id. Normalized embeddings ⇒ inner product ==
    cosine. Responses are never shared across scopes that would change the answer.
  - **TTL**: entries carry an expiry; expired hits are dropped on access.
  - **Max-size LRU eviction**: a global recency order caps total entries across all scopes.

The store is deliberately request-agnostic — `get`/`set` take an opaque `scope` string and the query
`text`. `build_scope_key`/`build_query_text` derive those from a request; keeping them separate lets
the pipeline layer (Phase 4 Group 2) decide, e.g., whether to prepend a tenant segment to the scope.
"""

import json
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import faiss  # OMP_NUM_THREADS is pinned to 1 in gateway/cache/__init__.py before this import
import numpy as np

from gateway.cache.embedder import Embedder
from gateway.schemas import ChatCompletionRequest, ChatCompletionResponse

# Request params that change the completion's content -> part of the cache scope.
_SCOPE_PARAMS = ("temperature", "top_p", "max_tokens", "stop", "n", "seed")


def build_query_text(request: ChatCompletionRequest) -> str:
    """Serialize the full message list (not just the last turn) so requests with different system
    prompts / history don't collide on the same final user message."""
    parts = []
    for message in request.messages:
        content = message.content
        if not isinstance(content, str):
            content = json.dumps(content, sort_keys=True, default=str)
        parts.append(f"{message.role}: {content}")
    return "\n".join(parts)


def build_scope_key(request: ChatCompletionRequest) -> str:
    """A stable key over the model alias + output-affecting params. Same prompt under different
    params ⇒ different scope ⇒ cache miss."""
    params = {name: getattr(request, name, None) for name in _SCOPE_PARAMS}
    return json.dumps({"model": request.model, "params": params}, sort_keys=True, default=str)


@dataclass
class _Entry:
    scope: str
    response: ChatCompletionResponse
    expires_at: float


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder,
        *,
        threshold: float = 0.85,
        ttl_s: float | None = None,
        max_entries: int = 10_000,
        now: Callable[[], float] = time.monotonic,
    ):
        self.embedder = embedder
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.now = now

        self._indexes: dict[str, faiss.IndexIDMap2] = {}
        self._entries: dict[int, _Entry] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()  # id -> None, MRU at the end
        self._next_id = 0
        self._dim: int | None = None

    def get(self, scope: str, text: str) -> ChatCompletionResponse | None:
        index = self._indexes.get(scope)
        if index is None or index.ntotal == 0:
            return None
        vec = self._embed_one(text)
        k = min(index.ntotal, 5)
        sims, ids = index.search(vec, k)
        now = self.now()
        for sim, entry_id in zip(sims[0], ids[0], strict=False):
            if entry_id == -1:
                continue
            entry = self._entries.get(int(entry_id))
            if entry is None:
                continue
            if now >= entry.expires_at:
                self._remove(int(entry_id))
                continue
            if sim >= self.threshold:
                self._lru.move_to_end(int(entry_id))
                return entry.response
            # FAISS returns descending similarity, so nothing further can clear the threshold.
            break
        return None

    def set(self, scope: str, text: str, response: ChatCompletionResponse) -> None:
        vec = self._embed_one(text)
        if self._dim is None:
            self._dim = vec.shape[1]
        index = self._indexes.get(scope)
        if index is None:
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dim))
            self._indexes[scope] = index

        entry_id = self._next_id
        self._next_id += 1
        index.add_with_ids(vec, np.array([entry_id], dtype="int64"))
        expires_at = self.now() + self.ttl_s if self.ttl_s else float("inf")
        self._entries[entry_id] = _Entry(scope, response, expires_at)
        self._lru[entry_id] = None
        self._evict_if_needed()

    def _embed_one(self, text: str) -> np.ndarray:
        return np.asarray(self.embedder.embed([text]), dtype="float32")

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            oldest_id, _ = self._lru.popitem(last=False)
            self._remove_from_index(oldest_id)
            self._entries.pop(oldest_id, None)

    def _remove(self, entry_id: int) -> None:
        self._remove_from_index(entry_id)
        self._entries.pop(entry_id, None)
        self._lru.pop(entry_id, None)

    def _remove_from_index(self, entry_id: int) -> None:
        entry = self._entries.get(entry_id)
        if entry is None:
            return
        index = self._indexes.get(entry.scope)
        if index is not None:
            index.remove_ids(np.array([entry_id], dtype="int64"))
