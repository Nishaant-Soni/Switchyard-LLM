"""Semantic cache package.

`faiss-cpu` and `torch` (pulled in by the MiniLM embedder) each bundle their own OpenMP runtime.
On macOS, whichever loads second aborts with a duplicate-libomp error, and multi-threaded faiss
ops alongside torch can segfault. Two env vars, set *before* either is imported, make them coexist:
  - `KMP_DUPLICATE_LIB_OK=TRUE` — allow the second OpenMP runtime to load instead of aborting.
  - `OMP_NUM_THREADS=1` — single-threaded faiss (ample for an in-process cache) so there's no
    parallel-region contention, which also neutralizes the KMP flag's "incorrect results" caveat.

This lives in the package `__init__` (Python runs it before `embedder` or `semantic_cache`), and
`setdefault` respects an explicit operator override.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
