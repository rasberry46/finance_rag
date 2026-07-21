"""
evaluation.py  —  STEP 4: Evaluation & Caching
==============================================
Two production concerns bundled together:

  A. EVALUATION — proving retrieval quality with numbers, not vibes.
     Precision@k, Recall@k, F1, MRR, Hit@k against a labeled test set.

  B. CACHING & MEMORY — the production-readiness layer:
     - TTLCache: repeat queries skip the whole pipeline (the "1000x" speedup).
     - ConversationMemory: last N turns, so follow-up questions have context.

Why this matters in the interview:
  "How do you know it works?" -> you show F1 on a golden set.
  "Is it production-ready?"   -> you show a cache hit is ~1000x faster than a
                                 cold call, and that you cap memory to N turns.
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field

from .config import CONFIG


# ============================================================================
# A. EVALUATION METRICS
# ============================================================================
def precision_recall_f1(retrieved: list[str], relevant: set[str]) -> tuple[float, float, float]:
    """Standard IR metrics for one query.
       precision = of what I returned, how much was relevant?
       recall    = of all relevant docs, how many did I find?
       f1        = harmonic mean (punishes lopsided scores)."""
    if not retrieved:
        return 0.0, 0.0, 0.0
    ret = set(retrieved)
    tp = len(ret & relevant)
    precision = tp / len(ret)
    recall = tp / len(relevant) if relevant else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/position of the first relevant hit.
       Rewards putting a correct answer near the top."""
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Did at least one relevant doc land in the top k? (1.0 or 0.0)"""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


@dataclass
class EvalReport:
    precision: float
    recall: float
    f1: float
    mrr: float
    hit_at_3: float
    n_cases: int

    def pretty(self) -> str:
        return (f"Precision={self.precision:.2f}  Recall={self.recall:.2f}  "
                f"F1={self.f1:.2f}  MRR={self.mrr:.2f}  Hit@3={self.hit_at_3:.2f}  "
                f"(n={self.n_cases})")


def evaluate(cases: list[dict], k: int = 5) -> EvalReport:
    """cases = [{"query":..., "retrieved":[ids], "relevant":{ids}}, ...]
       Averages each metric across all queries."""
    if not cases:
        return EvalReport(0, 0, 0, 0, 0, 0)
    P = R = F = M = H = 0.0
    for c in cases:
        ret = c["retrieved"][:k]
        rel = set(c["relevant"])
        p, r, f = precision_recall_f1(ret, rel)
        P += p; R += r; F += f
        M += mrr(ret, rel)
        H += hit_at_k(ret, rel, 3)
    n = len(cases)
    return EvalReport(P/n, R/n, F/n, M/n, H/n, n)


# ============================================================================
# B. TTL CACHE  (the "1000x latency reduction")
# ============================================================================
class TTLCache:
    """In-memory cache with time-to-live expiry. Stand-in for Redis SETEX.
       Key insight: an identical query within the TTL window skips embedding +
       retrieval + the LLM call entirely — turning a ~1-2s pipeline into a
       microsecond dict lookup."""

    def __init__(self, ttl_seconds: int = None):
        self.ttl = ttl_seconds or CONFIG.cache_ttl_seconds
        self._store: dict[str, tuple[float, object]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()

    def get(self, query: str):
        k = self._key(query)
        entry = self._store.get(k)
        if entry is None:
            self.misses += 1
            return None
        expires_at, value = entry
        if time.time() > expires_at:      # expired
            del self._store[k]
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set(self, query: str, value: object):
        self._store[self._key(query)] = (time.time() + self.ttl, value)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ============================================================================
# B2. CONVERSATION MEMORY (last N turns)
# ============================================================================
class ConversationMemory:
    """Sliding window of the last N (user, assistant) turns. Capping the window
       keeps prompt size + cost bounded while giving follow-ups context."""

    def __init__(self, max_turns: int = None):
        self.turns: deque = deque(maxlen=max_turns or CONFIG.conversation_max_turns)

    def add(self, user: str, assistant: str):
        self.turns.append((user, assistant))

    def as_context(self) -> str:
        return "\n".join(f"User: {u}\nAssistant: {a}" for u, a in self.turns)

    def __len__(self):
        return len(self.turns)


if __name__ == "__main__":
    # --- Evaluation demo on a tiny labeled set ---
    cases = [
        {"query": "Enterprise Q3 revenue", "retrieved": ["r_ent", "prose1", "r_sb"],
         "relevant": {"r_ent"}},
        {"query": "NRR rate", "retrieved": ["memo1", "r_nrr", "prose2"],
         "relevant": {"r_nrr", "memo1"}},
    ]
    print("EVAL:", evaluate(cases, k=5).pretty())

    # --- Cache demo: measure the real speedup ---
    cache = TTLCache(ttl_seconds=60)

    def expensive_pipeline(q):
        time.sleep(0.5)  # simulate embed + retrieve + LLM (~500ms)
        return f"answer to: {q}"

    q = "Explain the S&M variance"

    t0 = time.perf_counter()
    if (cached := cache.get(q)) is None:
        cached = expensive_pipeline(q); cache.set(q, cached)
    cold_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    _ = cache.get(q)                      # cache HIT
    warm_ms = (time.perf_counter() - t1) * 1000

    print(f"\nCold call: {cold_ms:8.2f} ms")
    print(f"Cache hit: {warm_ms:8.4f} ms")
    print(f"Speedup  : {cold_ms/warm_ms:,.0f}x   hit_rate={cache.hit_rate:.2f}")

    # --- Memory demo ---
    mem = ConversationMemory(max_turns=10)
    mem.add("What was S&M variance?", "Unfavorable 360 (+11.2%).")
    mem.add("Is that the largest?", "Yes, among opex line items.")
    print(f"\nMemory holds {len(mem)} turns:\n{mem.as_context()}")
