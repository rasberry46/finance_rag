# Step 4 — Evaluation & Caching — Colab

Builds on Steps 1–3. Measures the REAL cache speedup against actual Bedrock
round-trip latency, and computes retrieval quality metrics on a labeled set.

---

### Cell 1 — Evaluation metrics

```python
def precision_recall_f1(retrieved, relevant):
    if not retrieved: return 0.0, 0.0, 0.0
    ret = set(retrieved); tp = len(ret & relevant)
    p = tp/len(ret); r = tp/len(relevant) if relevant else 0.0
    f = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f

def mrr(retrieved, relevant):
    for i, d in enumerate(retrieved, 1):
        if d in relevant: return 1.0/i
    return 0.0

def evaluate(cases, k=5):
    P=R=F=M=0.0
    for c in cases:
        ret, rel = c["retrieved"][:k], set(c["relevant"])
        p,r,f = precision_recall_f1(ret, rel)
        P+=p; R+=r; F+=f; M+=mrr(ret, rel)
    n=len(cases)
    return dict(precision=P/n, recall=R/n, f1=F/n, mrr=M/n, n=n)
```

### Cell 2 — Build a labeled test set from YOUR chunks

We label which chunk_id is the "correct" answer for each query, then run real
retrieval and score it.

```python
# Helper: retrieve chunk indices for a query (reuses your working titan_embed)
import numpy as np
from collections import defaultdict

def rrf(lists, k=60):
    s = defaultdict(float)
    for lst in lists:
        for rank, idx in enumerate(lst):
            s[idx] += 1.0/(k+rank)
    return sorted(s.items(), key=lambda x: x[1], reverse=True)

def retrieve_ids(q, top_k=5):
    bm = list(np.argsort(bm25.get_scores(_tok(q)))[::-1][:20])
    qv = titan_embed([q]); faiss.normalize_L2(qv)
    _, fi = index.search(qv, 20)
    fused = rrf([bm, [int(i) for i in fi[0]]])
    return [chunks[idx].chunk_id for idx, _ in fused[:top_k]]

# Find the chunk_ids we EXPECT for each query (the ground truth)
def find_chunk(substring):
    return [c.chunk_id for c in chunks if substring.lower() in c.text.lower()]

cases = []
for q, needle in [
    ("What was Enterprise revenue in Q3?", "Enterprise = 2100"),   # the enterprise table row
    ("Net Revenue Retention rate",         "NRR"),
    ("Sales and Marketing budget variance","Sales & Marketing"),
]:
    relevant = set(find_chunk(needle))
    retrieved = retrieve_ids(q, top_k=5)
    cases.append({"query": q, "retrieved": retrieved, "relevant": relevant})
    print(f"{q[:40]:42} relevant={len(relevant)} retrieved_top5={len(retrieved)}")

print("\nEVAL:", evaluate(cases, k=5))
```

**What to observe:** Precision/Recall/F1/MRR on real retrieval. MRR near 1.0
means the right chunk is ranking at or near the top.

### Cell 3 — TTL Cache with REAL Bedrock latency

```python
import hashlib, time

class TTLCache:
    def __init__(self, ttl=3600):
        self.ttl=ttl; self.store={}; self.hits=0; self.misses=0
    def _k(self,q): return hashlib.sha256(q.strip().lower().encode()).hexdigest()
    def get(self,q):
        e=self.store.get(self._k(q))
        if not e or time.time()>e[0]:
            self.misses+=1; return None
        self.hits+=1; return e[1]
    def set(self,q,v): self.store[self._k(q)]=(time.time()+self.ttl, v)

cache = TTLCache()

def full_pipeline(q):
    """The real cold path: retrieve + Titan + Claude."""
    ids = retrieve_ids(q, top_k=5)
    ctx = [c.text for c in chunks if c.chunk_id in ids]
    return claude_generate(SYSTEM, build_prompt(q, ctx, []))

q = "What is the Net Revenue Retention rate?"

# COLD — full pipeline with real Bedrock calls
t0 = time.perf_counter()
ans = cache.get(q)
if ans is None:
    ans = full_pipeline(q); cache.set(q, ans)
cold_ms = (time.perf_counter()-t0)*1000

# WARM — cache hit
t1 = time.perf_counter()
_ = cache.get(q)
warm_ms = (time.perf_counter()-t1)*1000

print(f"Cold (real Bedrock): {cold_ms:10.1f} ms")
print(f"Warm (cache hit)   : {warm_ms:10.4f} ms")
print(f"Speedup            : {cold_ms/warm_ms:,.0f}x")
print(f"hit_rate           : {cache.hits/(cache.hits+cache.misses):.2f}")
```

**What to observe:** the cold call is now REAL — a full Titan + Claude round
trip, typically 1500-3000 ms. The cache hit is microseconds. Your actual
speedup will likely be **10,000x+**, dwarfing the workshop's "1000x" claim.

### Cell 4 — Conversation memory

```python
from collections import deque

class ConversationMemory:
    def __init__(self, max_turns=10): self.turns=deque(maxlen=max_turns)
    def add(self,u,a): self.turns.append((u,a))
    def as_context(self): return "\n".join(f"User: {u}\nAssistant: {a}" for u,a in self.turns)
    def __len__(self): return len(self.turns)

mem = ConversationMemory(max_turns=10)
mem.add(q, ans[:80])
mem.add("Is that above target?", "Yes, 112% vs 110% target.")
print(f"Memory holds {len(mem)} turns:\n{mem.as_context()}")
```

---

**Verify:** Cell 2 gives real F1/MRR; Cell 3 shows the real cold-vs-warm speedup
(likely 10,000x+). Then tell me to build **Step 5: Observability & Tracing** —
per-stage latency spans, the full breakdown (retrieve vs embed vs LLM), and
bottleneck identification. That's the final core step before Agentic RAG.
```
