# Step 5 — Observability & Tracing — Colab

Wraps your REAL pipeline (Titan + FAISS + Claude) in trace spans so you can see
exactly where time goes. Builds on Steps 1–4.

---

### Cell 1 — The Tracer

```python
import time
from contextlib import contextmanager

class Tracer:
    def __init__(self): self.spans = []
    @contextmanager
    def span(self, name):
        t = time.perf_counter()
        try: yield
        finally: self.spans.append((name, (time.perf_counter()-t)*1000))
    def summary(self):
        total = sum(ms for _, ms in self.spans)
        print(f"{'stage':<24}{'ms':>10}{'%':>8}")
        print("-"*42)
        for name, ms in self.spans:
            print(f"{name:<24}{ms:>10.1f}{ms/total*100:>7.1f}%")
        print("-"*42)
        print(f"{'TOTAL':<24}{total:>10.1f}{'100%':>8}")
        bn = max(self.spans, key=lambda x: x[1])
        print(f"\n🔎 bottleneck: {bn[0]} ({bn[1]:.0f} ms, {bn[1]/total*100:.0f}%)")
```

### Cell 2 — Trace one real end-to-end request

```python
import numpy as np
from collections import defaultdict

def rrf(lists, k=60):
    s = defaultdict(float)
    for lst in lists:
        for rank, idx in enumerate(lst):
            s[idx] += 1.0/(k+rank)
    return sorted(s.items(), key=lambda x: x[1], reverse=True)

q = "Explain the Sales & Marketing spend variance versus budget."
tracer = Tracer()

# 1. cache lookup (miss, first time)
with tracer.span("cache_lookup"):
    cached = cache.get(q) if 'cache' in dir() else None

# 2. embed the query (real Titan call)
with tracer.span("query_embedding"):
    qv = titan_embed([q]); faiss.normalize_L2(qv)

# 3. hybrid retrieval (BM25 + FAISS + RRF)
with tracer.span("hybrid_retrieval"):
    bm = list(np.argsort(bm25.get_scores(_tok(q)))[::-1][:20])
    _, fi = index.search(qv, 20)
    fused = rrf([bm, [int(i) for i in fi[0]]])
    top = [chunks[idx] for idx, _ in fused[:5]]

# 4. build prompt
with tracer.span("prompt_build"):
    ctx = [c.text for c in top]
    user = build_prompt(q, ctx, [comp])

# 5. LLM generation (real Claude call)
with tracer.span("llm_generation"):
    raw = claude_generate(SYSTEM, user)

# 6. hallucination guard
with tracer.span("hallucination_guard"):
    facts_text = comp if isinstance(comp, str) else comp.as_fact()
    risk = hallucination_risk(raw, ctx, facts_text)

tracer.summary()
```

**What you'll see:** `llm_generation` dominates — typically 85-95% of total time.
`query_embedding` is the second cost (the Titan round trip). BM25/FAISS/RRF are
near-instant. This is why caching the whole query (Step 4) matters: a cache hit
skips the entire ~2s bar.

### Cell 3 — Trace a CACHED request (see the bottleneck vanish)

```python
# Save the answer, then trace a repeat request
cache.set(q, raw)

tracer2 = Tracer()
with tracer2.span("cache_lookup"):
    hit = cache.get(q)      # HIT — returns instantly
with tracer2.span("(everything else skipped)"):
    pass

tracer2.summary()
print("\nCache hit returned:", hit[:60], "...")
```

**What you'll see:** total drops from ~2000 ms to a fraction of a millisecond.
The bottleneck (LLM) is gone entirely because we never call it. This is the
"1000x" from Step 4, now visible in the trace.

### Cell 4 — Aggregate across several requests (p50/p95)

```python
from statistics import median

def p95(v): 
    s = sorted(v); return s[min(len(s)-1, int(len(s)*0.95))]

by_stage = defaultdict(list)
for query in [
    "Explain the Sales & Marketing spend variance versus budget.",
    "What is the Net Revenue Retention rate?",
    "What was Enterprise revenue in Q3?",
]:
    tr = Tracer()
    with tr.span("query_embedding"):
        qv = titan_embed([query]); faiss.normalize_L2(qv)
    with tr.span("hybrid_retrieval"):
        bm = list(np.argsort(bm25.get_scores(_tok(query)))[::-1][:20])
        _, fi = index.search(qv, 20)
        fused = rrf([bm, [int(i) for i in fi[0]]])
        top = [chunks[idx] for idx, _ in fused[:5]]
    with tr.span("llm_generation"):
        _ = claude_generate(SYSTEM, build_prompt(query, [c.text for c in top], []))
    for name, ms in tr.spans:
        by_stage[name].append(ms)

print(f"{'stage':<24}{'p50':>10}{'p95':>10}")
print("-"*44)
for stage, vals in by_stage.items():
    print(f"{stage:<24}{median(vals):>10.1f}{p95(vals):>10.1f}")
```

**What you'll see:** stable p50/p95 per stage — the numbers you'd put on a
dashboard or SLO ("95% of queries answer in under X ms").

---

**Verify:** Cell 2 shows LLM as the bottleneck (~90%); Cell 3 shows it vanish on
a cache hit. That completes the **5 core production steps**. Then tell me to
build the **Agentic RAG layer (LangGraph)** — a supervisor that decides whether
a query needs retrieval, computation, or both, and routes accordingly. That's
the piece that maps directly to your Intuit multi-agent work.
```
