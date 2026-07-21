# Step 2 — Source Confidence Scoring — Colab Verification

Builds on Step 1's retriever. Assumes credentials are already set from Step 1
(Colab Secrets). Run in order.

---

### Cell 1 — Assign source types + dates to chunks

In production these come from ingestion metadata. Here we tag the two sample
docs differently so the scoring actually discriminates.

```python
from datetime import datetime, timezone, timedelta
now = datetime.now(timezone.utc)

for c in chunks:  # `chunks` from Step 1
    if c.doc_id.startswith("acme_10k"):
        c.metadata["source_type"] = "audited_filing"   # high trust
        c.metadata["doc_date"]    = now - timedelta(days=20)    # fresh
    else:
        c.metadata["source_type"] = "internal_memo"    # medium trust
        c.metadata["doc_date"]    = now - timedelta(days=200)   # older

print("✅ tagged", len(chunks), "chunks with source_type + doc_date")
```

### Cell 2 — The confidence scorer

```python
from src.prodrag.config import CONFIG
from src.prodrag.confidence import (
    filter_by_confidence, score_hits, TRUST_BY_SOURCE, freshness_score
)

print("Weights:", CONFIG.w_relevance, CONFIG.w_trust, CONFIG.w_freshness,
      "| threshold:", CONFIG.confidence_threshold)
print("Trust table:", TRUST_BY_SOURCE)
```

### Cell 3 — Score a query's retrieval hits

Uses the `retriever` from Step 1 (real Titan + FAISS + cross-encoder).

```python
hits = retriever.retrieve("What is the Net Revenue Retention rate?", top_k=5)
scored = filter_by_confidence(hits)

print(f"{'conf':>5} {'rel':>5} {'trust':>5} {'fresh':>5}  source          preview")
print("-"*85)
for s in scored:
    st = s.hit.chunk.metadata.get("source_type","?")
    print(f"{s.confidence:5.2f} {s.relevance:5.2f} {s.trust:5.2f} {s.freshness:5.2f}"
          f"  {st:15} {s.text[:35].replace(chr(10),' ')}")
```

**What to observe:**
- The `audited_filing` chunks get a **trust + freshness boost** (0.98 / 1.0).
- The `internal_memo` — even at relevance 1.0 — is penalized by age (freshness
  0.4) and lower trust (0.75), so its confidence lands lower.
- Ordering now reflects **trustworthiness**, not raw similarity.

### Cell 4 — Demonstrate the threshold dropping a low-confidence source

```python
# Simulate a low-trust, stale chat message that's highly relevant
from src.prodrag.ingestion import Chunk
from src.prodrag.retrieval import RetrievalHit

fake = Chunk(text="Someone on Slack said NRR is maybe 112 percent",
             chunk_type="prose", doc_id="slack",
             metadata={"source":"slack","source_type":"chat_message",
                       "doc_date": now - timedelta(days=500)})
fake_hit = RetrievalHit(chunk=fake, rerank_score=99.0, rrf_rank=0)  # very "relevant"

mixed = hits + [fake_hit]
kept = filter_by_confidence(mixed, threshold=0.5)   # raise bar to 0.5
print("Kept after threshold 0.5:")
for s in kept:
    print(f"  conf={s.confidence:.2f}  {s.hit.chunk.metadata['source_type']:15} "
          f"{s.text[:40]}")
print("\nNote: the highly-relevant Slack rumor is DROPPED — low trust + stale.")
```

---

**Verify:** confidence column is sorted descending, audited filing scores highest,
and the Slack rumor gets filtered at threshold 0.5. Then tell me to build
**Step 3: Constrained Generation, Citations & Hallucination Detection** — where
the deterministic-math-then-narrate principle lives, and where we make the FIRST
real Claude call.
