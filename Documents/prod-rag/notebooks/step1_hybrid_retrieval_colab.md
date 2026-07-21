# Step 1 — Hybrid Retrieval + Reranking — Colab Verification

Run these cells top to bottom in Google Colab. The first cell installs deps and
sets Bedrock credentials; the rest exercise real Titan embeddings + FAISS +
BM25 + RRF + a real cross-encoder.

---

### Cell 1 — Install

```python
!pip -q install boto3 faiss-cpu rank-bm25 sentence-transformers pdfplumber reportlab numpy
```

### Cell 2 — AWS credentials + provider = bedrock

```python
import os
os.environ["AWS_ACCESS_KEY_ID"]     = "YOUR_KEY"       # or use Colab secrets
os.environ["AWS_SECRET_ACCESS_KEY"] = "YOUR_SECRET"
os.environ["AWS_REGION"]            = "us-east-1"
os.environ["RAG_PROVIDER"]          = "bedrock"        # <-- turns on Titan + Claude

# Sanity: confirm Bedrock sees the models you expect
import boto3
bedrock = boto3.client("bedrock", region_name=os.environ["AWS_REGION"])
ids = [m["modelId"] for m in bedrock.list_foundation_models()["modelSummaries"]]
print("titan-embed-text-v2 available:", any("titan-embed-text-v2" in i for i in ids))
print("claude-3-5-sonnet available:", any("claude-3-5-sonnet" in i for i in ids))
```

### Cell 3 — Get the project code

```python
# Option A: upload the prod-rag.zip via the Files panel, then:
!unzip -q prod-rag.zip -d .
%cd prod-rag

# Option B: if you've pushed to GitHub:
# !git clone https://github.com/<you>/prod-rag.git && cd prod-rag
```

### Cell 4 — Generate sample PDFs (once)

```python
!python -m scripts.make_sample_pdfs
```

Expected:
```
wrote acme_10k_fy2024.pdf ... bytes
wrote saas_metrics_memo_q4.pdf ... bytes
```

### Cell 5 — Verify a single Titan embedding call

```python
from src.prodrag.providers import get_embedder
emb = get_embedder()
v = emb.embed(["deferred revenue under ASC 606"])
print("provider = bedrock, embedding dim =", len(v[0]))   # expect 1024
```

### Cell 6 — Ingest + build the hybrid index

```python
from src.prodrag.ingestion import ingest_dir
from src.prodrag.retrieval import HybridRetriever

chunks = ingest_dir()
print("ingested chunks:", len(chunks))

retriever = HybridRetriever(chunks)     # embeds all chunks via Titan, builds FAISS
print("FAISS dim:", retriever._dim,
      "| reranker:", "model" if retriever.reranker.model else "lexical-fallback")
```

Expected: `reranker: model` (the real cross-encoder loaded), FAISS dim `1024`.

### Cell 7 — Run the three test queries

```python
for q in ["How is revenue recognized under ASC 606?",
          "What was Enterprise revenue in Q3?",
          "What is the Net Revenue Retention rate?"]:
    print("Q:", q)
    for h in retriever.retrieve(q, top_k=3):
        print(f"   [{h.chunk.chunk_type:9}] {h.rerank_score:5.2f}  "
              f"{h.chunk.text[:70].replace(chr(10),' ')}")
    print()
```

**What to check on Bedrock (vs the sandbox):**
- "Enterprise revenue in Q3" should now rank the **table_row** with `Q3 = 2200`
  at or near the top — the real cross-encoder handles this far better than the
  offline lexical fallback.
- "Net Revenue Retention" should surface the SaaS memo / NRR table row even
  though the query says "Net Revenue Retention" and the doc says "NRR" — that's
  Titan's semantic embedding doing its job (the hashing stand-in can't).

### Cell 8 — Persist the index (so we don't re-embed next step)

```python
retriever.save()
print("saved FAISS index + chunks to data/faiss_index/")
```

---

Once Cells 5–7 look right on your Bedrock account, tell me and we'll build
**Step 2: Source Confidence Scoring** on top of these retrieval hits.
