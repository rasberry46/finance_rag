# prod-rag — Production RAG on AWS Bedrock

A real, runnable production RAG pipeline for financial documents — the system the
"10-step RAG" workshop *described* but shipped only as a basic chatbot. Every
stage here is real code, tested, and runs on **AWS Bedrock** (Titan embeddings +
Claude), with an offline `local` provider so the whole thing runs and tests
without credentials.

## The 5 core steps (+ agentic layer coming)
- [x] **Step 1 — Hybrid Retrieval**: BM25 + FAISS(ANN) + RRF + cross-encoder rerank
- [x] **Step 2 — Confidence Scoring**: 0.5·relevance + 0.3·trust + 0.2·freshness, threshold gate
- [x] **Step 3 — Constrained Generation**: deterministic math + citations + hallucination guard + fallback
- [x] **Step 4 — Evaluation & Caching**: P/R/F1, MRR, Hit@k, TTL cache, conversation memory
- [x] **Step 5 — Observability**: per-stage trace spans, latency breakdown, bottleneck, p50/p95
- [x] **Agentic layer (LangGraph)**: supervisor routes query, self-corrects, hard loop cap
- [x] **Continuous S3 ingestion** (architecture + stub): see `docs/ARCHITECTURE.md`

## Quickstart in VS Code

```bash
# 1. create + activate a virtualenv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install
pip install -r requirements.txt

# 3. generate the sample financial PDFs
python -m scripts.make_sample_pdfs

# 4a. run the FULL pipeline OFFLINE (no AWS needed — uses local stand-ins)
python -m src.prodrag.pipeline

# 4b. run it on REAL Bedrock
cp .env.example .env               # then edit .env with your AWS keys
export $(cat .env | xargs)         # or set RAG_PROVIDER=bedrock + AWS creds
python -m src.prodrag.pipeline

# 5. run the tests (offline, ~0.5s)
pytest -q
```

## Individual step demos
Each module runs standalone to show that step in isolation:
```bash
python -m src.prodrag.retrieval        # Step 1
python -m src.prodrag.confidence       # Step 2
python -m src.prodrag.generation       # Step 3
python -m src.prodrag.evaluation       # Step 4
python -m src.prodrag.observability    # Step 5
python -m src.prodrag.s3_ingestion     # continuous-ingestion dedup logic
```

## Provider switch
Set `RAG_PROVIDER`:
- `local` (default) — offline stand-ins (hashing embedder, rule-based LLM). For CI / sandbox / tests.
- `bedrock` — real Titan embeddings + Claude via `bedrock-runtime`. Needs AWS creds.

Everything else is identical between the two — only `providers.py` swaps.

## Layout
```
src/prodrag/
  config.py         # models, paths, tunable knobs
  providers.py      # Bedrock (Titan/Claude) + local fallbacks
  ingestion.py      # PDF → table-aware chunks (pdfplumber)
  retrieval.py      # Step 1: BM25 + FAISS + RRF + rerank
  confidence.py     # Step 2: trust/freshness scoring
  generation.py     # Step 3: deterministic math + guard
  evaluation.py     # Step 4: metrics + cache + memory
  observability.py  # Step 5: tracing + p50/p95
  s3_ingestion.py   # continuous S3 ingestion worker (stub)
  pipeline.py       # end-to-end orchestrator (ProductionRAG.ask)
scripts/make_sample_pdfs.py
notebooks/          # per-step Colab verification guides
docs/ARCHITECTURE.md   # batch → continuous S3 ingestion
tests/              # 16 tests, run offline
```

## Colab
`notebooks/step{1..5}_*_colab.md` walk through verifying each step on real
Bedrock, cell by cell with expected output.

## Notes on the sample data
Two generated PDFs — a mini 10-K (segment revenue tables, deferred revenue,
opex-vs-budget) and a SaaS metrics memo (ARR/NRR/GRR). They exercise the
table-aware parsing and the finance-specific reasoning.
