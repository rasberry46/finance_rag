# 3-Day Build Plan — from working prototype to full production system

Interview in 2-3 days. Goal: build all 7 features **properly**, one at a time,
each verified before moving on. Order is dependency-driven, not the wishlist order.

Rule for every step: **build → verify it runs → `git commit` → next.** Never
stack an unverified feature on another. If a step fights you >45 min, note it and
move on; a working 6-feature demo beats a broken 7-feature one.

---

## Day 1 — Real backend (unblocks everything)

### 1.1 — Live OpenSearch integration  ⭐ START HERE
Wire `opensearch_store.py` into the app so retrieval reads from the real domain
(already green + populated with 2,216 chunks from tonight).
- Swap `HybridRetriever` → `OpenSearchStore` behind a `RAG_STORE` env flag
  (`opensearch` | `faiss`), so FAISS stays as a fallback.
- Verify: `answer("total loans outstanding")` returns the cited 84,000 / 77,000.
- Commit: "feat: live OpenSearch backend with FAISS fallback"

### 1.2 — AWS Textract for clean multi-column extraction
Fixes the "total assets" gap (pdfplumber flattened the cell).
- Wire `textract_parser.py` async path (start_document_analysis → S3 → get).
- Re-ingest the 2 PDFs via Textract, re-index into OpenSearch.
- Verify: "total for assets" now returns a real number, not the fallback.
- Commit: "feat: Textract table extraction for complex layouts"

---

## Day 2 — Intelligence layer

### 2.1 — Agentic RAG over OpenSearch
Point `agentic.py`'s retrieve node at `OpenSearchStore`.
- Verify: the 4 routes (direct/retrieve/compute/retrieve_compute) all work on
  the real financial docs; the trace shows the path.
- Commit: "feat: agentic routing over OpenSearch"

### 2.2 — CRAG (Corrective RAG)
Extend the grade node: on a bad grade, don't just retry — **correct**.
- If retrieved context scores low → rewrite the query and re-retrieve.
- Optional: if still low → web-search fallback (or mark "outside knowledge base").
- You already have the grade + loop; this upgrades the retry into a correction.
- Verify: ask something partially answerable; watch it rewrite and improve.
- Commit: "feat: CRAG corrective retrieval loop"

### 2.3 — Multi-Agent RAG
Split the monolithic generate into specialized agents under the supervisor:
- **Retriever agent** (fetches + filters), **Analyst agent** (does the math +
  narrative), **Critic agent** (grades/guards). Supervisor orchestrates.
- Verify: trace shows agents handing off; critic catches a bad answer.
- Commit: "feat: multi-agent retriever/analyst/critic graph"

---

## Day 3 — Hardening + proof + polish

### 3.1 — Continuous ingestion (S3 → SQS → worker)
Near-real-time: drop a PDF in S3, it's indexed in seconds.
- Create SQS queue + S3 event notification (console or boto3).
- Run `s3_ingestion.py` worker; demo dropping a file live.
- Verify: upload a PDF → within seconds it's queryable.
- Commit: "feat: event-driven S3→SQS→OpenSearch ingestion"

### 3.2 — CloudWatch / Grafana monitoring
Export the trace spans you already emit.
- Push per-stage latency + cache-hit-rate as CloudWatch custom metrics.
- (Optional) a CloudWatch dashboard screenshot for the demo.
- Verify: metrics appear in CloudWatch after a few queries.
- Commit: "feat: CloudWatch metrics export"

### 3.3 — RAGAS-style evaluation
Add LLM-judged metrics on a small golden set.
- faithfulness, answer_relevancy, context_precision, context_recall.
- Build a 10-question golden set from your financial PDFs (known answers).
- Verify: a metrics table prints; scores are sensible.
- Commit: "feat: RAGAS-style evaluation on golden set"

### 3.4 — Wire the Streamlit UI to OpenSearch + final dry run
- Point `demo_app.py` at the live store; test all demo buttons.
- Full dry run of the demo script end to end. Fix anything that stutters.
- Commit: "chore: demo UI on live backend + dry run"

---

## Demo-day checklist (morning of)
- [ ] `.env` loaded, `pytest -q` green
- [ ] OpenSearch domain reachable (`cluster health` = green)
- [ ] `answer("total loans outstanding")` returns cited figures
- [ ] Streamlit app opens and all 4 example buttons work
- [ ] One clean end-to-end dry run done
- [ ] Know your 3 talking points cold (see cheat sheet)

## If something breaks on demo day
FAISS fallback (`RAG_STORE=faiss`) runs the whole thing offline-capable. You lose
"live OpenSearch" but keep every other feature. Never demo without a fallback.
