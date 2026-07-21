# 2-Day Plan — Optimized for THIS JD (Senior AI Engineer, GP/Altimetrik)

Interview in 2 days. The 7-feature wishlist is real, but 2 days means we build
what the JD **names** and what the **demo shows**, and we speak to the rest as
architecture. Every JD must-have is already covered; this plan closes the
highest-value gaps and makes the demo bulletproof.

Rule: build → verify → `git commit` → next. Keep the FAISS + known-good commit
as a fallback at every step. A working 5-feature demo beats a broken 7.

---

## What the JD rewards most (why this order)
- "AI agents, RAG, **natural language interfaces**, workflow automation" → agentic + UI
- "beyond proof-of-concepts or **chatbot implementations**" → deterministic math + guard
- "**Snowflake** / enterprise data" → a safe text-to-SQL piece (named skill: SQL)
- "**cloud: AWS**" → live Bedrock + OpenSearch (already done)
- "translate financial processes" → STAR stories (Day 2 evening)

---

## DAY 1 — Real backend + the JD's named skills

### 1.1 — Environment (VS Code) — 30 min  ⭐ FIRST
Follow `docs/VSCODE_SETUP.md`. venv + `.env` + `pytest -q` green. This ends the
Colab thrashing. Do NOT skip — it saves hours over the next 2 days.

### 1.2 — Live OpenSearch integration — 1 hr
Wire `opensearch_store.py` into the app behind `RAG_STORE=opensearch|faiss`.
Your domain is already green + populated (2,216 chunks). Just reconnect.
- Verify: `answer("total loans outstanding")` → cited 84,000 / 77,000.
- Commit.

### 1.3 — AWS Textract — 1.5 hr
Fixes the "total assets" gap; a NAMED enterprise-AWS skill.
- Wire `textract_parser.py` async path; re-ingest 2 PDFs; re-index.
- Verify: "total for assets" now returns a real number.
- Commit.

### 1.4 — Agentic RAG over OpenSearch — 1 hr
Point `agentic.py` retrieve node at OpenSearch. This is the JD's "AI agents."
- Verify: all 4 routes work on real financial docs; trace shows the path.
- Commit.

### 1.5 — Text-to-SQL safety piece — 1.5 hr  ⭐ JD-SPECIFIC
The JD names Snowflake + SQL. Build a small "NL → validated read-only SQL →
results" module: natural language question → LLM generates SQL → a guard blocks
writes/DDL and enforces read-only → run against a tiny SQLite/DuckDB finance
table → return results. Speak to it as "the same pattern I'd use over Snowflake
with row-level security and a SQL linter."
- Verify: "show total assets by category" → safe SELECT → result; a "DROP TABLE"
  attempt is blocked.
- Commit.

---

## DAY 2 — Intelligence, proof, polish

### 2.1 — CRAG (corrective loop) — 1 hr
Upgrade the grade node: on bad grade → rewrite query + re-retrieve (not just
retry). You're ~70% there.
- Verify: partially-answerable question → watch it rewrite and improve.
- Commit.

### 2.2 — Multi-Agent RAG — 1.5 hr
Split into retriever / analyst / critic agents under the supervisor.
- Verify: trace shows handoffs; critic catches a bad answer.
- Commit.

### 2.3 — RAGAS-style evaluation — 1 hr
faithfulness + answer_relevancy + context_precision on a 10-question golden set
built from your PDFs. Proves "how do you know it works."
- Verify: metrics table prints, scores sensible.
- Commit.

### 2.4 — Streamlit UI on live backend — 1 hr
Point `demo_app.py` at OpenSearch. Test all buttons. This IS the JD's "natural
language interface." Screen-share ready.
- Commit.

### 2.5 — Monitoring (light) — 45 min
Push per-stage latency + cache-hit-rate to CloudWatch custom metrics. One
dashboard screenshot for the demo. (Grafana optional — CloudWatch is faster.)
- Commit.

### 2.6 — Full dry run + STAR stories — evening
- End-to-end demo dry run; fix anything that stutters.
- Draft 2-3 STAR stories: "translated an FP&A process into an agent,"
  "prevented a hallucination in a financial workflow," "shipped multi-agent
  LangGraph to production at Intuit." (We'll do these together.)

---

## Deferred to "here's what I'd add next" (don't build, just speak to)
- **Continuous S3→SQS ingestion** — you have the stub + architecture doc. Say:
  "batch today; event-driven with S3 notifications → SQS for near-real-time."
  (Skipped because it needs infra setup with low demo payoff vs. the above.)
- **Grafana dashboards** — CloudWatch covers the monitoring story faster.
- **Prompt versioning** — "I'd use a prompt registry / LangSmith."

## Demo-day checklist
- [ ] `.env` loaded, `pytest -q` green
- [ ] OpenSearch green + `answer()` returns cited figures
- [ ] Textract fixed the assets question
- [ ] Streamlit opens, all buttons work, agentic trace visible
- [ ] text-to-SQL blocks a destructive query live
- [ ] RAGAS table prints
- [ ] FAISS fallback confirmed working (`RAG_STORE=faiss`)
- [ ] 3 STAR stories rehearsed
