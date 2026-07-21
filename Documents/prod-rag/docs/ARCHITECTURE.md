# Architecture: Batch prototype → Continuous S3 ingestion

## The question
"What if I connect this to S3, where documents arrive continuously? How does the
RAG pipeline work then?"

## Short answer
**Retrieval and generation don't change. Ingestion does.** You decouple the two:
documents flow into a continuously-updated vector store via an event-driven
worker, and queries always read the live store.

```
                        ┌─────────────── SERVING (unchanged) ───────────────┐
  query ──▶ cache ──▶ hybrid retrieve ──▶ confidence ──▶ generate ──▶ guard ──▶ answer
                          │
                          ▼ reads
                   ┌──────────────┐
                   │  Vector DB   │  ◀── continuously updated
                   │ (OpenSearch) │
                   └──────────────┘
                          ▲ upserts
                          │
  ┌───────────────── INGESTION (new, event-driven) ─────────────────┐
  new PDF ─▶ S3 ─▶ S3 Event ─▶ SQS ─▶ worker: parse → embed → upsert
```

## What changes vs. the batch prototype

| Concern | Batch prototype (this repo) | Continuous S3 |
|---|---|---|
| Index | FAISS in memory | OpenSearch Serverless / Pinecone / pgvector |
| Trigger | manual `ingest_dir()` | S3 ObjectCreated event → SQS → worker |
| Scope | embed whole corpus | embed only new/changed doc, upsert |
| Dedup | none needed | SHA-256 content hash; skip/reindex |
| Hybrid | BM25 (in-proc) + FAISS | OpenSearch native BM25 + kNN in one store |
| Freshness | static tag | real, from ingestion timestamp |

## Why each change

**FAISS → managed vector DB.** FAISS in a Python process can't take live writes
from multiple workers, doesn't persist cleanly, and won't scale past one box.
A managed store gives live upserts, persistence, and horizontal scale. Your
`retrieve()` logic is unchanged — only the backend swaps.

**Manual run → event-driven.** S3 emits an event on new objects. A Lambda (light
loads) or an SQS-backed worker (heavy loads) processes each file. Prefer SQS for
big PDFs: Textract on a large document can exceed Lambda's 15-minute limit, and
SQS gives you retries + dead-letter queues for free.

**Full → incremental indexing.** Embed only the new document's chunks and upsert
them. Classic BM25 needs corpus-wide term statistics, which is awkward when the
corpus grows continuously — so let OpenSearch do hybrid search (it computes BM25
and kNN over the live index in one query), instead of maintaining an in-process
BM25 object.

**Dedup + idempotency.** Continuous pipelines reprocess the same file (event
retries, duplicate deliveries). Hash each document; skip if that exact version
is already indexed. On update, delete old chunks by `doc_id` then insert new —
which is why every chunk carries `doc_id` in its metadata.

## The one-paragraph interview answer
> For a continuous S3 source I'd decouple ingestion from serving. S3 events
> trigger an async worker — Lambda for small files, SQS-backed for large ones —
> that parses with Textract or pdfplumber, embeds with Titan, and upserts into
> OpenSearch Serverless with content-hash dedup and doc_id-scoped updates.
> Queries always hit the live index, so freshness is automatic. FAISS is great
> for a fixed corpus, but once writes are continuous I'd move to OpenSearch,
> which gives live upserts and native hybrid BM25+kNN in one store.

See `src/prodrag/s3_ingestion.py` for the worker stub (real boto3 wiring; needs
AWS infra to run). Its dedup logic runs offline: `python -m src.prodrag.s3_ingestion`.
