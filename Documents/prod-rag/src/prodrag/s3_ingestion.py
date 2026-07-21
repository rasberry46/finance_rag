"""
s3_ingestion.py  —  Continuous document ingestion from S3 (architecture stub)
=============================================================================
This shows HOW the batch prototype becomes a production, event-driven pipeline
when documents land in S3 continuously. It is a STUB: the AWS wiring is real
boto3 code, but it needs actual S3/SQS/OpenSearch infrastructure to run. Its
purpose is to make the architecture concrete and speakable.

THE KEY INSIGHT
---------------
Retrieval + generation (the code you already built) DON'T change. What changes
is how documents GET IN. You decouple ingestion from serving:

    New PDF -> S3 -> (S3 event) -> SQS -> ingestion worker -> vector DB
                                                                   ^
    query -> retrieve() ------------------------------------------ /
             (always hits the live, continuously-updated index)

FOUR CHANGES FROM THE BATCH PROTOTYPE
-------------------------------------
1. FAISS-in-memory  ->  managed vector DB (OpenSearch Serverless / Pinecone /
   pgvector). Supports live upserts, persistence, multi-writer, horizontal
   scale. retrieve() logic is unchanged; only the backend swaps.

2. Manual ingest_dir()  ->  event-driven. S3 emits ObjectCreated events; a
   Lambda or SQS-backed worker processes each new file. Use SQS (not raw
   Lambda) for big PDFs, since Textract can exceed Lambda's 15-min limit.

3. Idempotency + dedup. Hash each document (SHA-256). Skip if that exact version
   is already indexed. On update, delete old chunks by doc_id, re-insert new.

4. Incremental indexing. Embed only the new/changed doc's chunks and upsert.
   BM25 needs corpus-wide stats, so let the vector DB do hybrid search:
   OpenSearch does BM25 + kNN natively in one continuously-updated store.

This module sketches the worker. Swap the OpenSearch calls for your chosen DB.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

# NOTE: boto3 import is deferred so the rest of the repo runs without AWS.


# ----------------------------------------------------------------------------
# Content-hash dedup
# ----------------------------------------------------------------------------
def document_fingerprint(content: bytes) -> str:
    """SHA-256 of raw bytes. Same file -> same fingerprint -> skip re-embedding."""
    return hashlib.sha256(content).hexdigest()


@dataclass
class IngestDecision:
    doc_id: str
    fingerprint: str
    action: str  # "index_new" | "reindex_changed" | "skip_duplicate"


def decide_action(doc_id: str, fingerprint: str,
                  known_fingerprints: dict[str, str]) -> IngestDecision:
    """Compare incoming fingerprint against what's already indexed for this doc_id."""
    existing = known_fingerprints.get(doc_id)
    if existing is None:
        return IngestDecision(doc_id, fingerprint, "index_new")
    if existing != fingerprint:
        return IngestDecision(doc_id, fingerprint, "reindex_changed")
    return IngestDecision(doc_id, fingerprint, "skip_duplicate")


# ----------------------------------------------------------------------------
# The ingestion worker (pulls from SQS, processes, upserts)
# ----------------------------------------------------------------------------
class S3IngestionWorker:
    """
    Event-driven worker. In production this runs as a container (ECS/Fargate) or
    a long-polling process, pulling S3-event messages from SQS.

    Requires (uncomment in requirements.txt): boto3, opensearch-py
    """

    def __init__(self, region: str = "us-east-1",
                 queue_url: str | None = None,
                 opensearch_endpoint: str | None = None,
                 index_name: str = "financial-rag"):
        import boto3  # deferred
        self.s3 = boto3.client("s3", region_name=region)
        self.sqs = boto3.client("sqs", region_name=region)
        self.textract = boto3.client("textract", region_name=region)
        self.bedrock = boto3.client("bedrock-runtime", region_name=region)
        self.queue_url = queue_url
        self.opensearch_endpoint = opensearch_endpoint
        self.index_name = index_name
        self._known_fingerprints: dict[str, str] = {}  # in prod: a metadata table

    # --- 1. embed via Titan (same model as the prototype) ---
    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            resp = self.bedrock.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({"inputText": t}))
            out.append(json.loads(resp["body"].read())["embedding"])
        return out

    # --- 2. parse a PDF from S3 (Textract for scanned, pdfplumber for digital) ---
    def parse_s3_pdf(self, bucket: str, key: str) -> list[dict]:
        """Returns chunk dicts. Reuse ingestion.py's table-aware logic here.
        For scanned docs use Textract async (start_document_analysis -> S3 ->
        get_document_analysis); for digital PDFs pdfplumber is cheaper."""
        raise NotImplementedError(
            "Wire to ingestion.ingest_pdf (digital) or textract async (scanned). "
            "Returns [{text, chunk_type, doc_id, metadata}, ...].")

    # --- 3. upsert into the vector DB ---
    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        """OpenSearch bulk upsert. On reindex, delete-by-query on doc_id first."""
        raise NotImplementedError(
            "OpenSearch: delete_by_query(doc_id) then bulk index "
            "{text, embedding (knn_vector), source_type, doc_date, doc_id}.")

    # --- 4. process one S3 event ---
    def process_event(self, bucket: str, key: str):
        obj = self.s3.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read()
        doc_id = key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        fp = document_fingerprint(content)

        decision = decide_action(doc_id, fp, self._known_fingerprints)
        if decision.action == "skip_duplicate":
            return f"skip {doc_id} (already indexed)"

        chunks = self.parse_s3_pdf(bucket, key)
        embeddings = self.embed([c["text"] for c in chunks])
        self.upsert_chunks(chunks, embeddings)
        self._known_fingerprints[doc_id] = fp
        return f"{decision.action} {doc_id}: {len(chunks)} chunks"

    # --- 5. the long-poll loop (SQS-backed) ---
    def run(self, max_iterations: int = None):
        """Poll SQS for S3 events, process each, delete the message on success.
        max_iterations caps the loop (useful for tests / graceful shutdown)."""
        i = 0
        while max_iterations is None or i < max_iterations:
            resp = self.sqs.receive_message(
                QueueUrl=self.queue_url, MaxNumberOfMessages=10,
                WaitTimeSeconds=20)  # long polling
            for msg in resp.get("Messages", []):
                body = json.loads(msg["Body"])
                for record in body.get("Records", []):
                    bucket = record["s3"]["bucket"]["name"]
                    key = record["s3"]["object"]["key"]
                    print(self.process_event(bucket, key))
                self.sqs.delete_message(QueueUrl=self.queue_url,
                                        ReceiptHandle=msg["ReceiptHandle"])
            i += 1


if __name__ == "__main__":
    # Offline demo of the DEDUP LOGIC (no AWS needed) — the part that keeps a
    # continuous pipeline from re-embedding the same file forever.
    known: dict[str, str] = {}

    for label, content in [
        ("first upload",     b"FY2024 10-K contents v1"),
        ("exact duplicate",  b"FY2024 10-K contents v1"),
        ("updated version",  b"FY2024 10-K contents v2 (restated)"),
    ]:
        fp = document_fingerprint(content)
        d = decide_action("acme_10k", fp, known)
        print(f"{label:18} -> {d.action}")
        if d.action != "skip_duplicate":
            known["acme_10k"] = fp
