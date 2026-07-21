"""
s3_loader.py  —  Load PDFs directly from an S3 bucket into the pipeline
======================================================================
Reads PDFs from s3://<bucket>/<prefix>, runs them through the same table-aware
ingestion used for local files, and returns Chunks ready for the retriever.

This is the DIRECT-READ path (batch): list bucket -> download -> parse -> chunk.
It's a real S3 integration you can demo live. The event-driven version
(S3 event -> SQS -> worker) lives in s3_ingestion.py; this is the simpler
foundation both share.

Usage:
    from src.prodrag.s3_loader import ingest_s3
    chunks = ingest_s3("rag81", prefix="pdfs/")           # local creds/env
    # or pass explicit creds (e.g. from Colab userdata):
    chunks = ingest_s3("rag81", prefix="pdfs/", key=KEY, secret=SECRET)
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from .config import CONFIG
from .ingestion import ingest_pdf, Chunk


def _s3_client(region: str, key: str | None, secret: str | None):
    import boto3
    kwargs = {"region_name": region}
    if key and secret:
        kwargs["aws_access_key_id"] = key
        kwargs["aws_secret_access_key"] = secret
    return boto3.client("s3", **kwargs)


def list_s3_pdfs(bucket: str, prefix: str = "", region: str = None,
                 key: str = None, secret: str = None) -> list[str]:
    """Return the S3 keys of all PDFs under bucket/prefix (handles pagination)."""
    region = region or CONFIG.aws_region
    s3 = _s3_client(region, key, secret)
    keys: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for o in resp.get("Contents", []):
            if o["Key"].lower().endswith(".pdf"):
                keys.append(o["Key"])
        if resp.get("IsTruncated"):
            token = resp["NextContinuationToken"]
        else:
            break
    return keys


def _infer_source_type(key: str) -> str:
    """Heuristic: financial statements are audited filings; everything else a memo.
    In production this comes from S3 object tags or a metadata table."""
    k = key.lower()
    if any(w in k for w in ["10-k", "10k", "annual", "financial-statement",
                            "financial_statement", "statements", "filing"]):
        return "audited_filing"
    if any(w in k for w in ["memo", "metrics", "internal"]):
        return "internal_memo"
    return "official_docs"


def ingest_s3(bucket: str, prefix: str = "", region: str = None,
              key: str = None, secret: str = None,
              progress=None) -> list[Chunk]:
    """
    Download every PDF under bucket/prefix and parse it into Chunks.

    progress: optional callable(msg:str) for UI feedback (e.g. Streamlit).
    """
    region = region or CONFIG.aws_region
    s3 = _s3_client(region, key, secret)
    pdf_keys = list_s3_pdfs(bucket, prefix, region, key, secret)

    if progress:
        progress(f"Found {len(pdf_keys)} PDF(s) in s3://{bucket}/{prefix}")

    all_chunks: list[Chunk] = []
    now = datetime.now(timezone.utc)

    for pk in pdf_keys:
        if progress:
            progress(f"Downloading + parsing {pk} …")
        obj = s3.get_object(Bucket=bucket, Key=pk)
        data = obj["Body"].read()
        last_modified = obj.get("LastModified", now)

        # ingest_pdf reads from a path, so write to a temp file first.
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        source_type = _infer_source_type(pk)
        try:
            chunks = ingest_pdf(tmp_path, source_type=source_type,
                                doc_date=last_modified)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Tag provenance so confidence scoring + citations know the S3 origin.
        for c in chunks:
            c.metadata["s3_key"] = pk
            c.metadata["source"] = pk.rsplit("/", 1)[-1]
            c.metadata.setdefault("source_type", source_type)
            c.metadata.setdefault("doc_date", last_modified)
        all_chunks.extend(chunks)
        if progress:
            progress(f"  → {len(chunks)} chunks from {pk}")

    if progress:
        progress(f"Total: {len(all_chunks)} chunks from {len(pdf_keys)} document(s)")
    return all_chunks


if __name__ == "__main__":
    import os
    bucket = os.environ.get("S3_BUCKET", "rag81")
    prefix = os.environ.get("S3_PREFIX", "pdfs/")
    print(f"Listing s3://{bucket}/{prefix} …")
    try:
        keys = list_s3_pdfs(bucket, prefix)
        print(f"PDFs: {keys}")
        chunks = ingest_s3(bucket, prefix, progress=print)
        types = {}
        for c in chunks:
            types[c.chunk_type] = types.get(c.chunk_type, 0) + 1
        print("\nChunk types:", types)
        # show one table chunk if present
        for c in chunks:
            if c.chunk_type == "table":
                print("\nExample table chunk:\n", c.text[:300])
                break
    except Exception as e:
        print("Error:", type(e).__name__, e)
        print("If AccessDenied: attach AmazonS3ReadOnlyAccess to the IAM user.")
