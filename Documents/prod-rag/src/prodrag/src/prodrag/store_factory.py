"""
store_factory.py — Pick the retrieval backend by env flag
RAG_STORE=opensearch -> live OpenSearch domain (production)
RAG_STORE=faiss      -> in-memory FAISS (offline fallback / demo safety net)
Both expose the same .retrieve(query, top_k) interface, so downstream is unchanged.
"""
from __future__ import annotations
import os
from datetime import datetime, timezone, timedelta


def get_store():
    backend = os.environ.get("RAG_STORE", "opensearch").lower()

    if backend == "opensearch":
        from .opensearch_store import OpenSearchStore
        return OpenSearchStore()

    # FAISS fallback
    from .ingestion import ingest_dir
    from .retrieval import HybridRetriever
    chunks = ingest_dir()
    now = datetime.now(timezone.utc)
    for c in chunks:
        if c.doc_id.startswith("acme_10k"):
            c.metadata.update(source_type="audited_filing", doc_date=now - timedelta(days=20))
        else:
            c.metadata.update(source_type="internal_memo", doc_date=now - timedelta(days=200))
    return HybridRetriever(chunks)