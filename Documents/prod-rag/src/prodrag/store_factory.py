from __future__ import annotations
import os
from datetime import datetime, timezone, timedelta


def get_store():
    backend = os.environ.get("RAG_STORE", "opensearch").lower()

    if backend == "opensearch":
        from .opensearch_store import OpenSearchStore
        from .confidence_store import ConfidenceStore
        return ConfidenceStore(OpenSearchStore())

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
