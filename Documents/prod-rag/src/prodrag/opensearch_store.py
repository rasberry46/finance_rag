"""
opensearch_store.py  —  OpenSearch as the vector store (production backend)
==========================================================================
Swaps the in-memory FAISS index for a managed OpenSearch domain that does
BM25 + k-NN vector search NATIVELY in one continuously-updatable store. This is
the production answer to "what if documents arrive continuously from S3?"

Auth: fine-grained access with a MASTER USER (basic auth over HTTPS). Simpler
than SigV4 — you just pass username/password.

What it provides:
  - create_index()  : k-NN index with the right mapping (knn_vector + text fields)
  - upsert_chunks() : embed + index chunks (delete-by-doc_id first on reindex)
  - hybrid_search() : BM25 + vector kNN combined, returns RetrievalHit list
                      → same interface as HybridRetriever, so the rest of the
                        pipeline (confidence, generation, agent) is UNCHANGED.

Requires: opensearch-py
    pip install opensearch-py

Env / args:
    OPENSEARCH_ENDPOINT = https://search-rag-demo-xxxx.us-east-1.es.amazonaws.com
    OPENSEARCH_USER, OPENSEARCH_PASSWORD  (the master user you created)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .config import CONFIG
from .providers import get_embedder
from .ingestion import Chunk
from .retrieval import RetrievalHit


INDEX_NAME = os.environ.get("OPENSEARCH_INDEX", "financial-rag")


def _client(endpoint: str = None, user: str = None, password: str = None):
    from opensearchpy import OpenSearch
    endpoint = endpoint or os.environ["OPENSEARCH_ENDPOINT"]
    user = user or os.environ["OPENSEARCH_USER"]
    password = password or os.environ["OPENSEARCH_PASSWORD"]
    host = endpoint.replace("https://", "").replace("http://", "")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=(user, password),
        use_ssl=True, verify_certs=True,
        timeout=30, max_retries=3, retry_on_timeout=True,
    )


class OpenSearchStore:
    def __init__(self, endpoint: str = None, user: str = None, password: str = None,
                 index: str = INDEX_NAME, dim: int = None):
        self.client = _client(endpoint, user, password)
        self.index = index
        self.dim = dim or CONFIG.embed_dim
        self.embedder = get_embedder()

    # --- index management ---
    def create_index(self, recreate: bool = False):
        if self.client.indices.exists(self.index):
            if recreate:
                self.client.indices.delete(self.index)
            else:
                return "exists"
        body = {
            "settings": {
                "index.knn": True,          # enable k-NN
                "number_of_shards": 1,
                "number_of_replicas": 0,    # single-node demo domain
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},               # BM25 side
                    "embedding": {                          # vector side
                        "type": "knn_vector",
                        "dimension": self.dim,
                        "method": {
                            "name": "hnsw", "space_type": "cosinesimil",
                            "engine": "lucene",
                        },
                    },
                    "doc_id": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "source_type": {"type": "keyword"},
                    "s3_key": {"type": "keyword"},
                    "page": {"type": "integer"},
                },
            },
        }
        self.client.indices.create(self.index, body=body)
        return "created"

    # --- ingestion ---
    def upsert_chunks(self, chunks: list[Chunk], batch: int = 96, progress=None):
        """Embed chunks and bulk-index them. On reindex of a doc, delete its old
        chunks first (delete-by-query on doc_id) to avoid duplicates."""
        from opensearchpy.helpers import bulk

        # delete existing chunks for these doc_ids (idempotent reindex)
        doc_ids = sorted({c.doc_id for c in chunks})
        for did in doc_ids:
            try:
                self.client.delete_by_query(
                    index=self.index,
                    body={"query": {"term": {"doc_id": did}}},
                    refresh=True, conflicts="proceed")
            except Exception:
                pass  # index may be empty first time

        total = 0
        for i in range(0, len(chunks), batch):
            group = chunks[i:i + batch]
            vecs = self.embedder.embed([c.text for c in group])
            actions = []
            for c, v in zip(group, vecs):
                actions.append({
                    "_index": self.index,
                    "_source": {
                        "text": c.text,
                        "embedding": v,
                        "doc_id": c.doc_id,
                        "chunk_type": c.chunk_type,
                        "source": c.metadata.get("source", c.doc_id),
                        "source_type": c.metadata.get("source_type", "unknown"),
                        "s3_key": c.metadata.get("s3_key", ""),
                        "page": c.metadata.get("page", 0),
                    },
                })
            bulk(self.client, actions)
            total += len(actions)
            if progress:
                progress(f"indexed {total}/{len(chunks)}")
        self.client.indices.refresh(self.index)
        return total

    # --- retrieval (same interface as HybridRetriever.retrieve) ---
    def retrieve(self, query: str, candidate_n: int = None,
                 top_k: int = None) -> list[RetrievalHit]:
        top_k = top_k or CONFIG.top_k
        qv = self.embedder.embed([query])[0]

        # Hybrid: a bool query combining BM25 (match) + kNN, both scored.
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {"match": {"text": {"query": query, "boost": 0.5}}},
                        {"knn": {"embedding": {"vector": qv, "k": top_k}}},
                    ]
                }
            },
        }
        resp = self.client.search(index=self.index, body=body)
        hits = []
        for rank, h in enumerate(resp["hits"]["hits"]):
            src = h["_source"]
            chunk = Chunk(
                text=src["text"], chunk_type=src.get("chunk_type", "prose"),
                doc_id=src.get("doc_id", ""),
                metadata={"source": src.get("source"),
                          "source_type": src.get("source_type", "unknown"),
                          "s3_key": src.get("s3_key", ""),
                          "page": src.get("page", 0)})
            hits.append(RetrievalHit(chunk=chunk, rerank_score=h["_score"], rrf_rank=rank))
        return hits


if __name__ == "__main__":
    # Smoke test — requires env vars set and the domain reachable.
    store = OpenSearchStore()
    print("create_index:", store.create_index())
    print("cluster:", store.client.cluster.health()["status"])
