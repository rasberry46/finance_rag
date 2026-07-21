"""
retrieval.py  —  STEP 1: Hybrid Retrieval + Reranking
=====================================================
The five sub-components the workshop advertises, built for real:

  1. BM25 keyword search      -> rank_bm25.BM25Okapi
  2. Semantic embeddings      -> providers.get_embedder() (Titan on Bedrock)
  3. ANN indexing             -> FAISS (IndexFlatIP on normalized vectors)
  4. RRF score merging        -> reciprocal_rank_fusion()
  5. Cross-encoder reranking  -> sentence_transformers.CrossEncoder
                                 (with a local lexical fallback for offline verify)

Flow:
    query
      ├── BM25 top-N (keyword)        ─┐
      └── FAISS top-N (semantic/ANN)  ─┤→ RRF merge ─→ cross-encoder rerank ─→ top-k
                                        │
                          (fusion uses rank position only, so BM25 and cosine
                           scores never need to be on the same scale)

Persistence: the FAISS index + chunk store can be saved/loaded so ingestion runs
once (mirrors production where you don't re-embed on every query).
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from .config import CONFIG
from .providers import get_embedder
from .ingestion import Chunk


_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


# ----------------------------------------------------------------------------
# RRF
# ----------------------------------------------------------------------------
def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = None) -> list[tuple[int, float]]:
    """Merge ranked lists of doc indices by rank position only.
       score(d) = sum 1/(k + rank_in_list(d))."""
    k = k or CONFIG.rrf_k
    fused: dict[int, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, doc_idx in enumerate(lst):
            fused[doc_idx] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# ----------------------------------------------------------------------------
# Cross-encoder reranker (real model, local fallback)
# ----------------------------------------------------------------------------
class Reranker:
    def __init__(self):
        self.model = None
        if CONFIG.provider == "bedrock":
            # In production we still use a HF cross-encoder (cheap, runs on CPU).
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(CONFIG.cross_encoder_id)
            except Exception as e:
                print(f"[Reranker] CrossEncoder unavailable ({e}); using lexical fallback")

    def score(self, query: str, docs: list[str]) -> list[float]:
        if self.model is not None:
            pairs = [(query, d) for d in docs]
            return [float(s) for s in self.model.predict(pairs)]
        # lexical fallback: query-term coverage, length-normalized so a short
        # precise table row isn't buried under a long prose block. (The real
        # cross-encoder on Bedrock/Colab replaces this entirely.)
        q = set(_tok(query))
        out = []
        for d in docs:
            dt = _tok(d)
            if not q or not dt:
                out.append(0.0); continue
            dset = set(dt)
            coverage = len(q & dset) / len(q)          # how many query terms present
            precision = len(q & dset) / len(dset)      # how focused the doc is on them
            out.append(coverage * 2.0 + precision)
        return out


# ----------------------------------------------------------------------------
# Hybrid retriever
# ----------------------------------------------------------------------------
@dataclass
class RetrievalHit:
    chunk: Chunk
    rerank_score: float
    rrf_rank: int


class HybridRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.texts = [c.text for c in chunks]
        self.embedder = get_embedder()
        self.reranker = Reranker()

        # BM25 over tokenized corpus
        self.bm25 = BM25Okapi([_tok(t) for t in self.texts])

        # FAISS ANN index over normalized embeddings (IP == cosine on unit vecs)
        vecs = np.array(self.embedder.embed(self.texts), dtype="float32")
        faiss.normalize_L2(vecs)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self._dim = vecs.shape[1]

    def _bm25_topn(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(_tok(query))
        order = np.argsort(scores)[::-1]
        return [int(i) for i in order[:n] if scores[i] > 0]

    def _faiss_topn(self, query: str, n: int) -> list[int]:
        q = np.array(self.embedder.embed([query]), dtype="float32")
        faiss.normalize_L2(q)
        _scores, idxs = self.index.search(q, n)
        return [int(i) for i in idxs[0] if i != -1]

    def retrieve(self, query: str, candidate_n: int = None, top_k: int = None) -> list[RetrievalHit]:
        candidate_n = candidate_n or CONFIG.candidate_n
        top_k = top_k or CONFIG.top_k

        bm25_list = self._bm25_topn(query, candidate_n)
        faiss_list = self._faiss_topn(query, candidate_n)

        fused = reciprocal_rank_fusion([bm25_list, faiss_list])
        candidate_idxs = [idx for idx, _ in fused[:candidate_n]]
        if not candidate_idxs:
            return []

        cand_texts = [self.texts[i] for i in candidate_idxs]
        rr_scores = self.reranker.score(query, cand_texts)

        ranked = sorted(zip(candidate_idxs, rr_scores), key=lambda x: x[1], reverse=True)
        hits = []
        for rank, (idx, score) in enumerate(ranked[:top_k]):
            hits.append(RetrievalHit(chunk=self.chunks[idx], rerank_score=float(score),
                                     rrf_rank=candidate_idxs.index(idx)))
        return hits

    # --- persistence ---
    def save(self, directory: str | Path = None):
        directory = Path(directory) if directory else CONFIG.index_dir
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, directory: str | Path = None) -> "HybridRetriever":
        directory = Path(directory) if directory else CONFIG.index_dir
        with open(directory / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        obj = cls.__new__(cls)
        obj.chunks = chunks
        obj.texts = [c.text for c in chunks]
        obj.embedder = get_embedder()
        obj.reranker = Reranker()
        obj.bm25 = BM25Okapi([_tok(t) for t in obj.texts])
        obj.index = faiss.read_index(str(directory / "index.faiss"))
        obj._dim = obj.index.d
        return obj


if __name__ == "__main__":
    from .ingestion import ingest_dir
    chunks = ingest_dir()
    r = HybridRetriever(chunks)
    print(f"Indexed {len(chunks)} chunks, FAISS dim={r._dim}, "
          f"reranker={'model' if r.reranker.model else 'lexical-fallback'}\n")

    for query in [
        "How is revenue recognized under ASC 606?",
        "What was Enterprise revenue in Q3?",
        "What is the Net Revenue Retention rate?",
    ]:
        print(f"Q: {query}")
        for h in r.retrieve(query, top_k=3):
            tag = h.chunk.chunk_type
            preview = h.chunk.text.replace("\n", " ")[:75]
            print(f"   [{tag:9} p{h.chunk.metadata.get('page','?')}] "
                  f"score={h.rerank_score:5.2f}  {preview}")
        print()
