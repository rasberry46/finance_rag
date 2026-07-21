from __future__ import annotations
from datetime import datetime, timezone
from .confidence import score_hits


def _tag(source):
    s = (source or "").lower()
    now = datetime.now(timezone.utc)
    if "sample_accounts" in s or "abc" in s:
        return datetime(1995, 12, 31, tzinfo=timezone.utc), "audited_filing"
    if "2024" in s or "financial-statements" in s:
        return datetime(2024, 12, 31, tzinfo=timezone.utc), "audited_filing"
    return now, "official_docs"


class ConfidenceStore:
    def __init__(self, store):
        self.store = store

    def retrieve(self, query, candidate_n=None, top_k=None):
        top_k = top_k or 5
        raw = self.store.retrieve(query, top_k=max(top_k * 2, 10))
        for h in raw:
            src = h.chunk.metadata.get("source") or h.chunk.doc_id
            doc_date, source_type = _tag(src)
            h.chunk.metadata["doc_date"] = doc_date
            h.chunk.metadata.setdefault("source_type", source_type)
        scored = score_hits(raw)
        scored.sort(key=lambda s: s.confidence, reverse=True)
        out = []
        for s in scored[:top_k]:
            s.hit.chunk.metadata["confidence"] = round(s.confidence, 3)
            s.hit.chunk.metadata["freshness"] = round(s.freshness, 2)
            s.hit.chunk.metadata["trust"] = round(s.trust, 2)
            out.append(s.hit)
        return out

    def __getattr__(self, name):
        return getattr(self.store, name)
