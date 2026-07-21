"""
confidence.py  —  STEP 2: Source Confidence Scoring
===================================================
Retrieval tells us what's RELEVANT. This layer tells us what's TRUSTWORTHY,
which is a different question that matters enormously in finance: a highly
relevant Slack rumor should NOT outrank a slightly-less-relevant audited filing.

Formula (from the workshop, wired to real metadata):

    confidence = 0.5 * relevance      # normalized rerank/RRF score, 0..1
               + 0.3 * trust          # by source_type (audited filing > memo > chat)
               + 0.2 * freshness      # decay by document age

Chunks scoring below CONFIG.confidence_threshold (0.3) are DROPPED before they
can reach the LLM. The kept set is what Step 3 cites from.

Why this is the right design for the interview:
  - It's deterministic and explainable ("we only cited sources scoring > 0.3").
  - Trust is data-driven from ingestion metadata, not hardcoded per query.
  - Freshness matters for financials: last quarter's ARR is not this quarter's.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .config import CONFIG
from .retrieval import RetrievalHit


# Trust priors by source type. In finance, the system of record wins.
TRUST_BY_SOURCE = {
    "audited_filing": 0.98,   # 10-K, 10-Q, audited statements
    "official_docs": 0.95,
    "erp_system": 0.92,       # SAP / Oracle / NetSuite of record
    "data_warehouse": 0.90,   # Snowflake curated tables
    "internal_memo": 0.75,    # our SaaS metrics memo
    "internal_wiki": 0.70,
    "blog": 0.60,
    "forum": 0.50,
    "chat_message": 0.40,     # Slack / Teams — lowest trust
    "unknown": 0.50,
}


def freshness_score(doc_date: datetime, now: datetime | None = None) -> float:
    """Step decay by age. Financial relevance drops as data ages."""
    now = now or datetime.now(timezone.utc)
    if doc_date.tzinfo is None:
        doc_date = doc_date.replace(tzinfo=timezone.utc)
    age_days = (now - doc_date).days
    if age_days <= 30:
        return 1.0
    if age_days <= 90:
        return 0.8
    if age_days <= 180:
        return 0.6
    if age_days <= 365:
        return 0.5
    # beyond a year: continuous linear decay by years (floor 0.05)
    years = age_days / 365.0
    return max(0.05, 0.5 - 0.04 * (years - 1))


def normalize_scores(raw: list[float]) -> list[float]:
    """Min-max normalize arbitrary rerank scores into 0..1 for the relevance term."""
    if not raw:
        return []
    lo, hi = min(raw), max(raw)
    if hi - lo < 1e-9:
        return [1.0] * len(raw)
    return [(s - lo) / (hi - lo) for s in raw]


@dataclass
class ScoredHit:
    hit: RetrievalHit
    relevance: float
    trust: float
    freshness: float
    confidence: float

    @property
    def text(self) -> str:
        return self.hit.chunk.text

    @property
    def source(self) -> str:
        return self.hit.chunk.metadata.get("source", "unknown")


def score_hits(hits: list[RetrievalHit], now: datetime | None = None) -> list[ScoredHit]:
    """Attach relevance/trust/freshness/confidence to each retrieval hit."""
    if not hits:
        return []
    rel = normalize_scores([h.rerank_score for h in hits])
    scored = []
    for i, h in enumerate(hits):
        meta = h.chunk.metadata
        source_type = meta.get("source_type", "unknown")
        doc_date = meta.get("doc_date") or datetime.now(timezone.utc)
        trust = TRUST_BY_SOURCE.get(source_type, TRUST_BY_SOURCE["unknown"])
        fresh = freshness_score(doc_date, now)
        conf = (CONFIG.w_relevance * rel[i]
                + CONFIG.w_trust * trust
                + CONFIG.w_freshness * fresh)
        scored.append(ScoredHit(hit=h, relevance=rel[i], trust=trust,
                                freshness=fresh, confidence=conf))
    return scored


def filter_by_confidence(hits: list[RetrievalHit], threshold: float = None,
                         now: datetime | None = None) -> list[ScoredHit]:
    """Score then drop anything below threshold, sorted best-first."""
    threshold = threshold if threshold is not None else CONFIG.confidence_threshold
    scored = score_hits(hits, now)
    kept = [s for s in scored if s.confidence >= threshold]
    kept.sort(key=lambda s: s.confidence, reverse=True)
    return kept


if __name__ == "__main__":
    from datetime import timedelta
    from .ingestion import ingest_dir
    from .retrieval import HybridRetriever

    # Assign differentiated source types + dates so scoring actually discriminates.
    chunks = ingest_dir()
    now = datetime.now(timezone.utc)
    for c in chunks:
        if c.doc_id.startswith("acme_10k"):
            c.metadata["source_type"] = "audited_filing"
            c.metadata["doc_date"] = now - timedelta(days=20)     # fresh
        else:
            c.metadata["source_type"] = "internal_memo"
            c.metadata["doc_date"] = now - timedelta(days=200)    # older

    r = HybridRetriever(chunks)
    hits = r.retrieve("What is the Net Revenue Retention rate?", top_k=5)

    print("Query: What is the Net Revenue Retention rate?\n")
    print(f"{'conf':>5} {'rel':>5} {'trust':>5} {'fresh':>5}  source            preview")
    print("-" * 90)
    for s in filter_by_confidence(hits):
        print(f"{s.confidence:5.2f} {s.relevance:5.2f} {s.trust:5.2f} "
              f"{s.freshness:5.2f}  {s.hit.chunk.metadata.get('source_type','?'):16} "
              f"{s.text[:35].replace(chr(10),' ')}")
