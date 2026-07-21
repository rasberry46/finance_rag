"""
pipeline.py  —  End-to-end orchestrator wiring all 5 steps together
===================================================================
One class that runs the full production RAG flow with tracing:

  cache -> embed -> hybrid retrieve -> confidence filter
        -> deterministic compute -> constrained generate -> hallucination guard
        -> trace + cache write

Run offline (local provider):     python -m src.prodrag.pipeline
Run on Bedrock:  RAG_PROVIDER=bedrock python -m src.prodrag.pipeline
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from .config import CONFIG
from .ingestion import ingest_dir
from .retrieval import HybridRetriever
from .confidence import filter_by_confidence
from .generation import answer as generate_answer, Computation, variance
from .evaluation import TTLCache, ConversationMemory
from .observability import Tracer, Metrics


class ProductionRAG:
    """The whole pipeline behind one .ask() method."""

    def __init__(self, retriever: HybridRetriever | None = None):
        self.cache = TTLCache()
        self.memory = ConversationMemory()
        self.metrics = Metrics()
        if retriever is None:
            chunks = self._tag(ingest_dir())
            retriever = HybridRetriever(chunks)
        self.retriever = retriever

    @staticmethod
    def _tag(chunks):
        """Assign source_type + doc_date so confidence scoring discriminates.
        In production these come from S3 metadata / the ingestion event."""
        now = datetime.now(timezone.utc)
        for c in chunks:
            if c.doc_id.startswith("acme_10k"):
                c.metadata.setdefault("source_type", "audited_filing")
                c.metadata.setdefault("doc_date", now - timedelta(days=20))
            else:
                c.metadata.setdefault("source_type", "internal_memo")
                c.metadata.setdefault("doc_date", now - timedelta(days=200))
        return chunks

    def ask(self, question: str, computations: list[Computation] | None = None) -> dict:
        tracer = Tracer()

        with tracer.span("cache_lookup"):
            cached = self.cache.get(question)
        if cached is not None:
            self.metrics.record(tracer.trace)
            return {"answer": cached, "cached": True, "trace": tracer.summary()}

        with tracer.span("hybrid_retrieval"):
            hits = self.retriever.retrieve(question, top_k=CONFIG.top_k)

        with tracer.span("confidence_filter"):
            scored = filter_by_confidence(hits)

        with tracer.span("generation_and_guard"):
            ans = generate_answer(question, scored, computations=computations or [])

        self.cache.set(question, ans.text)
        self.memory.add(question, ans.text)
        self.metrics.record(tracer.trace)

        return {
            "answer": ans.text,
            "cached": False,
            "risk": ans.risk,
            "fell_back": ans.fell_back,
            "citations": ans.citations,
            "contexts_used": ans.contexts_used,
            "trace": tracer.summary(),
        }


if __name__ == "__main__":
    print(f"Provider: {CONFIG.provider}\n")
    rag = ProductionRAG()

    # Variance question — math computed in code, narrated by the LLM
    comp = variance("Sales & Marketing", budget=3200, actual=3560, higher_is_better=False)
    out = rag.ask("Explain the Sales & Marketing spend variance versus budget.",
                  computations=[comp])
    print("Q1 ANSWER:", out["answer"])
    print(f"cached={out['cached']} risk={out.get('risk')} "
          f"citations={out.get('citations')}\n")
    print(out["trace"])

    # Same question again — cache hit
    print("\n--- asking the same question again ---")
    out2 = rag.ask("Explain the Sales & Marketing spend variance versus budget.",
                   computations=[comp])
    print("Q1-repeat cached:", out2["cached"])

    print("\n=== aggregate metrics ===")
    print(rag.metrics.report())
