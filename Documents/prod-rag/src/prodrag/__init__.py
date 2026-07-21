"""Production RAG on AWS Bedrock — all 5 steps + agentic layer."""
from .config import CONFIG
from .ingestion import ingest_dir, ingest_pdf, Chunk
from .retrieval import HybridRetriever, reciprocal_rank_fusion, RetrievalHit
from .confidence import filter_by_confidence, score_hits, ScoredHit
from .generation import answer, variance, growth_rate, hallucination_risk, Computation
from .evaluation import evaluate, TTLCache, ConversationMemory
from .observability import Tracer, Metrics
from .pipeline import ProductionRAG
from .agentic import AgenticRAG, build_agent

__all__ = [
    "CONFIG", "ingest_dir", "ingest_pdf", "Chunk",
    "HybridRetriever", "reciprocal_rank_fusion", "RetrievalHit",
    "filter_by_confidence", "score_hits", "ScoredHit",
    "answer", "variance", "growth_rate", "hallucination_risk", "Computation",
    "evaluate", "TTLCache", "ConversationMemory",
    "Tracer", "Metrics", "ProductionRAG", "AgenticRAG", "build_agent",
]
