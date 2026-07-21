"""
config.py
=========
Single source of truth for models, paths, and tunable RAG parameters.
Everything the pipeline needs is here so Colab / VS Code / Lambda all read the
same knobs. Override any field via environment variables.

Bedrock model IDs (verify current IDs in your region with:
    aws bedrock list-foundation-models --region us-east-1
):
  - Claude (generation): us.anthropic.claude-sonnet-4-6
  - Titan (embeddings) : amazon.titan-embed-text-v2:0   (1024-dim)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class Config:
    # --- Provider selection ---
    # "bedrock" for real AWS; "local" for offline sandbox verification.
    provider: str = field(default_factory=lambda: _env("RAG_PROVIDER", "local"))
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))

    # --- Bedrock model IDs ---
    bedrock_llm_id: str = field(
        default_factory=lambda: _env(
            "BEDROCK_LLM_ID", "us.anthropic.claude-sonnet-4-6"
        )
    )
    bedrock_embed_id: str = field(
        default_factory=lambda: _env(
            "BEDROCK_EMBED_ID", "amazon.titan-embed-text-v2:0"
        )
    )
    embed_dim: int = 1024  # Titan v2 default; local embedder matches this

    # --- Cross-encoder (HuggingFace) ---
    cross_encoder_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Retrieval knobs ---
    candidate_n: int = 20        # candidates from each retriever before fusion
    top_k: int = 5               # final chunks after reranking
    rrf_k: int = 60              # RRF dampening constant (from the paper)

    # --- Confidence scoring weights (must sum to 1.0) ---
    w_relevance: float = 0.5
    w_trust: float = 0.3
    w_freshness: float = 0.2
    confidence_threshold: float = 0.3

    # --- Generation guard ---
    hallucination_threshold: float = 0.5

    # --- Chunking ---
    prose_chunk_words: int = 120
    prose_overlap_words: int = 20

    # --- Cache / memory ---
    cache_ttl_seconds: int = 3600      # 1 hour
    conversation_max_turns: int = 10

    # --- Paths ---
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(init=False)
    sample_docs_dir: Path = field(init=False)
    index_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.root / "data"
        self.sample_docs_dir = self.data_dir / "sample_docs"
        self.index_dir = self.data_dir / "faiss_index"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        assert abs(self.w_relevance + self.w_trust + self.w_freshness - 1.0) < 1e-6, \
            "confidence weights must sum to 1.0"


CONFIG = Config()

if __name__ == "__main__":
    import json
    c = CONFIG
    print(json.dumps({
        "provider": c.provider, "region": c.aws_region,
        "llm": c.bedrock_llm_id, "embed": c.bedrock_embed_id,
        "embed_dim": c.embed_dim, "top_k": c.top_k, "rrf_k": c.rrf_k,
        "conf_weights": [c.w_relevance, c.w_trust, c.w_freshness],
        "threshold": c.confidence_threshold,
    }, indent=2))
