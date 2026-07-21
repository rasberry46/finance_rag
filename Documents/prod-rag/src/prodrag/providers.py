"""
providers.py
============
Abstraction over embeddings + LLM so the SAME pipeline runs on:
  - AWS Bedrock (Titan embeddings + Claude)  ← production
  - Local deterministic stand-ins            ← offline sandbox verification

Flip with CONFIG.provider ("bedrock" | "local") or RAG_PROVIDER env var.

The interfaces are tiny and stable:
    Embedder.embed(texts: list[str]) -> list[list[float]]
    LLM.generate(system: str, user: str) -> str

In Colab/VS Code, `pip install boto3`, set AWS creds + RAG_PROVIDER=bedrock,
and the real Titan/Claude calls activate with zero other changes.
"""

from __future__ import annotations

import json
import math
import re
from typing import Protocol

from .config import CONFIG


_TOKEN = re.compile(r"[a-z0-9]+")


# ============================================================================
# Interfaces
# ============================================================================
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...


# ============================================================================
# LOCAL stand-ins (for sandbox verification, no network/credentials)
# ============================================================================
class LocalEmbedder:
    """Deterministic feature-hashing embedder at the same dim as Titan (1024),
    so FAISS index shape matches production exactly. Not semantically strong;
    it exists to verify the *pipeline*, not embedding quality."""

    def __init__(self, dim: int = None):
        self.dim = dim or CONFIG.embed_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            vec = [0.0] * self.dim
            for tok in _TOKEN.findall(t.lower()):
                vec[hash(tok) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class LocalLLM:
    """Rule-following stub that mimics a well-behaved Claude: it obeys the
    constrained prompt, cites [S1], and refuses when the context can't support
    an answer (so the offline demo shows the same safety behavior as Bedrock)."""

    _STOP = {"what", "was", "the", "is", "a", "an", "of", "in", "for", "how",
             "explain", "under", "and", "s", "to", "on", "at", "which", "did"}

    def generate(self, system: str, user: str) -> str:
        facts = re.search(r"<computed_facts>\n(.*?)\n</computed_facts>", user, re.S)
        ctx = re.search(r"<context>\n(.*?)\n</context>", user, re.S)
        q = re.search(r"Question:\s*(.*)\s*$", user, re.S)
        fact_text = facts.group(1).strip() if facts else ""
        ctx_text = ctx.group(1).strip() if ctx else ""
        question = q.group(1).strip().lower() if q else ""

        # Computed facts present -> narrate them (deterministic math path).
        if fact_text and fact_text != "(none)":
            first = fact_text.splitlines()[0]
            return f"Based on the computed figures, {first} [S1]"

        if not ctx_text:
            return "I could not find sufficient evidence in the provided sources."

        # Grounding check: do the question's key terms actually appear in context?
        q_terms = {w for w in re.findall(r"[a-z0-9]+", question) if w not in self._STOP
                   and len(w) > 2}
        ctx_lower = ctx_text.lower()
        overlap = [t for t in q_terms if t in ctx_lower]
        if q_terms and len(overlap) / len(q_terms) < 0.34:
            # The corpus doesn't cover this question -> refuse rather than invent.
            return "I could not find sufficient evidence in the provided sources."

        first_ctx = ctx_text.splitlines()[0] if ctx_text else ""
        first_ctx = re.sub(r"^\[S\d+\]\s*", "", first_ctx)
        return f"{first_ctx} [S1]"


# ============================================================================
# BEDROCK implementations (production)
# ============================================================================
class BedrockEmbedder:
    """Amazon Titan Text Embeddings v2 via bedrock-runtime."""

    def __init__(self):
        import boto3
        self.client = boto3.client("bedrock-runtime", region_name=CONFIG.aws_region)
        self.model_id = CONFIG.bedrock_embed_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for t in texts:
            body = json.dumps({"inputText": t})
            resp = self.client.invoke_model(modelId=self.model_id, body=body)
            payload = json.loads(resp["body"].read())
            vectors.append(payload["embedding"])
        return vectors


class BedrockLLM:
    """Anthropic Claude on Bedrock via the Messages API."""

    def __init__(self):
        import boto3
        self.client = boto3.client("bedrock-runtime", region_name=CONFIG.aws_region)
        self.model_id = CONFIG.bedrock_llm_id

    def generate(self, system: str, user: str) -> str:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        })
        resp = self.client.invoke_model(modelId=self.model_id, body=body)
        payload = json.loads(resp["body"].read())
        return payload["content"][0]["text"]


# ============================================================================
# Factory
# ============================================================================
def get_embedder() -> Embedder:
    if CONFIG.provider == "bedrock":
        return BedrockEmbedder()
    return LocalEmbedder()


def get_llm() -> LLM:
    if CONFIG.provider == "bedrock":
        return BedrockLLM()
    return LocalLLM()


if __name__ == "__main__":
    emb = get_embedder()
    v = emb.embed(["deferred revenue under ASC 606"])
    print(f"provider={CONFIG.provider}  embedding_dim={len(v[0])}")
    llm = get_llm()
    out = llm.generate("You are helpful.",
                       "<computed_facts>\nRevenue variance = +12% [S1]\n</computed_facts>\n<context>\n[S1] x\n</context>\nQ?")
    print("llm:", out)
