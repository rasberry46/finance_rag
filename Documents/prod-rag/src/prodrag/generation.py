"""
generation.py  —  STEP 3: Constrained Generation, Citations & Hallucination Detection
=====================================================================================
The trustworthy shortlist from Step 2 arrives here. This module:

  1. deterministic_math()      — Python computes numbers. The LLM NEVER does.
  2. build_constrained_prompt()— injects context + computed numbers, forbids
                                  invention, requires [S1] citations.
  3. generate()                — calls Claude on Bedrock (via providers).
  4. hallucination_risk()      — scores the answer 0..1 for ungrounded claims.
  5. answer()                  — wires it together with a fallback when risk is high.

THE CORE PRINCIPLE (say this in the interview):
    A language model is good at writing a narrative. It is bad at arithmetic and
    will confidently produce wrong numbers. So we split the job: deterministic
    code does every calculation exactly, and the LLM only explains the results.
    That removes the highest-risk failure mode in financial workflows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import CONFIG
from .providers import get_llm
from .confidence import ScoredHit


# ============================================================================
# 1. Deterministic finance math — NEVER delegated to the LLM
# ============================================================================
@dataclass
class Computation:
    """A number computed in code, with a human-readable fact line for the prompt."""
    label: str
    value: float
    detail: str

    def as_fact(self) -> str:
        return f"{self.label}: {self.detail}"


def variance(line_item: str, budget: float, actual: float,
             higher_is_better: bool = True) -> Computation:
    """Budget vs actual variance — computed exactly in Python."""
    abs_var = actual - budget
    pct = (abs_var / budget * 100.0) if budget else float("inf")
    favorable = (abs_var >= 0) == higher_is_better
    detail = (f"budget={budget:,.0f}, actual={actual:,.0f}, "
              f"variance={abs_var:,.0f} ({pct:+.1f}%), "
              f"{'favorable' if favorable else 'unfavorable'}")
    return Computation(label=line_item, value=abs_var, detail=detail)


def growth_rate(label: str, prior: float, current: float) -> Computation:
    pct = ((current - prior) / prior * 100.0) if prior else float("inf")
    return Computation(label=label, value=pct,
                       detail=f"prior={prior:,.2f}, current={current:,.2f}, "
                              f"growth={pct:+.1f}%")


# ============================================================================
# 2. Constrained prompt
# ============================================================================
CONSTRAINED_SYSTEM = """You are a financial analysis assistant for an FP&A team.

HARD RULES:
- Use ONLY the numbers in <computed_facts> and the text in <context>.
- NEVER perform arithmetic yourself. All numbers are pre-computed; quote them exactly.
- Cite every factual claim with [S<n>] referring to the numbered context blocks.
- If the context does not support an answer, reply EXACTLY:
  "I could not find sufficient evidence in the provided sources."
- Be concise: 2-4 sentences. You are writing analysis, not a chatbot reply.
"""


def build_constrained_prompt(question: str, contexts: list[str],
                             computations: list[Computation]) -> tuple[str, str]:
    ctx = "\n".join(f"[S{i+1}] {c}" for i, c in enumerate(contexts))
    facts = "\n".join(c.as_fact() for c in computations) if computations else "(none)"
    user = (f"<computed_facts>\n{facts}\n</computed_facts>\n\n"
            f"<context>\n{ctx}\n</context>\n\n"
            f"Question: {question}")
    return CONSTRAINED_SYSTEM, user


# ============================================================================
# 3 + 4. Hallucination risk scoring
# ============================================================================
# "likely" deliberately excluded — too common in legitimate financial prose
# ("likely driven by") to use as a hallucination signal.
HEDGE_WORDS = ["probably", "might", "maybe", "i think", "presumably",
               "possibly", "could be", "i believe", "my guess"]


def _extract_numbers(text: str) -> set[str]:
    """Extract financial numbers for grounding checks, avoiding two false-positive
    traps learned in testing:
      1. Citation markers like [S2] leak their digits and look like invented
         figures -> strip them first.
      2. Format differences ("3,560" vs "3560" vs "3,560,") cause spurious
         mismatches -> normalize by removing commas and trailing punctuation.
    """
    text = re.sub(r"\[s\d+\]", " ", text, flags=re.IGNORECASE)  # drop citations
    cleaned = set()
    for n in re.findall(r"\d[\d,\.]*", text):
        n = n.rstrip(".,").replace(",", "")
        if n:
            cleaned.add(n)
    return cleaned


def hallucination_risk(answer: str, contexts: list[str],
                       computations: list[Computation]) -> float:
    """0 = fully grounded, 1 = high risk. Three cheap, explainable signals.

    Note: an over-eager detector is its own failure mode — it can force a
    correct, well-cited answer into an unnecessary fallback. These checks are
    tuned to minimize false positives (see _extract_numbers)."""
    a = answer.lower()
    risk = 0.0

    # (a) hedge language — a confident finance answer shouldn't waffle
    if any(h in a for h in HEDGE_WORDS):
        risk += 0.35

    # (b) no citation markers at all
    if not re.search(r"\[s\d+\]", a):
        risk += 0.35

    # (c) numbers in the answer grounded in NEITHER context nor computed facts
    grounded_text = " ".join(contexts) + " " + " ".join(c.detail for c in computations)
    grounded_numbers = _extract_numbers(grounded_text)
    fabricated = [n for n in _extract_numbers(answer) if n not in grounded_numbers]
    if fabricated:
        risk += min(0.4, 0.2 * len(fabricated))

    return min(1.0, risk)


# ============================================================================
# 5. End-to-end answer with fallback
# ============================================================================
FALLBACK = "I could not find sufficient evidence in the provided sources."


@dataclass
class Answer:
    text: str
    risk: float
    fell_back: bool
    citations: list[str] = field(default_factory=list)
    contexts_used: int = 0


def generate(system: str, user: str) -> str:
    """Single Claude (Bedrock) call via the provider layer."""
    return get_llm().generate(system, user)


def answer(question: str, scored_hits: list[ScoredHit],
           computations: list[Computation] = None,
           max_context: int = None) -> Answer:
    """Full Step 3: build prompt from trustworthy hits, generate, guard, fall back."""
    computations = computations or []
    max_context = max_context or CONFIG.top_k

    contexts = [s.text for s in scored_hits[:max_context]]
    system, user = build_constrained_prompt(question, contexts, computations)
    raw = generate(system, user)

    risk = hallucination_risk(raw, contexts, computations)
    citations = re.findall(r"\[S\d+\]", raw)

    if risk > CONFIG.hallucination_threshold:
        return Answer(text=FALLBACK, risk=risk, fell_back=True,
                      citations=[], contexts_used=len(contexts))
    return Answer(text=raw, risk=risk, fell_back=False,
                  citations=citations, contexts_used=len(contexts))


if __name__ == "__main__":
    from datetime import datetime, timezone, timedelta
    from .ingestion import ingest_dir
    from .retrieval import HybridRetriever
    from .confidence import filter_by_confidence

    chunks = ingest_dir()
    now = datetime.now(timezone.utc)
    for c in chunks:
        if c.doc_id.startswith("acme_10k"):
            c.metadata.update(source_type="audited_filing", doc_date=now - timedelta(days=20))
        else:
            c.metadata.update(source_type="internal_memo", doc_date=now - timedelta(days=200))

    r = HybridRetriever(chunks)

    # Q: variance question — math done in CODE, narrated by the LLM
    q = "Explain the Sales & Marketing spend variance versus budget."
    hits = r.retrieve(q, top_k=5)
    scored = filter_by_confidence(hits)
    comp = variance("Sales & Marketing", budget=3200, actual=3560, higher_is_better=False)
    print("COMPUTED IN CODE:", comp.as_fact())

    ans = answer(q, scored, computations=[comp])
    print("\nQ:", q)
    print("ANSWER:", ans.text)
    print(f"risk={ans.risk:.2f}  fell_back={ans.fell_back}  citations={ans.citations}")

    # Demonstrate the guard catching a fabricated number
    print("\n--- guard test: fabricated number ---")
    bad = hallucination_risk("Revenue was probably $9,999,999.",
                             ["Revenue was 4,200,000."], [])
    print(f"fabricated-number answer risk = {bad:.2f} (would fall back if > "
          f"{CONFIG.hallucination_threshold})")
