"""
agentic.py  —  Agentic RAG with LangGraph
=========================================
The 5-step pipeline runs a FIXED path for every query. This layer makes the
pipeline DECIDE how to answer each query — the difference between a script and
an agent.

THE GRAPH
---------
    supervisor ──▶ (route by intent)
        ├─ retrieve            "What is deferred revenue?"      (docs only)
        ├─ compute             "What is 12% of 3200?"           (math only)
        ├─ retrieve + compute  "Explain the S&M variance"       (docs + math)
        └─ direct              "Hello"                          (neither)
                    │
                    ▼
                generate  (constrained + guarded, from Step 3)
                    │
                    ▼
                 grade ── bad & retries left ──▶ back to retrieve (self-correct)
                    │
                    ▼  good, or out of retries
                 answer

INTERVIEW-CRITICAL DESIGN POINTS (this maps to LangGraph multi-agent work):
  - STATE PERSISTENCE: a typed AgentState flows through every node; each node
    reads and updates it. In production you attach a checkpointer (Redis/Postgres)
    so a run can pause/resume and survive restarts.
  - CONDITIONAL ROUTING: the supervisor is a router node; `add_conditional_edges`
    sends the query down the right branch.
  - SELF-CORRECTION LOOP: grade can route BACK to retrieve to try again — this is
    the cyclic graph LangGraph is built for.
  - LOOP LIMITS: `max_retries` + a retry counter in state guarantee termination.
    Without this, a grading loop could spin forever and burn thousands in tokens.
    This is the #1 production-readiness question about agents.

Runs offline with the `local` provider (rule-based supervisor + LLM stub) so the
GRAPH LOGIC is fully verifiable; on Bedrock the same nodes call Claude.
"""

from __future__ import annotations

import re
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END

from .config import CONFIG
from .retrieval import HybridRetriever
from .confidence import filter_by_confidence
from .generation import (build_constrained_prompt, generate as llm_generate,
                         hallucination_risk, variance, Computation, FALLBACK)
from .providers import get_llm


# ============================================================================
# State — flows through every node (this is what a checkpointer would persist)
# ============================================================================
class AgentState(TypedDict, total=False):
    question: str
    route: str                     # retrieve | compute | retrieve_compute | direct
    contexts: list[str]
    computations: list[Computation]
    answer: str
    risk: float
    grade: str                     # good | bad
    retries: int
    max_retries: int
    trace: list[str]               # human-readable path through the graph


# ============================================================================
# Supervisor — classifies intent and picks the route
# ============================================================================
COMPUTE_HINTS = ["variance", "budget", "actual", "growth", "difference",
                 "how much", "percent", "vs budget", "versus budget", "over budget"]
RETRIEVE_HINTS = ["what is", "explain", "deferred", "revenue", "recognition",
                  "nrr", "arr", "retention", "asc", "segment", "enterprise",
                  "policy", "how is", "describe", "which"]
GREETING_HINTS = ["hello", "hi", "hey", "thanks", "thank you", "good morning"]


def supervisor_node(state: AgentState) -> AgentState:
    """Rule-based router for offline verification. On Bedrock you can swap this
    for an LLM classification call — the routing contract is identical."""
    q = state["question"].lower()
    needs_compute = any(h in q for h in COMPUTE_HINTS)
    needs_retrieve = any(h in q for h in RETRIEVE_HINTS)
    is_greeting = (any(h in q for h in GREETING_HINTS)
                   and not any(h in q for h in COMPUTE_HINTS)
                   and not any(h in q for h in ["revenue", "deferred", "nrr", "arr",
                                                "asc", "variance", "segment", "retention"]))

    if is_greeting:
        route = "direct"
    elif needs_compute and needs_retrieve:
        route = "retrieve_compute"
    elif needs_compute:
        route = "compute"
    elif needs_retrieve:
        route = "retrieve"
    else:
        route = "retrieve"  # safe default: ground the answer in docs

    trace = state.get("trace", []) + [f"supervisor → route={route}"]
    return {**state, "route": route, "trace": trace}


def route_from_supervisor(state: AgentState) -> Literal["retrieve", "compute", "retrieve_compute", "direct"]:
    return state["route"]  # type: ignore


# ============================================================================
# Worker nodes
# ============================================================================
def make_retrieve_node(retriever: HybridRetriever):
    def retrieve_node(state: AgentState) -> AgentState:
        hits = retriever.retrieve(state["question"], top_k=CONFIG.top_k)
        scored = filter_by_confidence(hits)
        contexts = [s.text for s in scored[:CONFIG.top_k]]
        trace = state.get("trace", []) + [f"retrieve → {len(contexts)} contexts"]
        return {**state, "contexts": contexts, "trace": trace}
    return retrieve_node


def compute_node(state: AgentState) -> AgentState:
    """Extract budget/actual style numbers and compute deterministically.
    In production, structured params would come from an LLM tool-call or the
    query parser; here we detect the common 'budget X actual Y' shape."""
    q = state["question"]
    comps = list(state.get("computations", []))
    # If the caller already supplied computations, keep them.
    if not comps:
        nums = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", q)]
        if len(nums) >= 2:
            comps.append(variance("line item", budget=nums[0], actual=nums[1],
                                   higher_is_better=False))
    trace = state.get("trace", []) + [f"compute → {len(comps)} computation(s)"]
    return {**state, "computations": comps, "trace": trace}


def generate_node(state: AgentState) -> AgentState:
    contexts = state.get("contexts", [])
    comps = state.get("computations", [])
    if state["route"] == "direct":
        # No docs/math needed. On Bedrock this is a real Claude reply; offline
        # the stub is grounding-focused, so we give a simple direct response.
        if CONFIG.provider == "bedrock":
            answer = get_llm().generate(
                "You are a concise, friendly finance assistant.", state["question"])
        else:
            answer = ("Hello — I'm a financial analysis assistant. Ask me about "
                      "budgets, variances, revenue recognition, or SaaS metrics.")
        trace = state.get("trace", []) + ["generate → direct reply"]
        return {**state, "answer": answer, "risk": 0.0, "trace": trace}

    system, user = build_constrained_prompt(state["question"], contexts, comps)
    raw = llm_generate(system, user)
    risk = hallucination_risk(raw, contexts, comps)
    trace = state.get("trace", []) + [f"generate → risk={risk:.2f}"]
    return {**state, "answer": raw, "risk": risk, "trace": trace}


def grade_node(state: AgentState) -> AgentState:
    """Grade the answer. Bad if it fell back, hedged, or scored high risk."""
    risk = state.get("risk", 0.0)
    ans = state.get("answer", "")
    bad = (risk > CONFIG.hallucination_threshold
           or "could not find sufficient evidence" in ans.lower())
    grade = "bad" if bad else "good"
    trace = state.get("trace", []) + [f"grade → {grade} (risk={risk:.2f})"]
    return {**state, "grade": grade, "trace": trace}


def route_from_grade(state: AgentState) -> Literal["retry", "done"]:
    """Self-correction loop WITH A HARD LIMIT so it always terminates."""
    if state["grade"] == "good":
        return "done"
    if state.get("retries", 0) >= state.get("max_retries", 1):
        return "done"  # out of retries → return best effort / fallback
    return "retry"


def bump_retry_node(state: AgentState) -> AgentState:
    n = state.get("retries", 0) + 1
    trace = state.get("trace", []) + [f"retry #{n}"]
    return {**state, "retries": n, "trace": trace}


# ============================================================================
# Build the graph
# ============================================================================
def build_agent(retriever: HybridRetriever):
    g = StateGraph(AgentState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("retrieve", make_retrieve_node(retriever))
    g.add_node("compute", compute_node)
    g.add_node("generate", generate_node)
    g.add_node("grade", grade_node)
    g.add_node("bump_retry", bump_retry_node)

    g.set_entry_point("supervisor")

    # Supervisor routes to the right branch.
    g.add_conditional_edges("supervisor", route_from_supervisor, {
        "retrieve": "retrieve",
        "compute": "compute",
        "retrieve_compute": "retrieve",   # retrieve first, then compute
        "direct": "generate",
    })

    # retrieve_compute chains retrieve → compute; plain retrieve → generate.
    def after_retrieve(state: AgentState) -> Literal["compute", "generate"]:
        return "compute" if state["route"] == "retrieve_compute" else "generate"
    g.add_conditional_edges("retrieve", after_retrieve, {
        "compute": "compute", "generate": "generate",
    })

    g.add_edge("compute", "generate")

    # Direct replies skip grading (no grounding to check); others get graded.
    def after_generate(state: AgentState) -> Literal["grade", "done"]:
        return "done" if state["route"] == "direct" else "grade"
    g.add_conditional_edges("generate", after_generate, {
        "grade": "grade", "done": END,
    })

    # The self-correction loop with a hard cap.
    g.add_conditional_edges("grade", route_from_grade, {
        "retry": "bump_retry",
        "done": END,
    })
    g.add_edge("bump_retry", "retrieve")   # try retrieval again on a bad answer

    return g.compile()


class AgenticRAG:
    """Convenience wrapper around the compiled graph."""

    def __init__(self, retriever: HybridRetriever | None = None):
        if retriever is None:
            from .ingestion import ingest_dir
            from datetime import datetime, timezone, timedelta
            chunks = ingest_dir()
            now = datetime.now(timezone.utc)
            for c in chunks:
                if c.doc_id.startswith("acme_10k"):
                    c.metadata.update(source_type="audited_filing", doc_date=now - timedelta(days=20))
                else:
                    c.metadata.update(source_type="internal_memo", doc_date=now - timedelta(days=200))
            retriever = HybridRetriever(chunks)
        self.app = build_agent(retriever)

    def ask(self, question: str, computations: list[Computation] | None = None,
            max_retries: int = 1) -> dict:
        state: AgentState = {
            "question": question,
            "computations": computations or [],
            "retries": 0,
            "max_retries": max_retries,
            "trace": [],
        }
        final = self.app.invoke(state)
        return {
            "answer": final.get("answer", FALLBACK),
            "route": final.get("route"),
            "risk": final.get("risk", 0.0),
            "grade": final.get("grade"),
            "retries": final.get("retries", 0),
            "path": final.get("trace", []),
        }


if __name__ == "__main__":
    print(f"Provider: {CONFIG.provider}\n")
    agent = AgenticRAG()

    examples = [
        ("Hello there", None),
        ("What is deferred revenue under ASC 606?", None),
        ("Explain the Sales & Marketing spend variance versus budget.",
         [variance("Sales & Marketing", 3200, 3560, higher_is_better=False)]),
    ]
    for q, comps in examples:
        out = agent.ask(q, computations=comps)
        print(f"Q: {q}")
        print(f"   route={out['route']}  grade={out['grade']}  retries={out['retries']}")
        print(f"   path: {' | '.join(out['path'])}")
        print(f"   answer: {out['answer'][:90].strip()}")
        print()
