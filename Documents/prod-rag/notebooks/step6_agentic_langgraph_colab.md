# Agentic RAG (LangGraph) — Colab

The capstone. Turns the fixed pipeline into an agent that inspects each query and
routes it: retrieve, compute, both, or answer directly — with a self-correction
loop and a hard retry cap. Maps directly to production LangGraph multi-agent work.

Builds on the full repo. Assumes credentials + rebuilt state from earlier steps.

---

### Cell 1 — Install LangGraph

```python
!pip -q install langgraph
```

### Cell 2 — An LLM-based supervisor (optional upgrade over rule-based)

On Bedrock you can let Claude classify intent instead of keyword rules. This is
the more impressive version for the interview.

```python
import json

def llm_route(question):
    system = """Classify the finance question into exactly one route. Reply with ONLY the route word.
Routes:
- retrieve : needs document lookup (definitions, policies, "what is", metrics)
- compute : pure math on numbers in the question
- retrieve_compute : needs BOTH document lookup AND a calculation (variance, growth)
- direct : greeting or small talk, no docs/math needed"""
    r = claude_generate(system, f"Question: {question}\nRoute:")
    route = r.strip().lower().split()[0]
    return route if route in {"retrieve","compute","retrieve_compute","direct"} else "retrieve"

for q in ["Hello", "What is deferred revenue?",
          "Explain the S&M variance vs budget", "What is 12% of 3200?"]:
    print(f"{route:>18} ← {q}" if (route:=llm_route(q)) else q)
```

**Observe:** Claude routes each query correctly. This LLM router can replace the
rule-based `supervisor_node` in `agentic.py` — the graph is unchanged.

### Cell 3 — Run the compiled graph

```python
import sys
for m in list(sys.modules):
    if 'prodrag' in m: del sys.modules[m]
sys.path.insert(0, '.')

from src.prodrag.agentic import AgenticRAG
from src.prodrag.generation import variance

# NOTE: set RAG_PROVIDER=bedrock in the environment so nodes call Claude.
import os; os.environ["RAG_PROVIDER"] = "bedrock"

agent = AgenticRAG()

tests = [
    ("Hello there", None),
    ("What is deferred revenue under ASC 606?", None),
    ("Explain the Sales & Marketing spend variance versus budget.",
     [variance("Sales & Marketing", 3200, 3560, higher_is_better=False)]),
]
for q, comps in tests:
    out = agent.ask(q, computations=comps)
    print(f"\nQ: {q}")
    print(f"   route={out['route']}  grade={out['grade']}  retries={out['retries']}")
    print(f"   path: {' | '.join(out['path'])}")
    print(f"   answer: {out['answer'][:120].strip()}")
```

**Observe the `path`** — it shows the exact route through the graph for each
query. Different queries take different paths. That's the agent deciding.

### Cell 4 — Prove the loop limit (the key production point)

```python
# Temporarily force every answer to grade "bad" → confirm it STILL terminates.
from src.prodrag import agentic
orig = agentic.grade_node
agentic.grade_node = lambda s: {**s, "grade":"bad", "trace": s.get("trace",[])+["grade→bad(forced)"]}
try:
    agent2 = AgenticRAG()
    out = agent2.ask("What is deferred revenue?", max_retries=3)
    print("retries used:", out["retries"], "(capped at 3)")
    print("terminated cleanly:", out["retries"] <= 3)
finally:
    agentic.grade_node = orig
```

**Observe:** even when the grader always says "bad," the graph stops after
`max_retries` instead of looping forever. This is how you prevent an agent from
running up thousands in token costs — the #1 agent production-readiness question.

### Cell 5 — Visualize the graph (optional)

```python
from src.prodrag.agentic import build_agent
app = build_agent(agent.app.__wrapped__ if hasattr(agent.app,'__wrapped__') else None) if False else agent.app
try:
    print(app.get_graph().draw_ascii())
except Exception as e:
    print("ASCII draw needs grandalf; mermaid instead:")
    print(app.get_graph().draw_mermaid())
```

---

**This completes the whole project**: 5 production steps + agentic routing +
continuous-S3 architecture. For the interview, the story is:
"I built a hybrid-retrieval RAG on Bedrock with confidence scoring and a
deterministic-math-plus-guard generation layer, then wrapped it in a LangGraph
supervisor that routes each query and self-corrects — with hard loop limits for
cost control — and an event-driven S3 ingestion path for continuous documents."
```
