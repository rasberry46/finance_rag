# Step 3 — Constrained Generation, Citations & Hallucination Detection — Colab

This is where **Claude gets called for the first time**. Builds on Steps 1 & 2.
Assumes `chunks`, `retriever` (or `index`/`bm25`), and credentials exist from earlier.

---

### Cell 1 — Wire the real Claude client (same pattern as Titan)

```python
import json, re

# Reuse the `client` (bedrock-runtime) you built in Step 1.
CLAUDE_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"   # adjust if your list showed a different id

def claude_generate(system, user, max_tokens=512):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": 0,                 # deterministic for finance
        "system": system,
        "messages": [{"role": "user", "content": user}],
    })
    resp = client.invoke_model(modelId=CLAUDE_ID, body=body)
    return json.loads(resp["body"].read())["content"][0]["text"]

# Smoke test — first real Claude call
print(claude_generate("You are concise.", "Say 'Claude on Bedrock is working' and nothing else."))
```

Expected: `Claude on Bedrock is working`

### Cell 2 — The deterministic math (LLM never does this)

```python
def variance(line_item, budget, actual, higher_is_better=True):
    abs_var = actual - budget
    pct = (abs_var / budget * 100.0) if budget else float("inf")
    favorable = (abs_var >= 0) == higher_is_better
    detail = (f"budget={budget:,.0f}, actual={actual:,.0f}, "
              f"variance={abs_var:,.0f} ({pct:+.1f}%), "
              f"{'favorable' if favorable else 'unfavorable'}")
    return f"{line_item}: {detail}"

comp = variance("Sales & Marketing", budget=3200, actual=3560, higher_is_better=False)
print("COMPUTED IN CODE:", comp)   # Python did the math, not the LLM
```

Expected: `Sales & Marketing: budget=3,200, actual=3,560, variance=360 (+11.2%), unfavorable`

### Cell 3 — Constrained prompt + hallucination guard

```python
SYSTEM = """You are a financial analysis assistant for an FP&A team.
HARD RULES:
- Use ONLY the numbers in <computed_facts> and the text in <context>.
- NEVER perform arithmetic yourself. All numbers are pre-computed; quote them exactly.
- Cite every factual claim with [S<n>] referring to the numbered context blocks.
- If the context does not support an answer, reply EXACTLY:
  "I could not find sufficient evidence in the provided sources."
- Be concise: 2-4 sentences."""

def build_prompt(question, contexts, facts):
    ctx  = "\n".join(f"[S{i+1}] {c}" for i, c in enumerate(contexts))
    fct  = "\n".join(facts) if facts else "(none)"
    return f"<computed_facts>\n{fct}\n</computed_facts>\n\n<context>\n{ctx}\n</context>\n\nQuestion: {question}"

def hallucination_risk(answer, contexts, facts):
    a, risk = answer.lower(), 0.0
    if any(h in a for h in ["probably","might","maybe","i think","likely","possibly"]):
        risk += 0.35
    if not re.search(r"\[s\d+\]", a):
        risk += 0.35
    grounded = set(re.findall(r"\d[\d,\.]*", " ".join(contexts)+" "+" ".join(facts)))
    fabricated = [n for n in set(re.findall(r"\d[\d,\.]*", answer)) if n not in grounded]
    if fabricated: risk += min(0.4, 0.2*len(fabricated))
    return min(1.0, risk)
```

### Cell 4 — Full flow: retrieve → confidence → compute → Claude → guard

```python
from src.prodrag.confidence import filter_by_confidence

q = "Explain the Sales & Marketing spend variance versus budget."

# retrieve (use your Step 1 retriever OR the query() function)
hits = retriever.retrieve(q, top_k=5)
scored = filter_by_confidence(hits)
contexts = [s.text for s in scored[:5]]

# build prompt with the CODE-computed number
user = build_prompt(q, contexts, [comp])

# REAL Claude call
raw = claude_generate(SYSTEM, user)
risk = hallucination_risk(raw, contexts, [comp])

FALLBACK = "I could not find sufficient evidence in the provided sources."
final = FALLBACK if risk > 0.5 else raw

print("Q:", q)
print("\nCLAUDE ANSWER:\n", raw)
print(f"\nrisk={risk:.2f}  citations={re.findall(r'\\[S\\d+\\]', raw)}")
print("FINAL (after guard):", final)
```

**What to observe:**
- Claude explains *why* S&M was over budget, quoting the `360 (+11.2%)` number
  it was **handed** — it did not compute it.
- The answer contains `[S1]`-style citations.
- `risk` is low (grounded). If Claude had invented a number, risk would exceed
  0.5 and `final` would show the fallback.

### Cell 5 — Prove the guard: force a no-evidence question

```python
q2 = "What was the CEO's exact salary in 2019?"   # not in any document
hits2 = retriever.retrieve(q2, top_k=5)
scored2 = filter_by_confidence(hits2)
ctx2 = [s.text for s in scored2[:5]]
raw2 = claude_generate(SYSTEM, build_prompt(q2, ctx2, []))
print("Q:", q2)
print("ANSWER:", raw2)   # should be the exact fallback sentence
```

Expected: Claude returns *"I could not find sufficient evidence in the provided sources."*
because the constrained prompt forbids inventing an answer.

---

**Verify:** Cell 4 gives a cited, grounded narrative quoting the computed number;
Cell 5 returns the fallback. Then tell me to build **Step 4: Evaluation & Caching**
(Precision/Recall/F1, TTL cache, conversation memory, the "1000x latency reduction"
on cache hits).
```
