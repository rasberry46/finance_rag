# Live Demo Guide — Streamlit on AWS Bedrock

A screen-shareable web UI that demonstrates all three story beats in one flow:
full pipeline, agentic routing, and the finance safety guard.

---

## Run it locally (recommended for a screen-share)

```bash
# from the prod-rag/ folder, with your venv active
pip install -r requirements.txt          # includes streamlit + langgraph
python -m scripts.make_sample_pdfs        # generate the sample PDFs (once)

# --- OFFLINE (works anywhere, no AWS needed — good for a dry run) ---
streamlit run demo_app.py

# --- REAL BEDROCK (for the actual demo) ---
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
export RAG_PROVIDER=bedrock
streamlit run demo_app.py
```

Streamlit opens a browser tab at `http://localhost:8501`. Share that tab.

> The sidebar shows a green **"Provider: AWS Bedrock"** badge when credentials
> are set — good proof to the interviewer it's hitting real AWS, not a mock.

---

## The scripted demo flow (4 clicks, ~2 minutes)

The app has four buttons across the top. Click them in this order and narrate:

**1. 💬 Greeting** — "Hello, what can you help me with?"
   - Point out: route = **direct**. The supervisor recognized this needs no
     documents or math and skipped retrieval entirely. *"The agent decides the
     path per query — it's not a fixed pipeline."*

**2. 📄 Lookup** — "What is deferred revenue under ASC 606?"
   - Point out: route = **retrieve**. It pulled trustworthy chunks (confidence
     scored by source + freshness) and grounded the answer with a `[S1]` citation.
     *"Every claim is cited back to a source."*

**3. 🧮 Variance** — "Explain the Sales & Marketing spend variance versus budget."
   - Point out: route = **retrieve + compute**. Scroll to the
     **"Deterministic computation"** box: *"Python computed the 360 / +11.2%
     exactly — the LLM never did the arithmetic, it only wrote the narrative.
     That removes the biggest risk in financial AI."*

**4. 🛡️ Safety** — "What was the CEO's exact salary in 2019?"
   - Point out: the red **"Safety guard fired"** callout. *"This isn't in the
     corpus, so instead of inventing a number, the agent returns an honest
     fallback. Zero-hallucination behavior — critical for audit and compliance."*

Then optionally type a free-form question, e.g.
*"What is the Net Revenue Retention rate?"* to show it's not hard-coded.

---

## What to emphasize while screen-sharing

- **The trace panel** ("Agent trace: the path this query took") — this is the
  visible proof of routing. Different queries show different paths.
- **The metrics row** — route, hallucination risk, retries, latency — live.
- **The architecture diagram** in the sidebar ties it together.

---

## If something goes wrong (quick recovery)

- **App won't start / import error** → make sure you're in the `prod-rag/`
  folder and ran `pip install -r requirements.txt`.
- **Bedrock 403 / AccessDenied** → the demo still works fully in **offline mode**
  (just don't set `RAG_PROVIDER=bedrock`). The behavior is identical; only the
  embeddings/LLM are local stand-ins. For a screen-share, offline is a perfectly
  safe fallback and looks the same.
- **Model ID error** → confirm `us.anthropic.claude-sonnet-4-6` is enabled in
  your account, or set `BEDROCK_LLM_ID` to one you have.
- **Slow first load** → the index builds once on startup (ingest + embed). It's
  cached after that; subsequent questions are fast.

---

## One-liner to have ready if asked "is this your Intuit system?"
> "No — this is a reference implementation I built to demonstrate the patterns
> end to end: hybrid retrieval, confidence scoring, deterministic-math-plus-guard
> generation, and LangGraph routing with loop limits. The production patterns are
> the same ones I use at work."
