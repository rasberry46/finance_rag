# VS Code Local Setup — One-Time, ~10 Minutes

This ends the Colab thrashing for good. Files persist, credentials live in `.env`,
nothing drops on disconnect. Do this once tomorrow morning before building.

## 1. Get the project onto your machine

Unzip `prod-rag-final.zip` (or the latest) somewhere permanent, e.g.
`~/projects/prod-rag`. Open that folder in VS Code (`File → Open Folder`).

## 2. Create the virtual environment (once)

Open the VS Code terminal (`Ctrl+``) and run:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install opensearch-py python-dotenv streamlit   # extras for the full build
```

Tell VS Code to use this interpreter: `Cmd/Ctrl+Shift+P` → "Python: Select
Interpreter" → pick the one in `.venv`.

## 3. Create your .env (credentials live here, once — never in code)

Copy `.env.example` to `.env` and fill in:

```
RAG_PROVIDER=bedrock
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...

# OpenSearch (from tonight — the domain that's already green + indexed)
OPENSEARCH_ENDPOINT=https://search-rag-demo-XXXX.us-east-1.es.amazonaws.com
OPENSEARCH_USER=your-master-user
OPENSEARCH_PASSWORD=your-master-password
OPENSEARCH_INDEX=financial-rag

# S3
S3_BUCKET=rag81
S3_PREFIX=pdfs/
```

**Add `.env` to `.gitignore`** (already is) so credentials never get committed.

## 4. Load .env automatically

Two options — either works:

**A. Let the code load it.** Add this to the top of any script you run:
```python
from dotenv import load_dotenv; load_dotenv()
```
(We'll bake this into the entry points.)

**B. Export in the shell** before running:
```bash
export $(grep -v '^#' .env | xargs)
```

## 5. Verify everything works (the "nothing drops" test)

```bash
# offline sanity — should print 20 passing tests
pytest -q

# real Bedrock — should print a Titan dim of 1024
RAG_PROVIDER=bedrock python -c "from dotenv import load_dotenv; load_dotenv(); from src.prodrag.providers import get_embedder; print('dim:', len(get_embedder().embed(['test'])[0]))"
```

If both pass, your environment is bulletproof. No more re-uploads, no more
credential drops. Every file you create persists. You can close VS Code and
reopen it and everything is exactly where you left it.

## Why this is so much better than Colab for a multi-day build
- Files persist on disk — no re-uploading the repo every session.
- `.env` holds credentials once — no re-pasting keys, no "security token invalid."
- `git commit` after each working step — you can always roll back.
- The OpenSearch index you built tonight is already populated and persists
  independently — you just reconnect to it.

## First thing to run tomorrow
Once setup is done, we start **live OpenSearch integration** — wiring the real
store into the agent. See `PLAN_3DAY.md`.
