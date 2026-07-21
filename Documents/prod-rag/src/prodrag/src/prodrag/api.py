from __future__ import annotations
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()
os.environ.setdefault("RAG_STORE", "opensearch")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    tenant_id: str | None = Field(None, description="For multi-tenant isolation (future use)")

class AskResponse(BaseModel):
    answer: str
    route: str
    risk: float
    cached: bool
    latency_ms: float
    sources: list[str]
    path: list[str]

class SQLRequest(BaseModel):
    question: str

class SQLResponse(BaseModel):
    ok: bool
    generated_sql: str
    safe_sql: str
    reason: str
    rows: list

STATE: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.prodrag.store_factory import get_store
    from src.prodrag.agentic import AgenticRAG
    from src.prodrag.conversational_agent import ConversationalAgent
    store = get_store()
    STATE["store"] = store
    STATE["agent"] = ConversationalAgent(AgenticRAG(retriever=store), store)
    yield
    STATE.clear()

app = FastAPI(title="Production Financial RAG API",
              description="Agentic RAG over financial documents - AWS Bedrock + OpenSearch.",
              version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok",
            "provider": os.environ.get("RAG_PROVIDER", "local"),
            "store": os.environ.get("RAG_STORE", "opensearch"),
            "agent_ready": "agent" in STATE}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    agent = STATE.get("agent"); store = STATE.get("store")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")
    try:
        result = agent.ask(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"agent error: {e}")
    srcs = []
    for h in store.retrieve(req.question, top_k=5):
        s = h.chunk.metadata.get("source") or h.chunk.doc_id or "unknown"
        p = h.chunk.metadata.get("page")
        tag = f"{s} (p{p})" if p else s
        if tag not in srcs:
            srcs.append(tag)
    return AskResponse(answer=result["answer"], route=result.get("route", "-"),
                       risk=float(result.get("risk", 0)), cached=bool(result.get("cached", False)),
                       latency_ms=float(result.get("latency_ms", 0)), sources=srcs,
                       path=result.get("path", []))

@app.post("/sql", response_model=SQLResponse)
def sql(req: SQLRequest):
    from src.prodrag.text_to_sql import ask_sql
    try:
        r = ask_sql(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"sql error: {e}")
    return SQLResponse(ok=r.ok, generated_sql=r.generated_sql, safe_sql=r.safe_sql,
                       reason=r.reason, rows=[list(row) for row in r.rows])

@app.get("/metrics")
def metrics():
    agent = STATE.get("agent")
    if agent is None:
        return {"turns": 0, "cache_hit_rate": 0.0}
    return {"turns": agent.turn_count(), "cache_hit_rate": round(agent.cache.hit_rate, 3)}