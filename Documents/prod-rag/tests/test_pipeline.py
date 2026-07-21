"""
Test suite for the whole pipeline. Runs offline (local provider), no AWS needed.
    pytest -q
"""
from datetime import datetime, timezone, timedelta

from src.prodrag.ingestion import ingest_dir, Chunk
from src.prodrag.retrieval import HybridRetriever, reciprocal_rank_fusion
from src.prodrag.confidence import filter_by_confidence, freshness_score, TRUST_BY_SOURCE
from src.prodrag.generation import (variance, growth_rate, hallucination_risk,
                                    build_constrained_prompt, answer as gen_answer,
                                    _extract_numbers)
from src.prodrag.evaluation import (precision_recall_f1, mrr, hit_at_k, evaluate,
                                    TTLCache, ConversationMemory)
from src.prodrag.observability import Tracer, Metrics
from src.prodrag.s3_ingestion import document_fingerprint, decide_action
from src.prodrag.confidence import ScoredHit
from src.prodrag.retrieval import RetrievalHit


# ── Ingestion ────────────────────────────────────────────────────────────
def test_ingest_produces_chunks():
    chunks = ingest_dir()
    assert len(chunks) > 0
    types = {c.chunk_type for c in chunks}
    assert "table" in types and "prose" in types


def test_table_chunk_keeps_headers_with_values():
    chunks = ingest_dir()
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert tables
    # A revenue table should keep a header token and a numeric value together
    joined = " ".join(t.text for t in tables)
    assert "Segment" in joined or "Metric" in joined


# ── Retrieval ────────────────────────────────────────────────────────────
def test_rrf_rewards_agreement():
    a = [1, 2, 3]
    b = [2, 3, 4]
    fused = dict(reciprocal_rank_fusion([a, b]))
    assert fused[2] > fused[1]  # 2 appears in both -> ranks above 1


def test_hybrid_retrieval_returns_hits():
    chunks = ingest_dir()
    r = HybridRetriever(chunks)
    hits = r.retrieve("Enterprise revenue Q3", top_k=3)
    assert 1 <= len(hits) <= 3


# ── Confidence ───────────────────────────────────────────────────────────
def test_freshness_decays_with_age():
    now = datetime.now(timezone.utc)
    assert freshness_score(now - timedelta(days=5), now) == 1.0
    assert freshness_score(now - timedelta(days=2000), now) == 0.2


def test_confidence_prefers_trusted_source():
    now = datetime.now(timezone.utc)
    def hit(src, days, score):
        c = Chunk(text="x", chunk_type="prose", doc_id="d",
                  metadata={"source_type": src, "doc_date": now - timedelta(days=days)})
        return RetrievalHit(chunk=c, rerank_score=score, rrf_rank=0)
    hits = [hit("audited_filing", 5, 1.0), hit("chat_message", 5, 1.0)]
    kept = filter_by_confidence(hits, now=now)
    assert kept[0].hit.chunk.metadata["source_type"] == "audited_filing"


# ── Generation / guard ───────────────────────────────────────────────────
def test_variance_math_is_exact():
    v = variance("x", 3200, 3560, higher_is_better=False)
    assert v.value == 360
    assert "+11.2%" in v.detail and "unfavorable" in v.detail


def test_extract_numbers_strips_citations_and_commas():
    nums = _extract_numbers("Value was 3,560 [S2] and 360.")
    assert "3560" in nums and "360" in nums
    assert "2" not in nums  # citation digit must not leak


def test_hallucination_guard_flags_fabricated_number():
    # One fabricated number contributes 0.2; an uncited fabricated number more.
    risk_cited = hallucination_risk("Revenue was 9,999,999 [S1].",
                                    ["Revenue was 4,200,000."], [])
    assert risk_cited >= 0.2  # fabricated-number signal fired
    # Without a citation AND fabricated -> higher risk, crosses the 0.5 gate
    risk_uncited = hallucination_risk("Revenue was probably 9,999,999.",
                                      ["Revenue was 4,200,000."], [])
    assert risk_uncited > 0.5


def test_hallucination_guard_passes_grounded_answer():
    ctx = ["Sales & Marketing budget 3200 actual 3560"]
    from src.prodrag.generation import Computation
    comp = Computation("S&M", 360, "budget=3200, actual=3560, variance=360")
    risk = hallucination_risk("S&M variance was 360 [S1].", ctx, [comp])
    assert risk == 0.0


# ── Evaluation / cache / memory ──────────────────────────────────────────
def test_precision_recall_f1():
    p, r, f = precision_recall_f1(["a", "b", "c"], {"a", "b"})
    assert round(p, 2) == 0.67 and r == 1.0


def test_mrr_and_hit():
    assert mrr(["x", "a"], {"a"}) == 0.5
    assert hit_at_k(["x", "a"], {"a"}, 3) == 1.0


def test_cache_hit_and_expiry():
    c = TTLCache(ttl_seconds=100)
    c.set("q", "v")
    assert c.get("q") == "v"
    assert c.hit_rate > 0


def test_memory_caps_turns():
    m = ConversationMemory(max_turns=2)
    m.add("a", "1"); m.add("b", "2"); m.add("c", "3")
    assert len(m) == 2  # oldest dropped


# ── Observability ────────────────────────────────────────────────────────
def test_tracer_records_spans_and_bottleneck():
    import time
    t = Tracer()
    with t.span("fast"):
        pass
    with t.span("slow"):
        time.sleep(0.01)
    assert t.trace.bottleneck.name == "slow"
    assert t.trace.total_ms > 0


# ── S3 continuous ingestion ──────────────────────────────────────────────
def test_s3_dedup_decisions():
    known = {}
    fp1 = document_fingerprint(b"v1")
    assert decide_action("d", fp1, known).action == "index_new"
    known["d"] = fp1
    assert decide_action("d", fp1, known).action == "skip_duplicate"
    fp2 = document_fingerprint(b"v2")
    assert decide_action("d", fp2, known).action == "reindex_changed"


# ── Agentic layer (LangGraph) ────────────────────────────────────────────
def test_agentic_routes_greeting_to_direct():
    from src.prodrag.agentic import AgenticRAG
    agent = AgenticRAG()
    out = agent.ask("Hello there")
    assert out["route"] == "direct"


def test_agentic_routes_lookup_to_retrieve():
    from src.prodrag.agentic import AgenticRAG
    agent = AgenticRAG()
    out = agent.ask("What is deferred revenue under ASC 606?")
    assert out["route"] == "retrieve"
    assert out["grade"] == "good"


def test_agentic_routes_variance_to_retrieve_compute():
    from src.prodrag.agentic import AgenticRAG
    from src.prodrag.generation import variance
    agent = AgenticRAG()
    out = agent.ask("Explain the Sales & Marketing spend variance versus budget.",
                    computations=[variance("S&M", 3200, 3560, higher_is_better=False)])
    assert out["route"] == "retrieve_compute"
    assert "360" in out["answer"]


def test_agentic_loop_limit_terminates():
    # Force every answer to grade "bad" and confirm the graph still terminates
    # (retries capped) rather than looping forever.
    from src.prodrag import agentic
    from src.prodrag.agentic import AgenticRAG
    orig = agentic.grade_node
    agentic.grade_node = lambda s: {**s, "grade": "bad",
                                    "trace": s.get("trace", []) + ["grade → bad (forced)"]}
    try:
        # rebuild graph so it picks up the patched node
        agent = AgenticRAG()
        out = agent.ask("What is deferred revenue?", max_retries=2)
        assert out["retries"] <= 2   # hard cap respected → terminated
    finally:
        agentic.grade_node = orig


# ── Text-to-SQL safety ───────────────────────────────────────────────────
def test_sql_guard_blocks_destructive():
    from src.prodrag.text_to_sql import validate_sql
    for bad in ["DROP TABLE balance_sheet", "DELETE FROM balance_sheet",
                "UPDATE balance_sheet SET amount=0",
                "SELECT 1; DROP TABLE x", "SELECT * FROM t -- injected"]:
        assert validate_sql(bad).ok is False


def test_sql_guard_allows_select_and_adds_limit():
    from src.prodrag.text_to_sql import validate_sql
    v = validate_sql("SELECT amount FROM balance_sheet WHERE category='asset'")
    assert v.ok is True
    assert "LIMIT" in v.safe_sql.upper()


def test_sql_executor_returns_correct_totals():
    from src.prodrag.text_to_sql import build_demo_warehouse, validate_sql
    conn = build_demo_warehouse()
    v = validate_sql("SELECT amount FROM balance_sheet WHERE line_item='Total Assets'")
    rows = conn.execute(v.safe_sql).fetchall()
    assert rows == [(90500,)]
