import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("RAG_STORE", "opensearch")

st.set_page_config(page_title="Financial RAG Demo", layout="wide")
PROVIDER = os.environ.get("RAG_PROVIDER", "local")
STORE = os.environ.get("RAG_STORE", "opensearch")


@st.cache_resource(show_spinner="Connecting to OpenSearch + building agent...")
def load_agent():
    from src.prodrag.store_factory import get_store
    from src.prodrag.agentic import AgenticRAG
    from src.prodrag.conversational_agent import ConversationalAgent
    store = get_store()
    return ConversationalAgent(AgenticRAG(retriever=store), store, ttl_seconds=3600, max_turns=10), store


with st.sidebar:
    st.header("Status")
    st.success("LLM: AWS Bedrock" if PROVIDER == "bedrock" else "LLM: local")
    st.success("Store: OpenSearch (BM25 + kNN)" if STORE == "opensearch" else "Store: FAISS")
    st.divider()
    st.header("Agent graph")
    st.code("supervisor\n |- retrieve\n |- compute\n |- retrieve+compute\n |- direct\n     |\n  generate (guarded)\n     |\n   grade --bad--> retry\n     | good\n   answer")
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun()

st.title("Financial RAG - Agentic Demo")
st.caption("S3 . Titan . OpenSearch hybrid retrieval . Claude . LangGraph routing")

agent, store = load_agent()
st.success("Connected to live OpenSearch.")

if "chat" not in st.session_state:
    st.session_state.chat = []

for turn in st.session_state.chat:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        cols = st.columns(6)
        cols[0].metric("Route", turn.get("route", "-"))
        cols[1].metric("Risk", f"{turn.get('risk', 0):.2f}")
        cols[2].metric("Retries", turn.get("retries", 0))
        cols[3].metric("Latency", f"{turn.get('latency_ms', 0):.0f} ms")
        cols[4].metric("Cache", "HIT" if turn.get("cached") else "MISS")
        cols[5].metric("Follow-up", "YES" if turn.get("was_followup") else "no")
        with st.expander("Latency + sources + confidence + trace"):
            st.markdown("**Latency per layer:**")
            for layer, ms in turn.get("spans", []):
                st.markdown(f"- `{layer}` **{ms:.0f} ms**")
            if turn.get("sources"):
                st.markdown("**Sources:**")
                for s in turn["sources"]:
                    st.markdown(f"- {s}")
            if turn.get("chunks"):
                st.markdown("**Retrieved chunks (ranked by confidence = relevance + trust + freshness):**")
                for i, c in enumerate(turn["chunks"], 1):
                    st.markdown(f"- **[S{i}]** confidence={c['conf']:.3f} (rel={c['rel']:.2f} trust={c['trust']:.2f} fresh={c['fresh']:.2f}) - {c['src']}")
            st.markdown("**Agent trace:**")
            for step in turn.get("path", []):
                st.markdown(f"- `{step}`")

st.markdown("###### Examples (try a follow-up like 'what about L Oreal?' after):")
ex = st.columns(4)
examples = ["What is total equity?", "What are Nestle total assets?",
            "what about L Oreal?", "Hello, what can you do?"]
picked = None
for c, e in zip(ex, examples):
    if c.button(e, use_container_width=True):
        picked = e

typed = st.chat_input("Ask a question (or a follow-up)...")
question = picked or typed

if question:
    with st.spinner("Running the agent..."):
        result = agent.ask(question)
    hits = store.retrieve(question, top_k=5)
    srcs = []
    chunks_meta = []
    for h in hits:
        s = h.chunk.metadata.get("source") or h.chunk.doc_id or "unknown"
        p = h.chunk.metadata.get("page")
        tag = f"{s} (p{p})" if p else s
        if tag not in srcs:
            srcs.append(tag)
        chunks_meta.append({
            "conf": h.chunk.metadata.get("confidence", 0),
            "rel": h.rerank_score,
            "trust": h.chunk.metadata.get("trust", 0),
            "fresh": h.chunk.metadata.get("freshness", 0),
            "src": tag,
        })
    st.session_state.chat.append({
        "q": question, "answer": result["answer"], "route": result.get("route"),
        "risk": result.get("risk", 0), "retries": result.get("retries", 0),
        "latency_ms": result.get("latency_ms", 0), "cached": result.get("cached"),
        "was_followup": result.get("was_followup"), "spans": result.get("spans", []),
        "path": result.get("path", []), "sources": srcs, "chunks": chunks_meta,
    })
    st.rerun()
