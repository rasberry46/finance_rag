"""Standalone RAGAS-style eval — run: python run_ragas.py"""
import os, re
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("RAG_STORE", "opensearch")

from src.prodrag.store_factory import get_store
from src.prodrag.agentic import AgenticRAG
from src.prodrag.providers import get_llm

GOLDEN = [
    ("What are the total loans outstanding (gross) on the balance sheet?", "84,000"),
    ("What are the net loans outstanding?", "77,000"),
    ("What is the loan loss reserve?", "7,000"),
    ("What are the cash and bank current accounts?", "5,000"),
    ("What is total equity?", "45,500"),
]

def score(system, user):
    raw = get_llm().generate(system, user).strip()
    m = re.search(r"(?:0?\.\d+|0|1(?:\.0+)?)", raw)
    try: return max(0.0, min(1.0, float(m.group(0)))) if m else 0.0
    except: return 0.0

def faithfulness(ans, ctx):
    if "could not find" in ans.lower(): return 1.0
    return score("Score 0.0-1.0 how much the ANSWER is supported by CONTEXT. 1.0=all supported, 0.0=fabricated. Reply ONLY the number.",
                 f"CONTEXT:\n{chr(10).join(ctx)}\n\nANSWER:\n{ans}\n\nScore:")

def relevancy(ans, q):
    return score("Score 0.0-1.0 how well ANSWER addresses QUESTION (on-topic). Reply ONLY the number.",
                 f"QUESTION: {q}\n\nANSWER: {ans}\n\nScore:")

def ctx_precision(ctx, q):
    return score("Score 0.0-1.0: fraction of CONTEXT chunks relevant to QUESTION. Reply ONLY the number.",
                 f"QUESTION: {q}\n\nCONTEXT:\n{chr(10).join(ctx)}\n\nScore:")

def ctx_recall(ctx, gt):
    joined = " ".join(ctx)
    gt_num = gt.replace(",", "")
    nums = {n.replace(",", "") for n in re.findall(r"\d[\d,]*", joined)}
    return 1.0 if gt_num in nums else 0.0

store = get_store()
agent = AgenticRAG(retriever=store)

print("="*72)
print("RAGAS-STYLE EVALUATION (LLM-judged) - live OpenSearch")
print("="*72)
sums = {"f":0,"r":0,"p":0,"rc":0}
for q, gt in GOLDEN:
    res = agent.ask(q)
    ans = res["answer"]
    hits = store.retrieve(q, top_k=5)
    ctx = [h.chunk.text for h in hits]
    f, r, p, rc = faithfulness(ans,ctx), relevancy(ans,q), ctx_precision(ctx,q), ctx_recall(ctx,gt)
    sums["f"]+=f; sums["r"]+=r; sums["p"]+=p; sums["rc"]+=rc
    print(f"\nQ: {q}")
    print(f"   answer: {ans[:65].strip()}")
    print(f"   faithfulness={f:.2f} relevancy={r:.2f} ctx_precision={p:.2f} ctx_recall={rc:.2f}")
n = len(GOLDEN)
print("\n"+"-"*72)
print(f"AVERAGES: faithfulness={sums['f']/n:.2f} relevancy={sums['r']/n:.2f} ctx_precision={sums['p']/n:.2f} ctx_recall={sums['rc']/n:.2f}")
print("="*72)