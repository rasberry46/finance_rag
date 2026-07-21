import os
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("RAG_STORE", "opensearch")

from src.prodrag.store_factory import get_store

GOLDEN = [
    ("What are the total loans outstanding gross?", "84,000"),
    ("What are the net loans outstanding?", "77,000"),
    ("What is the loan loss reserve?", "7,000"),
    ("What are the cash and bank current accounts?", "5,000"),
    ("What is total equity?", "45,500"),
]

def is_relevant(chunk_text, needle):
    t = chunk_text.replace(",", "")
    return needle.replace(",", "") in t

store = get_store()
K = 5
P=R=F=MRR=HIT=0.0
n=len(GOLDEN)
print("="*70)
print("RETRIEVAL METRICS (hybrid BM25+kNN, top-%d) - live OpenSearch" % K)
print("="*70)
for q, needle in GOLDEN:
    hits = store.retrieve(q, top_k=K)
    rel_flags = [is_relevant(h.chunk.text, needle) for h in hits]
    n_rel = sum(rel_flags)
    precision = n_rel / len(hits) if hits else 0.0
    recall = 1.0 if n_rel > 0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    mrr = 0.0
    for i, flag in enumerate(rel_flags, 1):
        if flag:
            mrr = 1.0/i; break
    hit = 1.0 if n_rel > 0 else 0.0
    P+=precision; R+=recall; F+=f1; MRR+=mrr; HIT+=hit
    print("\nQ: %s" % q)
    print("   relevant in top-%d: %d/%d | precision=%.2f recall=%.2f f1=%.2f mrr=%.2f" %
          (K, n_rel, len(hits), precision, recall, f1, mrr))
print("\n"+"-"*70)
print("AVERAGES: Precision@%d=%.2f  Recall@%d=%.2f  F1=%.2f  MRR=%.2f  Hit@%d=%.2f" %
      (K, P/n, K, R/n, F/n, MRR/n, K, HIT/n))
print("="*70)
