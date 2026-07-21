from __future__ import annotations
import time
from .evaluation import TTLCache, ConversationMemory


class TracedAgent:
    def __init__(self, agent, store, ttl_seconds: int = 3600, max_turns: int = 10):
        self.agent = agent
        self.store = store
        self.cache = TTLCache(ttl_seconds=ttl_seconds)
        self.memory = ConversationMemory(max_turns=max_turns)

    def ask(self, question, **kwargs):
        t_start = time.perf_counter()
        cached = self.cache.get(question)
        if cached is not None:
            out = dict(cached)
            out["cached"] = True
            out["latency_ms"] = (time.perf_counter() - t_start) * 1000
            out["spans"] = [("cache lookup", out["latency_ms"])]
            out["cache_hit_rate"] = self.cache.hit_rate
            self.memory.add(question, out.get("answer", ""))
            return out
        spans = []
        t0 = time.perf_counter()
        _ = self.store.retrieve(question, top_k=5)
        spans.append(("retrieval: Titan embed + OpenSearch hybrid", (time.perf_counter() - t0) * 1000))
        t1 = time.perf_counter()
        result = self.agent.ask(question, **kwargs)
        agent_ms = (time.perf_counter() - t1) * 1000
        gen_ms = max(0.0, agent_ms - spans[0][1])
        spans.append(("generation: Claude + guard + grade", gen_ms))
        total = (time.perf_counter() - t_start) * 1000
        result["cached"] = False
        result["latency_ms"] = total
        result["spans"] = spans
        result["cache_hit_rate"] = self.cache.hit_rate
        self.cache.set(question, result)
        self.memory.add(question, result.get("answer", ""))
        return result

    def history(self):
        return list(self.memory.turns)

    def turns(self):
        return len(self.memory)
