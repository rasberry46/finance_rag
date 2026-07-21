from __future__ import annotations
import time
from .evaluation import TTLCache, ConversationMemory


class CachedAgent:
    def __init__(self, agent, ttl_seconds: int = 3600, max_turns: int = 10):
        self.agent = agent
        self.cache = TTLCache(ttl_seconds=ttl_seconds)
        self.memory = ConversationMemory(max_turns=max_turns)

    def ask(self, question: str, **kwargs) -> dict:
        t0 = time.perf_counter()
        cached = self.cache.get(question)
        if cached is not None:
            elapsed = (time.perf_counter() - t0) * 1000
            out = dict(cached)
            out["cached"] = True
            out["latency_ms"] = elapsed
            out["cache_hit_rate"] = self.cache.hit_rate
            self.memory.add(question, out.get("answer", ""))
            return out
        result = self.agent.ask(question, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        result["cached"] = False
        result["latency_ms"] = elapsed
        result["cache_hit_rate"] = self.cache.hit_rate
        self.cache.set(question, result)
        self.memory.add(question, result.get("answer", ""))
        return result

    def history(self) -> str:
        return self.memory.as_context()

    def turns(self) -> int:
        return len(self.memory)
