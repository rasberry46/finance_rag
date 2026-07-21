from __future__ import annotations
import time
from collections import deque
from .evaluation import TTLCache


class ConversationalAgent:
    def __init__(self, agent, store, ttl_seconds: int = 3600, max_turns: int = 10):
        self.agent = agent
        self.store = store
        self.cache = TTLCache(ttl_seconds=ttl_seconds)
        self.turns = deque(maxlen=max_turns)

    def _looks_like_followup(self, q):
        ql = q.lower().strip()
        cues = ["what about", "and ", "how about", "that", "those", "same for",
                "compare", "also", "then", "previous", "last one"]
        return len(ql.split()) <= 6 or any(ql.startswith(c) for c in cues)

    def _augment(self, question):
        if not self.turns or not self._looks_like_followup(question):
            return question
        recent = list(self.turns)[-3:]
        hist = "\n".join(f"Earlier Q: {q}\nEarlier A: {a[:200]}" for q, a in recent)
        return (f"Conversation so far:\n{hist}\n\nFollow-up question (resolve any "
                f"references using the conversation above): {question}")

    def ask(self, question, **kwargs):
        t_start = time.perf_counter()
        cached = self.cache.get(question)
        if cached is not None:
            out = dict(cached)
            out["cached"] = True
            out["latency_ms"] = (time.perf_counter() - t_start) * 1000
            out["spans"] = [("cache lookup", out["latency_ms"])]
            out["cache_hit_rate"] = self.cache.hit_rate
            self.turns.append((question, out.get("answer", "")))
            return out
        augmented = self._augment(question)
        was_followup = augmented != question
        spans = []
        t0 = time.perf_counter()
        _ = self.store.retrieve(augmented, top_k=5)
        spans.append(("retrieval: Titan embed + OpenSearch hybrid", (time.perf_counter() - t0) * 1000))
        t1 = time.perf_counter()
        result = self.agent.ask(augmented, **kwargs)
        agent_ms = (time.perf_counter() - t1) * 1000
        spans.append(("generation: Claude + guard + grade", max(0.0, agent_ms - spans[0][1])))
        result["cached"] = False
        result["latency_ms"] = (time.perf_counter() - t_start) * 1000
        result["spans"] = spans
        result["cache_hit_rate"] = self.cache.hit_rate
        result["was_followup"] = was_followup
        self.cache.set(question, result)
        self.turns.append((question, result.get("answer", "")))
        return result

    def history(self):
        return list(self.turns)

    def turn_count(self):
        return len(self.turns)
