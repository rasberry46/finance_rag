"""
observability.py  —  STEP 5: Observability, Tracing & Bottleneck Detection
==========================================================================
Right now the pipeline is a black box: query in, answer out. If a request takes
4 seconds, which stage was slow — embedding? retrieval? the LLM? This module
makes every stage measurable.

Provides:
  - Tracer / Span : time each stage of a request with a `with` block.
  - trace summary : per-stage latency breakdown + % of total.
  - bottleneck    : automatically flags the slowest stage.
  - metrics       : rolling aggregates across many requests (p50/p95).

In production this maps to OpenTelemetry spans exported to Grafana Tempo /
Jaeger / CloudWatch. Here it's stdlib-only so it runs anywhere, but the *shape*
(spans, nesting, attributes) is the same, so the mental model transfers.

The interview point: "I don't guess where latency comes from — I measure it per
span, find the bottleneck, and optimize that. Usually it's the LLM call, which
is why caching earlier stages matters."
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import median


@dataclass
class Span:
    """One timed stage of a request."""
    name: str
    duration_ms: float
    attributes: dict = field(default_factory=dict)


@dataclass
class Trace:
    """All spans for a single request, in order."""
    spans: list[Span] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans)

    @property
    def bottleneck(self) -> Span | None:
        """The single slowest stage — where to focus optimization."""
        return max(self.spans, key=lambda s: s.duration_ms) if self.spans else None

    def summary(self) -> str:
        if not self.spans:
            return "(no spans recorded)"
        total = self.total_ms
        lines = [f"{'stage':<26}{'ms':>10}{'% total':>10}"]
        lines.append("-" * 46)
        for s in self.spans:
            pct = (s.duration_ms / total * 100) if total else 0
            lines.append(f"{s.name:<26}{s.duration_ms:>10.1f}{pct:>9.1f}%")
        lines.append("-" * 46)
        lines.append(f"{'TOTAL':<26}{total:>10.1f}{'100.0%':>10}")
        bn = self.bottleneck
        if bn:
            lines.append(f"\n🔎 bottleneck: {bn.name} "
                         f"({bn.duration_ms:.1f} ms, {bn.duration_ms/total*100:.0f}% of total)")
        return "\n".join(lines)


class Tracer:
    """Collects spans for one request. Use the `span` context manager to time
    a stage; it records the elapsed wall-clock time automatically."""

    def __init__(self):
        self.trace = Trace()

    @contextmanager
    def span(self, name: str, **attributes):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.trace.spans.append(Span(name, elapsed_ms, attributes))

    def summary(self) -> str:
        return self.trace.summary()


# ----------------------------------------------------------------------------
# Rolling metrics across MANY requests (for dashboards / SLOs)
# ----------------------------------------------------------------------------
class Metrics:
    """Aggregates latency across requests so you can report p50/p95 per stage —
    the numbers you'd put on a dashboard or an SLO."""

    def __init__(self):
        self._by_stage: dict[str, list[float]] = {}
        self._totals: list[float] = []

    def record(self, trace: Trace):
        for s in trace.spans:
            self._by_stage.setdefault(s.name, []).append(s.duration_ms)
        self._totals.append(trace.total_ms)

    @staticmethod
    def _p(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = min(len(s) - 1, int(len(s) * pct))
        return s[idx]

    def report(self) -> str:
        if not self._totals:
            return "(no requests recorded)"
        lines = [f"{'stage':<26}{'p50':>10}{'p95':>10}{'calls':>8}"]
        lines.append("-" * 54)
        for stage, vals in self._by_stage.items():
            lines.append(f"{stage:<26}{median(vals):>10.1f}"
                         f"{self._p(vals,0.95):>10.1f}{len(vals):>8}")
        lines.append("-" * 54)
        lines.append(f"{'end-to-end':<26}{median(self._totals):>10.1f}"
                     f"{self._p(self._totals,0.95):>10.1f}{len(self._totals):>8}")
        return "\n".join(lines)


if __name__ == "__main__":
    import random

    # Simulate a few requests through a traced pipeline.
    metrics = Metrics()
    for _ in range(5):
        tracer = Tracer()
        with tracer.span("cache_lookup"):
            time.sleep(0.0005)
        with tracer.span("query_embedding", model="titan-v2"):
            time.sleep(random.uniform(0.08, 0.12))     # Titan round trip
        with tracer.span("hybrid_retrieval"):
            time.sleep(random.uniform(0.01, 0.03))     # BM25 + FAISS + RRF
        with tracer.span("confidence_scoring"):
            time.sleep(0.001)
        with tracer.span("llm_generation", model="claude-sonnet-4-6"):
            time.sleep(random.uniform(1.2, 2.0))       # Claude round trip
        metrics.record(tracer.trace)

    # Show the LAST request's detailed breakdown...
    print("=== Single request trace ===")
    print(tracer.summary())

    # ...and the aggregate across all requests.
    print("\n=== Aggregate metrics (p50/p95) ===")
    print(metrics.report())
