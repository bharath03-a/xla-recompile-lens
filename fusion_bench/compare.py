"""Measure both strategies on the same graph: quality *and* decision latency.

This is where the honest conclusion comes from. We report, per strategy:

* `group_count`   — fewer fused subgraphs = fewer compiles/launches.
* `peak_bytes`    — the largest group; lower = better balance / SRAM headroom.
* `latency_us`    — wall-clock to *decide* the grouping, median over repeats.

The decision latency is the crux: a compiler makes fusion decisions constantly,
so an optimizer that is 100x slower per decision is disqualified from the hot
path no matter how slightly better its output is.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

from .classical import greedy_fusion
from .graph import FusionGraph, Partition
from .planner import balanced_fusion


@dataclass(frozen=True, slots=True)
class StrategyResult:
    name: str
    group_count: int
    peak_bytes: int
    latency_us: float

    def render(self) -> str:
        return (
            f"{self.name:<28} groups={self.group_count:<4} "
            f"peak={self.peak_bytes:>12,} bytes  "
            f"decide={self.latency_us:>10.1f} us"
        )


def _time_strategy(
    fn: Callable[[], Partition], repeats: int
) -> tuple[Partition, float]:
    """Run `fn` `repeats` times, return its result and median latency (us)."""
    samples: list[float] = []
    result: Partition | None = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    assert result is not None
    return result, statistics.median(samples)


def compare_strategies(
    graph: FusionGraph,
    budget_bytes: int,
    repeats: int = 200,
) -> tuple[StrategyResult, StrategyResult]:
    """Compare greedy vs. the balanced optimizer at the same group count."""
    greedy_part, greedy_us = _time_strategy(
        lambda: greedy_fusion(graph, budget_bytes), repeats
    )
    k = greedy_part.group_count
    planner_part, planner_us = _time_strategy(
        lambda: balanced_fusion(graph, budget_bytes, k=k), repeats
    )

    greedy_res = StrategyResult(
        name="classical greedy (O(n))",
        group_count=greedy_part.group_count,
        peak_bytes=greedy_part.peak_group_bytes,
        latency_us=greedy_us,
    )
    planner_res = StrategyResult(
        name="global planner (DP, agent-like)",
        group_count=planner_part.group_count,
        peak_bytes=planner_part.peak_group_bytes,
        latency_us=planner_us,
    )
    return greedy_res, planner_res


def render_comparison(
    greedy: StrategyResult, planner: StrategyResult
) -> str:
    """Format the comparison plus the one-line honest verdict."""
    lines = ["=== fusion_bench: greedy vs. global planner ===", ""]
    lines.append(greedy.render())
    lines.append(planner.render())
    lines.append("")

    balance_gain = (
        (greedy.peak_bytes - planner.peak_bytes) / greedy.peak_bytes * 100
        if greedy.peak_bytes
        else 0.0
    )
    slowdown = (
        planner.latency_us / greedy.latency_us
        if greedy.latency_us
        else float("inf")
    )
    lines.append(
        f"Planner improves peak balance by {balance_gain:.1f}% "
        f"but is {slowdown:.0f}x slower per decision."
    )
    lines.append(
        "Verdict: greedy belongs in the compile loop; the global/agent search "
        "is an offline tool to discover heuristics, not a hot-path policy."
    )
    return "\n".join(lines)
