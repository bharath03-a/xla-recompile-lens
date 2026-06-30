"""fusion_bench — an honest comparison of fusion-grouping strategies.

TorchTPU's *Fused Eager* mode groups consecutive ops into denser fused XLA
subgraphs. Deciding the group boundaries is an optimization problem, and it is
*adjacent to my MLSys 2026 win* (multi-agent planning for memory-constrained
DAG scheduling). This package asks the honest question a compiler hiring manager
would ask: does the fancy planning approach actually belong in the compiler?

It compares two strategies on the same op graph under a fast-memory (SRAM)
budget:

* `classical.greedy_fusion` — a deterministic O(n) first-fit scheduler. This is
  the kind of algorithm that can run inside the compile loop.
* `planner.balanced_fusion` — an expensive global optimizer (stand-in for the
  multi-agent search) that minimizes the *largest* group, producing more
  balanced groups at much higher decision cost.

`compare.compare_strategies` measures both for quality *and* decision latency.
The expected, honest conclusion (see README): the global optimizer wins on
balance by a little, but its per-decision latency makes it unfit for the hot
compile path — it is an *offline* tool for discovering heuristics, which is
exactly the right role for the multi-agent approach.
"""

from __future__ import annotations

from .classical import greedy_fusion
from .compare import StrategyResult, compare_strategies
from .graph import FusionGraph, Op, Partition
from .planner import balanced_fusion

__all__ = [
    "FusionGraph",
    "Op",
    "Partition",
    "greedy_fusion",
    "balanced_fusion",
    "compare_strategies",
    "StrategyResult",
]
