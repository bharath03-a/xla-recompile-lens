"""Global fusion optimizer — stand-in for the multi-agent planner.

My MLSys 2026 submission used multiple LLM strategy planners plus a reflexion
loop to search the space of schedules for a memory-constrained DAG. That is a
*global, expensive* optimizer. To study whether that style of approach belongs
in a compiler, we model its essential character here without the LLM cost: an
algorithm that does real global optimization and pays for it in runtime.

`balanced_fusion` finds the contiguous partition into `k` groups that minimizes
the *largest* group footprint (the classic "split array largest sum" problem).
Unlike greedy first-fit, it considers all split positions via dynamic
programming — O(n^2 * k) — and returns the provably most balanced grouping.

This is the honest steelman of the planning approach: it produces equal or
better balance than greedy, at materially higher decision latency. The
comparison (see `compare.py`) then asks whether that quality is worth the cost
inside a compile loop that runs millions of times.
"""

from __future__ import annotations

from .classical import greedy_fusion
from .graph import FusionGraph, Partition

_INF = float("inf")


def balanced_fusion(
    graph: FusionGraph,
    budget_bytes: int,
    k: int | None = None,
) -> Partition:
    """Minimize the largest group footprint over contiguous `k`-partitions.

    If `k` is None, uses the minimum feasible group count (from greedy) so the
    two strategies are compared at the *same* group count — isolating balance
    quality from group-count differences.

    Raises `ValueError` if no op-respecting partition fits the budget.
    """
    n = len(graph)
    if n == 0:
        return Partition(graph=graph, groups=())

    if k is None:
        k = greedy_fusion(graph, budget_bytes).group_count
    k = min(k, n)

    sizes = graph.sizes
    # prefix[i] = sum of sizes[:i]; segment(j, i) = prefix[i] - prefix[j].
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + sizes[i]

    def seg(j: int, i: int) -> int:
        return prefix[i] - prefix[j]

    # dp[k][i] = min achievable max-group-sum for first i ops in k groups.
    # back[k][i] = split point achieving it (start index of the last group).
    dp = [[_INF] * (n + 1) for _ in range(k + 1)]
    back = [[0] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0
    for g in range(1, k + 1):
        for i in range(g, n + 1):  # need >= g ops to form g groups
            for j in range(g - 1, i):  # last group covers ops j..i-1
                last = seg(j, i)
                cand = max(dp[g - 1][j], last)
                if cand < dp[g][i]:
                    dp[g][i] = cand
                    back[g][i] = j

    if dp[k][n] == _INF or dp[k][n] > budget_bytes:
        raise ValueError(
            f"no {k}-group partition fits budget {budget_bytes} "
            f"(best max group = {dp[k][n]})."
        )

    # Reconstruct group boundaries from `back`.
    groups: list[tuple[int, ...]] = []
    i = n
    for g in range(k, 0, -1):
        j = back[g][i]
        groups.append(tuple(range(j, i)))
        i = j
    groups.reverse()
    return Partition(graph=graph, groups=tuple(groups))
