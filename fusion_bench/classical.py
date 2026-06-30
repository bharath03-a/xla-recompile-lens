"""Classical fusion: deterministic, O(n), compile-loop-safe.

`greedy_fusion` walks the chain once, growing the current fusion group while it
still fits the SRAM budget, and starting a new group the moment the next op
would overflow. This is first-fit bin packing on a sequence.

Properties that matter to a compiler:
* **Deterministic** — same input, same output, every time.
* **O(n)** — one pass, no search. Microseconds even for large graphs.
* **Greedy-optimal for *group count*** under a prefix budget: you cannot cover
  the chain in fewer groups without exceeding the budget.

Its weakness — and the reason `planner.balanced_fusion` exists — is *balance*:
greedy can leave one group nearly full and the next nearly empty, giving a high
`peak_group_bytes`. Whether that matters is exactly the question the comparison
answers.
"""

from __future__ import annotations

from .graph import FusionGraph, Partition


def greedy_fusion(graph: FusionGraph, budget_bytes: int) -> Partition:
    """First-fit contiguous grouping under an SRAM budget.

    Raises `ValueError` if a single op exceeds the budget (no valid grouping).
    """
    sizes = graph.sizes
    for i, s in enumerate(sizes):
        if s > budget_bytes:
            raise ValueError(
                f"op {graph.ops[i].name} ({s} bytes) exceeds budget "
                f"{budget_bytes}; no valid fusion exists."
            )

    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    current_bytes = 0
    for i, s in enumerate(sizes):
        if current and current_bytes + s > budget_bytes:
            groups.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(i)
        current_bytes += s
    if current:
        groups.append(tuple(current))

    return Partition(graph=graph, groups=tuple(groups))
