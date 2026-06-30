"""Runnable demo: `python -m fusion_bench.demo`.

Builds an op chain shaped like a stack of transformer blocks (each block =
qkv -> attn -> proj -> ffn1 -> ffn2, with realistic relative output sizes),
then compares greedy vs. the global planner under an SRAM-like budget.
"""

from __future__ import annotations

from .compare import compare_strategies, render_comparison
from .graph import FusionGraph, Op

# Relative output sizes (bytes) for one transformer block's ops. The feedforward
# expansion (ffn1) is the memory hot-spot, which is what makes balance matter.
_BLOCK = [
    ("qkv", 3_000_000),
    ("attn", 4_000_000),
    ("proj", 1_000_000),
    ("ffn1", 6_000_000),
    ("ffn2", 2_000_000),
]


def build_graph(num_blocks: int = 6) -> FusionGraph:
    ops: list[Op] = []
    for b in range(num_blocks):
        for name, size in _BLOCK:
            ops.append(Op(name=f"blk{b}.{name}", output_bytes=size))
    return FusionGraph(ops=tuple(ops))


def main() -> None:
    graph = build_graph(num_blocks=6)
    budget = 15_000_000  # 15 MB fast-memory (SRAM) budget
    greedy, planner = compare_strategies(graph, budget)
    print(render_comparison(greedy, planner))


if __name__ == "__main__":
    main()
