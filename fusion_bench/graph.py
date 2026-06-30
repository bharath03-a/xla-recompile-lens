"""Minimal op-graph model for fusion experiments.

We model a model's forward pass as a *linear chain* of ops, each with an output
size in bytes. Transformer blocks are largely sequential, so a contiguous
partition of the chain is a faithful, explainable stand-in for fusion groups: a
"fusion group" is a contiguous run of ops fused into one XLA subgraph.

The cost model: a fused group must hold its intermediate outputs in fast memory
(SRAM) simultaneously, so a group's memory footprint is the sum of its ops'
output bytes. A valid grouping keeps every group within the SRAM budget.

Everything here is immutable (frozen dataclasses, tuples) — a `Partition` is a
result you can trust, not a structure that gets mutated as algorithms run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Op:
    """One operation in the chain."""

    name: str
    output_bytes: int


@dataclass(frozen=True, slots=True)
class FusionGraph:
    """A linear chain of ops to be partitioned into fusion groups."""

    ops: tuple[Op, ...]

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(op.output_bytes for op in self.ops)

    def __len__(self) -> int:
        return len(self.ops)


@dataclass(frozen=True, slots=True)
class Partition:
    """A grouping of the chain into contiguous fusion groups.

    `groups` is a tuple of tuples of op indices, in order and covering the chain
    exactly once.
    """

    graph: FusionGraph
    groups: tuple[tuple[int, ...], ...]

    def group_bytes(self) -> tuple[int, ...]:
        """Memory footprint of each group."""
        sizes = self.graph.sizes
        return tuple(sum(sizes[i] for i in g) for g in self.groups)

    @property
    def group_count(self) -> int:
        """Number of fused subgraphs. Fewer = fewer compiles + launches."""
        return len(self.groups)

    @property
    def peak_group_bytes(self) -> int:
        """Largest group footprint — what must fit in SRAM."""
        gb = self.group_bytes()
        return max(gb) if gb else 0

    def is_valid(self, budget_bytes: int) -> bool:
        """True iff every group fits the budget and the chain is covered once."""
        flat = [i for g in self.groups for i in g]
        if flat != list(range(len(self.graph))):
            return False
        return all(b <= budget_bytes for b in self.group_bytes())
