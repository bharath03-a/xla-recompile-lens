"""Tests for the fusion_bench strategies and their invariants."""

from __future__ import annotations

import pytest

from fusion_bench import FusionGraph, Op, balanced_fusion, greedy_fusion
from fusion_bench.compare import compare_strategies


def _chain(sizes: list[int]) -> FusionGraph:
    return FusionGraph(ops=tuple(Op(f"op{i}", s) for i, s in enumerate(sizes)))


@pytest.mark.unit
def test_greedy_partition_is_valid_and_covers_chain() -> None:
    graph = _chain([3, 4, 1, 8, 2] * 3)
    part = greedy_fusion(graph, budget_bytes=10)
    assert part.is_valid(10)


@pytest.mark.unit
def test_greedy_raises_when_single_op_exceeds_budget() -> None:
    graph = _chain([3, 20, 1])
    with pytest.raises(ValueError):
        greedy_fusion(graph, budget_bytes=10)


@pytest.mark.unit
def test_planner_is_at_least_as_balanced_as_greedy() -> None:
    # At the same group count, the global optimizer's peak group must be <=
    # greedy's peak group — that is the whole point of optimizing balance.
    graph = _chain([3, 4, 1, 8, 2, 7, 3, 5, 6, 2])
    budget = 12
    g = greedy_fusion(graph, budget)
    p = balanced_fusion(graph, budget, k=g.group_count)
    assert p.group_count == g.group_count
    assert p.peak_group_bytes <= g.peak_group_bytes
    assert p.is_valid(budget)


@pytest.mark.unit
def test_compare_reports_both_strategies() -> None:
    graph = _chain([3, 4, 1, 8, 2, 7, 3, 5, 6, 2] * 2)
    greedy_res, planner_res = compare_strategies(graph, 16, repeats=10)
    assert greedy_res.group_count == planner_res.group_count
    # Planner is never more balanced-worse than greedy.
    assert planner_res.peak_bytes <= greedy_res.peak_bytes
    # Both report a positive measured latency.
    assert greedy_res.latency_us > 0
    assert planner_res.latency_us > 0
