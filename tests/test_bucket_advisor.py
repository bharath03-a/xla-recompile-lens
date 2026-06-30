"""Tests for the data-derived sequence bucketing advisor (pure, CPU).

These verify the optimization is correct and the cost/compile tradeoff behaves —
no TPU, no model. This is the 'significant' code's correctness gate.
"""

from __future__ import annotations

import pytest

from xla_recompile_lens import (
    CostModel,
    LengthHistogram,
    fit_cost_model,
    optimal_buckets,
    padded_cost,
    recommend_buckets,
)


def _hist(lengths: list[int]) -> LengthHistogram:
    return LengthHistogram.from_lengths(lengths)


@pytest.mark.unit
def test_histogram_basics() -> None:
    h = _hist([10, 10, 20, 5])
    assert h.lengths == (5, 10, 20)
    assert h.counts == (1, 2, 1)
    assert h.total == 4
    assert h.distinct == 3
    assert h.max_length == 20
    assert h.total_tokens == 5 + 20 + 20  # 5*1 + 10*2 + 20*1


@pytest.mark.unit
def test_histogram_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        _hist([10, 0, 5])


@pytest.mark.unit
def test_optimal_buckets_cover_and_sorted() -> None:
    h = _hist([3, 7, 11, 40, 41, 100])
    cost = CostModel(a=1.0, b=0.0)
    buckets = optimal_buckets(h, k=3, cost_fn=cost)
    assert list(buckets) == sorted(buckets)
    assert buckets[-1] == h.max_length  # covers the workload
    assert len(buckets) == 3


@pytest.mark.unit
def test_more_buckets_never_cost_more() -> None:
    # Refining the partition can only reduce (or hold) padded cost.
    h = _hist([1, 5, 9, 17, 33, 65, 129, 257])
    cost = CostModel(a=0.0, b=1.0)  # quadratic
    prev = float("inf")
    for k in range(1, h.distinct + 1):
        c = padded_cost(h, optimal_buckets(h, k, cost), cost)
        assert c <= prev + 1e-9
        prev = c


@pytest.mark.unit
def test_optimal_beats_or_matches_powers_of_two() -> None:
    # The data-derived buckets must be <= a fixed power-of-two set at equal K.
    lengths = [12, 13, 14, 15, 60, 61, 62, 200, 201, 202]
    h = _hist(lengths)
    cost = CostModel(a=0.0, b=1.0)
    pow2 = (16, 64, 256)  # 3 buckets covering max=202
    derived = optimal_buckets(h, k=3, cost_fn=cost)
    assert padded_cost(h, derived, cost) <= padded_cost(h, pow2, cost) + 1e-9


@pytest.mark.unit
def test_padded_cost_rejects_uncovering_buckets() -> None:
    h = _hist([10, 500])
    with pytest.raises(ValueError):
        padded_cost(h, (32, 64), CostModel(a=1.0, b=0.0))


@pytest.mark.unit
def test_fit_cost_model_recovers_coefficients() -> None:
    truth = CostModel(a=2.0, b=0.5)
    samples = [(n, truth(n)) for n in (8, 16, 32, 64, 128, 256)]
    fit = fit_cost_model(samples)
    assert abs(fit.a - 2.0) < 1e-3
    assert abs(fit.b - 0.5) < 1e-4


@pytest.mark.unit
def test_fit_cost_model_fallback_on_too_few_points() -> None:
    fit = fit_cost_model([(10, 1.0)])
    assert fit.a == 0.0 and fit.b == 1.0


@pytest.mark.unit
def test_recommend_fewer_buckets_as_compile_cost_rises() -> None:
    # Many distinct lengths; quadratic compute. Cheap compiles -> many buckets;
    # expensive compiles -> few buckets. The tradeoff must move monotonically.
    h = _hist(list(range(1, 200, 3)))
    cost = CostModel(a=0.0, b=1.0)
    cheap = recommend_buckets(h, cost, compile_cost=1.0, k_max=12)
    pricey = recommend_buckets(h, cost, compile_cost=1e9, k_max=12)
    assert pricey.k <= cheap.k
    assert pricey.k == 1  # compiles so costly that one bucket wins
    # The curve covers every K considered.
    assert len(cheap.curve) == min(12, h.distinct)


@pytest.mark.unit
def test_recommendation_total_is_consistent() -> None:
    h = _hist([5, 10, 20, 40, 80])
    cost = CostModel(a=1.0, b=0.1)
    rec = recommend_buckets(h, cost, compile_cost=5.0, k_max=5)
    assert rec.total_cost == pytest.approx(rec.compute_cost + rec.compile_cost_total)
    # The chosen K must be the argmin of the curve.
    best_k = min(rec.curve, key=lambda kv: kv[1])[0]
    assert rec.k == best_k
