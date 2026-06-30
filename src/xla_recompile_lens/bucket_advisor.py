"""Data-derived sequence bucketing: choose the bucket set that minimizes the
*total* serving cost for a real workload, instead of hardcoding powers of two.

The framework-level fix for shape-driven XLA recompilation is to pad inputs to a
small set of buckets (bounded dynamism). But *which* buckets? Powers of two are a
guess. This module derives them from the workload:

* Profile the real prompt-length distribution -> `LengthHistogram`.
* Model per-length compute cost with a coefficient fit to *measured* timings
  (`fit_cost_model`) — attention makes prefill roughly `a*len + b*len**2`.
* Trade off the two real costs: padding waste (more, smaller buckets reduce it)
  vs compilation (each bucket is one XLA compile). `recommend_buckets` sweeps the
  number of buckets K and picks the minimum-total-cost point.

`optimal_buckets` solves the core sub-problem exactly: partition the sorted
distinct lengths into K contiguous segments minimizing total padded compute,
where each segment pads up to its max (a contiguous-partition DP, same shape as
the scheduler in `fusion_bench/planner.py`).

Everything here is pure and CPU-testable — no torch, no TPU. The benchmark feeds
it measured numbers; the advice it returns is then validated end-to-end on device.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

# A cost function maps a (padded) sequence length to a relative compute cost.
CostFn = Callable[[int], float]

_INF = float("inf")


@dataclass(frozen=True, slots=True)
class LengthHistogram:
    """The empirical distribution of sequence lengths in a workload.

    Stored as sorted distinct lengths with parallel counts — the form the DP
    consumes. Immutable: profiling produces it once, the optimizer only reads it.
    """

    lengths: tuple[int, ...]
    counts: tuple[int, ...]

    @classmethod
    def from_lengths(cls, raw: Iterable[int]) -> LengthHistogram:
        hist: dict[int, int] = {}
        for length in raw:
            if length <= 0:
                raise ValueError(f"sequence length must be positive, got {length}")
            hist[length] = hist.get(length, 0) + 1
        if not hist:
            raise ValueError("cannot build a histogram from zero lengths")
        items = sorted(hist.items())
        return cls(
            lengths=tuple(k for k, _ in items),
            counts=tuple(v for _, v in items),
        )

    @property
    def total(self) -> int:
        """Number of requests."""
        return sum(self.counts)

    @property
    def distinct(self) -> int:
        """Number of distinct lengths == compiles the naive strategy would pay."""
        return len(self.lengths)

    @property
    def max_length(self) -> int:
        return self.lengths[-1]

    @property
    def total_tokens(self) -> int:
        """Sum of real (unpadded) token positions across all requests."""
        return sum(
            length * count
            for length, count in zip(self.lengths, self.counts, strict=True)
        )


@dataclass(frozen=True, slots=True)
class CostModel:
    """Per-length compute cost `a*len + b*len**2` (linear MLP + quadratic attn).

    Calibrated to measured prefill timings so the optimizer reasons in real time
    units, not an abstract proxy.
    """

    a: float
    b: float

    def __call__(self, length: int) -> float:
        return self.a * length + self.b * length * length


def fit_cost_model(samples: list[tuple[int, float]]) -> CostModel:
    """Least-squares fit of `a*len + b*len**2` to measured `(len, time)` points.

    Falls back to a pure-quadratic model (`a=0, b=1`) when there are too few
    points to fit — the optimizer still works, just on a coarser cost shape.
    """
    points = [(int(n), float(t)) for n, t in samples if n > 0]
    if len(points) < 2:
        return CostModel(a=0.0, b=1.0)

    import numpy as np

    x = np.array([n for n, _ in points], dtype=float)
    y = np.array([t for _, t in points], dtype=float)
    # Design matrix columns: [len, len**2].
    design = np.stack([x, x * x], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    # Compute cost must be non-decreasing in length; clamp tiny negative fits.
    return CostModel(a=max(a, 0.0), b=max(b, 0.0))


def padded_cost(
    hist: LengthHistogram, buckets: tuple[int, ...], cost_fn: CostFn
) -> float:
    """Total compute cost of serving `hist` with a given `buckets` set.

    Each length is padded up to the smallest bucket >= it; the largest bucket
    must cover `hist.max_length` or it is a configuration error.
    """
    ordered = sorted(buckets)
    if ordered[-1] < hist.max_length:
        raise ValueError(
            f"largest bucket {ordered[-1]} < max length {hist.max_length}; "
            "buckets must cover the workload."
        )
    total = 0.0
    for length, count in zip(hist.lengths, hist.counts, strict=True):
        cap = next(b for b in ordered if b >= length)
        total += count * cost_fn(cap)
    return total


def optimal_buckets(
    hist: LengthHistogram, k: int, cost_fn: CostFn
) -> tuple[int, ...]:
    """Exact min-padded-cost partition of the lengths into `k` buckets.

    Partition the sorted distinct lengths into `k` contiguous segments; each
    segment pads everything up to its largest length (the bucket cap). Minimize
    total `count * cost(cap)`. Solved with a contiguous-partition DP — the same
    structure as the fusion scheduler.

    Returns the bucket caps (sorted, last == max_length so all lengths are
    covered). `k` is clamped to the number of distinct lengths.
    """
    lengths = hist.lengths
    counts = hist.counts
    n = len(lengths)
    k = min(k, n)
    if k <= 0:
        raise ValueError("k must be >= 1")

    # prefix_count[i] = sum(counts[:i]) for O(1) segment count sums.
    prefix_count = [0] * (n + 1)
    for i in range(n):
        prefix_count[i + 1] = prefix_count[i] + counts[i]

    def seg_cost(j: int, i: int) -> float:
        # Segment covers lengths[j..i-1]; cap = lengths[i-1] (the segment max).
        requests = prefix_count[i] - prefix_count[j]
        return requests * cost_fn(lengths[i - 1])

    # dp[g][i] = min cost to cover first i lengths with g segments.
    dp = [[_INF] * (n + 1) for _ in range(k + 1)]
    back = [[0] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0.0
    for g in range(1, k + 1):
        for i in range(g, n + 1):
            for j in range(g - 1, i):
                cand = dp[g - 1][j] + seg_cost(j, i)
                if cand < dp[g][i]:
                    dp[g][i] = cand
                    back[g][i] = j

    # Reconstruct segment end caps.
    caps: list[int] = []
    i = n
    for g in range(k, 0, -1):
        j = back[g][i]
        caps.append(lengths[i - 1])
        i = j
    caps.reverse()
    return tuple(caps)


@dataclass(frozen=True, slots=True)
class Recommendation:
    """The advisor's output: chosen buckets and the cost-vs-K tradeoff curve."""

    buckets: tuple[int, ...]
    k: int
    compute_cost: float
    compile_cost_total: float
    curve: tuple[tuple[int, float], ...] = field(default_factory=tuple)
    """(K, total_estimated_cost) for each K considered — shows the knee."""

    @property
    def total_cost(self) -> float:
        return self.compute_cost + self.compile_cost_total


def recommend_buckets(
    hist: LengthHistogram,
    cost_fn: CostFn,
    compile_cost: float,
    k_max: int = 12,
) -> Recommendation:
    """Pick the bucket count K that minimizes padding cost + K * compile_cost.

    This is the real optimization: fewer buckets means fewer compiles but more
    padding waste; more buckets means the opposite. With a `compile_cost` and a
    `cost_fn` in the *same time units*, the minimizer is the predicted-fastest
    bucketization for this workload.
    """
    if compile_cost < 0:
        raise ValueError("compile_cost must be non-negative")
    if k_max < 1:
        raise ValueError("k_max must be >= 1")

    k_cap = min(k_max, hist.distinct)
    best: Recommendation | None = None
    curve: list[tuple[int, float]] = []
    for k in range(1, k_cap + 1):
        buckets = optimal_buckets(hist, k, cost_fn)
        compute = padded_cost(hist, buckets, cost_fn)
        compile_total = k * compile_cost
        total = compute + compile_total
        curve.append((k, total))
        if best is None or total < best.total_cost:
            best = Recommendation(
                buckets=buckets,
                k=k,
                compute_cost=compute,
                compile_cost_total=compile_total,
            )

    if best is None:  # unreachable given k_cap >= 1, but fail loudly if not
        raise RuntimeError("recommend_buckets found no candidate bucketization")
    # Re-create with the full curve attached (frozen dataclass).
    return Recommendation(
        buckets=best.buckets,
        k=best.k,
        compute_cost=best.compute_cost,
        compile_cost_total=best.compile_cost_total,
        curve=tuple(curve),
    )
