"""Remediations that turn a recompile cause into a measurable improvement.

The report tells you *why* you recompile; these helpers let you *prove the fix*.
The headline result of the project is a before/after: run a workload with
per-batch shapes (lots of recompiles), then run it again with bucketized shapes
(few recompiles), and show the measured `UncachedCompile` counter drop.

Right now this focuses on the `VARIABLE_SHAPE` cause, which is both the most
common in practice and the one with the cleanest fix: pad every input to the
next bucket boundary so XLA sees a small, fixed set of shapes.
"""

from __future__ import annotations

import torch


def next_bucket(length: int, buckets: tuple[int, ...]) -> int:
    """Smallest bucket >= length. Falls back to the largest bucket.

    Bucketing trades a little wasted compute (padding) for far fewer distinct
    shapes — exactly the bounded-dynamism tradeoff XLA wants.
    """
    for b in buckets:
        if b >= length:
            return b
    return buckets[-1]


def pad_to_bucket(
    x: torch.Tensor,
    buckets: tuple[int, ...] = (32, 64, 128, 256, 512),
    *,
    dim: int = -1,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad `x` along `dim` up to the next bucket size.

    With per-sample sequence lengths this collapses an unbounded set of shapes
    into at most `len(buckets)` shapes, so XLA compiles at most that many
    graphs instead of one per unique length.
    """
    size = x.shape[dim]
    target = next_bucket(size, buckets)
    if target == size:
        return x
    pad_amount = target - size
    # torch.nn.functional.pad pads the *last* dim first; build the pad spec so
    # only `dim` is padded. We normalize a negative dim to its positive index.
    ndim = x.dim()
    pos_dim = dim % ndim
    pad_spec: list[int] = [0, 0] * ndim
    # pad spec is ordered from last dim to first; index for `pos_dim`:
    slot = (ndim - 1 - pos_dim) * 2
    pad_spec[slot + 1] = pad_amount
    return torch.nn.functional.pad(x, pad_spec, value=pad_value)
