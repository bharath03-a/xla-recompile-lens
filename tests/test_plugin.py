"""Tests for the Dynamo-backend plugin: recompile counting + auto_bucket.

These run on CPU — the plugin rides Torch Dynamo, not any TPU API — so the core
claim ("counting backend calls counts recompiles, and bucketing reduces them")
is verifiable anywhere.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from xla_recompile_lens import auto_bucket, instrument, set_recompile_limit

# Raise Dynamo's recompile cap so tests measure the full count, not a clamp.
set_recompile_limit(256)


class _Tiny(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * 2.0).sum()


def _fresh() -> _Tiny:
    # A fresh module + reset avoids cross-test Dynamo cache contamination.
    torch._dynamo.reset()
    return _Tiny()


@pytest.mark.unit
def test_recompiles_equal_distinct_shapes() -> None:
    compiled, lens = instrument(_fresh())
    for n in [4, 4, 8, 8, 16, 4]:  # 3 distinct shapes
        compiled(torch.randn(n))
    assert lens.recompile_count == 3
    assert lens.distinct_shapes == 3


@pytest.mark.unit
def test_single_shape_compiles_once() -> None:
    compiled, lens = instrument(_fresh())
    for _ in range(5):
        compiled(torch.randn(10))
    assert lens.recompile_count == 1


@pytest.mark.unit
def test_auto_bucket_reduces_recompiles() -> None:
    lengths = [5, 12, 30, 7, 64, 100, 5, 12]  # 6 distinct raw shapes

    _, lens_raw = _measure(lengths)
    _, lens_fix = _measure(lengths, bucketed=True)

    assert lens_raw.recompile_count == 6
    # lengths fall into {32, 64, 128} -> at most 3 shapes.
    assert lens_fix.recompile_count <= 3
    assert lens_fix.recompile_count < lens_raw.recompile_count


def _measure(lengths, bucketed: bool = False):
    compiled, lens = instrument(_fresh())
    fn = auto_bucket(compiled, buckets=(32, 64, 128), dim=-1) if bucketed else compiled
    for n in lengths:
        fn(torch.randn(1, n))
    return fn, lens


@pytest.mark.unit
def test_events_are_immutable() -> None:
    compiled, lens = instrument(_fresh())
    compiled(torch.randn(4))
    with pytest.raises((AttributeError, TypeError)):
        lens.events[0].ordinal = 99  # type: ignore[misc]
