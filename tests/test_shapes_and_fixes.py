"""Tests for shape-signature tracking and the bucketizing fix."""

from __future__ import annotations

import pytest
import torch

from xla_recompile_lens import (
    ShapeSignatureTracker,
    next_bucket,
    pad_to_bucket,
    shape_signature,
)


@pytest.mark.unit
def test_shape_signature_ignores_non_tensors() -> None:
    sig = shape_signature((torch.zeros(2, 3), 5, "x", torch.zeros(4)))
    assert sig == ((2, 3), (4,))


@pytest.mark.unit
def test_tracker_counts_distinct_signatures() -> None:
    tracker = ShapeSignatureTracker()
    tracker.observe(torch.zeros(2, 10))
    tracker.observe(torch.zeros(2, 10))  # duplicate -> no new compile
    tracker.observe(torch.zeros(2, 12))  # new shape -> +1 compile
    assert tracker.distinct_signatures == 2
    assert tracker.shape_driven_recompiles == 1


@pytest.mark.unit
def test_bucketizing_collapses_many_shapes_to_few() -> None:
    # Simulate per-sample sequence lengths 1..100. Raw: ~100 distinct shapes.
    raw = ShapeSignatureTracker()
    bucketed = ShapeSignatureTracker()
    buckets = (32, 64, 128)
    for length in range(1, 101):
        x = torch.zeros(1, length)
        raw.observe(x)
        bucketed.observe(pad_to_bucket(x, buckets, dim=-1))
    assert raw.distinct_signatures == 100
    # All lengths 1..100 fall into {32, 64, 128} -> at most 3 shapes.
    assert bucketed.distinct_signatures <= 3
    assert bucketed.shape_driven_recompiles < raw.shape_driven_recompiles


@pytest.mark.unit
def test_next_bucket() -> None:
    assert next_bucket(10, (32, 64)) == 32
    assert next_bucket(40, (32, 64)) == 64
    assert next_bucket(999, (32, 64)) == 64  # falls back to largest


@pytest.mark.unit
def test_pad_to_bucket_pads_correct_dim() -> None:
    x = torch.ones(2, 10)
    out = pad_to_bucket(x, (32,), dim=-1)
    assert out.shape == (2, 32)
    # Original values preserved, pad region is zeros.
    assert torch.equal(out[:, :10], x)
    assert torch.equal(out[:, 10:], torch.zeros(2, 22))
