"""xla-recompile-lens: attribute XLA recompilations to their root cause.

Public API. Import the pieces you need:

    from xla_recompile_lens import analyze_module, render_text

The static analysis (`analyze_module`, `ShapeSignatureTracker`) runs anywhere.
The measured capture (`capture_steps`, `measure_compiles`) needs `torch_xla`
(TPU/Colab) and is guarded by `xla_available()`.

The plugin (`instrument`, `RecompileLens`, `auto_bucket`) rides the standard
Torch Dynamo backend seam, so it measures recompiles on any hardware — including
TorchTPU — with no vendor metrics.
"""

from __future__ import annotations

from .attribute import analyze_module, scan_graph
from .autobucket import auto_bucket
from .bucket_advisor import (
    CostModel,
    LengthHistogram,
    Recommendation,
    fit_cost_model,
    optimal_buckets,
    padded_cost,
    recommend_buckets,
)
from .capture import capture_steps, device_kind, measure_compiles, xla_available
from .fixes import next_bucket, pad_to_bucket
from .plugin import (
    RecompileEvent,
    RecompileLens,
    instrument,
    set_recompile_limit,
)
from .report import before_after_delta, plot_before_after, render_text
from .shapes import ShapeSignatureTracker, shape_signature
from .types import (
    SUGGESTED_FIX,
    Finding,
    MeasuredRecompiles,
    RecompileCause,
    Report,
)

__all__ = [
    "analyze_module",
    "scan_graph",
    "auto_bucket",
    "LengthHistogram",
    "CostModel",
    "Recommendation",
    "fit_cost_model",
    "optimal_buckets",
    "padded_cost",
    "recommend_buckets",
    "instrument",
    "set_recompile_limit",
    "RecompileLens",
    "RecompileEvent",
    "capture_steps",
    "measure_compiles",
    "xla_available",
    "device_kind",
    "next_bucket",
    "pad_to_bucket",
    "before_after_delta",
    "plot_before_after",
    "render_text",
    "ShapeSignatureTracker",
    "shape_signature",
    "Finding",
    "MeasuredRecompiles",
    "RecompileCause",
    "Report",
    "SUGGESTED_FIX",
]
