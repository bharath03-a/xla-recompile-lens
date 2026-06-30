"""Static fx-graph analysis: find recompile causes by reading the model.

This layer answers "what in this model *can* cause an XLA recompile, and
where?" using only `torch.fx` — so it runs on a plain CPU/Mac with no TPU. It
detects two of the three causes statically:

* `DATA_DEPENDENT_OP` — by scanning the traced graph for ops with
  value-dependent output shapes (`nonzero`, `masked_select`, `unique`,
  `.item()`).
* `CONTROL_FLOW` — by observing that `torch.fx.symbolic_trace` *fails* with a
  `TraceError` when the model branches on a tensor value. The trace failure is
  itself the signal, so we turn the exception into a `Finding` instead of
  crashing.

The third cause (`VARIABLE_SHAPE`) is data-driven and handled separately by
`shapes.ShapeSignatureTracker`.

Design note: we deliberately do not depend on `transformers`. Models that are
not directly fx-traceable (many HF models) should be traced by the caller (e.g.
`transformers.utils.fx.symbolic_trace`) and passed in as a `GraphModule`. That
keeps this module small, dependency-light, and unit-testable on tiny models.
"""

from __future__ import annotations

import torch
from torch import fx, nn

from .types import Finding, RecompileCause, Report

# fx node targets whose output shape depends on tensor *values*, not just input
# shapes. Matched by the function object where we can (robust to namespacing)
# and by name suffix as a fallback.
_DATA_DEPENDENT_FUNCS = {
    torch.nonzero,
    torch.masked_select,
    torch.unique,
    torch.unique_consecutive,
    torch.bincount,
}
_DATA_DEPENDENT_NAMES = {
    "nonzero",
    "masked_select",
    "unique",
    "unique_consecutive",
    "bincount",
    "item",  # call_method: forces a host sync + dynamic value
}


def _node_location(node: fx.Node) -> str | None:
    """Best-effort `file:line` for a node, from fx's recorded stack trace."""
    stack = getattr(node, "stack_trace", None)
    if not stack:
        return None
    # fx stores the trace as text; the last frame is the model code line.
    last = [ln for ln in stack.strip().splitlines() if ln.strip()]
    return last[-1].strip() if last else None


def _target_name(node: fx.Node) -> str:
    """Readable name for a node's target across call_function/method/module."""
    target = node.target
    if callable(target):
        return getattr(target, "__name__", repr(target))
    return str(target)


def scan_graph(gm: fx.GraphModule) -> tuple[Finding, ...]:
    """Find all data-dependent-op findings in an already-traced graph."""
    findings: list[Finding] = []
    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method"):
            continue
        name = _target_name(node)
        is_dd = (
            node.target in _DATA_DEPENDENT_FUNCS
            or name in _DATA_DEPENDENT_NAMES
        )
        if is_dd:
            findings.append(
                Finding(
                    cause=RecompileCause.DATA_DEPENDENT_OP,
                    op=name,
                    detail=(
                        f"'{name}' produces a value-dependent output shape; "
                        "XLA cannot fix the size at trace time."
                    ),
                    location=_node_location(node),
                )
            )
    return tuple(findings)


def analyze_module(
    model: nn.Module,
    example_inputs: tuple[object, ...],
    model_name: str | None = None,
    *,
    tpu_available: bool = False,
    device: str = "",
) -> Report:
    """Statically analyze a model for recompile causes.

    Tries to symbolically trace the model. On success, scans for data-dependent
    ops. On a `TraceError` (the fingerprint of value-dependent control flow), it
    records a `CONTROL_FLOW` finding instead of raising.

    `example_inputs` is accepted for parity with the dynamic API and to allow
    future shape-aware passes; tracing itself uses fx's symbolic inputs.
    """
    name = model_name or type(model).__name__
    findings: list[Finding] = []

    try:
        gm = fx.symbolic_trace(model)
        findings.extend(scan_graph(gm))
    except fx.proxy.TraceError as exc:  # value-dependent control flow
        findings.append(
            Finding(
                cause=RecompileCause.CONTROL_FLOW,
                op="<python control flow>",
                detail=(
                    "Model branches on a tensor value (fx could not trace a "
                    f"single graph): {exc}"
                ),
                location=None,
            )
        )
    except Exception as exc:  # noqa: BLE001 - report, never crash the tool
        findings.append(
            Finding(
                cause=RecompileCause.UNKNOWN,
                op="<trace failure>",
                detail=f"Could not trace model statically: {exc}",
                location=None,
            )
        )

    return Report(
        model_name=name,
        findings=tuple(findings),
        tpu_available=tpu_available,
        device=device,
    )
