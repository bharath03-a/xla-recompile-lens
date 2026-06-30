"""Core domain model for xla-recompile-lens.

Everything here is an *immutable* value object (frozen dataclass / enum). The
analysis layers (`attribute`, `capture`) produce these objects; the reporting
layer (`report`) only reads them. Keeping the data immutable means a `Report`
can never be mutated after it is built — what you analyzed is exactly what you
print, which matters when the whole point of the tool is trustworthy numbers.

Why a shared types module at all? It is the one place that defines the
vocabulary ("a recompile cause", "a finding", "a report") that every other
module speaks. New analysis passes plug in by emitting `Finding`s; new
reporters plug in by reading a `Report`. Neither needs to know about the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class RecompileCause(StrEnum):
    """The three documented root causes of XLA graph recompilation.

    Source: PyTorch/XLA "Source of recompilations" docs. We subclass `str` so
    the value serializes cleanly to JSON and prints readably, while still
    giving us a closed set of names to switch on.
    """

    VARIABLE_SHAPE = "variable_shape"
    """A new input *shape signature* appears, so XLA compiles a fresh graph.
    Classic trigger: padding sequences to their own length instead of to a
    bucket, so every batch is a slightly different shape."""

    DATA_DEPENDENT_OP = "data_dependent_op"
    """An op whose output shape depends on tensor *values*, not just input
    shapes: `nonzero`, `masked_select`, `unique`, `.item()`. XLA cannot know
    the output size at trace time, so it recompiles (or graph-breaks)."""

    CONTROL_FLOW = "control_flow"
    """Python control flow branching on a tensor value (`if x[0] == 3:`). The
    value must be materialized to decide the branch, which cuts the graph and
    forces separate compilations per path."""

    UNKNOWN = "unknown"
    """A recompile we observed but could not attribute statically. Reported
    honestly rather than guessed."""


# Human-readable, actionable fix for each cause. Kept beside the enum so the
# advice and the cause can never drift apart.
SUGGESTED_FIX: dict[RecompileCause, str] = {
    RecompileCause.VARIABLE_SHAPE: (
        "Bucketize or pad inputs to a fixed set of shapes (e.g. pad seq_len "
        "up to the next power of two) so XLA reuses a compiled graph."
    ),
    RecompileCause.DATA_DEPENDENT_OP: (
        "Use bounded dynamism: pad the dynamic dimension to a known maximum, "
        "or rewrite to avoid value-dependent output shapes."
    ),
    RecompileCause.CONTROL_FLOW: (
        "Replace value-dependent Python branching with functional control "
        "flow (torch.where, torch.cond) so the graph stays whole."
    ),
    RecompileCause.UNKNOWN: (
        "Inspect XLA debug logs (PT_XLA_DEBUG=1) for this step to attribute "
        "the recompile manually."
    ),
}


@dataclass(frozen=True, slots=True)
class Finding:
    """One attributed reason a graph would (or did) recompile.

    A `Finding` is produced by static fx analysis. It points at the specific op
    and, where possible, the Python source line that introduced the problem, so
    the report is actionable rather than just a count.
    """

    cause: RecompileCause
    op: str
    """The fx node target, e.g. 'torch.nonzero' or 'aten::masked_select'."""
    detail: str
    """One-line explanation specific to this occurrence."""
    location: str | None = None
    """`file:line` from fx stack traces when available, else None."""

    @property
    def fix(self) -> str:
        """The suggested remediation for this finding's cause."""
        return SUGGESTED_FIX[self.cause]


@dataclass(frozen=True, slots=True)
class MeasuredRecompiles:
    """Real recompile counts captured from a live `torch_xla` run.

    Distinct from `Finding`s on purpose: findings are *static predictions*
    ("this code can recompile and here's why"); this is the *measured truth*
    from the device ("we compiled N graphs over these steps"). Showing both —
    and that they agree — is what makes the artifact credible.
    """

    compile_count: int
    """Number of distinct XLA compilations observed."""
    steps: int
    """Number of model steps the count was measured over."""
    label: str = ""
    """e.g. 'before fix' / 'after bucketizing'."""

    @property
    def compiles_per_step(self) -> float:
        """Lower is better. ~1 total (amortized to ~0/step) is the goal."""
        return self.compile_count / self.steps if self.steps else float("nan")


@dataclass(frozen=True, slots=True)
class Report:
    """The full analysis result: static findings plus any measured counts.

    Built once, never mutated. `report.render()` (see `report.py`) turns this
    into a table; `report.before_after` holds the money chart's data.
    """

    model_name: str
    findings: tuple[Finding, ...] = field(default_factory=tuple)
    measured: tuple[MeasuredRecompiles, ...] = field(default_factory=tuple)
    tpu_available: bool = False
    """False when run on CPU/Mac: the report then carries static findings only
    and says so, instead of pretending to have device numbers."""
    device: str = ""
    """Actual XLA backend when measured: 'TPU', 'CUDA', 'CPU', etc. Reported
    verbatim so a result is never labeled 'TPU' unless it really ran on one
    (torch_xla silently falls back to XLA-CPU when no TPU is acquired)."""

    def causes_summary(self) -> dict[RecompileCause, int]:
        """Count of findings per cause — the static breakdown."""
        summary: dict[RecompileCause, int] = {}
        for f in self.findings:
            summary[f.cause] = summary.get(f.cause, 0) + 1
        return summary
