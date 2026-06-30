"""Render a `Report` as a readable table and an optional before/after chart.

Pure presentation: this module only *reads* immutable `Report` objects. It has
no analysis logic, so the numbers it prints are exactly the numbers that were
measured/derived upstream — there is nowhere here to accidentally change them.

`matplotlib` is an optional dependency (the `viz` extra). Text rendering always
works; the chart degrades gracefully with a clear message if matplotlib is
absent.
"""

from __future__ import annotations

from .types import MeasuredRecompiles, RecompileCause, Report

_CAUSE_TITLE = {
    RecompileCause.VARIABLE_SHAPE: "Variable input shape",
    RecompileCause.DATA_DEPENDENT_OP: "Data-dependent op",
    RecompileCause.CONTROL_FLOW: "Control flow on tensor value",
    RecompileCause.UNKNOWN: "Unattributed",
}


def render_text(report: Report) -> str:
    """Render the full report as a plain-text block."""
    lines: list[str] = []
    lines.append(f"=== xla-recompile-lens: {report.model_name} ===")
    if report.tpu_available:
        dev = report.device or "unknown"
        mode = f"measured via torch_xla (XLA backend: {dev})"
    else:
        mode = "static analysis only (no torch_xla)"
    lines.append(f"mode: {mode}")
    lines.append("")

    summary = report.causes_summary()
    if summary:
        lines.append("Static findings by cause:")
        for cause, count in summary.items():
            lines.append(f"  [{count}] {_CAUSE_TITLE[cause]}")
        lines.append("")
        lines.append("Details:")
        for f in report.findings:
            loc = f" ({f.location})" if f.location else ""
            lines.append(f"  - {_CAUSE_TITLE[f.cause]}: {f.op}{loc}")
            lines.append(f"      why: {f.detail}")
            lines.append(f"      fix: {f.fix}")
    else:
        lines.append("No static recompile causes found in the traced graph.")

    if report.measured:
        lines.append("")
        lines.append("Measured compilations:")
        for m in report.measured:
            tag = f" [{m.label}]" if m.label else ""
            lines.append(
                f"  {m.compile_count} compiles over {m.steps} steps"
                f" ({m.compiles_per_step:.2f}/step){tag}"
            )
    return "\n".join(lines)


def before_after_delta(
    before: MeasuredRecompiles, after: MeasuredRecompiles
) -> str:
    """One-line summary of the headline before/after result."""
    drop = before.compile_count - after.compile_count
    pct = (drop / before.compile_count * 100) if before.compile_count else 0.0
    return (
        f"Recompiles: {before.compile_count} -> {after.compile_count} "
        f"({drop} fewer, {pct:.0f}% reduction) over {before.steps} steps"
    )


def plot_before_after(
    before: MeasuredRecompiles,
    after: MeasuredRecompiles,
    path: str = "recompile_before_after.png",
) -> str | None:
    """Save the money chart. Returns the path, or None if matplotlib missing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(4, 4))
    labels = [before.label or "before", after.label or "after"]
    values = [before.compile_count, after.compile_count]
    ax.bar(labels, values, color=["#d62728", "#2ca02c"])
    ax.set_ylabel("XLA compilations")
    ax.set_title("Recompiles before vs. after fix")
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
