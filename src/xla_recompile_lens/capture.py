"""Measured recompile capture from a live `torch_xla` run.

This is the half of the tool that needs real hardware. `torch_xla` exposes a
metrics/counter registry; the `UncachedCompile` counter increments once per real
(uncached) XLA graph compilation. By reading that counter before and after a
block of model steps we get the *measured* number of compilations — the ground
truth that backs up the static findings from `attribute.py`.

`torch_xla` only installs on Linux/TPU (e.g. Colab), so every import here is
guarded. On a CPU/Mac dev box `xla_available()` returns False and the capture
APIs raise a clear error telling you to run on TPU — they never silently return
fake numbers, because fabricated metrics would defeat the entire purpose.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from .types import MeasuredRecompiles

# Guarded import: present on Colab/TPU, absent on a Mac.
try:  # pragma: no cover - environment dependent
    import torch_xla.core.xla_model as xm  # noqa: F401
    import torch_xla.debug.metrics as met

    _XLA = True
except ImportError:  # pragma: no cover - the common CPU case
    met = None  # type: ignore[assignment]
    _XLA = False


def xla_available() -> bool:
    """True iff `torch_xla` imported (i.e. we can capture real metrics)."""
    return _XLA


def device_kind() -> str:
    """Actual XLA backend in use: 'TPU', 'CUDA', 'CPU', or 'none'.

    Critical for honest reporting: torch_xla silently falls back to the XLA-CPU
    backend when no TPU is acquired (you'll see `Defaulting to PJRT_DEVICE=CPU`).
    The recompilation *mechanism* is identical across backends, but a result is
    only a "TPU" result if this returns 'TPU'. Never hard-code the device label.
    """
    if not _XLA:
        return "none"
    try:
        import torch_xla.runtime as xr

        return xr.device_type() or "UNKNOWN"
    except Exception:  # noqa: BLE001 - report honestly, never crash
        return "UNKNOWN"


def _compile_count() -> int:
    """Number of *uncached* XLA compilations so far (i.e. real recompiles).

    Important torch_xla distinction: `UncachedCompile` is a **counter** (an
    integer event count), while `CompileTime` is a **time metric**. Only
    counters are readable via `counter_value` — calling it on `CompileTime`
    returns None. So we read the `UncachedCompile` counter, and fall back to the
    `CompileTime` metric's sample count (`metric_data(...)[0]` = number of
    recorded samples = number of compiles) if the counter name is unavailable on
    a given torch_xla version. Counters return None before first touched, which
    we treat as zero.
    """
    if met is None:
        raise RuntimeError(
            "torch_xla is not available — compile metrics require a TPU "
            "runtime (e.g. free Google Colab with the TPU runtime selected)."
        )
    value = met.counter_value("UncachedCompile")
    if value is not None:
        return int(value)
    # Fallback: CompileTime is a time metric; its sample count == #compiles.
    data = met.metric_data("CompileTime")
    return int(data[0]) if data else 0


@contextmanager
def measure_compiles(steps: int, label: str = "") -> Iterator[list[int]]:
    """Measure how many XLA compilations happen inside the `with` block.

    Usage::

        with measure_compiles(steps=20, label="before fix") as out:
            for batch in batches:
                run_one_step(model, batch)
        result = MeasuredRecompiles(out[0], steps, label)  # see capture_steps

    The yielded list is filled with `[compile_count_delta]` on exit. Most
    callers should prefer `capture_steps`, which wraps this and returns the
    immutable `MeasuredRecompiles` directly.
    """
    start = _compile_count()
    delta_holder: list[int] = []
    try:
        yield delta_holder
    finally:
        delta_holder.append(_compile_count() - start)


def capture_steps(
    run_step,
    batches,
    label: str = "",
) -> MeasuredRecompiles:
    """Run `run_step(batch)` over `batches` and return measured compiles.

    `run_step` must trigger execution (call `xm.mark_step()` / read a result)
    so XLA actually compiles. This function is intentionally tiny and explicit:
    the caller owns the model and the step, we only own the measurement.
    """
    batch_list = list(batches)
    with measure_compiles(len(batch_list), label) as out:
        for batch in batch_list:
            run_step(batch)
    return MeasuredRecompiles(
        compile_count=out[0],
        steps=len(batch_list),
        label=label,
    )
