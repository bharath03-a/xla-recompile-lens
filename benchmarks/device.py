"""Device + measurement primitives for the serving benchmark.

Isolates everything torch_xla-specific so the rest of the harness is plain
PyTorch. On a TPU/XLA runtime we measure real compilation; on a plain-CPU dev
box (no torch_xla) the harness still runs eagerly so the pipeline is testable,
but there is no recompilation to measure — `--dry-run` exists for exactly that.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

try:  # pragma: no cover - environment dependent
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

    _XLA = True
except ImportError:  # pragma: no cover - plain CPU dev box
    xm = None  # type: ignore[assignment]
    met = None  # type: ignore[assignment]
    _XLA = False


@dataclass(frozen=True, slots=True)
class Device:
    """The active device and its honest backend label."""

    torch_device: torch.device
    kind: str  # "TPU" | "CUDA" | "CPU" | "XLA-CPU"
    is_xla: bool


def detect_device() -> Device:
    """Pick the device: XLA (TPU/XLA-CPU) if torch_xla is present, else CPU."""
    if _XLA:
        dev = xm.xla_device()
        try:
            import torch_xla.runtime as xr

            kind = xr.device_type() or "XLA"
        except Exception:  # noqa: BLE001
            kind = "XLA"
        # Distinguish a real TPU from XLA's CPU fallback.
        label = kind if kind == "TPU" else f"XLA-{kind}"
        return Device(torch_device=dev, kind=label, is_xla=True)
    return Device(torch_device=torch.device("cpu"), kind="CPU", is_xla=False)


def sync(device: Device, result: torch.Tensor) -> None:
    """Force pending work to actually execute and complete.

    torch_xla is lazy: without a sync the timer measures graph-building, not
    execution. `mark_step` flushes the graph; reading to CPU blocks until done.
    """
    if device.is_xla:
        xm.mark_step()
    _ = result.detach().to("cpu")


def uncached_compiles() -> int:
    """Total XLA compilations so far (the `UncachedCompile` counter), else 0.

    Counters return None before first touched; non-XLA backends have none.
    """
    if met is None:
        return 0
    value = met.counter_value("UncachedCompile")
    return int(value) if value is not None else 0


def timed(fn: Callable[[], torch.Tensor], device: Device) -> float:
    """Run `fn`, sync, and return wall-seconds for the completed work."""
    start = time.perf_counter()
    out = fn()
    sync(device, out)
    return time.perf_counter() - start
