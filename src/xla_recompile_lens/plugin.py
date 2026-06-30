"""The plugin: a drop-in `torch.compile` backend that measures recompiles live.

This is the piece a TorchTPU engineer can actually *test once and come back*.

How it works — the key mechanism:
    PyTorch's Torch Dynamo (the engine behind `torch.compile`, and the same one
    TorchTPU uses to capture FX graphs) calls its *backend* exactly once per
    distinct compilation. Every time the input guards change (a new shape, a new
    branch), Dynamo recompiles and calls the backend again. So if we slip a thin
    backend in front of the real one, **counting our invocations counts the
    recompiles** — directly, with no profiler and no vendor metrics.

Because this rides the standard Dynamo seam, the *same* wrapper runs on CPU
today and should map directly to TorchTPU: there, you pass their TPU/XLA backend
as the `inner` to get the recompile breakdown for a real TPU run (same
mechanism, pending confirmation on real TPU hardware).

    lens = RecompileLens()
    compiled = torch.compile(model, backend=lens.backend())  # one line
    for batch in batches:
        compiled(batch)
    print(lens.render())   # how many recompiles, and the distinct shapes

Pair it with `auto_bucket` (see below) to *fix* shape-driven recompiles without
touching the data pipeline, and show the count drop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import fx

from .shapes import ShapeSig, shape_signature

# A Dynamo backend is a callable: (GraphModule, example_inputs) -> callable.
Backend = Callable[[fx.GraphModule, list[object]], Callable[..., object]]


def set_recompile_limit(n: int) -> None:
    """Raise Dynamo's per-frame recompile cap so we can *measure* full thrash.

    The config field was renamed across PyTorch versions: it is
    `cache_size_limit` on older releases (e.g. torch 2.8, the version the Colab
    TPU wheel pins) and `recompile_limit` on newer ones. We set whichever exist
    so the demos and the notebook run unchanged on either — otherwise the lens
    would hit the default cap (8) and undercount, and a hard-coded attribute
    name would crash on the other version.
    """
    cfg = torch._dynamo.config
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, n)


def _eager_backend(gm: fx.GraphModule, _inputs: list[object]) -> Callable[..., object]:
    """Default inner backend: just run the captured graph eagerly.

    Dependency-free and hardware-agnostic — perfect for diagnosis. For real
    speed (or on TPU) pass a production backend as `inner`, e.g. "inductor" or
    TorchTPU's XLA backend; this wrapper measures it without changing it.
    """
    return gm.forward


def _resolve_backend(inner: Backend | str) -> Backend:
    """Accept a backend callable or a registered name (e.g. "openxla").

    On Colab TPU you pass `inner="openxla"` (torch_xla's Dynamo backend); this
    looks it up in PyTorch's backend registry so the lens measures a real
    XLA-compiled run.
    """
    if callable(inner):
        return inner
    return torch._dynamo.lookup_backend(inner)


@dataclass(frozen=True, slots=True)
class RecompileEvent:
    """One observed recompilation."""

    ordinal: int
    """1 for the first compile, 2 for the next, ... — order matters."""
    shape_sig: ShapeSig
    """Input shape signature that triggered this compile."""
    is_new_shape: bool
    """True if this signature was never seen before (a shape-driven recompile);
    False would mean a recompile for some other guard at a known shape."""


@dataclass(slots=True)
class RecompileLens:
    """Live recompile recorder. Wrap a model's backend with `.backend()`.

    Mutable by design — it is the accumulator that watches a run. Each recorded
    event is an immutable `RecompileEvent`; the summary it produces is read-only.
    """

    events: list[RecompileEvent] = field(default_factory=list)
    _seen: set[ShapeSig] = field(default_factory=set)

    def backend(self, inner: Backend | str = _eager_backend) -> Backend:
        """Return a Dynamo backend that records, then delegates to `inner`.

        `inner` may be a backend callable or a registered name like "openxla"
        (the torch_xla backend on Colab TPU).
        """
        resolved = _resolve_backend(inner)

        def _instrumented(
            gm: fx.GraphModule, example_inputs: list[object]
        ) -> Callable[..., object]:
            sig = shape_signature(example_inputs)
            is_new = sig not in self._seen
            self._seen.add(sig)
            self.events.append(
                RecompileEvent(
                    ordinal=len(self.events) + 1,
                    shape_sig=sig,
                    is_new_shape=is_new,
                )
            )
            return resolved(gm, example_inputs)

        return _instrumented

    @property
    def recompile_count(self) -> int:
        """Total compilations Dynamo performed (1 is ideal; more = thrashing)."""
        return len(self.events)

    @property
    def distinct_shapes(self) -> int:
        return len(self._seen)

    def render(self) -> str:
        """Human-readable summary of what the run cost in recompiles."""
        lines = ["=== RecompileLens ==="]
        lines.append(f"recompiles: {self.recompile_count}")
        lines.append(f"distinct input shapes: {self.distinct_shapes}")
        if self.distinct_shapes and self.recompile_count > 1:
            lines.append(
                f"  -> {self.distinct_shapes} of these look shape-driven; "
                "bucketize inputs (see auto_bucket) to collapse them."
            )
        for e in self.events:
            flag = "new shape" if e.is_new_shape else "same shape, other guard"
            lines.append(f"  #{e.ordinal}: {e.shape_sig}  ({flag})")
        return "\n".join(lines)


def instrument(
    fn: Callable[..., object],
    inner: Backend | str = _eager_backend,
    *,
    dynamic: bool = False,
) -> tuple[Callable[..., object], RecompileLens]:
    """One-call setup: compile `fn` with a lens attached.

    Returns `(compiled_fn, lens)`. Run your workload through `compiled_fn`, then
    read `lens.render()`. `dynamic=False` keeps Dynamo's default shape
    specialization so you see the real per-shape recompiles. Pass
    `inner="openxla"` on Colab TPU to measure a real XLA-compiled run.
    """
    lens = RecompileLens()
    compiled = torch.compile(fn, backend=lens.backend(inner), dynamic=dynamic)
    return compiled, lens
