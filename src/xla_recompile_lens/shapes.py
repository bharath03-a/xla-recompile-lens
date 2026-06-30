"""Variable-input-shape detection.

The first recompile cause (`VARIABLE_SHAPE`) is special: it is *not* a property
of the model's code, it is a property of the *data you feed it*. A model that is
perfectly static will still recompile on every batch if you hand it a new tensor
shape each time. So we cannot find this cause by reading the fx graph — we have
to watch the shapes that actually flow through.

`ShapeSignatureTracker` records the "shape signature" of each call (the tuple of
input shapes) and reports how many *distinct* signatures it saw. XLA compiles
one graph per distinct signature, so `distinct_signatures - 1` is a direct lower
bound on shape-driven recompiles.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

# PEP 695 type alias (Python 3.12+). A shape signature is the ordered tuple of
# the input tensors' shapes — naming it keeps the nested generics readable.
type ShapeSig = tuple[tuple[int, ...], ...]


def shape_signature(inputs: Iterable[object]) -> ShapeSig:
    """Return the tuple of shapes for the tensor inputs in `inputs`.

    Non-tensor args are ignored: they do not drive XLA shape specialization.
    Two calls with the same signature reuse the same compiled graph; two calls
    with different signatures each cost a compilation.
    """
    sig: list[tuple[int, ...]] = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            sig.append(tuple(x.shape))
    return tuple(sig)


@dataclass(slots=True)
class ShapeSignatureTracker:
    """Accumulates distinct shape signatures across many calls.

    This is intentionally *mutable* — it is an accumulator, the one place where
    we collect observations over time. Everything it produces (the count, the
    set of signatures) is then frozen into an immutable `Report`/`Finding`.
    """

    _seen: set[ShapeSig]

    def __init__(self) -> None:
        self._seen = set()

    def observe(self, *inputs: object) -> None:
        """Record one call's input shapes."""
        self._seen.add(shape_signature(inputs))

    @property
    def distinct_signatures(self) -> int:
        return len(self._seen)

    @property
    def shape_driven_recompiles(self) -> int:
        """Lower bound on recompiles caused purely by shape variation.

        The first signature is the unavoidable initial compile; every
        *additional* distinct signature is one extra compilation.
        """
        return max(0, self.distinct_signatures - 1)

    def signatures(self) -> tuple[ShapeSig, ...]:
        """All distinct signatures seen, for reporting."""
        return tuple(sorted(self._seen))
