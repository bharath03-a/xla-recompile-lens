"""auto_bucket: collapse shape-driven recompiles without touching your data.

The diagnostic (`plugin.RecompileLens`) tells you that variable input shapes are
causing recompiles. `auto_bucket` is the *fix as a plugin*: wrap your callable
and it pads each input up to the next bucket size before the model runs, so the
model only ever sees a small fixed set of shapes — and Dynamo only compiles that
many graphs.

    compiled, lens = instrument(model)          # measure
    fast = auto_bucket(compiled, dim=-1)         # fix, no pipeline changes
    for batch in batches:
        fast(batch)
    print(lens.render())                          # recompiles collapsed

This mirrors TorchTPU's "bounded dynamism" idea at the framework level: trade a
little padding compute for far fewer compilations.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

import torch

from .fixes import pad_to_bucket

DEFAULT_BUCKETS = (32, 64, 128, 256, 512)


def auto_bucket(
    fn: Callable[..., object],
    buckets: tuple[int, ...] = DEFAULT_BUCKETS,
    *,
    dim: int = -1,
    pad_value: float = 0.0,
) -> Callable[..., object]:
    """Wrap `fn` so every tensor arg is padded to the next bucket along `dim`.

    Only tensor positional args are padded; everything else passes through
    untouched. Returns a new callable — the original is not modified.
    """

    @wraps(fn)
    def _wrapped(*args: object, **kwargs: object) -> object:
        new_args = tuple(
            pad_to_bucket(a, buckets, dim=dim, pad_value=pad_value)
            if isinstance(a, torch.Tensor)
            else a
            for a in args
        )
        return fn(*new_args, **kwargs)

    return _wrapped
