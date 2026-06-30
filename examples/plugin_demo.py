"""The plugin in action — measure recompiles, then kill them. Runs on CPU.

    uv run python examples/plugin_demo.py

This is the "wrap your model, run once, see it" story. We compile a tiny model
with a RecompileLens attached and feed it variable-length inputs (lots of
recompiles). Then we wrap the same model with `auto_bucket` and feed the *same*
inputs — and the recompile count collapses. No data-pipeline changes, one line.

The exact same wrapping works on TorchTPU by passing their XLA backend as
`inner=` to `instrument` — the measurement rides Torch Dynamo, not any vendor
API.
"""

from __future__ import annotations

import torch
from torch import nn

from xla_recompile_lens import auto_bucket, instrument, set_recompile_limit

# Dynamo gives up recompiling a frame after this many tries (default 8) — itself
# a symptom of the thrashing problem. Raise it so we can *measure* the full cost.
set_recompile_limit(256)

# Variable-length inputs: the real-world cause of TPU recompile thrashing.
LENGTHS = [5, 12, 5, 30, 12, 7, 64, 5, 100, 12]


class TinyMLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sum over the variable dim so any length is valid.
        return (x * 2.0).relu().sum()


def run(compiled, transform=lambda x: x) -> None:
    for n in LENGTHS:
        compiled(transform(torch.randn(1, n)))


def main() -> None:
    model = TinyMLP()

    # BEFORE: feed raw, variable shapes.
    compiled_raw, lens_raw = instrument(model)
    run(compiled_raw)

    # AFTER: same model, same inputs, but auto-bucketed to {32, 64, 128}.
    compiled_fix, lens_fix = instrument(model)
    fast = auto_bucket(compiled_fix, buckets=(32, 64, 128), dim=-1)
    run(fast)

    print(f"Fed {len(LENGTHS)} variable-length inputs.\n")
    print(f"  WITHOUT plugin fix: {lens_raw.recompile_count} recompiles "
          f"({lens_raw.distinct_shapes} distinct shapes)")
    print(f"  WITH auto_bucket:   {lens_fix.recompile_count} recompiles "
          f"({lens_fix.distinct_shapes} distinct shapes)")
    saved = lens_raw.recompile_count - lens_fix.recompile_count
    print(f"\n  => {saved} recompiles eliminated by a one-line wrapper.")
    print("\n--- detail (before) ---")
    print(lens_raw.render())


if __name__ == "__main__":
    main()
