"""See the recompilation problem — and the fix — on your own laptop.

Run me:  uv run python examples/see_it_work.py

No TPU needed. XLA compiles one graph per distinct input *shape*, so the number
of distinct shapes you feed a model is a direct lower bound on how many times it
must recompile. `ShapeSignatureTracker` counts exactly that — real counting, no
simulation. This script shows the count exploding with raw inputs and collapsing
once we bucketize, which is the same effect you'll measure for real on a TPU.
"""

from __future__ import annotations

import torch

from xla_recompile_lens import ShapeSignatureTracker, pad_to_bucket

# Imagine 100 text inputs whose lengths vary from 1 to 100 tokens — exactly the
# real-world situation that makes TPUs thrash.
SENTENCE_LENGTHS = list(range(1, 101))
BUCKETS = (32, 64, 128)


def main() -> None:
    raw = ShapeSignatureTracker()
    bucketed = ShapeSignatureTracker()

    for length in SENTENCE_LENGTHS:
        x = torch.zeros(1, length)  # one input of this length
        raw.observe(x)                              # feed it as-is
        bucketed.observe(pad_to_bucket(x, BUCKETS, dim=-1))  # padded to a bucket

    print("Fed the model 100 inputs of varying length.\n")
    print(f"  WITHOUT fix: {raw.distinct_signatures:>3} distinct shapes "
          f"-> ~{raw.shape_driven_recompiles} recompiles")
    print(f"  WITH fix:    {bucketed.distinct_signatures:>3} distinct shapes "
          f"-> ~{bucketed.shape_driven_recompiles} recompiles")

    saved = raw.shape_driven_recompiles - bucketed.shape_driven_recompiles
    print(f"\n  => Bucketizing avoided ~{saved} recompiles.")
    print("     On a TPU each avoided recompile is seconds of wasted compute.")


if __name__ == "__main__":
    main()
