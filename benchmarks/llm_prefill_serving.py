"""End-to-end serving benchmark: data-derived bucketing vs naive vs powers-of-two.

Run:
    uv run python -m benchmarks.llm_prefill_serving --dry-run      # CPU pipeline smoke
    uv run python -m benchmarks.llm_prefill_serving --n 300 --max-len 256   # on TPU

What it measures (prefill, batch=1, bf16):
  * a fitted compute cost model t(len) ~ a*len + b*len^2 from a measured sweep,
  * the measured per-compile cost (cold - warm),
  * for each strategy, the wall-time / #compiles / padding to serve a stream of
    real-distribution prompt lengths:
      - exact   : pad each prompt to its own length (max compiles, min padding)
      - pow2    : fixed {16,32,...} buckets (standard practice)
      - derived : buckets chosen by the advisor to minimize total cost
  * the advisor's K-vs-cost curve (the optimization).

Honesty: strategies are run derived -> pow2 -> exact so the headline (derived)
is measured cold/fairly; minor cross-strategy cache sharing can only *understate*
exact (conservative for our claim). Calibration also pre-warms a few sweep lengths
(some of which may equal pow2 caps); any resulting cache sharing favors pow2 over
derived, again keeping our claim conservative. The active backend (TPU vs XLA-CPU)
is printed and every number is measured on it.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch

from xla_recompile_lens import (
    CostModel,
    LengthHistogram,
    fit_cost_model,
    next_bucket,
    recommend_buckets,
)

from .device import Device, detect_device, timed, uncached_compiles
from .workload import load_model, make_inputs, prefill_forward, prompt_lengths

POW2_BUCKETS = (16, 32, 64, 128, 256, 512)


def pow2_buckets_covering(max_length: int) -> tuple[int, ...]:
    """Power-of-two buckets that always cover `max_length` (no silent clipping)."""
    top = 1 << max(0, max_length - 1).bit_length()  # next power of two >= max_length
    return tuple(b for b in POW2_BUCKETS if b < top) + (top,)


def _run_once(model, length: int, pad_to: int, vocab: int, device: Device) -> float:
    inputs = make_inputs(length, pad_to, vocab, device)
    with torch.no_grad():
        return timed(lambda: prefill_forward(model, inputs), device)


def calibrate(
    model: torch.nn.Module, vocab: int, device: Device, max_len: int
) -> tuple[CostModel, float]:
    """One measured sweep -> (fitted cost model, per-compile cost in seconds)."""
    sweep = sorted({max(1, int(max_len * f)) for f in (0.1, 0.25, 0.4, 0.6, 0.8, 1.0)})
    warm_samples: list[tuple[int, float]] = []
    compile_samples: list[float] = []
    for length in sweep:
        cold = _run_once(model, length, length, vocab, device)  # first = compiles
        warm = statistics.median(
            _run_once(model, length, length, vocab, device) for _ in range(3)
        )
        warm_samples.append((length, warm))
        compile_samples.append(max(cold - warm, 0.0))
    cost = fit_cost_model(warm_samples)
    compile_cost = statistics.median(compile_samples) if compile_samples else 0.0
    return cost, compile_cost


@dataclass(frozen=True, slots=True)
class StrategyResult:
    name: str
    wall_time: float           # cold pass: serve the stream incl. compiles (headline)
    compiles: int
    padded_tokens: int
    real_tokens: int
    warm_mean: float = 0.0     # steady-state: mean per-stream wall over warm passes
    warm_std: float = 0.0      # spread across warm passes (measurement noise)
    warm_samples: int = 0      # number of warm passes averaged

    @property
    def pad_overhead(self) -> float:
        return self.padded_tokens / self.real_tokens if self.real_tokens else 1.0


def _serve_stream(
    model: torch.nn.Module,
    lengths: list[int],
    pad_fn: Callable[[int], int],
    vocab: int,
    device: Device,
) -> float:
    """One full pass over the stream; returns total wall-time."""
    total = 0.0
    for length in lengths:
        total += _run_once(model, length, pad_fn(length), vocab, device)
    return total


def run_strategy(
    name: str,
    model: torch.nn.Module,
    lengths: list[int],
    pad_fn: Callable[[int], int],
    vocab: int,
    device: Device,
    warm_repeats: int = 0,
) -> StrategyResult:
    """Serve the whole stream under a padding policy; measure wall-time + compiles.

    The first pass is *cold* — every distinct shape pays its compile once, which is
    the real cold/diverse-serving cost and the headline number. Any `warm_repeats`
    further passes hit the warm cache (no compiles); their mean±std is the honest
    steady-state cost, reported separately so the cold win is never conflated with it.
    """
    base = uncached_compiles()
    # Running sums over a streaming loop — collecting all at once would needlessly
    # double memory for long serving streams; the result is frozen below.
    padded = real = 0
    cold = 0.0
    for length in lengths:
        pad_to = pad_fn(length)
        cold += _run_once(model, length, pad_to, vocab, device)
        padded += pad_to
        real += length
    compiles = uncached_compiles() - base

    warm = [_serve_stream(model, lengths, pad_fn, vocab, device)
            for _ in range(warm_repeats)]
    return StrategyResult(
        name=name,
        wall_time=cold,
        compiles=compiles,
        padded_tokens=padded,
        real_tokens=real,
        warm_mean=statistics.mean(warm) if warm else 0.0,
        warm_std=statistics.stdev(warm) if len(warm) > 1 else 0.0,
        warm_samples=len(warm),
    )


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Everything a run produces — so callers (CLI, notebook) can print or plot."""

    backend: str
    model_name: str
    dataset: str
    length_source: str
    histogram: LengthHistogram
    cost_a: float
    cost_b: float
    compile_cost: float
    derived_buckets: tuple[int, ...]
    pow2_buckets: tuple[int, ...]
    results: tuple[StrategyResult, ...]
    curve: tuple[tuple[int, float], ...]
    chosen_k: int

    def by_name(self) -> dict[str, StrategyResult]:
        return {r.name: r for r in self.results}


def run_benchmark(
    n: int,
    max_len: int,
    *,
    dataset: str = "alpaca",
    dry_run: bool = False,
    model_name: str | None = None,
    warm_repeats: int = 0,
) -> BenchmarkReport:
    """Run the full serving benchmark and return all measured objects."""
    device = detect_device()
    loaded = load_model(device, model_name=model_name, dry_run=dry_run)
    lengths, length_source = prompt_lengths(
        n, max_len, dataset=dataset, dry_run=dry_run
    )
    hist = LengthHistogram.from_lengths(lengths)
    cost, compile_cost = calibrate(loaded.model, loaded.vocab_size, device, max_len)

    rec = recommend_buckets(hist, cost, compile_cost, k_max=12)
    derived = rec.buckets
    pow2 = pow2_buckets_covering(hist.max_length)

    # Run derived -> pow2 -> exact (see module docstring on ordering/honesty).
    results = tuple(
        run_strategy(
            name, loaded.model, lengths, pad_fn, loaded.vocab_size, device,
            warm_repeats=warm_repeats,
        )
        for name, pad_fn in (
            ("derived", lambda n: next_bucket(n, derived)),
            ("pow2", lambda n: next_bucket(n, pow2)),
            ("exact", lambda n: n),
        )
    )
    return BenchmarkReport(
        backend=device.kind,
        model_name=loaded.name,
        dataset=dataset,
        length_source=length_source,
        histogram=hist,
        cost_a=cost.a,
        cost_b=cost.b,
        compile_cost=compile_cost,
        derived_buckets=derived,
        pow2_buckets=pow2,
        results=results,
        curve=rec.curve,
        chosen_k=rec.k,
    )


def print_report(rep: BenchmarkReport) -> None:
    """Human-readable summary of a `BenchmarkReport`."""
    print(f"backend: {rep.backend}  model: {rep.model_name}  dataset: {rep.dataset}")
    if rep.length_source == "synthetic":
        print("WARNING: lengths are SYNTHETIC (dataset did not load) — not a "
              "real-data result. Do not report as real.")
    else:
        print(f"length source: real ({rep.length_source})")
    if rep.backend != "TPU":
        print("NOTE: not a real TPU — recompilation cost is only meaningful under "
              "XLA. Treat non-TPU numbers as a pipeline check, not a result.")
    h = rep.histogram
    print(f"prompts: {h.total}  distinct lengths: {h.distinct}  max: {h.max_length}")
    print(f"cost(len) = {rep.cost_a:.3e}*len + {rep.cost_b:.3e}*len^2 ; "
          f"compile_cost = {rep.compile_cost*1e3:.1f} ms")
    print(f"derived buckets (K={rep.chosen_k}): {rep.derived_buckets}")
    print(f"pow2 buckets: {rep.pow2_buckets}")

    by = rep.by_name()
    baseline = by["exact"].wall_time
    print("\n=== cold serving (incl. compiles — headline; lower is better) ===")
    for r in rep.results:
        speedup = baseline / r.wall_time if r.wall_time else float("inf")
        print(f"{r.name:<9} {r.wall_time:8.3f}s  compiles={r.compiles:<4} "
              f"pad={r.pad_overhead:4.2f}x  speedup={speedup:4.2f}x")

    if any(r.warm_samples for r in rep.results):
        print("\n=== warm steady-state (cache hot, no compiles; "
              "mean +/- std over passes) ===")
        for r in rep.results:
            print(f"{r.name:<9} {r.warm_mean:8.3f}s +/- {r.warm_std:.3f}  "
                  f"(n={r.warm_samples})")

    d, p = by["derived"], by["pow2"]
    if d.wall_time:
        print(f"\nderived vs exact : {baseline / d.wall_time:.2f}x faster, "
              f"{by['exact'].compiles - d.compiles} fewer compiles")
    if p.wall_time:
        print(f"derived vs pow2  : {p.wall_time / d.wall_time:.2f}x "
              f"(pad {p.pad_overhead:.2f}x -> {d.pad_overhead:.2f}x)")

    print("\nK-vs-total-cost curve (advisor):")
    for k, total in rep.curve:
        mark = "  <- chosen" if k == rep.chosen_k else ""
        print(f"  K={k:<2} est_total={total:.4f}{mark}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500, help="number of prompts")
    ap.add_argument("--max-len", type=int, default=256, help="clip prompt length")
    ap.add_argument("--dataset", default="alpaca", choices=["alpaca", "dolly", "cnn"],
                    help="prompt-length source; long-tailed (cnn/dolly) favors derived")
    ap.add_argument("--warm-repeats", type=int, default=0,
                    help="extra warm passes per strategy for steady-state mean+/-std")
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny random model + few prompts on CPU (pipeline smoke)")
    ap.add_argument("--model", default=None,
                    help="model name (e.g. gpt2); default tries Llama then TinyLlama")
    args = ap.parse_args()
    if args.dry_run:
        args.n, args.max_len = min(args.n, 40), min(args.max_len, 64)
    print_report(
        run_benchmark(
            args.n, args.max_len, dataset=args.dataset, dry_run=args.dry_run,
            model_name=args.model, warm_repeats=args.warm_repeats,
        )
    )


if __name__ == "__main__":
    main()
