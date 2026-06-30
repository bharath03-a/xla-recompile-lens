# xla-recompile-lens

[![CI](https://github.com/bharath03-a/xla-recompile-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/bharath03-a/xla-recompile-lens/actions/workflows/ci.yml)

**Classify every XLA recompilation by root cause — on a real TPU run — and prove the fix with measured before/after numbers.**

## Measured result (free Colab TPU)

A 2-layer transformer encoder fed variable sequence lengths (1..40), before vs.
after one-line input bucketing, measured two independent ways on a real TPU:

| Measurement path | Recompiles before | after | reduction |
|------------------|------------------:|------:|----------:|
| Lazy-tensor `UncachedCompile` counter | 39 / 40 steps | **1** | **97%** |
| Dynamo-backend plugin | 40 / 40 steps | **2** | **95%** |

Reproduce: [`notebooks/colab_tpu_demo.ipynb`](notebooks/colab_tpu_demo.ipynb)
(Runtime → TPU → Run all).

## Data-derived sequence bucketing (real LLM, real distributions)

Bucketing fixes recompilation — but *which* buckets? Powers of two are a guess.
The [`bucket_advisor`](src/xla_recompile_lens/bucket_advisor.py) **derives** the
bucket set from the workload: it fits a compute cost model to measured prefill
timings, measures the per-compile cost, and solves a contiguous-partition DP to
trade padding waste against compile count — then picks the bucket count K at the
knee of the cost curve.

The [serving benchmark](benchmarks/llm_prefill_serving.py) validates it
end-to-end on Llama-3.2-1B prefill across **three real prompt-length
distributions**, comparing `exact` (per-length) vs `pow2` vs `derived` bucketing
by **measured TPU wall-time**.

### Measured on a real TPU (Kaggle v5e, Llama-3.2-1B-Instruct)

The headline is **naive → bucketed**: collapsing the recompile storm is a 4–10×
win. Cold serving wall-time (incl. compilation), speedup vs naive per-length:

| Dataset (regime) | naive | `pow2` | `derived` | compiles (naive→derived) |
|------------------|------:|-------:|----------:|-------------------------:|
| Alpaca (short)   | 1.00× | 4.83×  | 4.31×     | 110 → 7 |
| Dolly (spread)   | 1.00× | 6.80×  | 6.65×     | 156 → 7 |
| CNN/DM (saturated) | 1.00× | 10.42× | 10.14×  | 116 → 1 |

![bucketing collapses the recompile storm](docs/assets/headline_speedup.png)

**The honest finding:** I hypothesized data-derived buckets would *beat*
power-of-two. Across all three real distributions they **matched it within ~2%**
(pow2 marginally ahead) — on smooth real prompt lengths, pow2 is already
near-optimal. The advisor's value is that it *derives* the set + count
automatically (and collapses to K=1 when there's nothing to bucket), not that it
beats the heuristic on a benchmark.

![data-derived matches pow2 within ~2%](docs/assets/derived_vs_pow2.png)

Why it ties: `derived` takes fewer compiles but more padding; the fitted cost
model is linear, so it under-penalizes padding and over-trades it for compiles.
A cost-model post-mortem is in the report.

**Full writeup (the journey + post-mortem):**
[rendered HTML report](https://htmlpreview.github.io/?https://github.com/bharath03-a/xla-recompile-lens/blob/main/docs/report.html)
· plain-text [`docs/REPORT.md`](docs/REPORT.md)
· source [`docs/report.html`](docs/report.html).

Reproduce (Colab or Kaggle): [`notebooks/llm_serving_tpu.ipynb`](notebooks/llm_serving_tpu.ipynb)
(switch `DATASET` ∈ {alpaca, dolly, cnn}).
Local pipeline check: `uv run python -m benchmarks.llm_prefill_serving --dry-run`.
Charts regenerate from [`results/measured_tpu.json`](results/measured_tpu.json)
via `uv run python scripts/render_report.py`.

XLA graph recompilation is the documented bottleneck for dynamic PyTorch
workloads on TPU: pods routinely sit at ~50–55% MFU purely because the
compiler re-traces and re-compiles whenever a shape, a data-dependent op, or a
control-flow branch changes. This is the problem TorchTPU's *bounded dynamism*
and *Fused Eager* work is built to attack.

Today, finding *why* a workload recompiles means hand-reading XLA debug logs.
`xla-recompile-lens` automates that: it attributes each recompile to one of the
three known root causes, emits an actionable report, and shows the recompile
count drop after you apply the suggested fix.

## The plugin (start here) — wrap your model, run once, see the cost

The headline is a **drop-in `torch.compile` backend**. Torch Dynamo — the graph
capture engine behind `torch.compile`, and the one TorchTPU uses — calls its
backend exactly once per recompilation. Slip a thin lens in front of the real
backend and **counting our calls counts the recompiles**, with no profiler and
no vendor metrics. Because it rides the standard Dynamo seam, the *same* wrapper
runs on CPU today and should map directly to TorchTPU by passing their XLA
backend as `inner` — same mechanism, pending confirmation on real TPU hardware.

```python
from xla_recompile_lens import instrument, auto_bucket

compiled, lens = instrument(model)        # one line: measure recompiles
for batch in batches:
    compiled(batch)
print(lens.render())                       # how many, and the distinct shapes

fast = auto_bucket(compiled, dim=-1)       # the fix, no data-pipeline changes
```

Try it now, no TPU:

```bash
uv run python examples/plugin_demo.py
# WITHOUT plugin fix: 6 recompiles  ->  WITH auto_bucket: 3 recompiles
```

On TorchTPU, the only change is the inner backend:

```python
compiled, lens = instrument(model, inner=torch_tpu_backend)  # measures a real TPU run
```

## The three root causes (from the PyTorch/XLA docs)

| Cause | Trigger | Suggested fix |
|-------|---------|---------------|
| **Variable input shape** | New input shape signature each step (e.g. seq-length padding) | Bucketize / pad to a bound |
| **Data-dependent op** | `nonzero`, `masked_select`, `.item()` → dynamic output shape | Bounded dynamism / pad to max |
| **Control flow on tensor value** | `if x[0] == 3:` materializes a value, breaks the graph | Replace with `torch.where` / functional control flow |

## Quickstart (free Colab TPU)

Open `notebooks/colab_tpu_demo.ipynb` in Colab, set runtime to **TPU**, and run.
It installs `torch_xla`, runs a real model, and prints the attributed report
plus the before/after recompile chart.

## Local (CPU) usage

The fx-graph analysis (data-dependent ops, control-flow detection, shape
signatures) runs anywhere `torch` is installed — no TPU required:

```python
from xla_recompile_lens import analyze_module, render_text
report = analyze_module(model, example_inputs)
print(render_text(report))
```

The *measured recompile counts* require `torch_xla` (TPU/Colab); without it the
tool falls back to static fx analysis and says so explicitly.

## Why this exists

Built as a focused study of TorchTPU's hardest problem. See
[`fusion_bench/`](fusion_bench/) for an honest experiment comparing a classical
memory-aware fusion scheduler against a multi-agent LLM planner — including why
the classical algorithm wins inside the compile loop.

## Layout

```
src/xla_recompile_lens/
  plugin.py        drop-in torch.compile backend (RecompileLens) — counts recompiles
  autobucket.py    auto_bucket: the fix as a one-line wrapper
  attribute.py     fx-graph 3-cause classifier (static, CPU)
  shapes.py        shape-signature tracking (variable-shape cause)
  capture.py       real torch_xla UncachedCompile counter hooks (guarded)
  fixes.py         bucketize / pad helpers
  report.py        report table + matplotlib chart
  types.py         immutable domain model
  cli.py           `xla-recompile-lens --demo`
examples/          CPU-runnable demos (see_it_work.py, plugin_demo.py)
fusion_bench/      classical vs. multi-agent fusion experiment
notebooks/         Colab TPU demo (lazy-tensor + openxla paths)
docs/              WALKTHROUGH.md (plain-English tour) + issue draft
tests/             31 tests, all CPU-runnable
```

## Development

```bash
uv sync --extra viz      # install (torch, matplotlib, dev tools)
uv run ruff check .      # lint
uv run pytest            # 31 tests, no TPU needed
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint + tests +
smoke-runs every CPU demo on each push/PR to `main`. TPU paths live in the
notebook (no TPU CI runner).

This repo ships Claude Code helpers in [`.claude/`](.claude/): a
`recompile-analyst` agent, a `tpu-honesty-reviewer` agent (guards against
fabricated/overclaimed numbers), and `verify` / `colab-run` skills.

### Contributing principle

The project's value is **honest, measured** numbers. Never fabricate a
measurement, never describe a static prediction as a measured recompile, and
always qualify "works on TorchTPU" as validated-on-the-Dynamo-seam until
confirmed on real TorchTPU hardware.
