# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this is

`xla-recompile-lens` — a diagnostic tool that attributes XLA graph
recompilations to their root cause and proves the fix with **measured**
before/after numbers. Built as a portfolio artifact targeting Google's TorchTPU
team: it demonstrates real compiler-internals understanding, not a toy.

Two parts:
- **`src/xla_recompile_lens/`** — the installable library (the product).
- **`fusion_bench/`** — a separate research experiment comparing a classical
  fusion scheduler against a multi-agent-style global planner. Its conclusion is
  deliberately self-critical (the planner is too slow for the compile loop).

## The core principle: never fake numbers

The entire credibility of this project rests on real measurements. The capture
layer reads `torch_xla`'s actual `UncachedCompile` counter (note: `CompileTime`
is a *time metric*, not a counter — do not read it via `counter_value`). When
`torch_xla` is absent (CPU/Mac), the tool falls back to **static fx analysis**
and says so — it must **never** fabricate, estimate, or simulate device metrics.
Any change that hand-waves a number defeats the purpose. Results are only labeled
**TPU** when `device_kind() == "TPU"` (torch_xla silently falls back to XLA-CPU).

## Architecture (one-way data flow)

```
attribute.py / capture.py / shapes.py   →  produce immutable types.py objects
                                         →  report.py only READS them
```

- `types.py` — frozen dataclasses + `RecompileCause` (StrEnum). The vocabulary.
- `attribute.py` — static fx scan: data-dependent ops + control-flow breaks.
- `shapes.py` — `ShapeSignatureTracker` for the variable-shape cause.
- `capture.py` — real `torch_xla` compile-count hooks (guarded import).
- `fixes.py` — `pad_to_bucket` (the bounded-dynamism remediation).
- `report.py` — text table + before/after chart (presentation only).
- `cli.py` — `xla-recompile-lens --demo`.

## Conventions (enforced)

- **Immutable data:** frozen dataclasses / tuples. Mutable accumulators (e.g.
  `ShapeSignatureTracker`) are the rare, explicit exception — and their output
  is frozen into a `Report`.
- **Type annotations** on every signature. `from __future__ import annotations`.
- **Small focused files.** New analysis passes emit `Finding`s; new reporters
  read a `Report`. Don't couple them.
- **torch_xla is optional** — guard every import; the package must install and
  the static path must run on a Mac with no TPU.
- Comments explain the *why* (this is also a teaching artifact), not the *what*.

- The **plugin** (`plugin.py`, `autobucket.py`) rides the Torch Dynamo backend
  seam, so it measures recompiles on any hardware (CPU/GPU/TPU) with no vendor
  metrics. On TorchTPU you pass their backend as `inner=`.
- The **bucket advisor** (`bucket_advisor.py`) is pure/CPU-testable: a fitted
  cost model + a contiguous-partition DP that derives the optimal bucket set for
  a workload. The **serving benchmark** (`benchmarks/`) validates it end-to-end
  on a real LLM (Llama-3.2-1B-Instruct, TinyLlama fallback) over real Alpaca
  prompt lengths, measuring wall-time on the active backend. `--dry-run` exercises
  the whole pipeline on CPU with a tiny random model (no download/TPU).

## Commands

```bash
uv sync --extra viz          # set up env (torch + matplotlib + dev tools)
uv run pytest                # 31 tests, all CPU-runnable (no TPU needed)
uv run ruff check .          # lint (notebooks excluded)
uv run xla-recompile-lens --demo      # static analysis demo
uv run python examples/see_it_work.py   # core idea: 100 -> 3 shapes
uv run python examples/plugin_demo.py   # plugin: recompiles drop via auto_bucket
uv run python -m fusion_bench.demo    # classical vs. planner comparison
```

Measured TPU numbers come from `notebooks/colab_tpu_demo.ipynb` on a free
Colab TPU runtime.

## Project agents (`.claude/agents/`)

- **recompile-analyst** — reasons about *why* a workload recompiles and which
  fix applies. Use when interpreting `RecompileLens` output or sizing buckets.
- **tpu-honesty-reviewer** — guards the project's credibility: no fabricated
  numbers, no unqualified "works on TorchTPU" claims. Run before committing
  changes to measurements, the capture path, or README claims.

## Project skills (`.claude/skills/`)

- **verify** — the full CPU verification loop (sync, ruff, pytest, smoke-run all
  demos). Use before committing.
- **colab-run** — prepare/interpret the Colab TPU run and backfill the real
  measured numbers into the docs (honestly).

## CI (`.github/workflows/ci.yml`)

On push/PR to `main`: uv sync → ruff → pytest → smoke-run every CPU demo. Mirrors
the `verify` skill. TPU paths are not in CI (no TPU runner); they live in the
notebook. Keep CI green; never weaken an honesty assertion to pass it.

## Open follow-ups

- Run `notebooks/colab_tpu_demo.ipynb` on Colab TPU to capture the real
  before/after count; backfill it into `docs/pytorch-xla-issue-draft.md`
  (currently `~N`). The `colab-run` skill walks this through.
- Optional `transformers` extra + a real BERT static-analysis example.
- File the issue in `docs/pytorch-xla-issue-draft.md` after pushing to GitHub.
