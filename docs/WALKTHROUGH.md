# Walkthrough — understand every piece (plain English)

Read this top to bottom once. By the end you'll be able to explain the whole
project to anyone, including Kat.

## The problem in one paragraph

A TPU runs a model by first having the XLA compiler build a specialized program
for one *exact* input shape. Change the shape (e.g. a longer sentence) and XLA
throws that program away and builds a new one — a **recompilation**. Real
workloads have constantly-changing shapes, so they recompile over and over and
waste most of their time. This is the #1 documented performance problem on TPU,
and the exact thing Google's TorchTPU team is working to fix ("bounded
dynamism", "Fused Eager"). Our tool diagnoses it.

## The three reasons a recompile happens

1. **Variable input shape** — you feed different-sized tensors each step.
2. **Data-dependent op** — an op like `nonzero` whose *output size depends on
   the values*, so the compiler can't know the size ahead of time.
3. **Control flow on a tensor value** — `if x[0] > 0:` forces the program to
   stop and look at the data, which breaks the compiled graph in two.

Remember these three. The whole tool is organized around them.

## The files, in the order they matter

### `src/xla_recompile_lens/types.py` — the vocabulary
Defines the words everything else uses: `RecompileCause` (the three reasons
above + "unknown"), a `Finding` (one detected problem + its fix), and a
`Report` (all findings + any measured numbers). These are *frozen* (immutable):
once built, they can't be changed — so a report you print is exactly what was
analyzed. Nothing can secretly alter the numbers.

### `src/xla_recompile_lens/attribute.py` — the static doctor
Reads your model *without running it* (using `torch.fx`, which turns a model
into an inspectable graph) and looks for reasons #2 and #3. Finds a `nonzero`?
That's a `Finding` with cause `DATA_DEPENDENT_OP` and a suggested fix. Model
branches on a tensor value? fx can't trace it — we catch that and report it as
`CONTROL_FLOW` instead of crashing. **Runs on your Mac, no TPU.**

### `src/xla_recompile_lens/shapes.py` — counting shape variety
Reason #1 isn't in the code, it's in the *data you feed*. `ShapeSignatureTracker`
watches the shapes flowing through and counts how many *distinct* ones it sees.
Distinct shapes = lower bound on recompiles. This is what `examples/see_it_work.py`
uses to show 100 → 3.

### `src/xla_recompile_lens/fixes.py` — the remedy
`pad_to_bucket` rounds an input up to the next standard size (32, 64, 128...).
Turns "100 different shapes" into "3 shapes". This is the fix you *prove* works.

### `src/xla_recompile_lens/capture.py` — the real measurement (needs TPU)
On a real TPU, `torch_xla` keeps a counter called `UncachedCompile` that ticks
up once per real recompile. This file reads that counter before and after a run
to get the **true** recompile count. On a Mac (no `torch_xla`) it refuses to run
and tells you to use Colab — it never makes up a number. That honesty is the point.

### `src/xla_recompile_lens/report.py` — showing results
Takes a `Report` and prints a readable table, and draws the before/after bar
chart. Only reads data, never computes — so it can't distort anything.

### `src/xla_recompile_lens/cli.py` — the one-command demo
`uv run xla-recompile-lens --demo` runs the static doctor on a tiny model so
you see output instantly.

### `src/xla_recompile_lens/plugin.py` — THE PLUGIN (the part they test once)
This is the piece a TorchTPU engineer can drop into their own code. `torch.compile`
has a pluggable "backend", and PyTorch's Dynamo engine calls that backend
**once every time it recompiles**. So we insert a thin lens in front of the real
backend: every call = one recompile, which we record (with its input shape).
You attach it in one line — `compiled, lens = instrument(model)` — run your
workload, then `lens.render()` tells you how many recompiles and which shapes
caused them. It needs no TPU and no special metrics; on TorchTPU you just pass
their backend as `inner=`, and it measures a real TPU run the same way.

### `src/xla_recompile_lens/autobucket.py` — the fix as a one-liner
`auto_bucket(compiled)` wraps your model so every input is padded to a standard
bucket size before it runs. Variable shapes collapse to a handful, recompiles
collapse with them — and you didn't touch your data loader. This is the
"after" half of the before/after.

## How to test it / see results

| What | Command | Needs |
|------|---------|-------|
| See the core idea (100 → 3) | `uv run python examples/see_it_work.py` | nothing |
| **See the plugin (6 → 3 recompiles)** | `uv run python examples/plugin_demo.py` | nothing |
| See the static doctor | `uv run xla-recompile-lens --demo` | nothing |
| Run the test suite | `uv run pytest` | nothing |
| See the agent comparison | `uv run python -m fusion_bench.demo` | nothing |
| **Real measured numbers** | open `notebooks/colab_tpu_demo.ipynb` in Colab, runtime = TPU, Run all | free Colab TPU |

The first four work on your laptop today. The last is the one that produces the
*measured* before/after — that's the screenshot you send Kat.

## The agents part (`fusion_bench/`) — and your competition

Your MLSys win used multiple AI "agents" planning together to schedule a graph
under memory limits. TorchTPU's *Fused Eager* groups operations together — the
same kind of problem. So the obvious idea: point your agents at it.

`fusion_bench/` does the honest version of that question. It compares:
- a **simple, instant** scheduler (the kind a compiler can actually use), vs.
- an **expensive, smarter** optimizer (standing in for your multi-agent search).

Result: the smart one is ~7% better but ~120x slower *per decision*. Since a
compiler makes these decisions constantly, the slow one is disqualified from the
live path — but it's perfect as an *offline* tool to discover good rules.

**Why this matters for you:** anyone can say "I'll throw AI at it." Showing you
know *where AI helps and where it doesn't* — with measured numbers — is what a
compiler team respects. It turns your competition strength into a sign of
judgment, not a one-trick hammer.

## Can you add more agents? Yes — but carefully

A legitimate extension: an **offline "fusion advisor"** — your multi-agent
planner runs once, off the hot path, analyzes a model, and *suggests* bucket
sizes or fusion groups that a human or the fast scheduler then uses. That fits
the honest framing above (AI offline, fast rules online). It would be a strong
follow-up. What you should NOT do is put an LLM call inside the compile loop —
that's the thing the benchmark shows is a bad idea.

## How to show Kat you're genuinely into this

1. **Run it and read this doc** until you can explain recompilation in your own
   words without notes. Understanding > the code itself.
2. **Get the real Colab number** and put the before/after chart in the README.
3. **Send the email**: "XLA recompilation is TorchTPU's documented bottleneck,
   so I built a tool that classifies every recompile by cause and proves the fix
   with measured numbers on a TPU. I also tested whether my multi-agent approach
   belongs in the compiler — it doesn't in the hot path, but it's a great
   offline advisor. Repo + chart attached."
4. **Be ready to talk**: she may ask "what are the three causes?" or "why not
   just use your agents everywhere?" — this doc gives you both answers.
