# Draft: pytorch/xla GitHub issue

> Post this as a **feature request** issue on https://github.com/pytorch/xla
> *before* opening a PR, to check interest and align on scope. Tone: helpful,
> specific, humble. Link the repo once it's public.

---

**Title:** Tooling to attribute XLA recompilations to their root cause

**Labels (suggest):** enhancement, debuggability

**Body:**

### Problem

Recompilation is the dominant performance cliff for dynamic PyTorch/XLA
workloads — the docs already enumerate the three root causes (variable input
shape, data-dependent ops, value-dependent control flow). But when a real
training run is slow, finding *which* cause is firing — and *where* in the model
— still means manually reading `PT_XLA_DEBUG=1` logs and correlating compile
events by hand. There's no first-class way to get an attributed summary.

### Proposal

A small diagnostic utility that:

1. **Statically** scans an fx graph and flags ops that force dynamic shapes
   (`nonzero`, `masked_select`, `unique`, `.item()`) and value-dependent
   control flow, with the source location and the documented fix for each.
2. **At runtime** reads the `UncachedCompile` counter across steps to report the
   *measured* compile count, and demonstrates the before/after when a fix
   (e.g. bucketized padding) is applied.

Output is a per-model report: cause breakdown, the specific op/line, a suggested
remediation, and the measured recompile delta after the fix.

### Prior art / what I've built

I prototyped this as a standalone package and validated the before/after on a
free Colab TPU: a 2-layer transformer encoder fed per-length (1..40) vs.
bucketized sequences shows the `UncachedCompile` counter drop from **39 to 1
over 40 steps (97% fewer compiles)**, with the Dynamo-backend plugin
independently measuring 40 → 2. Happy to:

- contribute it under `torch_xla/debug/` or `contrib/` if that's wanted, or
- keep it standalone and link it from the recompilation docs.

Would the team find this useful, and if so, where should it live? I'd like to
align on scope before sending a PR.

---

## Notes for Bharath (not part of the issue)

- File the issue from your GitHub account; link the repo once pushed.
- If a maintainer responds positively, the PR is: move `src/xla_recompile_lens`
  attribution + capture into the agreed location, add their CI conventions.
- Even if they prefer it standalone, the issue + repo is a public, dated
  artifact showing you engaging the real codebase — link it to Kat.
- Secondary contribution if you want more surface area: the op-coverage gap
  report (run real models through `torch.compile(backend="openxla")`, rank ATen
  ops that graph-break). Empirical and immediately useful.
