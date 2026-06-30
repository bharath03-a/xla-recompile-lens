---
name: colab-run
description: Guide and validate the Colab TPU run that produces the real measured before/after numbers, then backfill them into the docs. Use when the user is about to run, or has just run, notebooks/colab_tpu_demo.ipynb on a TPU.
---

# colab-run

The one step that turns this from "static analysis" into "measured on a real
TPU." This skill prepares the notebook, interprets the output, and updates the
docs with the real number.

## Before the run — validate the notebook

```bash
python3 -c "import json; json.load(open('notebooks/colab_tpu_demo.ipynb')); print('valid')"
```

Confirm cell 1's install line points at the right source:
- pushed repo: `pip install -q git+https://github.com/<user>/xla-recompile-lens.git`
- uploaded: `pip install -q -e /content/<repo-dir>`

Remind the user: **Runtime → Change runtime type → TPU**, then Run all.

## Two measurement paths in the notebook (both should agree)

1. **Lazy-tensor path** — `capture_steps` + `before_after_delta`, reads
   torch_xla's `CompileTime`. Produces the bar chart PNG.
2. **Plugin path** — `instrument(fwd, inner='openxla')`, counts via Dynamo.

Expect both to show recompiles dropping from ~40 (per-length) to ~3
(bucketized).

## After the run — backfill real numbers

The READMEs and walkthrough currently use placeholders (`~N`). Replace them with
the actual measured before/after counts the user reports:

- `README.md` (plugin quickstart + any measured claim)
- `fusion_bench/README.md` is unaffected (CPU latency, already real)
- `docs/pytorch-xla-issue-draft.md` ("from ~N to ~3" line)

## Honesty guardrails

- Only write numbers the user actually observed on the TPU run. If they didn't
  run it, leave placeholders.
- Keep the openxla-vs-TorchTPU qualifier in any claim: measured on torch_xla's
  openxla backend; same Dynamo seam as TorchTPU, real-TorchTPU confirmation
  pending.
- If the plugin (openxla) cell errors on a given torch_xla version, fall back to
  the lazy-tensor chart — note which path produced the number.
