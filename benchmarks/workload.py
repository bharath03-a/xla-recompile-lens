"""Model + workload setup for the serving benchmark.

Three concerns, kept honest:

* **Model.** Primary `meta-llama/Llama-3.2-1B-Instruct` (gated; auth via HF
  login / Colab secret). Falls back to ungated `TinyLlama-1.1B-Chat` so the
  benchmark runs without a token. `--dry-run` builds a tiny *random* Llama (no
  download) purely to exercise the pipeline.
* **Lengths.** Sampled from a real dataset (`tatsu-lab/alpaca`) and tokenized to
  get true token lengths. Only the *length distribution* is used — prefill cost
  is a function of shape, not token values — so we run synthetic ids of those
  lengths. That keeps the benchmark about the compiler effect, honestly.
* **Prefill.** One forward pass with an attention mask; right-padding keeps
  causal outputs at real positions correct (see tests/test_padding_correctness).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .device import Device

PRIMARY_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# Dataset registry: name -> (hf_path, config, split, text_fn). text_fn maps a row
# to the prompt string we tokenize for its length. Short sets (alpaca) cluster
# below ~100 tokens where power-of-two bucketing is already near-optimal; long,
# heavy-tailed sets (cnn, dolly-with-context) spread mass into the awkward gaps
# *above* each power of two, where pow2 over-pads and the advisor's data-derived
# buckets pull ahead. Which wins is an empirical question we measure, not assume.
def _alpaca_text(r: dict) -> str:
    return r["instruction"] + "\n" + (r.get("input") or "")


def _dolly_text(r: dict) -> str:
    return r["instruction"] + "\n" + (r.get("context") or "")


def _cnn_text(r: dict) -> str:
    return r["article"]


DATASETS: dict[str, tuple[str, str | None, str, Callable[[dict], str]]] = {
    "alpaca": ("tatsu-lab/alpaca", None, "train", _alpaca_text),
    "dolly": ("databricks/databricks-dolly-15k", None, "train", _dolly_text),
    "cnn": ("abisee/cnn_dailymail", "3.0.0", "train", _cnn_text),
}
DATASET = "alpaca"  # default; long-context runs pass dataset="cnn" (or "dolly")


@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: torch.nn.Module
    vocab_size: int
    name: str


def load_model(
    device: Device,
    *,
    model_name: str | None = None,
    dry_run: bool = False,
    dtype=torch.bfloat16,
) -> LoadedModel:
    """Load a model onto `device`.

    `model_name` forces a specific model (e.g. "gpt2", which runs reliably on
    free v5e TPUs where Llama currently segfaults under torch_xla 2.8). If None,
    tries the gated Llama then the ungated TinyLlama fallback. `dry_run` builds a
    tiny random Llama (no download) to exercise the pipeline.
    """
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    if dry_run:
        # Tiny random Llama — no download, exercises the exact code path.
        cfg = LlamaConfig(
            vocab_size=512, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=1024,
        )
        model = LlamaForCausalLM(cfg).to(device.torch_device).eval()
        return LoadedModel(
            model=model, vocab_size=cfg.vocab_size, name="tiny-random-llama"
        )

    candidates = [model_name] if model_name else [PRIMARY_MODEL, FALLBACK_MODEL]
    last_err: Exception | None = None
    for name in candidates:
        try:
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
            model = model.to(device.torch_device).eval()
            return LoadedModel(
                model=model, vocab_size=model.config.vocab_size, name=name
            )
        except Exception as exc:  # noqa: BLE001 - try the fallback, report at end
            last_err = exc
            print(f"[workload] could not load {name}: {exc}")
    raise RuntimeError(f"failed to load any model; last error: {last_err}")


def prompt_lengths(
    n: int,
    max_len: int,
    *,
    dataset: str = DATASET,
    dry_run: bool = False,
    seed: int = 0,
) -> tuple[list[int], str]:
    """Prompt token-lengths and their source label `(lengths, source)`.

    `source` is the dataset name when real lengths loaded, or `"synthetic"` when
    we fell back to a lognormal draw (dry-run, or `datasets` unavailable). Callers
    surface this so a synthetic run is never mislabeled as real data — the whole
    credibility of the result depends on that distinction.
    """
    rng = random.Random(seed)
    if not dry_run:
        try:
            return _dataset_lengths(n, max_len, seed, dataset), dataset
        except Exception as exc:  # noqa: BLE001 - fall back to synthetic, disclose
            print(f"[workload] dataset unavailable ({exc}); using synthetic lengths")
    lengths = [
        max(1, min(max_len, int(round(rng.lognormvariate(3.4, 0.6)))))
        for _ in range(n)
    ]
    return lengths, "synthetic"


def _dataset_lengths(n: int, max_len: int, seed: int, dataset: str) -> list[int]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choices: {sorted(DATASETS)}")
    path, config, split, text_fn = DATASETS[dataset]
    tok = AutoTokenizer.from_pretrained(FALLBACK_MODEL)  # ungated tokenizer is fine
    # Stream + shuffle-buffer instead of a full download: cnn_dailymail is ~1.3 GB,
    # and we only need the first n shuffled rows' token lengths.
    ds = (load_dataset(path, config, split=split, streaming=True) if config
          else load_dataset(path, split=split, streaming=True))
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    lengths: list[int] = []
    for row in ds:
        length = len(tok(text_fn(row), add_special_tokens=True)["input_ids"])
        lengths.append(max(1, min(max_len, length)))
        if len(lengths) >= n:
            break
    return lengths


def make_inputs(
    length: int, pad_to: int, vocab: int, device: Device
) -> dict[str, torch.Tensor]:
    """Build a single right-padded prefill batch of real `length`, padded to `pad_to`.

    Token *values* are arbitrary (prefill cost depends on shape); the attention
    mask marks the `length` real positions so padding never affects them.
    """
    ids = torch.randint(0, vocab, (1, pad_to), dtype=torch.long)
    mask = torch.zeros(1, pad_to, dtype=torch.long)
    mask[:, :length] = 1
    return {
        "input_ids": ids.to(device.torch_device),
        "attention_mask": mask.to(device.torch_device),
    }


def prefill_forward(model: torch.nn.Module, inputs: dict[str, torch.Tensor]):
    """Run prefill through the transformer *backbone* and return hidden states.

    We deliberately skip the LM head: its logits span the full vocab (128k for
    Llama), so `[1, seq, 128256]` tensors cached across many distinct compiled
    shapes exhaust TPU HBM and segfault — while contributing nothing to the
    shape-driven recompilation we are measuring. The backbone (`model.model`)
    exercises the same attention/MLP compute that recompiles per sequence length,
    at ~60x less memory. Returns `last_hidden_state`.
    """
    # Llama/TinyLlama expose `.model`; GPT-2 exposes `.transformer`.
    backbone = getattr(model, "model", None) or getattr(model, "transformer", model)
    return backbone(**inputs).last_hidden_state
