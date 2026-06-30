"""Command-line entry point: `xla-recompile-lens`.

Thin wrapper over the library. It runs the *static* analysis on a built-in demo
model so you can see output with zero setup, on any machine. The full measured
TPU before/after lives in the Colab notebook, where `torch_xla` is available.
"""

from __future__ import annotations

import argparse
from typing import override

import torch
from torch import nn

from .attribute import analyze_module
from .capture import xla_available
from .report import render_text


class _DemoModel(nn.Module):
    """Tiny model that exhibits a data-dependent op (`nonzero`) on purpose."""

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.nonzero(x > 0)  # data-dependent: dynamic output shape
        return idx.float().sum()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="xla-recompile-lens",
        description="Statically attribute XLA recompile causes in a model.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Analyze the built-in demo model (default if no other input).",
    )
    parser.parse_args()

    model = _DemoModel()
    report = analyze_module(
        model,
        example_inputs=(torch.randn(8),),
        model_name="DemoModel",
        tpu_available=xla_available(),
    )
    print(render_text(report))
    if not xla_available():
        print(
            "\n[note] torch_xla not found — static analysis only. "
            "Run notebooks/colab_tpu_demo.ipynb on a TPU for measured counts."
        )
