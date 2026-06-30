"""Tests for the static fx-graph recompile-cause classifier.

All of these run on CPU — no TPU/torch_xla needed — which is the point: the
attribution logic is verifiable anywhere.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from xla_recompile_lens import RecompileCause, analyze_module


class DataDependentModel(nn.Module):
    """Uses `nonzero` -> a data-dependent (dynamic output shape) op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(x > 0).float().sum()


class MaskedSelectModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.masked_select(x, x > 0).sum()


class StaticModel(nn.Module):
    """Pure static ops — should produce zero findings."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).relu()


class ControlFlowModel(nn.Module):
    """Branches on a tensor value -> fx cannot trace one graph."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:  # value-dependent control flow
            return x * 2
        return x * 3


@pytest.mark.unit
def test_detects_nonzero_as_data_dependent() -> None:
    report = analyze_module(DataDependentModel(), (torch.randn(8),))
    causes = report.causes_summary()
    assert causes.get(RecompileCause.DATA_DEPENDENT_OP) == 1
    assert report.findings[0].op == "nonzero"
    assert "bounded dynamism" in report.findings[0].fix.lower()


@pytest.mark.unit
def test_detects_masked_select() -> None:
    report = analyze_module(MaskedSelectModel(), (torch.randn(8),))
    assert RecompileCause.DATA_DEPENDENT_OP in report.causes_summary()


@pytest.mark.unit
def test_static_model_has_no_findings() -> None:
    report = analyze_module(StaticModel(), (torch.randn(2, 4),))
    assert report.causes_summary() == {}


@pytest.mark.unit
def test_control_flow_is_attributed_not_crashed() -> None:
    # fx raises on value-dependent control flow; we must turn that into a
    # finding, never let it crash the tool.
    report = analyze_module(ControlFlowModel(), (torch.randn(8),))
    causes = report.causes_summary()
    assert (
        RecompileCause.CONTROL_FLOW in causes
        or RecompileCause.UNKNOWN in causes
    )


@pytest.mark.unit
def test_report_is_immutable() -> None:
    report = analyze_module(StaticModel(), (torch.randn(2, 4),))
    with pytest.raises((AttributeError, TypeError)):
        report.model_name = "mutated"  # type: ignore[misc]
