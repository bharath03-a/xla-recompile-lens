"""Render regime-study charts from measured TPU results.

Reads results/measured_tpu.json (real numbers from TPU runs across three prompt-
length distributions) and writes publication-quality PNGs to docs/assets/ for the
README and the HTML report. Rendering is separate from measuring on purpose: the
JSON is the source of truth, this only draws it -- update the JSON after a new run
and re-render.

    uv run python scripts/render_report.py

Honest framing baked into the charts: the headline is naive -> bucketed (the
recompile-storm collapse); derived vs pow2 is shown as the near-tie it actually
is, not spun as a win.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "measured_tpu.json"
OUT = ROOT / "docs" / "assets"

ORDER = ["alpaca", "dolly", "cnn"]
C_POW2 = "#ff7f0e"
C_DERIVED = "#2ca02c"
C_EXACT = "#d62728"

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def _labels(ds: dict) -> list[str]:
    return [ds[k]["label"] for k in ORDER]


def _headline(ds: dict) -> None:
    """The real win: naive per-length vs bucketed, per dataset."""
    labels = _labels(ds)
    derived = [ds[k]["strategies"]["derived"]["speedup"] for k in ORDER]
    x = range(len(ORDER))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    bars = ax.bar(x, derived, color=C_DERIVED, width=0.6)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="naive per-length (1.00x)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("cold-serving speedup vs naive (x)")
    ax.set_title("Bucketing collapses the recompile storm\n"
                 "(Llama-3.2-1B prefill, real prompts, TPU v5e)")
    for b, v, k in zip(bars, derived, ORDER, strict=True):
        comp = ds[k]["strategies"]["exact"]["compiles"]
        bc = ds[k]["strategies"]["derived"]["compiles"]
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}x\n{comp}->{bc}\ncompiles",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(derived) * 1.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "headline_speedup.png", dpi=150)
    plt.close(fig)


def _derived_vs_pow2(ds: dict) -> None:
    """The honest near-tie: derived matches the pow2 heuristic within ~2%."""
    labels = _labels(ds)
    pow2 = [ds[k]["strategies"]["pow2"]["speedup"] for k in ORDER]
    derived = [ds[k]["strategies"]["derived"]["speedup"] for k in ORDER]
    x = range(len(ORDER))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.4))
    b1 = ax.bar([i - w / 2 for i in x], pow2, w, label="pow2 (heuristic)", color=C_POW2)
    b2 = ax.bar([i + w / 2 for i in x], derived, w, label="derived (advisor)",
                color=C_DERIVED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("cold-serving speedup vs naive (x)")
    ax.set_title("Data-derived MATCHES power-of-two within ~2%\n"
                 "on smooth real distributions (honest null result)")
    for bars, vals in ((b1, pow2), (b2, derived)):
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}x",
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(pow2 + derived) * 1.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "derived_vs_pow2.png", dpi=150)
    plt.close(fig)


def _compiles(ds: dict) -> None:
    """Compile-count collapse: the mechanism behind the speedup."""
    labels = _labels(ds)
    exact = [ds[k]["strategies"]["exact"]["compiles"] for k in ORDER]
    derived = [ds[k]["strategies"]["derived"]["compiles"] for k in ORDER]
    x = range(len(ORDER))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar([i - w / 2 for i in x], exact, w, label="naive per-length", color=C_EXACT)
    ax.bar([i + w / 2 for i in x], derived, w, label="bucketed (derived)",
           color=C_DERIVED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("XLA compilations (UncachedCompile)")
    ax.set_title("The mechanism: variable shapes -> one compile each,\n"
                 "bucketing collapses them to a handful")
    for i, (e, d) in enumerate(zip(exact, derived, strict=True)):
        ax.text(i - w / 2, e, str(e), ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, d, str(d), ha="center", va="bottom", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "compiles_collapse.png", dpi=150)
    plt.close(fig)


def _kcurve(ds: dict) -> None:
    """Advisor K-vs-cost curve (dolly: the spread regime with a real knee)."""
    d = ds["dolly"]
    curve = d["curve"]
    ks = [k for k, _ in curve]
    totals = [t for _, t in curve]
    chosen = d["chosen_k"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(ks, totals, marker="o", color="#1f77b4")
    ax.axvline(chosen, color=C_DERIVED, ls="--", label=f"chosen K={chosen}")
    ax.scatter([chosen], [dict(curve)[chosen]], color=C_DERIVED, zorder=5, s=80)
    ax.set_xlabel("number of buckets (K)")
    ax.set_ylabel("advisor estimated total cost (s)")
    ax.set_title("Advisor finds the knee (Dolly): padding vs compile tradeoff\n"
                 "(estimated on a linear cost proxy -- see post-mortem)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "k_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    data = json.loads(DATA.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    ds = data["datasets"]
    _headline(ds)
    _derived_vs_pow2(ds)
    _compiles(ds)
    _kcurve(ds)
    print(f"wrote regime-study charts to {OUT}")


if __name__ == "__main__":
    main()
