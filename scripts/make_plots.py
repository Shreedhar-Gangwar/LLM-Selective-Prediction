"""Phase 4 figures for the report — two clean, colorblind-safe matplotlib panels.

  1. risk_coverage.png  — selective accuracy vs coverage for all four signals on one
     axis, with the chosen operating point marked. The headline of the whole project.
  2. reliability_logprob.png — reliability diagram for the logprob max-softmax signal:
     is a stated confidence of p actually right about p of the time?

Colors are the Okabe-Ito palette, colorblind-safe by construction; every series is also
direct-labeled or legended so identity never rests on color alone. Reads the committed
report/risk_coverage.json and the cached Phase 2 test records.

Usage:  python -m scripts.make_plots
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cache import CACHE_DIR

REPORT_DIR = Path(__file__).resolve().parent.parent / "report"

# Okabe-Ito: colorblind-safe categorical hues, assigned in fixed order by signal rank.
COLORS = {
    "logprob_margin": "#0072B2",   # blue
    "logprob_max": "#56B4E9",      # sky blue
    "self_consistency": "#E69F00", # orange
    "verbalized": "#D55E00",       # vermillion
}
GRID = "#D9D9D9"
INK = "#222222"


def plot_risk_coverage() -> None:
    data = json.loads((REPORT_DIR / "risk_coverage.json").read_text())
    op = json.loads((REPORT_DIR / "operating_point.json").read_text())

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    # The logprob signals sweep the whole coverage range (many distinct scores) so they
    # read as lines; self-consistency and verbalized take few discrete values and occupy
    # only a narrow high-coverage corner, so markers make their short extent visible.
    styles = {
        "logprob_margin": dict(lw=2.4, zorder=4),
        "logprob_max": dict(lw=1.8, zorder=3),
        "self_consistency": dict(lw=2, marker="o", markersize=5, zorder=5),
        "verbalized": dict(lw=2, marker="s", markersize=5, zorder=5),
    }
    for field in ("logprob_margin", "logprob_max", "self_consistency", "verbalized"):
        c = data["curves"][field]
        cov = np.array(c["coverage"])
        acc = np.array(c["selective_accuracy"])
        keep = cov > 0.02  # tiny-coverage tail is noise
        ax.plot(cov[keep] * 100, acc[keep] * 100, color=COLORS[field],
                label=c["name"], **styles[field])

    # base accuracy (answer everything) and the chosen operating point
    ax.axhline(op["test_base_accuracy"] * 100, color="#999", lw=1.3, ls="--", zorder=0)
    ax.annotate(f"base accuracy {op['test_base_accuracy']:.0%} (answer everything)",
                xy=(70, op["test_base_accuracy"] * 100 - 0.15), fontsize=8, color="#666",
                va="top", ha="center")
    ax.scatter([op["test_coverage"] * 100], [op["test_accepted_accuracy"] * 100],
               color=INK, zorder=6, s=60, marker="o")
    ax.annotate(
        f"operating point\n{op['test_accepted_accuracy']:.1%} acc @ "
        f"{op['test_coverage']:.0%} coverage",
        xy=(op["test_coverage"] * 100, op["test_accepted_accuracy"] * 100),
        xytext=(op["test_coverage"] * 100 + 2.5, op["test_accepted_accuracy"] * 100 + 1.6),
        fontsize=8.5, color=INK, ha="left", va="bottom",
        arrowprops=dict(arrowstyle="->", color=INK, lw=1),
    )

    ax.set_xlabel("Coverage — % of tickets answered")
    ax.set_ylabel("Selective accuracy — % correct among answered")
    ax.set_title("Risk-coverage: only the logprob signal buys accuracy by abstaining",
                 fontsize=12, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(88, 100.5)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.01, 0.03), fontsize=9,
              title="confidence signal", title_fontsize=9, alignment="left")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "risk_coverage.png", dpi=150)
    plt.close(fig)


def plot_reliability(n_bins: int = 10) -> None:
    rows = [json.loads(l) for l in (CACHE_DIR / "phase2_test_n1000.jsonl").open()]
    conf = np.array([r["logprob_max"] for r in rows])
    correct = np.array([r["correct"] for r in rows], dtype=float)

    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 5:
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
            ns.append(int(m.sum()))

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls="--", zorder=0)
    ax.annotate("perfect calibration", xy=(0.62, 0.6), rotation=34, fontsize=8,
                color="#888", va="bottom")
    sizes = 40 + 240 * np.array(ns) / max(ns)
    ax.scatter(xs, ys, s=sizes, color=COLORS["logprob_margin"], alpha=0.85,
               edgecolor="white", lw=1, zorder=3)
    ax.plot(xs, ys, color=COLORS["logprob_margin"], lw=1.5, zorder=2)

    ax.set_xlabel("Stated confidence (max-softmax probability)")
    ax.set_ylabel("Empirical accuracy in bin")
    ax.set_title("Reliability of the logprob signal (marker size = bin count)",
                 fontsize=11.5, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "reliability_logprob.png", dpi=150)
    plt.close(fig)


def plot_group_conditional() -> None:
    """Per-group accepted accuracy under one marginal threshold vs per-group thresholds.

    The story: pooling hides the `transfers` group below the 90% target; a per-group
    threshold restores it. Bars are accuracy; the coverage cost is in the report table.
    """
    path = REPORT_DIR / "group_conditional.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    target = (1 - data["alpha"]) * 100
    recs = sorted(data["groups"], key=lambda r: (r["marginal_accuracy"] or 0))
    names = [r["group"].replace("_", " ") for r in recs]
    x = np.arange(len(recs))
    w = 0.38

    m_acc = [(r["marginal_accuracy"] or 0) * 100 for r in recs]
    g_acc = [(r["gc_accuracy"] * 100 if r["gc_accuracy"] is not None else None) for r in recs]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.bar(x - w / 2, m_acc, w, color="#BBBBBB", label="one marginal threshold")
    for xi, v in zip(x, g_acc):
        if v is None:  # group-conditional abstains on everything (insufficient data)
            ax.bar(xi + w / 2, target, w, color="none", edgecolor="#999",
                   hatch="//", lw=1)
            ax.annotate("abstains", (xi + w / 2, target + 0.4), ha="center",
                        fontsize=6.5, color="#777", rotation=90, va="bottom")
        else:
            ax.bar(xi + w / 2, v, w, color=COLORS["logprob_margin"],
                   label="per-group threshold" if xi == 0 else None)

    ax.axhline(target, color="#D55E00", lw=1.5, ls="--", zorder=5)
    ax.annotate(f"{target:.0f}% target", xy=(len(recs) - 0.4, target + 0.3),
                fontsize=8.5, color="#D55E00", ha="right", va="bottom")
    # call out the group pooling hid
    bad = next((i for i, r in enumerate(recs) if r["group"] == "transfers"), None)
    if bad is not None:
        ax.annotate("pooling hid this\nfailure", xy=(bad - w / 2, m_acc[bad]),
                    xytext=(bad - 0.1, 82.5), fontsize=8, color=INK, ha="center",
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1))

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Accepted accuracy on test (%)")
    ax.set_title("Group-conditional calibration restores subgroups the marginal one hides",
                 fontsize=11.5, color=INK)
    ax.set_ylim(80, 101)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9,
              facecolor="white", edgecolor="none")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "group_conditional.png", dpi=150)
    plt.close(fig)


def main() -> int:
    REPORT_DIR.mkdir(exist_ok=True)
    plot_risk_coverage()
    plot_reliability()
    plot_group_conditional()
    print(f"wrote {REPORT_DIR}/risk_coverage.png, reliability_logprob.png, "
          f"group_conditional.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
