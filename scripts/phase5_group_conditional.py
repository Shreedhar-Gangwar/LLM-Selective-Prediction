"""Phase 5 (stretch) — group-conditional coverage.

A marginal guarantee controls accepted-set risk *pooled over all 77 intents*. Pooling can
hide a subgroup that misses the target. This script shows that, then fixes it.

Demonstrated at the 90% target (accepted accuracy >= 90%), which is what ~100-150
calibration examples per group can actually certify distribution-free:

  1. MARGINAL — one threshold (calibrated on all groups pooled) applied to every group.
     The `transfers` group lands at 86% accepted accuracy — below 90% — even though the
     pooled number clears it. Pooling hid a real failure.
  2. GROUP-CONDITIONAL — a SEPARATE threshold per group, each calibrated on that group's
     calibration data with Learn-then-Test. `transfers` gets a stricter threshold that
     restores >=90% accuracy, at the cost of lower coverage; 7 of 8 groups certify.

Two honesties are reported, not hidden:
  * the stricter 95% target is NOT per-group certifiable here — subgroup guarantees need
    subgroup-sized calibration data, quantified below;
  * a smaller threshold grid is used per group than in the marginal case, because ~150
    examples don't support 100 candidate thresholds under Bonferroni (coarser grid = less
    multiplicity, still valid).

Runs from cache — no GPU.  Usage:  python -m scripts.phase5_group_conditional
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.cache import CACHE_DIR
from src.conformal import evaluate_threshold, ltt_threshold
from src.groups import GROUP_NAMES, group_of

REPORT_DIR = Path(__file__).resolve().parent.parent / "report"
SIGNAL = "logprob_margin"
ALPHA = 0.10          # target accepted error: >=90% accepted accuracy (per-group achievable)
DELTA = 0.10
GROUP_GRID = 30       # coarser grid, matched to ~150 calibration examples per group


def load(split: str) -> list[dict]:
    return [json.loads(l) for l in (CACHE_DIR / f"phase2_{split}_n1000.jsonl").open()]


def arrays(rows: list[dict], group: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    sel = [r for r in rows if group is None or group_of(r["gold"]) == group]
    return (np.array([r[SIGNAL] for r in sel], dtype=float),
            np.array([r["correct"] for r in sel], dtype=bool))


def main() -> int:
    cal, test = load("calibration"), load("test")

    # marginal threshold: one threshold from all calibration data pooled, at the 90% target
    cal_s, cal_c = arrays(cal)
    tau_marginal = ltt_threshold(cal_s, cal_c, ALPHA, DELTA).tau

    print(f"Group-conditional coverage — signal={SIGNAL}, target accepted-acc >= "
          f"{1-ALPHA:.0%}, delta={DELTA}")
    print(f"Marginal threshold (all groups pooled): tau = {tau_marginal:.4f}\n")

    hdr = (f"{'group':18s} {'nte':>4s} | {'MARGINAL threshold':^24s} | "
           f"{'GROUP-CONDITIONAL threshold':^30s}")
    print(hdr)
    print(f"{'':18s} {'':>4s} | {'cov':>7s} {'acc':>7s} {'>=90%?':>7s} | "
          f"{'tau':>6s} {'cov':>7s} {'acc':>7s} {'>=90%?':>7s}")
    print("-" * len(hdr))

    records = []
    for g in GROUP_NAMES:
        cs, cc = arrays(cal, g)
        ts, tc = arrays(test, g)

        m_cov, m_acc, m_n, m_k = evaluate_threshold(ts, tc, tau_marginal)
        m_ok = "yes" if (m_n and (1 - m_acc) <= ALPHA) else "NO"

        gc = ltt_threshold(cs, cc, ALPHA, DELTA, grid_size=GROUP_GRID)
        if gc.controlled:
            g_cov, g_acc, g_n, g_k = evaluate_threshold(ts, tc, gc.tau)
            g_ok = "yes" if (g_n and (1 - g_acc) <= ALPHA) else "NO"
            gc_cells = f"{gc.tau:6.3f} {g_cov:7.1%} {g_acc:7.1%} {g_ok:>7s}"
        else:
            g_cov = g_acc = None
            gc_cells = f"{'—':>6s} {'abstains-all':>15s} {'—':>7s}"

        print(f"{g:18s} {len(ts):4d} | {m_cov:7.1%} {m_acc:7.1%} {m_ok:>7s} | {gc_cells}")
        records.append({
            "group": g, "n_test": len(ts),
            "marginal_coverage": m_cov, "marginal_accuracy": m_acc if m_n else None,
            "gc_controlled": gc.controlled,
            "gc_tau": gc.tau if gc.controlled else None,
            "gc_coverage": g_cov, "gc_accuracy": g_acc,
        })

    below = [r for r in records if r["marginal_accuracy"] is not None
             and (1 - r["marginal_accuracy"]) > ALPHA]
    fixed = [r for r in below if r["gc_accuracy"] is not None
             and (1 - r["gc_accuracy"]) <= ALPHA]
    n_cert = sum(r["gc_controlled"] for r in records)

    # honest limitation: the stricter 95% target per group
    cert95 = sum(ltt_threshold(*arrays(cal, g), 0.05, DELTA, grid_size=GROUP_GRID).controlled
                 for g in GROUP_NAMES)

    print("\n" + "=" * 78)
    print(f"Marginal threshold leaves {len(below)} group below {1-ALPHA:.0%}: "
          f"{[r['group'] for r in below]}  (pooling hid it).")
    print(f"Group-conditional calibration certifies {n_cert}/{len(GROUP_NAMES)} groups and "
          f"restores {len(fixed)}/{len(below)} failing group(s) to target.")
    print(f"Limitation: at the stricter 95% target, only {cert95}/{len(GROUP_NAMES)} groups "
          f"are per-group certifiable — subgroup guarantees need subgroup-sized data.")
    print("=" * 78)

    out = REPORT_DIR / "group_conditional.json"
    out.write_text(json.dumps({"alpha": ALPHA, "delta": DELTA, "signal": SIGNAL,
                               "tau_marginal": tau_marginal, "groups": records}, indent=2))
    print(f"\nreport artifact -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
