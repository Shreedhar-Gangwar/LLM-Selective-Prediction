"""Phase 3 — calibrate the abstention threshold and report risk-coverage.

Reads the cached Phase 2 records (no GPU). For each confidence signal and each target
accepted-error level alpha:
  1. calibrate a threshold on the CALIBRATION split with Learn-then-Test (src/conformal),
  2. freeze it and apply it to the TEST split,
  3. report achieved accepted-accuracy AND coverage on test, and check the risk promise.

Calibration/test hygiene: the threshold sees only calibration data; test is touched once,
to report. The whole risk-coverage curve is emitted (for the Phase 4 plots) before any
single operating point is chosen.

Usage:  python -m scripts.phase3_calibrate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.cache import CACHE_DIR
from src.conformal import (
    evaluate_threshold,
    ltt_threshold,
    naive_threshold,
    risk_coverage_curve,
)

DELTA = 0.10          # the guarantee holds w.p. >= 1 - DELTA over the calibration draw
ALPHAS = [0.10, 0.05, 0.02]   # target accepted-error: >=90%, >=95%, >=98% accuracy
PRIMARY = "logprob_margin"
HEADLINE_ALPHA = 0.05

# (record field, display name); order = the signal ranking from Phase 2.
SIGNALS = [
    ("logprob_margin", "logprob margin"),
    ("logprob_max", "logprob max-softmax"),
    ("self_consistency", "self-consistency"),
    ("verbalized", "verbalized"),
]


def load(split: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    path = CACHE_DIR / f"phase2_{split}_n1000.jsonl"
    rows = [json.loads(l) for l in path.open()]
    scores = {f: np.array([r[f] for r in rows], dtype=float) for f, _ in SIGNALS}
    correct = np.array([r["correct"] for r in rows], dtype=bool)
    return scores, correct


def main() -> int:
    cal_scores, cal_correct = load("calibration")
    test_scores, test_correct = load("test")
    n_cal, n_test = len(cal_correct), len(test_correct)

    print(f"Calibration n={n_cal} (acc {cal_correct.mean():.1%}), "
          f"test n={n_test} (acc {test_correct.mean():.1%})")
    print(f"LTT calibration at delta={DELTA}; guarantee: accepted accuracy >= 1-alpha "
          f"w.p. >= {1-DELTA:.0%}.\n")

    print(f"{'signal':20s} {'alpha':>6s} {'target':>7s} | "
          f"{'cal cov':>7s} | {'TEST cov':>8s} {'TEST acc':>8s} {'held?':>6s}")
    print("-" * 78)

    for field, name in SIGNALS:
        for alpha in ALPHAS:
            t = ltt_threshold(cal_scores[field], cal_correct, alpha, DELTA)
            if not t.controlled:
                print(f"{name:20s} {alpha:6.2f} {1-alpha:6.0%}  |  "
                      f"{'—':>6s} |  abstains on everything (target not achievable)")
                continue
            cov, acc, n, k = evaluate_threshold(test_scores[field], test_correct, t.tau)
            held = "yes" if (n == 0 or (1 - acc) <= alpha) else "NO"
            print(f"{name:20s} {alpha:6.2f} {1-alpha:6.0%}  |  "
                  f"{t.calib_coverage:6.1%} |  {cov:7.1%} {acc:7.1%} {held:>6s}")
        print()

    # -- naive contrast on the primary signal at the headline alpha --------------
    print("Why the finite-sample correction matters (primary signal, "
          f"alpha={HEADLINE_ALPHA}):")
    tau_ltt = ltt_threshold(cal_scores[PRIMARY], cal_correct, HEADLINE_ALPHA, DELTA).tau
    tau_naive = naive_threshold(cal_scores[PRIMARY], cal_correct, HEADLINE_ALPHA)
    for label, tau in (("LTT (finite-sample)", tau_ltt), ("naive (empirical)", tau_naive)):
        cov, acc, n, _ = evaluate_threshold(test_scores[PRIMARY], test_correct, tau)
        print(f"  {label:22s} tau={tau:.3f}  ->  test coverage {cov:.1%}, "
              f"test accepted-acc {acc:.1%}")

    # -- the chosen operating point ---------------------------------------------
    t = ltt_threshold(cal_scores[PRIMARY], cal_correct, HEADLINE_ALPHA, DELTA)
    cov, acc, n, k = evaluate_threshold(test_scores[PRIMARY], test_correct, t.tau)
    print("\n" + "=" * 78)
    print(f"OPERATING POINT — {PRIMARY}, target accepted-accuracy >= {1-HEADLINE_ALPHA:.0%}")
    print(f"  threshold tau = {t.tau:.4f}  (calibrated on calibration, frozen)")
    print(f"  TEST: accepted accuracy {acc:.1%} on {cov:.1%} coverage "
          f"({n}/{n_test} answered, {k} wrong)")
    print(f"  vs base accuracy {test_correct.mean():.1%} answering 100%.")
    print(f"  Human-review load: {1-cov:.1%} of tickets routed to a human "
          f"(down from 100% with no automation).")
    print("=" * 78)

    # -- emit risk-coverage curves for the Phase 4 plots -------------------------
    curves = {}
    for field, name in SIGNALS:
        rc = risk_coverage_curve(test_scores[field], test_correct)
        curves[field] = {
            "name": name,
            "coverage": rc.coverage.tolist(),
            "selective_accuracy": rc.selective_accuracy.tolist(),
        }
    out = CACHE_DIR / "phase3_curves.json"
    out.write_text(json.dumps({"delta": DELTA, "alphas": ALPHAS, "curves": curves}))
    print(f"\nrisk-coverage curves (test) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
