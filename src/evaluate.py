"""Batch evaluation harness — the whole pipeline from cache, no GPU.

Reads the cached Phase 2 signal records for a split and the frozen Phase 3 operating
point, then reports the three numbers that define the service: raw accuracy (answer
everything), accepted accuracy (answer only above threshold), and coverage — plus the
human-review-load reduction that coverage implies.

Usage:  python -m src.evaluate [--split test]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.cache import CACHE_DIR
from src.conformal import evaluate_threshold

OPERATING_POINT = Path(__file__).resolve().parent.parent / "report" / "operating_point.json"


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    path = CACHE_DIR / f"phase2_{split}_n1000.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts.phase2_signals --split {split}")
    rows = [json.loads(l) for l in path.open()]
    return (
        np.array([r["logprob_margin"] for r in rows], dtype=float),
        np.array([r["correct"] for r in rows], dtype=bool),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "calibration"])
    args = ap.parse_args()

    op = json.loads(OPERATING_POINT.read_text())
    scores, correct = load_split(args.split)
    n = len(correct)

    raw_acc = float(correct.mean())
    cov, acc, n_acc, n_err = evaluate_threshold(scores, correct, op["tau"])

    print(f"Split: {args.split}  (n={n})")
    print(f"Operating point: {op['signal']}, tau={op['tau']:.4f}, "
          f"target accepted-accuracy >= {op['target_accepted_accuracy']:.0%}\n")
    print(f"  raw accuracy (answer all)   : {raw_acc:.1%}   [{correct.sum()}/{n}]")
    print(f"  accepted accuracy (answer)  : {acc:.1%}   [{n_acc - n_err}/{n_acc}]")
    print(f"  coverage (automation rate)  : {cov:.1%}   [{n_acc}/{n}]")
    print(f"  human-review load           : {1 - cov:.1%}   [{n - n_acc}/{n} routed to a human]")
    print(f"\n  Lift from abstention: {acc - raw_acc:+.1%} accuracy on answered tickets, "
          f"at the cost of routing {1 - cov:.1%} to humans.")
    held = "holds" if (1 - acc) <= op["alpha"] else "VIOLATED"
    print(f"  Risk promise on this split: accepted error {1 - acc:.1%} "
          f"vs target <= {op['alpha']:.0%}  ->  {held}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
