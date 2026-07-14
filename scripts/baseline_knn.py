"""Honesty check: how much of the retrieval-augmented accuracy is the LLM actually adding?

The retrieved-shot prompt reaches 90% on test. But the retriever alone already surfaces
the gold intent ~97% of the time — so a plain TF-IDF k-NN vote over the same neighbours
might score nearly as well *without any LLM*. If it does, the LLM is decoration and the
headline number is not an LLM result.

This runs the no-LLM baseline over the identical retrieval index and examples. CPU only.

Usage:  python -m scripts.baseline_knn [--n 500]
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from src.data import label_maps, load_split, sample
from src.retrieval import ShotRetriever
from scripts.phase1_accuracy import EVAL_SEED, wilson_ci


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    canon, _, _ = label_maps()
    examples = sample(load_split("test"), 500, seed=EVAL_SEED)[: args.n]
    gold = [ex.label for ex in examples]

    r = ShotRetriever()
    sims = linear_kernel(r.vectorizer.transform([e.text for e in examples]), r.matrix)

    print(f"TF-IDF k-NN baseline (no LLM) on {len(examples)} test examples")
    print("for comparison: LLM with k=16 retrieved shots = 90.0% (n=150)\n")

    for k in (1, 8, 16, 32):
        preds = []
        for row in sims:
            top = row.argsort()[-k:][::-1]
            # Similarity-weighted vote, so a close neighbour outranks a distant one.
            votes: Counter = Counter()
            for i in top:
                votes[r.pool[i].label] += row[i]
            preds.append(votes.most_common(1)[0][0])
        correct = np.array([p == g for p, g in zip(preds, gold)])
        c, n = int(correct.sum()), len(correct)
        lo, hi = wilson_ci(c, n)
        tag = "1-NN" if k == 1 else f"{k}-NN weighted vote"
        print(f"  {tag:22s} {c/n:6.1%}  [{c}/{n}]  95% CI {lo:.0%}-{hi:.0%}")

    print("\nIf these land far below the LLM's 90%, the LLM is doing the real work:")
    print("retrieval supplies candidates, the LLM discriminates among confusable intents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
