"""Phase 1 — raw top-1 accuracy on banking77. No abstention logic yet.

Runs the locked classifier (see src/config.py: LLM + 16 TF-IDF-retrieved few-shot examples,
length-normalized log-prob scoring over the 77 intents) and reports:

  * raw top-1 accuracy — the anchor for the whole risk-coverage story,
  * the no-LLM TF-IDF k-NN baseline on the SAME examples, so the LLM's contribution is
    stated honestly rather than implied,
  * a paired McNemar test on that difference, so "the LLM adds N points" is backed by a
    significance test and not by two overlapping confidence intervals,
  * a first look at whether the confidence signal separates correct from wrong answers —
    the property Phase 3's abstention will exploit.

Every per-example scoring result is cached by (example_id, params); re-running is free.

Usage:  python -m scripts.phase1_accuracy [--n 500] [--split test]
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from src.cache import JsonlCache
from src.config import BASELINE_KNN_16, K_SHOTS, NORMALIZATION, cache_name, normalize, scoring_params
from src.data import Example, label_maps, load_split, sample, to_natural_language
from src.model import LabelScorer
from src.retrieval import ShotRetriever

EVAL_SEED = 7  # fixed: the evaluation subsample must be reproducible


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion — honest error bars on accuracy."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test on two paired boolean correctness vectors.

    Only the disagreements carry information: b01 = a right & b wrong, b10 = the reverse.
    Under H0 (the systems are equally accurate) each disagreement is a fair coin, so the
    p-value is a two-sided binomial test on b01 out of (b01 + b10).
    Returns (b01, b10, p_value).
    """
    from scipy.stats import binomtest

    b01 = int(np.sum(a & ~b))
    b10 = int(np.sum(~a & b))
    if b01 + b10 == 0:
        return b01, b10, 1.0
    return b01, b10, float(binomtest(b01, b01 + b10, 0.5).pvalue)


def knn_predict(retriever: ShotRetriever, examples: list[Example], k: int) -> list[str]:
    """Similarity-weighted k-NN vote over the retrieval pool. No LLM involved."""
    sims = linear_kernel(
        retriever.vectorizer.transform([e.text for e in examples]), retriever.matrix
    )
    preds = []
    for row in sims:
        votes: Counter = Counter()
        for i in row.argsort()[-k:][::-1]:
            votes[retriever.pool[i].label] += row[i]
        preds.append(votes.most_common(1)[0][0])
    return preds


def score_examples(
    scorer: LabelScorer,
    retriever: ShotRetriever,
    examples: list[Example],
    nl: list[str],
    cache: JsonlCache,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sum_logprobs, n_tokens) for every example, using and filling the cache."""
    sums = np.empty((len(examples), len(nl)))
    ntok = np.empty(len(nl))
    n_hit = 0
    t0 = time.time()

    for i, ex in enumerate(examples):
        hit = cache.get(ex.example_id)
        if hit is None:
            shots = [
                (n.text, to_natural_language(n.label))
                for n in retriever.retrieve(ex.text, K_SHOTS)
            ]
            res = scorer.score_labels(ex.text, nl, shots)
            hit = {"sum": res.sum_logprobs.tolist(), "ntok": res.n_tokens.tolist()}
            cache.set(ex.example_id, hit)
        else:
            n_hit += 1
        sums[i] = hit["sum"]
        ntok = np.asarray(hit["ntok"])

        done = i + 1
        if done % 50 == 0 or done == len(examples):
            miss = done - n_hit
            eta = ((time.time() - t0) / max(miss, 1)) * (len(examples) - done)
            print(
                f"  {done}/{len(examples)}  (cache hits {n_hit})"
                + (f"  ~{eta/60:.1f} min left" if miss and done < len(examples) else ""),
                flush=True,
            )

    print(f"  {n_hit} from cache, {len(examples)-n_hit} scored on GPU "
          f"({time.time()-t0:.0f}s)")
    return sums, ntok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--split", default="test", choices=["test", "calibration"])
    args = ap.parse_args()

    canon, nl, _ = label_maps()
    print(f"banking77: {len(canon)} intents")
    for s in ("fewshot_pool", "calibration", "test"):
        print(f"  {s:13s} {len(load_split(s)):5d}")

    examples = sample(load_split(args.split), args.n, seed=EVAL_SEED)
    gold = [ex.label for ex in examples]

    print(f"\nClassifier: LLM + {K_SHOTS} retrieved shots, {NORMALIZATION}-normalized "
          f"logprob scoring over {len(canon)} intents")
    print(f"Scoring {len(examples)} '{args.split}' examples ...")

    retriever = ShotRetriever()
    scorer = LabelScorer()
    cache = JsonlCache(cache_name(), scoring_params())
    sums, ntok = score_examples(scorer, retriever, examples, nl, cache)

    scores = normalize(sums, ntok)
    llm_correct = np.array([canon[i] == g for i, g in zip(scores.argmax(axis=1), gold)])
    k, n = int(llm_correct.sum()), len(llm_correct)
    lo, hi = wilson_ci(k, n)

    print("\n" + "=" * 62)
    print(f"RAW TOP-1 ACCURACY ({args.split}, n={n}, no abstention)")
    print(f"  {k/n:.1%}   [{k}/{n}]   95% CI: {lo:.1%} - {hi:.1%}")
    print("=" * 62)

    # Honesty check: the retriever alone already surfaces the gold intent most of the time.
    knn_correct = np.array([p == g for p, g in zip(knn_predict(retriever, examples, 16), gold)])
    kb = int(knn_correct.sum())
    b01, b10, p = mcnemar(llm_correct, knn_correct)
    print(f"\nNo-LLM baseline (TF-IDF 16-NN vote, same retrieval index)")
    print(f"  {kb/n:.1%}   [{kb}/{n}]")
    print(f"  LLM advantage: {(k-kb)/n:+.1%}")
    print(f"  McNemar (paired): LLM-only-right {b01}, baseline-only-right {b10}, "
          f"p = {p:.2g}")
    print("  -> retrieval supplies the candidates; the LLM's job is discriminating among "
          "confusable\n     intents. Both numbers belong in the write-up.")

    # First look at the signal Phase 3 will threshold on. Not yet a calibration claim.
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    conf = probs.max(axis=1)
    print(f"\nConfidence (max softmax): correct {conf[llm_correct].mean():.3f} vs "
          f"wrong {conf[~llm_correct].mean():.3f}")
    print("That gap is what abstention will exploit in Phase 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
