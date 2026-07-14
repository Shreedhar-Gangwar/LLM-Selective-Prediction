"""Phase 1 diagnostic — why is raw accuracy only 44%?

Raw zero-shot accuracy with mean-normalized label scoring came in at 44.4% (n=500),
far below the 70-85% the plan expects. This script tests the three suspects head-to-head
on one shared subset of test examples:

  prompt variants   zero-shot | 5-shot (random) | 77-shot (one example per intent)
  normalizations    mean (per-token) | sum | PMI (prior-corrected), from ONE scoring pass
  sanity baseline   greedy generation, to confirm scoring isn't itself the problem

Each prompt variant needs one scoring pass per example; all normalizations are derived
from the cached raw sums, so they are free.

Usage:  python -m scripts.diagnose_scoring [--n 150]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from src.cache import JsonlCache
from src.data import (
    Example,
    label_maps,
    load_split,
    pick_shots,
    pick_shots_one_per_class,
    sample,
    to_natural_language,
)
from src.model import MODEL_ID, MODEL_REVISION, LabelScorer
from src.prompt import PROMPT_VERSION, build_prompt
from scripts.phase1_accuracy import EVAL_SEED, wilson_ci


def score_split(
    scorer: LabelScorer,
    examples: list[Example],
    nl: list[str],
    shots: list[tuple[str, str]] | None,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sums, ntok, prior_sums) for a prompt variant, caching per example."""
    params = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "prompt": PROMPT_VERSION,
        "label_form": "natural_language",
        "variant": variant,
        "n_shots": len(shots or []),
    }
    cache = JsonlCache(f"score_{variant}", params)

    sums = np.empty((len(examples), len(nl)))
    ntok = None
    t0 = time.time()
    for i, ex in enumerate(examples):
        hit = cache.get(ex.example_id)
        if hit is None:
            res = scorer.score_labels(ex.text, nl, shots)
            hit = {"sum": res.sum_logprobs.tolist(), "ntok": res.n_tokens.tolist()}
            cache.set(ex.example_id, hit)
        sums[i] = hit["sum"]
        ntok = np.asarray(hit["ntok"])
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(examples)}  ({time.time()-t0:.0f}s)", flush=True)

    prior = scorer.label_priors(nl, shots).sum_logprobs
    return sums, ntok, prior


def report(name: str, scores: np.ndarray, canon: list[str], gold: list[str]) -> dict:
    """Accuracy of argmax over a (n, 77) score matrix."""
    preds = [canon[i] for i in scores.argmax(axis=1)]
    correct = np.array([p == g for p, g in zip(preds, gold)])
    k, n = int(correct.sum()), len(correct)
    lo, hi = wilson_ci(k, n)
    print(f"  {name:34s} {k/n:6.1%}   [{k}/{n}]   95% CI {lo:.0%}-{hi:.0%}")
    return {"name": name, "acc": k / n, "correct": correct}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    canon, nl, _ = label_maps()
    # A prefix of the same seeded sample used for the 500-example headline run, so results
    # are directly comparable with it.
    examples = sample(load_split("test"), 500, seed=EVAL_SEED)[: args.n]
    gold = [ex.label for ex in examples]
    print(f"Diagnostic on {len(examples)} test examples\n")

    scorer = LabelScorer()

    shot_sets: dict[str, list[tuple[str, str]] | None] = {
        "zeroshot": None,
        "fewshot5": [(e.text, to_natural_language(e.label)) for e in pick_shots(5)],
        "fewshot77": [
            (e.text, to_natural_language(e.label)) for e in pick_shots_one_per_class()
        ],
    }

    results = []
    for variant, shots in shot_sets.items():
        ntoks = len(scorer.tok(build_prompt(scorer.tok, "hi", nl, shots)).input_ids)
        print(f"[{variant}]  prompt = {ntoks} tokens, {len(shots or [])} shots")
        sums, ntok, prior = score_split(scorer, examples, nl, shots, variant)

        print(f"  --- accuracy by normalization ({variant}) ---")
        results.append(report(f"{variant} / mean (current)", sums / ntok, canon, gold))
        results.append(report(f"{variant} / sum", sums, canon, gold))
        results.append(report(f"{variant} / PMI mean", (sums - prior) / ntok, canon, gold))
        results.append(report(f"{variant} / PMI sum", sums - prior, canon, gold))
        print()

    # Sanity baseline: does free generation beat scoring? If so, the scoring path is suspect.
    print("[greedy generation, zero-shot]  (sanity baseline)")
    nl_to_canon = {n: c for n, c in zip(nl, canon)}
    t0 = time.time()
    gen_correct = []
    for i, ex in enumerate(examples):
        out = scorer.generate_label(ex.text, nl, temperature=0.0).strip().lower()
        gen_correct.append(nl_to_canon.get(out) == ex.label)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(examples)}  ({time.time()-t0:.0f}s)", flush=True)
    k, n = int(np.sum(gen_correct)), len(gen_correct)
    lo, hi = wilson_ci(k, n)
    print(f"  {'greedy generation (exact match)':34s} {k/n:6.1%}   [{k}/{n}]   "
          f"95% CI {lo:.0%}-{hi:.0%}")

    print("\n" + "=" * 72)
    print("RANKING")
    for r in sorted(results, key=lambda r: -r["acc"]):
        print(f"  {r['acc']:6.1%}  {r['name']}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
