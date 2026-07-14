"""Phase 1 diagnostic, part 2 — do retrieved few-shot examples beat a fixed 77-shot prompt?

Baseline to beat: 77-shot (one example per intent) / sum = 64.7% on these same 150 test
examples, at a 2,536-token prompt (~3.0 s/example).

Retrieved shots are per-message, so the prompt is much shorter. If accuracy holds or
improves, we win on accuracy AND on the GPU budget for every later phase.

Usage:  python -m scripts.diagnose_retrieval [--n 150]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from src.cache import JsonlCache
from src.data import label_maps, load_split, sample, to_natural_language
from src.model import MODEL_ID, MODEL_REVISION, LabelScorer
from src.prompt import PROMPT_VERSION, build_prompt
from src.retrieval import ShotRetriever
from scripts.phase1_accuracy import EVAL_SEED, wilson_ci

K_VALUES = [8, 16, 32]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    canon, nl, _ = label_maps()
    examples = sample(load_split("test"), 500, seed=EVAL_SEED)[: args.n]
    gold = [ex.label for ex in examples]

    print(f"Retrieval diagnostic on {len(examples)} test examples")
    print("baseline: fewshot77 / sum = 64.7%  (2536-token prompt, ~3.0 s/example)\n")

    retriever = ShotRetriever()
    print(f"retrieval index: {len(retriever.pool)} few-shot-pool examples\n")

    scorer = LabelScorer()
    results = []

    for k in K_VALUES:
        params = {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "prompt": PROMPT_VERSION,
            "label_form": "natural_language",
            "variant": "retrieved",
            "k": k,
            "retriever": "tfidf_char_wb_3_5",
        }
        cache = JsonlCache(f"score_retrieved{k}", params)

        sums = np.empty((len(examples), len(nl)))
        ntok = None
        # How many of the retrieved shots actually carry the gold intent? A useful
        # diagnostic: it upper-bounds how much the retrieved context can help.
        gold_in_shots = 0
        prompt_tokens = []
        t0 = time.time()

        for i, ex in enumerate(examples):
            neighbours = retriever.retrieve(ex.text, k)
            gold_in_shots += any(n.label == ex.label for n in neighbours)
            shots = [(n.text, to_natural_language(n.label)) for n in neighbours]

            hit = cache.get(ex.example_id)
            if hit is None:
                res = scorer.score_labels(ex.text, nl, shots)
                hit = {"sum": res.sum_logprobs.tolist(), "ntok": res.n_tokens.tolist()}
                cache.set(ex.example_id, hit)
            sums[i] = hit["sum"]
            ntok = np.asarray(hit["ntok"])
            if i == 0:
                prompt_tokens.append(
                    len(scorer.tok(build_prompt(scorer.tok, ex.text, nl, shots)).input_ids)
                )

        elapsed = time.time() - t0
        print(f"[retrieved k={k}]  prompt ~{prompt_tokens[0]} tokens, "
              f"{elapsed/len(examples):.2f} s/example")
        print(f"  gold intent present among retrieved shots: "
              f"{gold_in_shots/len(examples):.1%}")

        for name, scores in (("mean", sums / ntok), ("sum", sums)):
            preds = [canon[j] for j in scores.argmax(axis=1)]
            correct = np.array([p == g for p, g in zip(preds, gold)])
            kk, n = int(correct.sum()), len(correct)
            lo, hi = wilson_ci(kk, n)
            print(f"  retrieved{k} / {name:5s}  {kk/n:6.1%}  [{kk}/{n}]  "
                  f"95% CI {lo:.0%}-{hi:.0%}")
            results.append((kk / n, f"retrieved k={k} / {name}"))
        print()

    print("=" * 64)
    print("RANKING (vs fewshot77/sum = 64.7%)")
    for acc, name in sorted(results, reverse=True):
        flag = "  <-- beats 77-shot" if acc > 0.647 else ""
        print(f"  {acc:6.1%}  {name}{flag}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
