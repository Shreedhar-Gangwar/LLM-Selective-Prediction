"""Phase 2 — compute the three confidence signals over a split, cached.

For each example, using the SAME 16 retrieved shots across all signals:
  * logprob        — reuse the Phase 1 label-scoring cache; derive prediction, max-softmax,
                     and margin. This also fixes the prediction the other signals score.
  * self-consistency — sample the label SC_K times at SC_TEMPERATURE; agreement fraction.
  * verbalized     — one greedy generation asking for 'intent | confidence'; parse it.

Each signal has its own cache keyed by its params, so a re-run recomputes nothing and a
parameter change invalidates only the affected signal. The consolidated per-example
records (scores + correctness) are written to cache/phase2_<split>.jsonl for Phase 3.

Usage:  python -m scripts.phase2_signals --split test --n 1000
        python -m scripts.phase2_signals --split test --n 5 --dry-run   # quick sanity
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.cache import CACHE_DIR, JsonlCache
from src.config import K_SHOTS, cache_name, normalize, scoring_params
from src.data import Example, label_maps, load_split, sample, to_natural_language
from src.model import MODEL_ID, MODEL_REVISION, LabelScorer
from src.prompt import PROMPT_VERSION, build_verbalized_prompt
from src.retrieval import ShotRetriever
from src.signals import (
    SC_K,
    SC_SEED,
    SC_TEMPERATURE,
    ExampleSignals,
    logprob_signals,
    match_label,
    parse_verbalized,
    self_consistency,
)

EVAL_SEED = 7


def sc_params() -> dict:
    p = scoring_params()
    p.update({"signal": "self_consistency", "k": SC_K, "temp": SC_TEMPERATURE, "seed": SC_SEED})
    return p


def verb_params() -> dict:
    return {
        "model": MODEL_ID, "revision": MODEL_REVISION, "prompt": PROMPT_VERSION,
        "signal": "verbalized", "label_form": "natural_language",
        "variant": "retrieved", "k": K_SHOTS,
    }


def compute(
    scorer: LabelScorer, retriever: ShotRetriever, examples: list[Example],
    canon: list[str], nl: list[str], nl_to_canon: dict[str, str],
) -> list[ExampleSignals]:
    lp_cache = JsonlCache(cache_name(), scoring_params())
    sc_cache = JsonlCache("selfconsistency", sc_params())
    vb_cache = JsonlCache("verbalized", verb_params())

    records: list[ExampleSignals] = []
    t0 = time.time()
    n_gpu = 0

    for i, ex in enumerate(examples):
        shots = [(n.text, to_natural_language(n.label)) for n in retriever.retrieve(ex.text, K_SHOTS)]
        did_gpu = False

        # 1) logprob — reuse Phase 1 cache where present
        hit = lp_cache.get(ex.example_id)
        if hit is None:
            res = scorer.score_labels(ex.text, nl, shots)
            hit = {"sum": res.sum_logprobs.tolist(), "ntok": res.n_tokens.tolist()}
            lp_cache.set(ex.example_id, hit)
            did_gpu = True
        mean_lp = normalize(np.asarray(hit["sum"]), np.asarray(hit["ntok"]))
        predicted = canon[int(mean_lp.argmax())]
        lp_max, lp_margin = logprob_signals(mean_lp)

        # 2) self-consistency
        sc_hit = sc_cache.get(ex.example_id)
        if sc_hit is None:
            samples = scorer.sample_labels(ex.text, nl, SC_K, SC_TEMPERATURE, SC_SEED, shots=shots)
            sc_cache.set(ex.example_id, {"samples": samples})
            sc_hit = {"samples": samples}
            did_gpu = True
        sc_score, sc_modal = self_consistency(sc_hit["samples"], predicted, nl, nl_to_canon)

        # 3) verbalized
        vb_hit = vb_cache.get(ex.example_id)
        if vb_hit is None:
            prompt = build_verbalized_prompt(scorer.tok, ex.text, nl, shots)
            raw = scorer.generate_from_prompt(prompt, max_new_tokens=24)
            vb_cache.set(ex.example_id, {"raw": raw})
            vb_hit = {"raw": raw}
            did_gpu = True
        verbalized = parse_verbalized(vb_hit["raw"])

        records.append(ExampleSignals(
            example_id=ex.example_id, gold=ex.label, predicted=predicted,
            correct=(predicted == ex.label), logprob_max=lp_max, logprob_margin=lp_margin,
            self_consistency=sc_score, sc_modal=sc_modal, verbalized=verbalized,
        ))
        n_gpu += did_gpu

        done = i + 1
        if done % 25 == 0 or done == len(examples):
            eta = ((time.time() - t0) / max(n_gpu, 1)) * (len(examples) - done)
            print(f"  {done}/{len(examples)}  (GPU {n_gpu})"
                  + (f"  ~{eta/60:.1f} min left" if n_gpu and done < len(examples) else ""),
                  flush=True)
    return records


def summarize(records: list[ExampleSignals]) -> None:
    correct = np.array([r.correct for r in records])
    acc = correct.mean()
    print(f"\n  accuracy on this split: {acc:.1%}  ({correct.sum()}/{len(records)})")
    print("  signal separation (AUROC, correct vs wrong; 0.5 = useless, 1.0 = perfect):")
    if correct.all() or not correct.any():
        print("    [all-correct or all-wrong subset — AUROC undefined]")
        return
    for name, vals in (
        ("logprob max-softmax", [r.logprob_max for r in records]),
        ("logprob margin", [r.logprob_margin for r in records]),
        ("self-consistency", [r.self_consistency for r in records]),
        ("verbalized", [r.verbalized for r in records]),
    ):
        auroc = roc_auc_score(correct, vals)
        print(f"    {name:22s} {auroc:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "calibration"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true", help="small run, don't write results file")
    args = ap.parse_args()

    canon, nl, nl_to_canon = label_maps()
    examples = sample(load_split(args.split), args.n, seed=EVAL_SEED)
    print(f"Phase 2 signals: {len(examples)} '{args.split}' examples, "
          f"SC k={SC_K}@T={SC_TEMPERATURE}\n")

    retriever = ShotRetriever()
    scorer = LabelScorer()
    records = compute(scorer, retriever, examples, canon, nl, nl_to_canon)
    summarize(records)

    if not args.dry_run:
        out = CACHE_DIR / f"phase2_{args.split}_n{args.n}.jsonl"
        with Path(out).open("w") as f:
            for r in records:
                f.write(json.dumps(r.__dict__) + "\n")
        print(f"\n  wrote {len(records)} records -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
