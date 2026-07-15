# LLM Selective Prediction — calibrated abstention for intent classification

A customer-support intent classifier that **knows when to stay silent**. A local 4B LLM
classifies banking77 messages (77 intents), attaches a calibrated confidence, and
**abstains** — routing to a human — whenever confidence is too low. The abstention
threshold carries a finite-sample guarantee:

> **On the tickets it answers, accuracy is ≥95% — and it answers 80% of them.**
> (Test: 98.1% accepted accuracy at 80.2% coverage; base accuracy answering everything is
> 90.2%.)

Most "LLM portfolio project" repos stop at *"I built a RAG chatbot."* The differentiator
here is the **statistical layer on top**: abstaining under a calibrated uncertainty
guarantee, which turns a mediocre-looking classifier into a controllable cost lever — you
choose the error rate you can tolerate, and the system tells you how many tickets it can
safely automate.

---

## What it does

```
ticket ──▶ retrieve 16 nearest examples ──▶ 4-bit LLM scores 77 intents
                                                     │
                                          confidence = margin to runner-up
                                                     │
                                     margin ≥ τ ?  ──┴──▶  answer  (predicted intent)
                                                   └────▶  abstain (route to a human)
```

τ is calibrated once on held-out data with a risk-controlling procedure, then frozen.

---

## Results

| | |
|---|---|
| Raw top-1 accuracy (answer everything) | **90.2%** |
| No-LLM TF-IDF k-NN baseline | 80.6% |
| Accepted accuracy at the operating point | **98.1%** |
| Coverage (tickets automated) | **80.2%** |
| Human-review load | 19.8% |

![Risk-coverage curves](report/risk_coverage.png)

The full write-up — how the classifier was built, the three confidence signals compared,
the calibration method, and every caveat — is in **[report/findings.md](report/findings.md)**.

---

## The interesting parts

- **Retrieval-augmented, honestly benchmarked.** 16 TF-IDF-retrieved few-shot examples take
  the classifier from 44% (naive zero-shot) to 89%. But a no-LLM k-NN baseline over the
  same retrieval index already scores 81%, so the LLM's *real* contribution is +8.8 points
  (McNemar p = 5.7e-07) — stated up front rather than buried.
- **Three confidence signals, one honest ranking.** Log-prob margin (AUROC 0.92) ≫
  self-consistency (0.65) ≫ verbalized confidence (0.52, a coin flip). The model's
  *self-reported* confidence is worthless; its *log-probs* are not.
- **A real guarantee, stated with its assumptions.** The threshold is chosen by Learn-then-
  Test (RCPS family): with probability ≥90%, accepted accuracy ≥ target, assuming
  exchangeability. Distribution-free and finite-sample — no reliance on the score being a
  calibrated probability (it isn't; it's under-confident, which is why we calibrate).
- **Group-conditional coverage.** The marginal guarantee is shown to *hide* a subgroup
  failure (the `transfers` intents sit below target while the pooled number clears it);
  per-group calibration restores it, at a measured coverage cost. See §5 of the findings.
- **Runs on a laptop.** RTX 4060 Mobile, 8 GB VRAM. 4-bit NF4, full GPU offload, KV-cache
  reuse for label scoring.

---

## How to run

```bash
python -m venv .venv && source .venv/bin/activate          # WSL2 on Windows
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python scripts/check_env.py          # verify GPU/CUDA/VRAM/deps first
python scripts/smoke_test.py         # confirm 4-bit load + label scoring

# fetch the canonical banking77 CSVs (see SETUP.md)
python -m scripts.phase1_accuracy    # raw accuracy + the no-LLM baseline
python -m scripts.phase2_signals --split calibration --n 1000   # GPU: compute signals
python -m scripts.phase2_signals --split test        --n 1000
python -m scripts.phase3_calibrate   # calibrate the threshold, emit report artifacts
python -m scripts.phase5_group_conditional   # group-conditional coverage (stretch)
python -m scripts.make_plots         # the figures in report/

python -m src.evaluate               # raw / accepted accuracy / coverage from cache
pytest -q                            # the calibration math
uvicorn src.serve:app                # the classify-or-abstain endpoint
```

Model calls are cached per `(example_id, signal, params)`; re-running never re-queries the
GPU blindly. Seeds and the model revision are pinned.

### The service

```bash
curl -s localhost:8000/classify -H 'content-type: application/json' \
  -d '{"message": "my new card still has not arrived, how long does delivery take?"}'
```
```json
{
  "predicted_intent": "card_delivery_estimate",
  "confidence": 0.8063,
  "margin": 0.7635,
  "threshold": 0.3473,
  "decision": "answer",
  "guarantee": "On answered tickets, accuracy >= 95% with probability >= 90% (exchangeability assumed)."
}
```

A vague message falls below the threshold and is routed to a human instead of guessed:

```bash
curl -s localhost:8000/classify -H 'content-type: application/json' \
  -d '{"message": "hi, there is a problem and I need help please"}'
# -> "decision": "abstain"   (margin 0.05, well under the 0.3473 threshold)
```

---

## Layout

```
src/
  data.py        banking77 loading, seeded disjoint splits, label mapping
  retrieval.py   TF-IDF few-shot retriever (indexed on the few-shot pool only)
  model.py       4-bit load; length-normalized label scoring + sampling primitives
  signals.py     the three confidence signals
  conformal.py   Learn-then-Test calibration + risk-coverage curves  ← the core
  groups.py      intent grouping for the group-conditional guarantee
  evaluate.py    batch harness (from cache, no GPU)
  serve.py       FastAPI classify-or-abstain endpoint
scripts/         env check, smoke test, the phase-by-phase pipeline, plotting
tests/           the calibration math, incl. a Monte-Carlo guarantee check
report/          findings.md, figures, the frozen operating point
```

---

## Methodology guardrails (why the numbers are trustworthy)

- **Calibration/test hygiene.** The threshold is chosen on the calibration split only,
  never on test. Few-shot examples are drawn from a *third* disjoint pool, so they can't
  leak into either.
- **The guarantee is a risk-control claim, stated with its assumption** (exchangeability),
  not an unqualified promise.
- **The whole risk-coverage curve is reported**, then one operating point is chosen and
  justified — no cherry-picking a single flattering number.
- **Nothing is inflated.** Negative results (verbalized confidence, PMI correction) and the
  no-LLM baseline are reported next to the headline.

---

## Companion project

This pairs with a **credit-risk** project (tabular XGBoost, probability calibration, SHAP)
that covers classic tabular ML. This one deliberately covers the other axis: **GenAI +
serving + uncertainty quantification.** Together they span structured-data modelling and
LLM systems, both with calibration taken seriously.
