# CLAUDE.md

Standing context for this repository. Read this before every task. The step-by-step
roadmap lives in `PLAN.md` — consult it for *what* to build next; this file governs
*how* to build it.

---

## What this project is

An **LLM selective-prediction service**: it classifies customer-support messages
(banking77, 77 intents) with a local LLM, attaches a calibrated confidence score, and
**abstains** — routing to a human — whenever confidence is too low. Calibration is done
so we can state a real guarantee, e.g. "on the tickets it chooses to answer, accuracy is
>=90% at ~85% coverage."

It is a **portfolio project for a Data Science / ML-Engineer resume**. It pairs with a
separate credit-risk project (tabular XGBoost, calibration, SHAP). That one covers
classic tabular ML; **this one must cover GenAI + serving + uncertainty**. Do not turn
it into another tabular-ML notebook.

The intellectual core — abstaining under calibrated uncertainty with a finite-sample
guarantee — is the differentiator. Most LLM portfolio projects are "I built a RAG
chatbot." This one adds statistical rigor and a real cost lever (controlling error and
human-review load). Keep that framing central.

---

## Hardware and model constraints (do not violate)

Target machine: laptop, **RTX 4060 Mobile, 8 GB VRAM, 16 GB DDR4**. This dictates:

- **Model: a ~3-4B instruct model**, default `Qwen/Qwen2.5-3B-Instruct`. This is chosen
  specifically because it is **natively implemented in transformers (no `trust_remote_code`)**
  — the model code ships with the library, so it can't desync from it. `microsoft/Phi-4-mini-instruct`
  was tried first and its bundled remote code broke twice across transformers versions
  (an import error, then a generation-loop shape error), so avoid remote-code models here.
  A slightly weaker model is *fine and even desirable*: it produces a richer accept/abstain
  tradeoff than a near-perfect classifier.
- **Runtime: HuggingFace `transformers` with 4-bit quantization (bitsandbytes, NF4).**
  Not Ollama. We need clean per-token logit access to score candidate labels; Ollama is
  generation-oriented and fights that. One runtime, one loading path.
- **Full GPU offload is mandatory.** The whole model must sit in VRAM. If layers spill
  to system RAM over PCIe, decode speed collapses ~30x. Stay at <=4B; never reach for a
  13B "because it might be better."
- banking77 inputs are short (a message + a 77-label prompt, ~500 tokens), so KV-cache
  pressure is low — this workload is a good fit for 8 GB. Keep prompts compact.

bitsandbytes 4-bit is smoothest on Linux. On Windows, run inside **WSL2**.

---

## Directory layout

```
llm-selective-prediction/
├── data/                       # banking77 pulled via `datasets`; cache lives here
├── src/
│   ├── model.py                # load 4-bit model; label-scoring + generation primitives
│   ├── signals.py              # 3 confidence signals: logprob, self-consistency, verbalized
│   ├── conformal.py            # calibrate abstention threshold; risk-coverage curves
│   ├── evaluate.py             # batch eval harness over calibration/test splits
│   └── serve.py                # FastAPI endpoint: classify-or-abstain
├── tests/
│   └── test_conformal.py       # tests on the calibration math
├── report/
│   └── findings.md             # results: risk-coverage curves, operating point, cost lever
├── scripts/
│   ├── check_env.py            # GPU/CUDA/VRAM/deps verification (run first)
│   └── smoke_test.py           # load model in 4-bit, score labels for one example
├── cache/                      # cached model outputs (per-example, keyed) — never re-run blind
├── README.md
├── requirements.txt
├── CLAUDE.md
└── PLAN.md
```

---

## The dataset

`PolyAI/banking77` via the `datasets` library. ~13k short online-banking queries, each
labeled with one of **77 fine-grained intents**. Canonical split: ~10,003 train / 3,080
test. **Carve a calibration split out of train** — calibration data must be disjoint
from both the prompt's few-shot examples and the test set.

Note the class count: 77 intents, imbalanced. Watch for classes that are rare in
calibration — marginal guarantees can hide per-class failures (see stretch goal).

---

## The three confidence signals

All three feed one comparison on a single risk-coverage curve:

1. **Logprob (primary).** For each of the 77 labels, score the model's length-normalized
   log-likelihood of that label string given the prompt; softmax over labels -> a
   probability. Confidence = max class probability (or margin to runner-up). This is a
   *scoring* forward pass, not free generation — the reason we use transformers.
2. **Self-consistency.** Sample the label `k` times at temperature > 0; confidence = the
   fraction agreeing with the modal answer. Multiplies inference by `k`; cache results.
3. **Verbalized confidence.** Ask the model to emit a 0-1 confidence with its answer;
   parse it. Cheapest, usually worst-calibrated — showing that is a legitimate finding.

---

## Non-negotiable methodological guardrails

1. **Calibration/test hygiene.** The threshold is chosen on the calibration split only.
   Never tune it on test. Never let few-shot prompt examples leak into calibration or test.

2. **The guarantee is about selective risk, stated honestly.** "Accuracy >= 1-alpha on
   accepted predictions" is a *risk-control* claim. Calibrate the abstention threshold on
   the calibration set with a finite-sample-valid procedure (split-conformal quantile on
   the confidence score, or an LTT / risk-controlling calibration for the error-rate
   version), and report the achieved risk **and** coverage. Do not claim a guarantee the
   procedure doesn't deliver; state the assumption (exchangeability) plainly.

3. **Report the whole risk-coverage curve, not a single cherry-picked point.** Then choose
   one operating point and justify it.

4. **Determinism where it matters.** Fixed seeds for sampling and splits; pin the model
   revision. The logprob-scoring path is deterministic (temperature 0 / pure scoring) —
   keep it that way so results reproduce.

5. **Cache model outputs.** Every model call is expensive on a laptop GPU. Key outputs per
   (example_id, signal, params) and reuse. Re-running the eval must not re-query blindly.

6. **Reporting integrity.** Never invent, round-up, or inflate a metric. Any number not yet
   computed is a clearly marked placeholder (e.g. `[X%]`) until the code produces it. Every
   resume-bound number must be reproducible from this repo.

---

## Coding standards

- Logic lives in `src/` as pure, testable functions. Scripts and the API import from it;
  no analysis logic buried in a notebook.
- Type hints and short docstrings on public functions, stating inputs, outputs, and any
  statistical assumption.
- Statistics functions take arrays and return numbers / small dataclasses — no printing or
  plotting inside them. Presentation is separate.
- Model I/O isolated in `model.py`; the rest depends on its interface, not on transformers
  internals, so the model could be swapped without touching calibration.
- Readable over clever. An interviewer will read this.

---

## How to run

```bash
python -m venv .venv && source .venv/bin/activate     # WSL2 on Windows
pip install -r requirements.txt

python scripts/check_env.py        # verify GPU/CUDA/VRAM/deps FIRST
python scripts/smoke_test.py       # confirm 4-bit load + label scoring works

python -m src.evaluate             # run signals over calibration + test, cache outputs
python -m src.conformal            # calibrate threshold, emit risk-coverage curves
pytest -q
uvicorn src.serve:app --reload     # the classify-or-abstain endpoint
```

---

## Definition of done (per component)

- **check_env.py / smoke_test.py** — confirm the model loads in 4-bit within VRAM and that
  label-scoring returns sane per-class logprobs for one example.
- **model.py** — `score_labels(text, labels)` returns a probability over the 77 intents via
  length-normalized logprob scoring; `generate_label(text, temperature)` for sampling.
- **signals.py** — all three confidence signals, each returning a per-example score in [0,1].
- **conformal.py** — given calibration scores+correctness, returns the abstention threshold
  for a target risk, plus the risk-coverage curve; honest about the guarantee type.
- **evaluate.py** — runs signals over splits, caches, reports raw accuracy, and accuracy vs
  coverage at the chosen threshold.
- **serve.py** — POST a ticket -> {predicted_intent, confidence, decision: answer|abstain}.
- **tests** — calibration math checked against inputs with known answers.
- **findings.md** — raw accuracy, the three signals compared, the chosen operating point and
  its guarantee, and the human-review-load reduction it implies.

---

## Tone for written artifacts

Plain, senior, honest. "We answer 85% of tickets at 91% accuracy and route the rest to a
human; the logprob signal calibrates best, verbalized confidence worst" beats hedged or
inflated phrasing. State the exchangeability assumption and the abstention tradeoff before
anyone asks. That posture is the differentiator.

---

## Optional stretch (authentic, high-value)

Add **class- or group-conditional coverage**: instead of only a marginal guarantee, control
risk within intent groups, exposing where marginal calibration hides per-class failures.
This directly mirrors group-conditional conformal methods and is a strong, genuine talking
point — but only after the marginal pipeline works end-to-end.
