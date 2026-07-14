# PLAN.md

Phased build roadmap. Read `CLAUDE.md` first for conventions and guardrails, then work
through these phases in order. Each task has an acceptance check — don't advance a phase
until its checks pass. Tick items as you go.

---

## Phase 0 — Environment and smoke test

The setup kit (`scripts/check_env.py`, `scripts/smoke_test.py`, `requirements.txt`,
`SETUP.md`) exists to make this phase turnkey. Run it, don't rebuild it.

- [ ] Create the venv (WSL2 on Windows), install `requirements.txt`.
- [ ] `python scripts/check_env.py` — confirm CUDA is visible, the GPU is the 4060, and
      >=7 GB VRAM is free. If CUDA is not available, stop and fix drivers/torch build
      before anything else.
- [ ] `python scripts/smoke_test.py` — confirm the 4-bit model loads inside VRAM and that
      label-scoring returns sane per-class logprobs for one hardcoded example.

**Acceptance:** smoke test prints a predicted intent and a confidence for one example,
using the logprob-scoring path, without OOM.

---

## Phase 1 — Data and base classifier (no abstention yet)

- [ ] Load `PolyAI/banking77` via `datasets`. Confirm 77 labels; record train/test sizes.
- [ ] Carve a **calibration split** out of train (seeded). Keep few-shot prompt examples
      (if any) disjoint from calibration and test.
- [ ] `src/model.py`: implement `score_labels(text, labels)` — length-normalized logprob
      of each label string given the prompt, softmax to a distribution over the 77 intents.
      Also `generate_label(text, temperature)` for later sampling.
- [ ] Build a compact classification prompt (task instruction + the 77 label names;
      optionally a few few-shot examples). Keep it short — VRAM and speed depend on it.
- [ ] Measure raw top-1 accuracy on a test subset. No abstention logic yet.

**Acceptance:** raw accuracy on banking77 is in a sane range for a 4B model (expect
roughly 70–85% zero/few-shot). Record the number — it anchors the whole risk-coverage story.

**Guardrail:** cache every model output keyed by (example_id, signal, params). You will
re-run evaluation many times; never re-query the GPU blindly.

---

## Phase 2 — The three confidence signals

Each returns a per-example score in [0,1]. Implement in `src/signals.py`.

- [ ] **Logprob** (primary): confidence = max softmax prob over labels (also store the
      margin to the runner-up; it often calibrates better).
- [ ] **Self-consistency**: sample the label `k` times (start k=5) at temperature ~0.7;
      confidence = fraction agreeing with the modal answer. Cache all k samples.
- [ ] **Verbalized**: prompt the model to emit its answer plus a 0–1 confidence; parse
      robustly (handle malformed outputs — default to low confidence, don't crash).

**Acceptance:** for a shared set of examples, all three signals are computed and cached,
and correctness labels are recorded alongside each.

---

## Phase 3 — Conformal selective calibration (the core)

Implement in `src/conformal.py`.

- [ ] On the **calibration split**, for a target risk level alpha (e.g. 0.10), choose the
      abstention threshold on the confidence score using a finite-sample-valid procedure:
      split-conformal quantile on the score, or an LTT / risk-controlling calibration for
      the error-among-accepted version. Document which and its assumption (exchangeability).
- [ ] Produce the **risk-coverage curve** on test for each of the three signals: as the
      threshold sweeps, plot selective error (or accuracy) vs coverage.
- [ ] Verify on test that the accepted-set error is at/below the target the calibration
      promised (within sampling noise). Report achieved risk AND coverage — never one alone.
- [ ] Choose one operating point and justify it (e.g. the max coverage meeting >=90%
      accepted-accuracy).

**Acceptance:** a threshold calibrated on calibration data holds its risk promise on test;
the three signals are ranked by their risk-coverage curves; the logprob signal is expected
to lead and verbalized to trail — but report what you actually find.

---

## Phase 4 — Serving and packaging

- [ ] `src/serve.py`: FastAPI endpoint. POST a ticket -> {predicted_intent, confidence,
      decision: answer|abstain} using the frozen threshold. Load the model once at startup.
- [ ] `src/evaluate.py`: batch harness that runs the whole pipeline over a split from cache
      and prints raw accuracy, accepted-accuracy, and coverage at the chosen threshold.
- [ ] `tests/test_conformal.py`: calibration math against known-answer inputs (e.g. a
      threshold that must accept a synthetic set at a known rate; monotonicity of the
      risk-coverage curve).
- [ ] Plots to `report/`: the three risk-coverage curves on one axis; a reliability/
      calibration plot for the logprob signal. Keep it to a few clean figures.
- [ ] `report/findings.md`: raw accuracy, the signal comparison, the chosen operating
      point and its guarantee, and the implied **human-review-load reduction** (coverage
      is the automation rate; 1 - coverage is what routes to a human).
- [ ] `README.md`: the pitch, the honest framing (local model, logprob scoring, conformal
      calibration, exchangeability caveat), how to run, and why it complements the tabular
      credit-risk project.

**Acceptance:** a reader opens the repo cold and, within five minutes, understands the
decision the service makes, trusts the guarantee, and sees the GenAI + serving + uncertainty
range.

---

## Optional stretch (only after Phase 4 works)

- [ ] **Class/group-conditional coverage**: control risk within intent groups, not just
      marginally; show where marginal calibration hides per-class failures. Mirrors
      group-conditional conformal methods — a strong, authentic talking point.
- [ ] **Cost dashboard**: a small view of automation rate and projected review-hours saved
      across operating points.

---

## Resume lines this project should earn

Fill placeholders from actual computed results — never before.

- "Built a selective-prediction LLM service (local 4B model, FastAPI) with conformal
  calibration guaranteeing >=[90]% accuracy on non-abstained tickets at ~[85]% coverage."
- "Compared logprob, self-consistency, and verbalized-confidence signals via risk-coverage
  analysis; abstention routing cut projected human-review load by ~[X]%."
