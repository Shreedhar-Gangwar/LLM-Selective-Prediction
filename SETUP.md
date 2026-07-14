# SETUP.md — one-time setup on your laptop (RTX 4060 Mobile, 8 GB VRAM)

Run these on **your machine**, not in any sandbox. Whole thing is ~15 minutes, mostly
the model download. Do the steps in order; don't skip the two check scripts.

---

## 0. Windows only: use WSL2

`bitsandbytes` 4-bit is painful on native Windows and clean on Linux. If you're on
Windows, do everything below **inside WSL2** (Ubuntu):

```powershell
wsl --install -d Ubuntu     # if you don't already have it; then reboot
```

Then open the Ubuntu terminal and continue. Your NVIDIA driver on Windows already exposes
the GPU to WSL2 — you do NOT install a separate Linux GPU driver, just the CUDA-enabled
torch wheel below. If you're already on Linux, ignore this section.

---

## 1. Project folder + virtual environment

```bash
cd llm-selective-prediction        # the folder holding CLAUDE.md, PLAN.md, requirements.txt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Install PyTorch with CUDA (do this BEFORE requirements.txt)

Install the CUDA build of torch from PyTorch's own index so you don't get the CPU-only
wheel by accident:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

(If your driver is older and cu124 fails, try the cu121 index instead. Newer driver is
fine — cu124 wheels run on newer drivers.)

## 3. Install the rest

```bash
pip install -r requirements.txt
```

## 4. Verify the environment (do not skip)

```bash
python scripts/check_env.py
```

You want **RESULT: PASS**, with CUDA available, the GPU shown as your 4060, and >=6 GB
VRAM free. If it says CUDA is not available, the CPU-only torch got installed — redo
step 2. On Windows-not-in-WSL2, bitsandbytes is the usual failure; go back to step 0.

## 5. Smoke-test the real path

```bash
python scripts/smoke_test.py
```

First run downloads the ~4B model (a few GB) into your HuggingFace cache; later runs are
instant. Success looks like a short softmax table over candidate intents and a predicted
label with a confidence, printed without any out-of-memory error. That proves the two
primitives the project depends on — 4-bit loading and logprob label-scoring — work on
your GPU.

---

## Model notes

- The model is `Qwen/Qwen2.5-3B-Instruct`, pinned to revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1` in `src/model.py`. It is natively implemented
  in transformers (no `trust_remote_code`), so it cannot desync from the library —
  `microsoft/Phi-4-mini-instruct` was rejected for exactly that reason (see CLAUDE.md).
- Change `MODEL_ID` in one place (`src/model.py`) to swap it. Avoid `-VL` vision variants.

## Dataset note

`datasets` >= 4.0 no longer executes dataset scripts, and the `PolyAI/banking77` repo ships
only a script — so `load_dataset("PolyAI/banking77")` fails. We read the canonical CSVs
(the same ones that script downloads) directly instead:

```bash
mkdir -p data
curl -sSfL -o data/train.csv https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv
curl -sSfL -o data/test.csv  https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv
```

That gives the canonical 10,003 train / 3,080 test split over 77 intents.

## If you hit out-of-memory

- Close other GPU users (browser with hardware accel, games, other notebooks).
- Keep the classification prompt short (fewer/no few-shot examples).
- Drop to an even smaller model temporarily to unblock, then scale back up.
- Confirm the model actually loaded in 4-bit (check_env shows free VRAM; a 3.8B in NF4
  should occupy only ~3-4 GB).

---

## What's next

Environment green? Open Claude Code in this folder, point it at `PLAN.md`, and start at
Phase 1. Phase 0 is already done by the steps above.
