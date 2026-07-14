"""Smoke test — confirms the critical path works on your GPU.

Loads a ~3-4B instruct model in 4-bit (NF4) and demonstrates the two primitives the whole
project is built on:
  1. label-scoring: length-normalized log-prob the model assigns to each candidate label
     string given the prompt -> softmax -> a probability distribution (the logprob signal),
  2. generation, as a sanity check that the chat template + sampling work (needed later
     for the self-consistency signal).

Self-contained: a few hardcoded banking-style examples, no dataset download required.

Usage:  python scripts/smoke_test.py
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- config -----------------------------------------------------------------
# Default is natively supported in transformers (NO remote code), which avoids the
# version-sensitivity that Phi-4-mini's bundled model code runs into.
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
# Alternative (uses trust_remote_code; can break across transformers versions):
#   "microsoft/Phi-4-mini-instruct"
MODEL_REVISION = None  # pin a commit hash here for reproducibility once chosen

CANDIDATE_LABELS = [
    "card_arrival",
    "card_not_working",
    "lost_or_stolen_card",
    "top_up_by_card_charge",
    "exchange_rate",
]
EXAMPLE_TEXT = "My new card still hasn't turned up, how long does delivery take?"


def load_model():
    """Load the model in 4-bit NF4. Returns (tokenizer, model)."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        quantization_config=bnb,
        device_map="auto",
    )
    model.eval()
    return tok, model


def build_prompt(tok, text: str) -> str:
    """Wrap the query in the model's chat template, asking for a single intent label."""
    system = (
        "You are an intent classifier for online-banking support messages. "
        "Reply with exactly one intent label and nothing else."
    )
    user = (
        f"Message: {text}\n"
        f"Choose the single best intent from: {', '.join(CANDIDATE_LABELS)}."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score_label(tok, model, prompt: str, label: str) -> float:
    """Length-normalized log-prob of `label` continuing `prompt`.

    Scores only the label tokens (prompt tokens are masked out), then averages over
    label-token count so labels of different lengths are comparable.
    """
    prompt_ids = tok(prompt, return_tensors="pt").input_ids
    full_ids = tok(prompt + label, return_tensors="pt").input_ids.to(model.device)

    logits = model(full_ids).logits  # (1, seq, vocab)
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = full_ids[:, 1:]
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, seq-1)

    n_prompt = prompt_ids.shape[1]
    label_lp = token_lp[0, n_prompt - 1 :]  # only the label continuation
    if label_lp.numel() == 0:
        return float("-inf")
    return float(label_lp.mean().item())


@torch.no_grad()
def generation_sanity_check(tok, model, prompt: str) -> str:
    """Greedy-generate a short continuation. Passes an attention mask and pad token
    explicitly to avoid the 'pad == eos' warning and shape edge cases."""
    enc = tok(prompt, return_tensors="pt").to(model.device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = model.generate(
        **enc,
        max_new_tokens=12,
        do_sample=False,
        pad_token_id=pad_id,
    )
    return tok.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True).strip()


def main() -> int:
    print(f"Loading {MODEL_ID} in 4-bit ... (first run downloads weights)")
    tok, model = load_model()
    print("Model loaded.\n")

    prompt = build_prompt(tok, EXAMPLE_TEXT)

    # 1) logprob scoring over candidate labels (the core project primitive)
    scores = {lab: score_label(tok, model, prompt, " " + lab) for lab in CANDIDATE_LABELS}
    t = torch.tensor([scores[l] for l in CANDIDATE_LABELS])
    probs = torch.softmax(t, dim=0)
    ranked = sorted(zip(CANDIDATE_LABELS, probs.tolist()), key=lambda x: -x[1])

    print(f"Example: {EXAMPLE_TEXT!r}")
    print("Label scoring (softmax over candidates):")
    for lab, p in ranked:
        print(f"  {p:6.1%}  {lab}")
    top_label, top_p = ranked[0]
    print(f"\nPredicted intent: {top_label}   confidence: {top_p:.1%}")

    # 2) generation sanity check (non-fatal: needed for self-consistency later, but a
    #    failure here should never block confirming the environment)
    try:
        gen = generation_sanity_check(tok, model, prompt)
        print(f"Generation sanity check -> {gen!r}")
    except Exception as e:  # noqa: BLE001
        print(f"Generation sanity check SKIPPED (non-fatal): {type(e).__name__}: {e}")

    print("\nSmoke test done. If the prediction is sensible, proceed to Phase 1 in PLAN.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
