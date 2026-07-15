"""Model I/O: 4-bit loading, label scoring, and generation.

This is the only module that touches transformers. Everything downstream (signals,
conformal calibration, serving) depends on this interface, not on transformers
internals, so the model could be swapped without touching the calibration math.

Two primitives:
  * `score_labels`  — deterministic, pure scoring. Length-normalized log-likelihood of
                      each candidate label string given the prompt, softmaxed into a
                      distribution over the 77 intents. This is the logprob signal.
  * `generate_label` — sampling, used later by the self-consistency and verbalized signals.

Performance note: scoring 77 labels naively means 77 full forward passes over a ~500-token
prompt. Instead we run the prompt **once**, keep its KV cache, and replay only the short
label continuations against that cache in batches. That is ~10x cheaper and is what makes
this feasible on an 8 GB laptop GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache

from src.prompt import build_prompt

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"  # pinned for reproducibility

# How many candidate labels to score in one batched forward pass. The prompt KV cache is
# replicated across the batch, so this trades VRAM for speed; 16 is comfortable in 8 GB.
LABEL_BATCH = 16


@dataclass
class ScoredLabels:
    """Raw scoring output for one message, index-aligned with the label list passed in.

    We deliberately store the *unnormalized* pieces — the summed token log-prob and the
    token count — rather than a single pre-normalized number. Every normalization we might
    want (mean, sum, PMI/prior-corrected) is a cheap function of these two, so exploring
    scoring variants never costs another GPU pass.
    """

    sum_logprobs: np.ndarray  # (n_labels,) total log P(label tokens | prompt)
    n_tokens: np.ndarray  # (n_labels,) tokens per label

    @property
    def mean_logprobs(self) -> np.ndarray:
        """Length-normalized score: comparable across labels of different token lengths."""
        return self.sum_logprobs / self.n_tokens

    def probs(self) -> np.ndarray:
        """Softmax over the length-normalized scores -> distribution over the intents."""
        return _softmax(self.mean_logprobs)

    @property
    def top_index(self) -> int:
        return int(self.mean_logprobs.argmax())


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class LabelScorer:
    """Loads the model in 4-bit NF4 and exposes the scoring/generation primitives."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: str | None = MODEL_REVISION,
        quantize: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """`quantize=False` loads unquantized at `dtype` — used only to verify the scoring
        maths without 4-bit kernel noise in the way; the project itself runs quantized."""
        self.model_id = model_id
        self.revision = revision
        kwargs: dict = {}
        if quantize:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["torch_dtype"] = dtype
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, device_map="auto", **kwargs
        )
        self.model.eval()
        self.pad_id = self.tok.pad_token_id or self.tok.eos_token_id
        self._label_tokens: dict[str, list[int]] = {}

    # -- internals ----------------------------------------------------------------

    def _tokenize_label(self, label: str) -> list[int]:
        """Token ids of a label as it would appear as the assistant's reply (cached)."""
        if label not in self._label_tokens:
            ids = self.tok(label, add_special_tokens=False).input_ids
            if not ids:
                raise ValueError(f"label tokenized to nothing: {label!r}")
            self._label_tokens[label] = ids
        return self._label_tokens[label]

    @torch.no_grad()
    def _prompt_cache(self, prompt: str) -> tuple[DynamicCache, torch.Tensor, int]:
        """Run the prompt once. Returns (KV cache, log-probs for the *first* label token,
        prompt length)."""
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.model.device)
        try:
            # Only the final position's logits are needed; skip materialising (P, vocab).
            out = self.model(ids, use_cache=True, num_logits_to_keep=1)
        except TypeError:  # older/newer signature without the kwarg
            out = self.model(ids, use_cache=True)
        first_lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        return out.past_key_values, first_lp, ids.shape[1]

    @staticmethod
    def _replicate(cache: DynamicCache, batch: int) -> DynamicCache:
        """Copy a batch-1 KV cache across `batch` rows.

        `repeat` allocates new tensors, so the original prompt cache is never mutated by
        the forward pass that consumes the copy — it stays reusable for the next chunk.
        """
        out = DynamicCache()
        for layer, (k, v) in enumerate(cache.to_legacy_cache()):
            out.update(k.repeat(batch, 1, 1, 1), v.repeat(batch, 1, 1, 1), layer)
        return out

    # -- primitives ---------------------------------------------------------------

    @torch.no_grad()
    def score_labels(
        self, text: str, nl_labels: list[str], shots: list[tuple[str, str]] | None = None
    ) -> ScoredLabels:
        """Total log-prob of each label string given the message.

        Deterministic: a pure scoring pass, no sampling. Returns raw sums plus token
        counts; normalization (mean / PMI) is the caller's choice.
        """
        prompt = build_prompt(self.tok, text, nl_labels, shots)
        cache, first_lp, n_prompt = self._prompt_cache(prompt)

        token_ids = [self._tokenize_label(l) for l in nl_labels]
        sums = np.empty(len(nl_labels), dtype=np.float64)
        ntok = np.array([len(t) for t in token_ids], dtype=np.float64)

        for start in range(0, len(nl_labels), LABEL_BATCH):
            chunk = token_ids[start : start + LABEL_BATCH]
            b, width = len(chunk), max(len(t) for t in chunk)

            inp = torch.full((b, width), self.pad_id, dtype=torch.long)
            for j, t in enumerate(chunk):
                inp[j, : len(t)] = torch.tensor(t, dtype=torch.long)
            inp = inp.to(self.model.device)

            # Right-padding is safe under causal attention: a padded tail position can
            # only affect positions after it, and we never read those log-probs.
            attn = torch.ones((b, n_prompt + width), dtype=torch.long, device=self.model.device)
            pos = (
                torch.arange(n_prompt, n_prompt + width, device=self.model.device)
                .unsqueeze(0)
                .expand(b, width)
            )
            out = self.model(
                inp,
                past_key_values=self._replicate(cache, b),
                attention_mask=attn,
                position_ids=pos,
            )
            # logits[:, i] predicts label token i+1; token 0 comes from the prompt pass.
            lp = torch.log_softmax(out.logits.float(), dim=-1)

            for j, t in enumerate(chunk):
                total = first_lp[t[0]]
                for i in range(1, len(t)):
                    total = total + lp[j, i - 1, t[i]]
                sums[start + j] = float(total)

        return ScoredLabels(sum_logprobs=sums, n_tokens=ntok)

    def label_priors(self, nl_labels: list[str], shots: list[tuple[str, str]] | None = None) -> ScoredLabels:
        """Score the labels against a content-free message.

        This is each label's *prior* fluency under the prompt — how likely the model is to
        say it regardless of what the customer asked. Subtracting it (PMI / contextual
        calibration) removes the model's bias toward a priori fluent labels, which is a
        known failure mode of raw label scoring.
        """
        return self.score_labels("N/A", nl_labels, shots)

    @torch.no_grad()
    def generate_label(
        self,
        text: str,
        nl_labels: list[str],
        temperature: float = 0.0,
        max_new_tokens: int = 16,
        seed: int | None = None,
        shots: list[tuple[str, str]] | None = None,
    ) -> str:
        """Free-generate a label. temperature = 0 is greedy; > 0 samples (for
        self-consistency). Returns the raw decoded string — parsing/matching to a
        canonical intent is the caller's job."""
        prompt = build_prompt(self.tok, text, nl_labels, shots)
        return self._generate(prompt, n=1, temperature=temperature,
                              max_new_tokens=max_new_tokens, seed=seed)[0]

    @torch.no_grad()
    def sample_labels(
        self,
        text: str,
        nl_labels: list[str],
        k: int,
        temperature: float,
        seed: int,
        max_new_tokens: int = 16,
        shots: list[tuple[str, str]] | None = None,
    ) -> list[str]:
        """Draw k label samples at temperature > 0 for the self-consistency signal.

        All k are produced in one `generate` call (num_return_sequences=k), so the prompt
        is encoded once rather than k times. Seeded, so the samples reproduce.
        """
        prompt = build_prompt(self.tok, text, nl_labels, shots)
        return self._generate(prompt, n=k, temperature=temperature,
                              max_new_tokens=max_new_tokens, seed=seed)

    @torch.no_grad()
    def generate_from_prompt(
        self, prompt: str, max_new_tokens: int = 24, seed: int | None = None
    ) -> str:
        """Greedy generation from an already-built prompt (used by the verbalized signal,
        which needs its own instruction). Returns the raw decoded continuation."""
        return self._generate(prompt, n=1, temperature=0.0,
                              max_new_tokens=max_new_tokens, seed=seed)[0]

    @torch.no_grad()
    def _generate(
        self, prompt: str, n: int, temperature: float, max_new_tokens: int, seed: int | None
    ) -> list[str]:
        """Low-level generation: n sequences from one prompt. Returns decoded strings."""
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        if seed is not None:
            torch.manual_seed(seed)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            num_return_sequences=n,
            pad_token_id=self.pad_id,
        )
        start = enc.input_ids.shape[1]
        return [self.tok.decode(row[start:], skip_special_tokens=True).strip() for row in out]
