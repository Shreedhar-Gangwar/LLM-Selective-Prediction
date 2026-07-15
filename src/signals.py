"""The three confidence signals.

All three score confidence in ONE prediction — the locked logprob-argmax classifier from
Phase 1 (the 89.4% system). Keeping the prediction fixed means the Phase 3 risk-coverage
curves isolate a single variable: which signal ranks the classifier's errors best. Each
signal returns a score in [0, 1] where higher = more confident.

  1. logprob      — max softmax over the 77 labels (primary); also the margin to the
                    runner-up, which often calibrates better.
  2. self-consistency — sample the label k times at temperature > 0; score = fraction of
                    samples agreeing with the prediction. (PLAN says "modal answer"; we
                    score agreement with the fixed prediction instead so all signals rank
                    the *same* classifier, and store the modal label for reference.)
  3. verbalized   — ask the model to state a 0-1 confidence; parse it. Cheapest, usually
                    worst-calibrated — showing that is a legitimate finding.

Pure functions here (parsing, matching, scoring from raw outputs). The GPU calls and
caching live in scripts/phase2_signals.py.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

# Self-consistency sampling parameters — part of the cache key.
SC_K = 5
SC_TEMPERATURE = 0.7
SC_SEED = 100


@dataclass
class ExampleSignals:
    """Everything Phase 3 needs for one example."""

    example_id: str
    gold: str
    predicted: str  # locked classifier's prediction (logprob argmax)
    correct: bool
    logprob_max: float  # max softmax probability
    logprob_margin: float  # top prob - runner-up prob
    self_consistency: float  # fraction of k samples == predicted
    sc_modal: str | None  # most common sampled label (reference only)
    verbalized: float  # model's self-reported confidence


# -- label matching --------------------------------------------------------------

_PUNCT = re.compile(r"[^a-z0-9 ]")


def _normalize(s: str) -> str:
    return _PUNCT.sub("", s.lower()).strip()


def match_label(raw: str, nl_labels: list[str], nl_to_canon: dict[str, str]) -> str | None:
    """Map a decoded model output to a canonical intent, or None if it matches none.

    Tries exact (normalized) equality first, then a unique substring match — the model
    sometimes wraps the label in a short phrase. Ambiguous or absent -> None.
    """
    norm = _normalize(raw)
    if not norm:
        return None
    norm_nl = {_normalize(nl): nl for nl in nl_labels}
    if norm in norm_nl:
        return nl_to_canon[norm_nl[norm]]
    hits = [nl for n, nl in norm_nl.items() if n and n in norm]
    if len(hits) == 1:
        return nl_to_canon[hits[0]]
    return None


# -- signal computation from raw outputs -----------------------------------------

def logprob_signals(mean_logprobs: np.ndarray) -> tuple[float, float]:
    """(max softmax prob, margin to runner-up) from length-normalized label log-probs."""
    e = np.exp(mean_logprobs - mean_logprobs.max())
    probs = e / e.sum()
    top2 = np.sort(probs)[-2:]
    return float(top2[-1]), float(top2[-1] - top2[-2])


def self_consistency(
    samples: list[str], predicted: str, nl_labels: list[str], nl_to_canon: dict[str, str]
) -> tuple[float, str | None]:
    """Fraction of k samples equal to `predicted`, plus the modal sampled label.

    Unparseable samples count against agreement (they stay in the denominator), so a model
    that samples gibberish is correctly scored as low-confidence rather than crashing.
    """
    mapped = [match_label(s, nl_labels, nl_to_canon) for s in samples]
    agree = sum(m == predicted for m in mapped)
    valid = [m for m in mapped if m is not None]
    modal = Counter(valid).most_common(1)[0][0] if valid else None
    return agree / len(samples), modal


_NUM = re.compile(r"([01](?:\.\d+)?|0?\.\d+)")


def parse_verbalized(raw: str) -> float:
    """Extract the self-reported confidence from a 'intent | confidence' reply.

    Robust to malformed output: takes the last number in [0,1] it can find, defaulting to
    0.0 (maximally abstain) when the model emits no usable number — never raises.
    """
    tail = raw.split("|")[-1] if "|" in raw else raw
    candidates = [float(m) for m in _NUM.findall(tail)]
    valid = [c for c in candidates if 0.0 <= c <= 1.0]
    return valid[-1] if valid else 0.0
