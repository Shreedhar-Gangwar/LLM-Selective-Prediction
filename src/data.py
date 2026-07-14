"""banking77 loading, label handling, and the seeded calibration split.

The canonical dataset (10,003 train / 3,080 test, 77 intents) is the pair of CSVs
published by PolyAI. We read them directly rather than through the `PolyAI/banking77`
HuggingFace loader, because that repo ships only a Python loading script and `datasets`
>= 4.0 no longer executes dataset scripts. The CSVs are the same upstream source the
script itself downloads, so the split is identical and canonical.

Split hygiene (see CLAUDE.md): the calibration split is carved out of *train* with a
fixed seed. It is disjoint from the test split and from the few-shot pool, so the
abstention threshold is never chosen on data the model was prompted with or evaluated on.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

TRAIN_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
TEST_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

N_LABELS = 77
CALIB_SIZE = 2000  # carved out of train; the rest is the few-shot pool
SPLIT_SEED = 20240  # fixed: the calibration/few-shot partition must be reproducible

Split = Literal["calibration", "fewshot_pool", "test"]


@dataclass(frozen=True)
class Example:
    """One banking77 message.

    `example_id` is stable across runs (split name + original CSV row index), so cached
    model outputs stay valid when the sampling size changes.
    """

    example_id: str
    text: str
    label: str  # canonical snake_case intent, the ground truth


def canonical_labels() -> list[str]:
    """The 77 intent names, sorted — a fixed, reproducible ordering."""
    rows = _read_csv(TRAIN_CSV)
    labels = sorted({r["category"] for r in rows})
    if len(labels) != N_LABELS:
        raise ValueError(f"expected {N_LABELS} labels, found {len(labels)}")
    return labels


def to_natural_language(label: str) -> str:
    """Canonical snake_case intent -> readable phrase used in the prompt.

    'lost_or_stolen_card'    -> 'lost or stolen card'
    'Refund_not_showing_up'  -> 'refund not showing up'   (upstream capitalises this one)
    'reverted_card_payment?' -> 'reverted card payment'   (upstream has a stray '?')

    The mapping must be injective so a scored phrase maps back to exactly one intent;
    `label_maps()` asserts that.
    """
    return label.lower().rstrip("?").replace("_", " ").strip()


def label_maps() -> tuple[list[str], list[str], dict[str, str]]:
    """Return (canonical labels, natural-language labels, nl -> canonical).

    Both lists share the same index order, so an argmax over scores indexes straight
    back into the canonical label list.
    """
    canon = canonical_labels()
    nl = [to_natural_language(c) for c in canon]
    if len(set(nl)) != len(nl):
        raise ValueError("natural-language label mapping is not injective")
    return canon, nl, dict(zip(nl, canon))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Fetch the canonical CSVs:\n"
            f"  curl -sSfL -o {path} "
            f"{TRAIN_URL if path.name == 'train.csv' else TEST_URL}"
        )
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load(path: Path, prefix: str) -> list[Example]:
    return [
        Example(example_id=f"{prefix}-{i:05d}", text=r["text"], label=r["category"])
        for i, r in enumerate(_read_csv(path))
    ]


def load_split(split: Split) -> list[Example]:
    """Load one split.

    'test'         — the canonical 3,080-row test set, untouched.
    'calibration'  — CALIB_SIZE rows drawn from train under SPLIT_SEED.
    'fewshot_pool' — the remaining train rows; the only place few-shot examples may come
                     from, keeping them disjoint from calibration and test.
    """
    if split == "test":
        return _load(TEST_CSV, "test")

    train = _load(TRAIN_CSV, "train")
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(train))
    calib_idx = set(perm[:CALIB_SIZE].tolist())

    if split == "calibration":
        return [train[i] for i in sorted(calib_idx)]
    if split == "fewshot_pool":
        return [train[i] for i in range(len(train)) if i not in calib_idx]
    raise ValueError(f"unknown split: {split}")


def sample(examples: list[Example], n: int, seed: int) -> list[Example]:
    """A seeded subsample. Returns all examples if n >= len(examples)."""
    if n >= len(examples):
        return list(examples)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(examples), size=n, replace=False)
    return [examples[i] for i in sorted(idx.tolist())]


def pick_shots(k: int, seed: int = 11) -> list[Example]:
    """Seeded few-shot examples, drawn only from the few-shot pool.

    Drawing exclusively from `fewshot_pool` is what keeps prompt examples disjoint from
    both the calibration and test splits (CLAUDE.md guardrail 1).
    """
    pool = load_split("fewshot_pool")
    return sample(pool, k, seed=seed)


def pick_shots_one_per_class(seed: int = 11) -> list[Example]:
    """One few-shot example per intent (77 shots), drawn from the few-shot pool.

    Random k-shot only demonstrates k of 77 intents. One-per-class instead shows the model
    every intent in use, which is the setting that can actually disambiguate confusable
    pairs like 'card arrival' vs 'card delivery estimate'. Costs a much longer prompt.
    """
    pool = load_split("fewshot_pool")
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[Example]] = {}
    for ex in pool:
        by_label.setdefault(ex.label, []).append(ex)
    shots = []
    for label in canonical_labels():
        candidates = by_label[label]
        shots.append(candidates[int(rng.integers(len(candidates)))])
    return shots
