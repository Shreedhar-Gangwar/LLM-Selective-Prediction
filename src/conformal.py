"""Conformal selective calibration — the statistical core.

We have a classifier that always predicts, and a confidence score s(x) in [0, 1] for each
prediction. We *accept* a prediction when s(x) >= tau and otherwise *abstain* (route to a
human). We want a threshold tau with a finite-sample guarantee on the accepted set:

    accepted-set error rate <= alpha        (equivalently: accepted accuracy >= 1 - alpha)

The guarantee is a RISK-CONTROL claim, not a coverage claim: we control the *error among
accepted predictions*. The procedure is Learn-then-Test (Angelopoulos et al. 2021), in the
same family as RCPS (Bates et al. 2021): each candidate threshold is a hypothesis
"risk >= alpha", we compute an exact binomial p-value for it on the calibration set, and we
reject with a Bonferroni correction over a fixed grid so the whole family is valid at once.
Among the thresholds we can reject (risk certified below alpha) we take the one with the
highest coverage.

Why not fixed sequential testing (which would avoid the Bonferroni penalty)? Here the
low-risk end of the threshold range is the SPARSE end — accept only the few most confident
predictions — which has no statistical power (you cannot certify a rate from one point).
Sequential testing from that end stalls immediately. Bonferroni over a grid sidesteps the
ordering problem entirely and is simple to verify.

Guarantee, stated honestly:
    With probability >= 1 - delta over the draw of the calibration set, the threshold this
    procedure returns has true selective risk <= alpha.
The only assumption is EXCHANGEABILITY of the calibration and test data (they are drawn
from the same distribution). No assumption on the model or the score's calibration — a
badly calibrated score just yields a conservative threshold (low coverage), never a
broken guarantee.

Everything here is a pure function of arrays: no I/O, no plotting, no model. Higher score
= more confident throughout; ties are handled by accepting the whole tie group.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

# tau = +inf means "abstain on everything" — the fallback when no threshold controls risk.
ABSTAIN_ALL = float("inf")


@dataclass
class SelectiveThreshold:
    """A calibrated threshold and what it achieved on the calibration set."""

    tau: float  # accept predictions with score >= tau
    alpha: float  # target selective risk (accepted error rate)
    delta: float  # the guarantee holds with probability >= 1 - delta
    controlled: bool  # False if no threshold could control risk (=> abstain on all)
    calib_coverage: float  # fraction of calibration accepted at tau
    calib_selective_risk: float  # accepted error rate on calibration at tau


@dataclass
class RiskCoverage:
    """A risk-coverage curve: parallel arrays, one entry per candidate threshold.

    Sorted by increasing coverage. `selective_risk` is the error rate among accepted
    (NaN where coverage is 0). This is descriptive — no guarantee attaches to a curve.
    """

    thresholds: np.ndarray
    coverage: np.ndarray
    selective_risk: np.ndarray
    selective_accuracy: np.ndarray


def binomial_selective_pvalue(k_errors: int, n_accepted: int, alpha: float) -> float:
    """Exact p-value for H0: true selective risk >= alpha, against H1: risk < alpha.

    Conditional on the accepted set, the error count under H0 is stochastically at least
    Binomial(n, alpha), so seeing few errors is evidence against H0:

        p = P(Binomial(n, alpha) <= k_errors)

    This is exact and finite-sample valid (no normal approximation). With no accepted
    points there is no evidence, so p = 1.
    """
    if n_accepted == 0:
        return 1.0
    return float(binom.cdf(k_errors, n_accepted, alpha))


def _accept_stats(scores: np.ndarray, correct: np.ndarray, tau: float) -> tuple[int, int]:
    """(n_accepted, n_errors) when accepting score >= tau."""
    acc = scores >= tau
    n = int(acc.sum())
    k = int((~correct[acc]).sum())
    return n, k


def _threshold_grid(scores: np.ndarray, grid_size: int) -> np.ndarray:
    """A fixed grid of candidate thresholds spanning the score range.

    Score quantiles rather than a linspace, so the grid tracks where the scores actually
    are (confidence scores can be bunched near 0 or 1). The lowest grid point is the
    minimum score, i.e. accept-everything. Deduplicated. The grid defines the multiple-
    testing family, so its size — not the sample size — sets the Bonferroni penalty.
    """
    qs = np.quantile(scores, np.linspace(0.0, 1.0, grid_size))
    return np.unique(qs)


def ltt_threshold(
    scores: np.ndarray,
    correct: np.ndarray,
    alpha: float,
    delta: float = 0.10,
    grid_size: int = 100,
) -> SelectiveThreshold:
    """Max-coverage threshold whose accepted-set risk is certified <= alpha at level delta.

    Learn-then-Test with Bonferroni: for every threshold on a fixed grid, form the exact
    binomial p-value for H0 "risk >= alpha" and reject it if p <= delta / m (m = grid
    size). Rejecting means the risk is certified below alpha at that threshold. Among all
    rejected thresholds we return the one with the highest coverage (lowest tau). Because
    Bonferroni controls the family-wise error at delta, the probability that ANY returned
    threshold has true risk > alpha is <= delta — so it holds for the one we pick.

    Low-power thresholds (accept only a handful of points) simply never get rejected and
    are skipped. Returns `controlled=False`, tau=+inf when nothing is certifiable (e.g. an
    uninformative score, or alpha below the model's best achievable accepted error).
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)

    grid = _threshold_grid(scores, grid_size)
    level = delta / len(grid)  # Bonferroni-corrected per-hypothesis level

    best: float | None = None  # lowest tau (max coverage) that is certifiable
    for tau in grid:
        n, k = _accept_stats(scores, correct, tau)
        if binomial_selective_pvalue(k, n, alpha) <= level:
            if best is None or tau < best:
                best = float(tau)

    if best is None:
        return SelectiveThreshold(ABSTAIN_ALL, alpha, delta, False, 0.0, float("nan"))

    n, k = _accept_stats(scores, correct, best)
    return SelectiveThreshold(best, alpha, delta, True, n / len(scores), k / n)


def naive_threshold(
    scores: np.ndarray, correct: np.ndarray, alpha: float
) -> float:
    """Smallest tau whose *empirical* calibration risk <= alpha. No finite-sample slack.

    Included only as a foil: it ignores sampling error, so on a fresh test set its true
    risk exceeds alpha far more than one might expect — which is exactly what the RCPS
    correction fixes. Never use this for the guarantee.
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    best = ABSTAIN_ALL
    for tau in np.unique(scores)[::-1]:
        n, k = _accept_stats(scores, correct, tau)
        if n > 0 and k / n <= alpha:
            best = float(tau)
        else:
            break
    return best


def evaluate_threshold(
    scores: np.ndarray, correct: np.ndarray, tau: float
) -> tuple[float, float, int, int]:
    """Apply a fixed tau to a fresh split. Returns (coverage, selective_accuracy,
    n_accepted, n_errors). selective_accuracy is NaN when nothing is accepted."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n, k = _accept_stats(scores, correct, tau)
    cov = n / len(scores)
    acc = (n - k) / n if n > 0 else float("nan")
    return cov, acc, n, k


def risk_coverage_curve(scores: np.ndarray, correct: np.ndarray) -> RiskCoverage:
    """Sweep every candidate threshold and record coverage vs accepted error/accuracy.

    Includes the accept-everything point (tau just below the minimum score). Sorted by
    increasing coverage for plotting. Descriptive only.
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)

    taus = np.concatenate([np.unique(scores), [-np.inf]])  # -inf => accept all
    cov = np.empty(len(taus))
    risk = np.empty(len(taus))
    for i, tau in enumerate(taus):
        c, a, n, _ = evaluate_threshold(scores, correct, tau)
        cov[i] = c
        risk[i] = (1.0 - a) if n > 0 else np.nan

    order = np.argsort(cov)
    cov, risk, taus = cov[order], risk[order], taus[order]
    return RiskCoverage(
        thresholds=taus, coverage=cov, selective_risk=risk, selective_accuracy=1.0 - risk
    )
