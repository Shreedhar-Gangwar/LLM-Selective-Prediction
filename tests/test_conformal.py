"""Tests for the conformal calibration math (src/conformal.py).

Checked against inputs with known answers and, for the guarantee itself, a Monte-Carlo
replay: across many calibration draws, RCPS must violate its risk target no more than
delta of the time, while the naive threshold violates far more often.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

from src.conformal import (
    ABSTAIN_ALL,
    binomial_selective_pvalue,
    evaluate_threshold,
    naive_threshold,
    ltt_threshold,
    risk_coverage_curve,
)


# -- the exact p-value -----------------------------------------------------------

def test_pvalue_matches_binomial_cdf():
    assert np.isclose(binomial_selective_pvalue(3, 40, 0.1), binom.cdf(3, 40, 0.1))
    assert np.isclose(binomial_selective_pvalue(0, 10, 0.05), 0.95**10)


def test_pvalue_empty_accept_is_one():
    assert binomial_selective_pvalue(0, 0, 0.1) == 1.0


def test_pvalue_monotone_in_errors():
    # More observed errors => weaker evidence against "risk is high" => larger p.
    ps = [binomial_selective_pvalue(k, 50, 0.1) for k in range(0, 20)]
    assert all(b >= a for a, b in zip(ps, ps[1:]))


# -- threshold selection on known inputs -----------------------------------------

def test_all_correct_large_n_accepts_everything():
    # 500 always-correct predictions: risk is 0, so the accept-all threshold is controlled.
    scores = np.linspace(0, 1, 500)
    correct = np.ones(500, dtype=bool)
    t = ltt_threshold(scores, correct, alpha=0.05, delta=0.1)
    assert t.controlled
    assert t.tau <= scores.min()          # accept all
    assert np.isclose(t.calib_coverage, 1.0)
    assert t.calib_selective_risk == 0.0


def test_small_n_all_correct_is_not_enough():
    # Finite-sample honesty: 10 correct-in-a-row cannot justify a 95% guarantee, because
    # P(Binom(10, 0.05) = 0) = 0.95^10 ~ 0.60 > delta. So it must NOT accept all 10.
    scores = np.linspace(0, 1, 10)
    correct = np.ones(10, dtype=bool)
    t = ltt_threshold(scores, correct, alpha=0.05, delta=0.1)
    assert not t.controlled or t.calib_coverage < 1.0


def test_all_wrong_cannot_control():
    scores = np.linspace(0, 1, 200)
    correct = np.zeros(200, dtype=bool)
    t = ltt_threshold(scores, correct, alpha=0.05, delta=0.1)
    assert not t.controlled
    assert t.tau == ABSTAIN_ALL


def test_rcps_at_least_as_conservative_as_naive():
    # RCPS adds finite-sample slack, so it can never accept MORE than the naive rule.
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 400)
    correct = rng.uniform(0, 1, 400) < scores  # higher score => likelier correct
    t = ltt_threshold(scores, correct, alpha=0.1, delta=0.1)
    tau_naive = naive_threshold(scores, correct, alpha=0.1)
    assert t.tau >= tau_naive


# -- risk-coverage curve ---------------------------------------------------------

def test_curve_coverage_monotonic_and_bounded():
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, 300)
    correct = rng.uniform(0, 1, 300) < scores
    rc = risk_coverage_curve(scores, correct)
    assert np.all(np.diff(rc.coverage) >= -1e-12)     # sorted, non-decreasing
    assert np.isclose(rc.coverage.max(), 1.0)         # includes accept-all
    assert np.all((rc.coverage >= 0) & (rc.coverage <= 1))


def test_curve_risk_decreases_as_we_accept_less_when_signal_informative():
    # With an informative score, tightening the threshold (less coverage) should not
    # increase accepted error on average: compare the two ends of the curve.
    rng = np.random.default_rng(2)
    scores = rng.uniform(0, 1, 2000)
    correct = rng.uniform(0, 1, 2000) < scores
    rc = risk_coverage_curve(scores, correct)
    high_cov_risk = rc.selective_risk[rc.coverage >= 0.9].mean()
    low_cov_risk = np.nanmean(rc.selective_risk[rc.coverage <= 0.3])
    assert low_cov_risk < high_cov_risk


def test_evaluate_threshold_counts():
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    correct = np.array([False, True, False, True])
    cov, acc, n, k = evaluate_threshold(scores, correct, tau=0.5)
    assert (n, k) == (2, 1)               # accepts 0.6 (wrong) and 0.9 (right)
    assert cov == 0.5 and acc == 0.5


# -- the actual guarantee, by Monte-Carlo replay ---------------------------------

def _draw(rng, n, base_risk):
    """Synthetic split: score ~ U(0,1); a prediction is correct with prob increasing in
    score, tuned so the marginal error rate is ~base_risk."""
    scores = rng.uniform(0, 1, n)
    p_correct = np.clip(1.0 - base_risk * 2.0 * (1.0 - scores), 0.0, 1.0)
    correct = rng.uniform(0, 1, n) < p_correct
    return scores, correct


def test_rcps_controls_risk_across_draws():
    # Over many (calibrate, test) splits, the LTT threshold's TEST selective risk should
    # exceed alpha at most ~delta of the time. The naive threshold should violate more.
    alpha, delta = 0.1, 0.1
    rng = np.random.default_rng(12345)
    trials = 300
    rcps_viol = naive_viol = 0
    for _ in range(trials):
        sc, co = _draw(rng, 1000, base_risk=0.15)
        st, ct = _draw(rng, 1000, base_risk=0.15)

        t = ltt_threshold(sc, co, alpha, delta)
        if t.controlled:
            _, acc, n, _ = evaluate_threshold(st, ct, t.tau)
            if n > 0 and (1 - acc) > alpha:
                rcps_viol += 1

        tau_naive = naive_threshold(sc, co, alpha)
        _, acc_n, n_n, _ = evaluate_threshold(st, ct, tau_naive)
        if n_n > 0 and (1 - acc_n) > alpha:
            naive_viol += 1

    # RCPS violation rate stays around/below delta (small slack for MC noise).
    assert rcps_viol / trials <= delta + 0.03
    # And RCPS is meaningfully safer than the uncorrected rule.
    assert rcps_viol < naive_viol
