"""Risk-parity construction and significance testing for Sharpe comparisons.

Two gaps this fills.

1. The comparison table ranks five methods by Sharpe and never asks whether the
   differences are real. A Sharpe estimated over ~7 years has a standard error
   near 0.4, so "0.57 beats 0.24" is not by itself a finding -- and picking the
   best of five tried strategies inflates the winner even when none of them has
   skill. `deflated_sharpe` and `sharpe_diff_test` answer both objections.

2. HRP is in the table but true equal-risk-contribution risk parity is not, even
   though it is the standard answer to "diversify by risk, not by capital".
   `erc_weights` adds it.

Run `python riskstats.py` for the self-check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

TRADING_DAYS = 252
EULER = 0.5772156649015329


# --------------------------------------------------------------- risk parity

def risk_contributions(w: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Euler decomposition of portfolio volatility onto positions.

    RC_i = w_i * (S w)_i / sqrt(w' S w), and sum(RC_i) == portfolio vol
    exactly, because volatility is homogeneous of degree 1 in the weights.
    Equal-weight means equal CAPITAL; risk parity means equal RC.
    """
    vol = float(np.sqrt(w @ S @ w))
    return w * (S @ w) / vol if vol > 0 else np.zeros_like(w)


def erc_weights(S: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    """Equal-risk-contribution portfolio, long-only, fully invested.

    Solved in the convex log-barrier form (Spinu 2013):

        min  0.5 w'Sw - (1/n) sum(log w_i),   w > 0

    The first-order condition is (Sw)_i = c / w_i, i.e. w_i (Sw)_i equal for
    every i -- exactly equal risk contributions. Normalising afterwards keeps
    that property because RC is homogeneous.

    Minimising the squared spread of risk contributions directly is the
    obvious alternative and is non-convex; it lands in local minima on
    ill-conditioned covariance matrices. This form does not.
    """
    n = len(S)
    c = 1.0 / n

    def f(w):
        return 0.5 * w @ S @ w - c * np.log(w).sum()

    def g(w):
        return S @ w - c / w

    w0 = np.full(n, 1.0 / n) / np.sqrt(np.diag(S).mean())
    res = optimize.minimize(f, w0, jac=g, method="L-BFGS-B",
                            bounds=[(1e-12, None)] * n,
                            options={"maxiter": 5000, "ftol": tol, "gtol": tol})
    return res.x / res.x.sum()


# ------------------------------------------------- Sharpe significance tests

def _sr(r: np.ndarray) -> float:
    """Per-period Sharpe of a raw return series (not annualised)."""
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


per_period_sharpe = _sr        # public alias; _sr is the internal short name


def probabilistic_sharpe(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """Bailey & Lopez de Prado (2012) Probabilistic Sharpe Ratio.

    P(true SR > benchmark), correcting the estimator for the two things that
    break the normal-iid assumption behind the usual SR standard error:

        PSR = Phi[ (SR - SR*) sqrt(T-1) / sqrt(1 - g3*SR + (g4-1)/4 * SR^2) ]

    Negative skew and fat tails (high g4) both INFLATE the denominator, so a
    strategy with a good Sharpe built out of many small gains and rare large
    losses is correctly penalised. That is the exact profile of most naive
    option-selling and momentum strategies.

    sr_benchmark is in the same (per-period) units as SR.
    """
    r = np.asarray(r, float)
    T = len(r)
    sr = _sr(r)
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))       # raw, not excess
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    if denom <= 0 or T < 3:
        return np.nan
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1)
                                / np.sqrt(denom)))


def expected_max_sharpe(trial_sharpes, n_trials: int | None = None) -> float:
    """Expected maximum of N independent trial Sharpes under the null of NO skill.

        E[max SR] ~ sqrt(V[SR]) * [ (1-g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(Ne)) ]

    This is the number the winner has to beat. Try enough strategies and the
    best one looks good by construction; this quantifies exactly how good
    "good by luck" is for the number of attempts made.
    """
    s = np.asarray(list(trial_sharpes), float)
    s = s[np.isfinite(s)]
    N = int(n_trials or len(s))
    if N < 2 or len(s) < 2:
        return 0.0
    v = float(s.std(ddof=1))
    return v * ((1 - EULER) * stats.norm.ppf(1 - 1 / N)
                + EULER * stats.norm.ppf(1 - 1 / (N * np.e)))


def deflated_sharpe(r: np.ndarray, trial_sharpes, n_trials: int | None = None):
    """Deflated Sharpe Ratio: PSR measured against the selection-adjusted
    benchmark rather than against zero.

    DSR < 0.95 means the strategy is not distinguishable from the best of N
    lucky coin flips at the 5% level -- regardless of how good its raw Sharpe
    looks. This is the single most useful defence against backtest overfitting,
    and it is cheap: one number, computed from data you already have.
    """
    sr0 = expected_max_sharpe(trial_sharpes, n_trials)
    return {"SR": _sr(np.asarray(r, float)),
            "SR0": sr0,
            "PSR": probabilistic_sharpe(r, 0.0),
            "DSR": probabilistic_sharpe(r, sr0)}


def sharpe_diff_test(r1: np.ndarray, r2: np.ndarray, n_boot: int = 5000,
                     block: int | None = None, seed: int = 0) -> dict:
    """Is SR(r1) - SR(r2) significantly different from zero?

    Circular block bootstrap rather than the closed-form Jobson-Korkie /
    Memmel statistic, deliberately: JK assumes iid normal returns, and the
    whole reason PSR exists is that returns are neither. Resampling in blocks
    preserves the autocorrelation and the fat tails instead of assuming them
    away.

    Returns the observed difference and a two-sided bootstrap p-value.
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    n = min(len(r1), len(r2))
    r1, r2 = r1[-n:], r2[-n:]
    obs = _sr(r1) - _sr(r2)
    if block is None:
        block = max(2, int(n ** (1 / 3)))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        # Same index into both series: the resample preserves the
        # cross-correlation between the two strategies, which is large here
        # (they hold overlapping assets) and would otherwise inflate the
        # apparent standard error of the difference.
        diffs[b] = _sr(r1[idx]) - _sr(r2[idx])
    # Standard recentred bootstrap p-value: the null is imposed by shifting the
    # bootstrap distribution to zero, not by centring the input series (which
    # would test SR1 = SR2 = 0 rather than SR1 = SR2).
    p = float(np.mean(np.abs(diffs - obs) >= abs(obs)))
    return {"sr1": _sr(r1), "sr2": _sr(r2), "diff": obs, "p_boot": p,
            "block": block}


# ------------------------------------------------------------------ self-check

def demo():
    rng = np.random.default_rng(0)

    # 1. Risk contributions are a true Euler decomposition: they sum to vol.
    A = rng.standard_normal((400, 4))
    S = np.cov(A, rowvar=False)
    w = np.array([0.4, 0.3, 0.2, 0.1])
    assert abs(risk_contributions(w, S).sum() - np.sqrt(w @ S @ w)) < 1e-12

    # 2. ERC equalises risk contributions, which equal weighting does not.
    we = erc_weights(S)
    rc = risk_contributions(we, S)
    assert abs(we.sum() - 1) < 1e-9 and (we > 0).all()
    assert rc.std() / rc.mean() < 1e-6, rc
    rc_eq = risk_contributions(np.full(4, 0.25), S)
    assert rc_eq.std() / rc_eq.mean() > rc.std() / rc.mean()

    # 3. With uncorrelated assets ERC must give w_i proportional to 1/sigma_i.
    D = np.diag([0.04, 0.01, 0.0025, 0.16])
    wd = erc_weights(D)
    inv_sig = 1 / np.sqrt(np.diag(D))
    assert np.allclose(wd, inv_sig / inv_sig.sum(), atol=1e-6), wd

    # 4. PSR: negative skew must lower it at an IDENTICAL Sharpe. Both series
    #    are standardised to the same sample mean and sd, so SR is equal by
    #    construction and skew is the only thing left that can move PSR.
    def _fix(x, m=0.0004, s=0.01):
        x = np.asarray(x, float)
        return (x - x.mean()) / x.std(ddof=1) * s + m

    n = 2000
    sym = _fix(rng.standard_normal(n))
    skw = _fix(-rng.gamma(2.0, 1.0, n))            # left-skewed, fat left tail
    assert abs(_sr(sym) - _sr(skw)) < 1e-9, (_sr(sym), _sr(skw))
    assert stats.skew(skw) < -1.0 < 0 < stats.skew(sym) + 1.0
    assert probabilistic_sharpe(skw) < probabilistic_sharpe(sym)

    # 5. PSR rises with sample length for the same underlying Sharpe.
    short = rng.standard_normal(120) * 0.01 + 0.0008
    long_ = rng.standard_normal(3000) * 0.01 + 0.0008
    assert probabilistic_sharpe(long_) > probabilistic_sharpe(short)

    # 6. The deflation benchmark grows with the number of trials -- that IS
    #    the multiple-testing correction.
    trials = rng.standard_normal(20) * 0.05
    assert expected_max_sharpe(trials, 100) > expected_max_sharpe(trials, 5) > 0

    # 7. A genuinely skill-free strategy picked as best of many must not
    #    survive deflation, even though its raw PSR looks fine.
    cands = [rng.standard_normal(1500) * 0.01 for _ in range(40)]
    srs = [_sr(c) for c in cands]
    best = cands[int(np.argmax(srs))]
    d = deflated_sharpe(best, srs)
    assert d["PSR"] > 0.90, d          # looks significant on its own
    assert d["DSR"] < 0.95, d          # and is not, once selection is priced in

    # 8. Sharpe difference test: same series -> p ~ 1; clearly better -> p small
    a = rng.standard_normal(2000) * 0.01
    assert sharpe_diff_test(a, a.copy(), n_boot=500)["p_boot"] > 0.9
    b = a + 0.0015
    t = sharpe_diff_test(b, a, n_boot=2000)
    assert t["p_boot"] < 0.05 and t["diff"] > 0, t

    print("riskstats self-check OK")


if __name__ == "__main__":
    demo()
