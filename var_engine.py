"""
VaR / Expected Shortfall engine with regulatory-grade backtesting.

Conventions (fixed everywhere, no exceptions):
  r_t      return, negative = loss
  alpha    tail probability. alpha=0.01 -> "99% VaR"
  var, es  POSITIVE loss magnitudes
  breach   r_t < -var_t

Every forecast for time t uses information up to t-1 only. There is no
in-sample number anywhere in this file -- that is the whole point.

Regulatory grounding:
  Basel III / FRTB capitalises on Expected Shortfall at 97.5%, but backtesting
  still runs on VaR at 99% and 97.5% (250 obs, desk level): <=12 breaches at
  99% AND <=30 at 97.5% to keep internal-model approval. See frtb_desk_test.

References:
  Kupiec (1995)             proportion-of-failures test
  Christoffersen (1998)     independence / conditional coverage
  Acerbi & Szekely (2014)   Z1/Z2 direct ES backtests (MC critical values)
  Barone-Adesi et al (1999) filtered historical simulation
  Glosten-Jagannathan-Runkle (1993) asymmetric GARCH

Run `python var_engine.py` for the self-check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln, xlogy

# --------------------------------------------------------------------------
# standardised Student-t (unit variance) -- the distribution that actually
# fits equity returns. scipy's t has variance nu/(nu-2), so rescale by that.
# --------------------------------------------------------------------------


def std_t_ppf(alpha: float, nu: float) -> float:
    return stats.t.ppf(alpha, nu) / np.sqrt(nu / (nu - 2.0))


def std_t_es(alpha: float, nu: float) -> float:
    """ES of a unit-variance Student-t, as a positive magnitude."""
    q = stats.t.ppf(alpha, nu)
    es_raw = (nu + q ** 2) / (nu - 1.0) * stats.t.pdf(q, nu) / alpha
    return es_raw / np.sqrt(nu / (nu - 2.0))


def empirical_es(z: np.ndarray, alpha: float) -> tuple[float, float]:
    """(var, es) as positive magnitudes from a sample."""
    q = np.quantile(z, alpha)
    tail = z[z <= q]
    return -q, -tail.mean()


# --------------------------------------------------------------------------
# forecast container
# --------------------------------------------------------------------------


@dataclass
class Forecast:
    name: str
    idx: np.ndarray      # positions in the original return array
    r: np.ndarray        # realised return at each idx
    var: np.ndarray      # positive loss magnitude
    es: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    draw: Callable[[], np.ndarray]   # ONE simulated path of standardised
                                     # innovations, length len(self)

    @property
    def breach(self) -> np.ndarray:
        return self.r < -self.var

    def __len__(self) -> int:
        return len(self.idx)


# --- null-hypothesis samplers ---------------------------------------------
# draw() must return a path whose element k comes from the predictive
# distribution actually in force on day k. Sampling every day from one fixed
# distribution instead makes the simulated null too narrow and manufactures
# rejections in acerbi_szekely -- which is exactly the bug a window-sensitivity
# sweep surfaced, so it is worth the extra bookkeeping.


def _chunks(n: int, size: int) -> np.ndarray:
    """Map each of n days to the index of its refit block."""
    return np.arange(n) // size


def _pool_sampler(pools, pool_of, rng):
    def draw():
        out = np.empty(len(pool_of))
        for j, p in enumerate(pools):
            m = pool_of == j
            k = int(m.sum())
            if k:
                out[m] = rng.choice(p, k)
        return out
    return draw


def _t_sampler(nus, pool_of, rng):
    def draw():
        out = np.empty(len(pool_of))
        for j, v in enumerate(nus):
            m = pool_of == j
            k = int(m.sum())
            if k:
                out[m] = rng.standard_t(v, k) / np.sqrt(v / (v - 2))
        return out
    return draw


# --------------------------------------------------------------------------
# models -- identical signature, so they are directly comparable
# --------------------------------------------------------------------------


def historical(r: np.ndarray, alpha: float, window: int = 500) -> Forecast:
    """Rolling empirical quantile. No distributional assumption, no vol model.
    Fails Christoffersen inside vol clusters: the window reacts too slowly."""
    n = len(r)
    idx = np.arange(window, n)
    var = np.empty(len(idx))
    es = np.empty(len(idx))
    sig = np.empty(len(idx))
    pool_of = _chunks(len(idx), 21)
    pools = []
    for k, t in enumerate(idx):
        w = r[t - window:t]
        var[k], es[k] = empirical_es(w, alpha)
        sig[k] = w.std(ddof=1)
        if k % 21 == 0:
            pools.append((w - w.mean()) / w.std(ddof=1))
    rng = np.random.default_rng(0)
    return Forecast("Historical", idx, r[idx], var, es,
                    np.zeros(len(idx)), sig, _pool_sampler(pools, pool_of, rng))


def gaussian(r: np.ndarray, alpha: float, window: int = 500) -> Forecast:
    """Variance-covariance / parametric normal. The textbook method, and the
    one that under-states the tail on real returns."""
    s = pd.Series(r)
    mu = s.rolling(window).mean().shift(1).to_numpy()
    sd = s.rolling(window).std(ddof=1).shift(1).to_numpy()
    idx = np.arange(window, len(r))
    mu, sd = mu[idx], sd[idx]
    z = stats.norm.ppf(alpha)
    var = -(mu + sd * z)
    es = -mu + sd * stats.norm.pdf(z) / alpha
    rng = np.random.default_rng(1)
    n_oos = len(idx)
    return Forecast("Gaussian", idx, r[idx], var, es, mu, sd,
                    lambda: rng.standard_normal(n_oos))


def ewma(r: np.ndarray, alpha: float, window: int = 500,
         lam: float = 0.94) -> Forecast:
    """RiskMetrics (1996). One parameter, nothing estimated, reacts to vol
    immediately -- usually fixes independence but stays normal-tailed."""
    n = len(r)
    s2 = np.empty(n)
    s2[0] = r[:window].var(ddof=1)
    for t in range(1, n):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2
    idx = np.arange(window, n)
    sd = np.sqrt(s2[idx])
    z = stats.norm.ppf(alpha)
    var = -sd * z
    es = sd * stats.norm.pdf(z) / alpha
    rng = np.random.default_rng(2)
    n_oos = len(idx)
    return Forecast(f"EWMA(lam={lam})", idx, r[idx], var, es,
                    np.zeros(len(idx)), sd, lambda: rng.standard_normal(n_oos))


# ---- GJR-GARCH(1,1) with standardised-t errors, MLE by hand ---------------


def _gjr_filter(r: np.ndarray, om: float, al: float, ga: float,
                be: float, s2_0: float) -> np.ndarray:
    """sigma^2_t = om + (al + ga*1{r_{t-1}<0}) r_{t-1}^2 + be*sigma^2_{t-1}

    The gamma term is the leverage effect: a down move raises tomorrow's
    variance more than an up move of the same size. Symmetric GARCH misses it,
    and it is exactly what bites in a crash.
    """
    n = len(r)
    s2 = np.empty(n)
    s2[0] = s2_0
    for t in range(1, n):
        e = r[t - 1]
        s2[t] = om + (al + (ga if e < 0.0 else 0.0)) * e * e + be * s2[t - 1]
    return s2


def _t_const(nu: float) -> float:
    return float(gammaln((nu + 1) / 2) - gammaln(nu / 2)
                 - 0.5 * np.log(np.pi * (nu - 2.0)))


def _gjr_nll(theta: np.ndarray, r: np.ndarray, s2_0: float) -> float:
    om, al, ga, be, nu = theta
    if om <= 0 or al < 0 or be < 0 or al + ga < 0 or nu <= 2.05:
        return 1e10
    if al + 0.5 * ga + be >= 0.999:                     # stationarity
        return 1e10
    s2 = _gjr_filter(r, om, al, ga, be, s2_0)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e10
    z2 = r * r / s2
    ll = (_t_const(nu) - 0.5 * np.log(s2)
          - 0.5 * (nu + 1.0) * np.log1p(z2 / (nu - 2.0)))
    return -float(ll.sum())


def fit_gjr(r: np.ndarray, start: np.ndarray | None = None) -> np.ndarray:
    """MLE of (omega, alpha, gamma, beta, nu). Warm-startable.

    Hand-rolled rather than `arch`: the likelihood is 6 lines, and owning it
    means you can answer "what does gamma do" in an interview.
    """
    v = float(r.var(ddof=1))
    if start is None:
        start = np.array([v * 0.05, 0.04, 0.08, 0.88, 7.0])
    bounds = [(1e-14, v * 5), (0.0, 0.4), (-0.2, 0.6),
              (0.0, 0.999), (2.1, 60.0)]
    res = optimize.minimize(_gjr_nll, start, args=(r, v), method="L-BFGS-B",
                            bounds=bounds, options={"maxiter": 300})
    if not np.isfinite(res.fun) or res.fun >= 1e9:
        res = optimize.minimize(
            _gjr_nll, np.array([v * 0.1, 0.05, 0.05, 0.85, 8.0]),
            args=(r, v), method="Nelder-Mead", options={"maxiter": 2000})
    return np.asarray(res.x, float)


_PATH_CACHE: dict = {}


def gjr_path(r: np.ndarray, window: int = 500, refit_every: int = 21):
    """One MLE sweep -> next-day sigma forecast and the standardised-residual
    pool in force on each out-of-sample day.

    Split out from the VaR calculation because the fit does not depend on
    alpha or on the tail treatment: every confidence level and both the
    analytic-t and FHS variants reuse a single sweep. Cached on
    (sample, window, refit_every).

    The filter runs forward across the whole sample carrying sigma^2 between
    days, resynced from the estimation window at each refit -- which is both
    what a production risk system does and O(n) instead of O(n*window).

    ponytail: refit monthly with warm start. Daily refit is ~20x the compute
    for a parameter path that barely moves; set refit_every=1 to check that.
    """
    key = (hash(r.tobytes()), window, refit_every)
    if key in _PATH_CACHE:
        return _PATH_CACHE[key]
    n = len(r)
    idx = np.arange(window, n)
    sd = np.empty(len(idx))
    pool_of = np.empty(len(idx), int)
    pools: list[np.ndarray] = []
    thetas: list[np.ndarray] = []
    theta = None
    s2 = float(r[:window].var(ddof=1))
    for k, t in enumerate(idx):
        if k % refit_every == 0:
            w = r[t - window:t]
            theta = fit_gjr(w, start=theta)
            s2w = _gjr_filter(w, *theta[:4], float(w.var(ddof=1)))
            pools.append(w / np.sqrt(s2w))
            thetas.append(theta)
            s2 = float(s2w[-1])                 # resync to the fresh fit
        om, al, ga, be, _ = theta
        e = r[t - 1]
        s2 = om + (al + (ga if e < 0.0 else 0.0)) * e * e + be * s2
        sd[k] = np.sqrt(s2)
        pool_of[k] = len(pools) - 1
    out = (idx, sd, pools, pool_of, np.array(thetas))
    _PATH_CACHE[key] = out
    return out


def gjr_garch(r: np.ndarray, alpha: float, window: int = 500,
              refit_every: int = 21, fhs: bool = True) -> Forecast:
    """Asymmetric GARCH volatility + fat tails.

    fhs=True  -> Filtered Historical Simulation: quantile of the *empirical*
                 standardised residuals. No tail-shape assumption at all.
                 This is what desks actually run for FRTB IMA.
    fhs=False -> analytic standardised-t quantile.

    Note the window requirement differs between the two. FHS reads the alpha
    quantile straight off the residual pool, so a 500-day window leaves only
    ~5 observations below the 1% level -- too few for a stable quantile. The
    analytic-t variant borrows strength from all 500 points via nu and does
    not have that problem. Use window>=1000 for FHS at 99%.
    """
    idx, sd, pools, pool_of, thetas = gjr_path(r, window, refit_every)
    if fhs:
        qe = np.array([empirical_es(p, alpha) for p in pools])
        q, e_ = qe[pool_of, 0], qe[pool_of, 1]
        rng = np.random.default_rng(3)
        drawf = _pool_sampler(pools, pool_of, rng)
        name = "GJR-GARCH-FHS"
    else:
        nus = thetas[:, 4]
        q = np.array([-std_t_ppf(alpha, v) for v in nus])[pool_of]
        e_ = np.array([std_t_es(alpha, v) for v in nus])[pool_of]
        rng = np.random.default_rng(4)
        drawf = _t_sampler(nus, pool_of, rng)
        name = f"GJR-GARCH-t(nu={float(nus[-1]):.1f})"
    return Forecast(name, idx, r[idx], sd * q, sd * e_,
                    np.zeros(len(idx)), sd, drawf)


# ---- conditional EVT: GARCH filter + Generalised Pareto tail --------------
# McNeil & Frey (2000). FHS has a hard ceiling: its worst possible forecast is
# the worst residual it has ever seen, because it reads an empirical quantile.
# Extreme value theory removes that ceiling. The Pickands-Balkema-de Haan
# theorem says exceedances over a high threshold converge to a Generalised
# Pareto distribution regardless of the parent distribution, so fit a GPD to
# the tail only and you can extrapolate past the sample maximum. That is the
# difference between "1-in-100 day" (which FHS can read off 1000 points) and
# "1-in-2000 day" (which it cannot).


def _gpd_fit(z: np.ndarray, u_q: float = 0.90) -> tuple:
    """Fit a GPD to the left tail of standardised residuals z.

    Returns (u, xi, beta, n, n_u) in LOSS space (y = -z), where u is the
    threshold, xi the shape (tail index) and beta the scale.

    xi is the number that matters: xi > 0 is heavy-tailed (power law),
    xi = 0 exponential, xi < 0 bounded. Equity residuals typically give
    xi ~ 0.1-0.3. xi >= 1 implies infinite mean and ES does not exist.
    """
    y = -z
    u = float(np.quantile(y, u_q))
    exc = y[y > u] - u
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    return u, float(xi), float(beta), len(y), len(exc)


def _gpd_var_es(fit: tuple, alpha: float) -> tuple[float, float]:
    """Tail quantile and ES from a fitted GPD, as positive magnitudes.

        VaR = u + (beta/xi) * [ ((n/n_u) * alpha)^(-xi) - 1 ]
        ES  = VaR/(1-xi) + (beta - xi*u)/(1-xi)

    The ES formula is where EVT pays off: once xi is estimated, the mean of
    the tail beyond ANY level follows in closed form.
    """
    u, xi, beta, n, n_u = fit
    xi = min(xi, 0.95)                    # ponytail: xi>=1 => ES infinite.
    if abs(xi) < 1e-8:                    # exponential limit
        var = u + beta * np.log((n_u / n) / alpha)
        return var, var + beta
    var = u + (beta / xi) * (((n / n_u) * alpha) ** (-xi) - 1.0)
    return var, var / (1 - xi) + (beta - xi * u) / (1 - xi)


def _gpd_rvs(xi: float, beta: float, size: int, rng) -> np.ndarray:
    """GPD draws by inverse CDF. Vectorised, because scipy's rvs called once
    per refit block per simulation is the difference between 3s and 3min."""
    uu = rng.random(size)
    if abs(xi) < 1e-8:
        return -beta * np.log1p(-uu)
    return beta / xi * ((1 - uu) ** (-xi) - 1.0)


def _evt_sampler(pools, fits, pool_of, rng):
    """Null sampler for the semi-parametric model: empirical in the body,
    GPD in the left tail. Bootstrapping the raw pool instead would sample a
    tail the model does not actually claim."""
    def draw():
        out = np.empty(len(pool_of))
        for j, p in enumerate(pools):
            m = pool_of == j
            k = int(m.sum())
            if not k:
                continue
            s = rng.choice(p, k)
            u, xi, beta, _, _ = fits[j]
            hit = s < -u
            nh = int(hit.sum())
            if nh:
                s[hit] = -(u + _gpd_rvs(min(xi, 0.95), beta, nh, rng))
            out[m] = s
        return out
    return draw


def evt_pot(r: np.ndarray, alpha: float, window: int = 1000,
            refit_every: int = 21, u_q: float = 0.90) -> Forecast:
    """Conditional EVT / peaks-over-threshold.

    Two-stage, and the order matters. GARCH first removes volatility
    clustering, which is what makes raw returns non-iid; EVT theory needs
    roughly iid data, so applying a GPD to raw returns (as plenty of repos do)
    violates its own assumption. Filter, then fit the tail of the residuals.
    """
    idx, sd, pools, pool_of, _ = gjr_path(r, window, refit_every)
    fits = [_gpd_fit(p, u_q) for p in pools]
    qe = np.array([_gpd_var_es(ft, alpha) for ft in fits])
    q, e_ = qe[pool_of, 0], qe[pool_of, 1]
    rng = np.random.default_rng(5)
    return Forecast(f"GJR-EVT-POT(xi={fits[-1][1]:.2f})", idx, r[idx],
                    sd * q, sd * e_, np.zeros(len(idx)), sd,
                    _evt_sampler(pools, fits, pool_of, rng))


# --------------------------------------------------------------------------
# backtests
# --------------------------------------------------------------------------


def kupiec(breach: np.ndarray, alpha: float) -> dict:
    """Unconditional coverage. H0: breach rate == alpha. LR ~ chi2(1)."""
    T, N = len(breach), int(breach.sum())
    pi = N / T
    ll0 = xlogy(T - N, 1 - alpha) + xlogy(N, alpha)
    ll1 = xlogy(T - N, 1 - pi) + xlogy(N, pi)
    lr = float(-2.0 * (ll0 - ll1))
    return {"T": T, "breaches": N, "rate": pi, "expected": alpha * T,
            "LR_uc": lr, "p_uc": float(1 - stats.chi2.cdf(lr, 1))}


def christoffersen(breach: np.ndarray, alpha: float) -> dict:
    """Independence (breaches must not cluster) + conditional coverage.

    A model can pass Kupiec and still be useless: the right *number* of
    breaches, all stacked into one crisis week. LR_ind catches exactly that,
    and clustering is the signature of a model that is not tracking vol.
    """
    b = breach.astype(int)
    prev, cur = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    p01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    ll1 = (xlogy(n00, 1 - p01) + xlogy(n01, p01)
           + xlogy(n10, 1 - p11) + xlogy(n11, p11))
    lr_ind = max(float(-2.0 * (ll0 - ll1)), 0.0)
    lr_uc = kupiec(breach, alpha)["LR_uc"]
    lr_cc = lr_uc + lr_ind
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "LR_ind": lr_ind, "p_ind": float(1 - stats.chi2.cdf(lr_ind, 1)),
            "LR_cc": lr_cc, "p_cc": float(1 - stats.chi2.cdf(lr_cc, 2))}


def dq_test(f: Forecast, alpha: float, lags: int = 4) -> dict:
    """Engle & Manganelli (2004) dynamic quantile test.

    Christoffersen only looks at whether breach t-1 predicts breach t -- a
    first-order Markov chain, and nothing else. A model can pass it while its
    breaches are predictable from four days ago, or from the level of VaR
    itself (systematically breaching when VaR is low is a real failure mode
    that LR_ind cannot see).

    DQ regresses the demeaned hit series on a constant, `lags` lagged hits,
    and the contemporaneous VaR. Under correct specification the hit series is
    an iid Bernoulli(alpha), so EVERY coefficient should be zero:

        Wald = b' X'X b / (alpha (1-alpha))  ~  chi2(k)

    This subsumes both Kupiec (via the constant) and Christoffersen (via the
    first lag), which is why it is the test most papers report now.
    """
    hit = f.breach.astype(float) - alpha
    T = len(hit)
    cols = [np.ones(T - lags)]
    cols += [hit[lags - i:T - i] for i in range(1, lags + 1)]
    cols.append(f.var[lags:])
    X = np.column_stack(cols)
    y = hit[lags:]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    stat = float(b @ (X.T @ X) @ b / (alpha * (1 - alpha)))
    return {"DQ": stat, "p_dq": float(1 - stats.chi2.cdf(stat, X.shape[1]))}


_BASEL_MULT = {0: 3.00, 1: 3.00, 2: 3.00, 3: 3.00, 4: 3.00,
               5: 3.40, 6: 3.50, 7: 3.65, 8: 3.75, 9: 3.85}


def basel_traffic_light(breach: np.ndarray) -> dict:
    """Basel traffic light over the most recent 250 observations at 99% VaR.
    Green <=4, Amber 5-9 (capital multiplier steps up), Red >=10."""
    b = breach[-250:]
    n = int(b.sum())
    zone = "GREEN" if n <= 4 else ("AMBER" if n <= 9 else "RED")
    return {"window": len(b), "breaches": n, "zone": zone,
            "multiplier": _BASEL_MULT.get(n, 4.00)}


def frtb_desk_test(breach_99: np.ndarray, breach_975: np.ndarray) -> dict:
    """FRTB desk-level IMA eligibility: over the most recent 250 days a desk
    keeps internal-model approval only with <=12 breaches at 99% AND <=30 at
    97.5%. Failing either pushes the desk onto the standardised approach --
    materially more capital. This is the test that has money attached."""
    n99 = int(breach_99[-250:].sum())
    n975 = int(breach_975[-250:].sum())
    ok = bool(n99 <= 12 and n975 <= 30)
    return {"exc_99": n99, "limit_99": 12, "exc_975": n975, "limit_975": 30,
            "ima_eligible": ok,
            "verdict": "IMA retained" if ok else "IMA revoked -> SA capital"}


# FRTB liquidity horizons in days (BCBS d457 Table 2, risk-factor buckets).
LIQ_HORIZONS = (10, 20, 40, 60, 120)


def frtb_liquidity_adjusted_es(es_by_horizon: dict) -> float:
    """BCBS d457 para 33 aggregation across liquidity horizons.

        ES = sqrt( ES(P,1)^2 + SUM_{j>=2} [ ES(P,j) * sqrt((LH_j - LH_j-1)/10) ]^2 )

    The logic: not every risk factor can be exited in 10 days. FRTB assigns
    each factor a liquidity horizon and computes ES(P,j) using only factors
    with horizon >= LH_j, all shocked over a 10-day base. The square-root
    weights convert each incremental horizon slice to its own holding period
    and the sum-of-squares assumes the slices are independent.

    Takes {horizon_days: es} and returns the aggregated charge. Passing a
    single {10: es} returns es unchanged, which is the check in demo().

    ponytail: real LH assignment comes from the risk-factor taxonomy (equity
    large-cap 10d, credit-spread HY 60d, ...). This implements the AGGREGATION
    exactly; the bucket mapping is an input, not a guess made here.
    """
    lh = [h for h in LIQ_HORIZONS if h in es_by_horizon]
    if not lh:
        raise ValueError("no recognised liquidity horizons")
    total = es_by_horizon[lh[0]] ** 2
    for prev, cur in zip(lh, lh[1:]):
        total += (es_by_horizon[cur] * np.sqrt((cur - prev) / 10.0)) ** 2
    return float(np.sqrt(total))


def stressed_es(r: np.ndarray, alpha: float = 0.025,
                win: int = 250) -> dict:
    """FRTB calibrates ES to a stress period, not to the current calm one.

    Finds the worst `win`-day window in the sample by realised ES and reports
    the stress multiplier ES_stressed / ES_current. A model calibrated only on
    recent data understates capital by exactly this factor -- which is the
    procyclicality problem the stressed calibration exists to fix.
    """
    n = len(r)
    worst_i, worst = 0, -np.inf
    for i in range(0, n - win, 5):                # ponytail: 5d stride
        _, e = empirical_es(r[i:i + win], alpha)
        if e > worst:
            worst_i, worst = i, e
    _, cur = empirical_es(r[-win:], alpha)
    return {"stressed_es": worst, "current_es": cur,
            "start": int(worst_i), "end": int(worst_i + win),
            "multiplier": float(worst / cur) if cur > 0 else np.nan}


def acerbi_szekely(f: Forecast, alpha: float, n_sim: int = 2000,
                   seed: int = 7) -> dict:
    """Acerbi-Szekely (2014) Z1/Z2 -- backtests ES itself, not VaR.

    Kupiec and Christoffersen only ever test the VaR *threshold*. They are
    blind to how bad losses beyond it are, which is the entire reason Basel
    moved to ES. Z1/Z2 close that gap.

        Z1 = mean over breaches of (r_t / ES_t) + 1        [conditional]
        Z2 = sum(r_t * 1{breach} / ES_t) / (T*alpha) + 1   [unconditional]

    E[Z] = 0 under H0; Z < 0 means realised tail losses exceeded forecast ES.
    ES is not elicitable and has no closed-form critical values, so the null
    distribution is simulated from each model's own predictive distribution.
    """
    b = f.breach
    n_b = int(b.sum())
    z1 = float(np.mean(f.r[b] / f.es[b]) + 1.0) if n_b else np.nan
    z2 = float(np.sum(f.r * b / f.es) / (len(f) * alpha) + 1.0)

    T = len(f)
    sim1 = np.empty(n_sim)
    sim2 = np.empty(n_sim)
    for i in range(n_sim):
        rs = f.mu + f.sigma * f.draw()           # draw under H0: model is true
        bs = rs < -f.var
        k = int(bs.sum())
        sim1[i] = (np.mean(rs[bs] / f.es[bs]) + 1.0) if k else np.nan
        sim2[i] = np.sum(rs * bs / f.es) / (T * alpha) + 1.0
    s1 = sim1[~np.isnan(sim1)]
    p1 = float(np.mean(s1 <= z1)) if (n_b and len(s1)) else np.nan
    p2 = float(np.mean(sim2 <= z2))
    return {"Z1": z1, "p_Z1": p1, "Z2": z2, "p_Z2": p2, "breaches": n_b}


# --------------------------------------------------------------------------
# comparative evaluation -- is model A actually BETTER than model B?
# --------------------------------------------------------------------------
# Everything above is absolute: a model passes or fails against its own null.
# That cannot rank two models that both pass, and "mine passed, yours didn't"
# is a weaker claim than "mine beats yours with p < 0.001".
#
# Ranking needs a scoring function that is *strictly consistent* for the
# functional being forecast -- one whose expected value is minimised by the
# true value. Squared error does this for the mean, pinball loss for a
# quantile. Gneiting (2011) proved ES has NO such scoring function: it is not
# elicitable. This is not a technicality, it is why you cannot just define an
# "ES MSE" and rank on it.
#
# Fissler & Ziegel (2016) found the way out: the PAIR (VaR, ES) is jointly
# elicitable even though ES alone is not. FZ0 below is the 0-homogeneous
# member of that family (Nolde & Ziegel 2017), which is the one to use for
# returns because the loss is scale-invariant -- it does not silently reward
# whichever model happens to run on the higher-volatility sample.


def fz0_loss(f: Forecast, alpha: float) -> np.ndarray:
    """Fissler-Ziegel 0-homogeneous joint loss for (VaR, ES). Lower is better.

        FZ0 = -1/(alpha*e) * 1{r<=v} * (v-r) + v/e + log(-e) - 1

    with v = -var and e = -es in return space (both negative). Minimised in
    expectation at the true (VaR, ES) -- asserted numerically in demo().
    """
    v, e = -f.var, -f.es
    hit = (f.r <= v).astype(float)
    return (-1.0 / (alpha * e)) * hit * (v - f.r) + v / e + np.log(-e) - 1.0


def diebold_mariano(l1: np.ndarray, l2: np.ndarray, lag: int | None = None):
    """Diebold-Mariano test on a loss differential d_t = l1_t - l2_t.

    H0: E[d] = 0, the two forecasts are equally good. The variance of dbar
    needs a HAC (Newey-West) estimator because loss differentials are
    autocorrelated -- tail losses arrive in clusters, so treating them as iid
    understates the standard error and overstates significance.

    DM > 0 means model 1 has the higher loss, i.e. model 2 wins.
    """
    d = np.asarray(l1) - np.asarray(l2)
    T = len(d)
    if lag is None:
        lag = int(np.ceil(T ** (1 / 3)))
    dc = d - d.mean()
    v = float(dc @ dc) / T
    for h in range(1, lag + 1):                       # Bartlett kernel
        v += 2.0 * (1 - h / (lag + 1)) * float(dc[h:] @ dc[:-h]) / T
    if v <= 0:
        return {"DM": 0.0, "p_dm": 1.0}
    stat = float(d.mean() / np.sqrt(v / T))
    return {"DM": stat, "p_dm": float(2 * (1 - stats.norm.cdf(abs(stat))))}


def compare(forecasts: list, alpha: float, benchmark: int = 0) -> pd.DataFrame:
    """Rank by mean FZ0 loss and DM-test every model against a FIXED benchmark.

    The benchmark is `forecasts[benchmark]`, defaulting to the first model
    passed -- which in report() is Historical Simulation, the industry default
    and the obvious a-priori baseline.

    It deliberately is NOT the best-scoring model. Testing everything against
    the ex-post winner is data snooping: the winner was selected by looking at
    these same losses, so its advantage is biased upward and the p-values are
    not valid. Fixing the benchmark in advance removes that entirely, and
    "beats the industry standard" is the claim worth making anyway. Ranking
    the whole table against each other properly would need Hansen's Model
    Confidence Set; against one pre-specified baseline, plain DM is correct.

    Forecasts are trimmed to a common length so models with different
    estimation windows stay comparable.
    """
    n = min(len(f) for f in forecasts)
    losses = {f.name: fz0_loss(f, alpha)[-n:] for f in forecasts}
    bench = forecasts[benchmark].name
    best = min(losses, key=lambda k: losses[k].mean())
    rows = []
    for name, l in losses.items():
        dm = diebold_mariano(losses[bench], l)   # >0 => this model beats bench
        rows.append({"model": name, "FZ0 loss": round(float(l.mean()), 5),
                     "DM vs bench": round(dm["DM"], 2),
                     "p_dm": round(dm["p_dm"], 4),
                     "note": ("benchmark" if name == bench
                              else "<- lowest loss" if name == best else "")})
    return pd.DataFrame(rows).sort_values("FZ0 loss").set_index("model")


# --------------------------------------------------------------------------
# risk attribution -- "which position owns my tail?"
# --------------------------------------------------------------------------


def component_es(R: np.ndarray, w: np.ndarray, alpha: float) -> np.ndarray:
    """Euler allocation of portfolio ES to positions.

    ES_i = -w_i * E[R_i | portfolio is in its alpha-tail], and sum(ES_i) == ES
    exactly, because ES is homogeneous of degree 1. Note this is NOT variance
    contribution: a position can be a small share of variance and a large
    share of the tail. That gap is the whole reason to compute it.
    """
    rp = R @ w
    q = np.quantile(rp, alpha)
    tail = rp <= q
    return -w * R[tail].mean(axis=0)


def stress(R: np.ndarray, w: np.ndarray, dates: pd.DatetimeIndex,
           windows: dict) -> pd.DataFrame:
    """Historical scenario replay: apply a crisis window's realised returns to
    today's weights. No model, no assumption -- 'this book, that week'."""
    rp = pd.Series(R @ w, index=dates)
    out = {}
    for name, (a, b) in windows.items():
        seg = rp.loc[a:b]
        if not len(seg):
            continue
        growth = (1 + seg).cumprod()
        out[name] = {"days": len(seg),
                     "cum_pnl": float(growth.iloc[-1] - 1),
                     "worst_day": float(seg.min()),
                     "max_dd": float((growth / growth.cummax() - 1).min())}
    return pd.DataFrame(out).T


CRISES = {
    "GFC 2008":        ("2008-09-01", "2009-03-31"),
    "COVID crash":     ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-01", "2022-12-31"),
}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def report(r: np.ndarray, alpha: float = 0.01, window: int = 1000,
           models=None, es_sim: int = 1000, with_fits: bool = False):
    """window defaults to 1000, not the more common 500: FHS reads the alpha
    quantile straight off the residual pool and 500 days leaves ~5 points
    below the 1% level. See run_report.py `sweep`."""
    if models is None:
        models = [historical, gaussian, ewma,
                  lambda x, a, w: gjr_garch(x, a, w, fhs=False),
                  gjr_garch, evt_pot]
    rows, fits = [], []
    for m in models:
        f = m(r, alpha, window)
        fits.append(f)
        b = f.breach
        k = kupiec(b, alpha)
        c = christoffersen(b, alpha)
        tl = basel_traffic_light(b)
        az = acerbi_szekely(f, alpha, n_sim=es_sim)
        dq = dq_test(f, alpha)
        rows.append({
            "model": f.name,
            "breach": k["breaches"],
            "exp": round(k["expected"], 1),
            "rate": f"{k['rate']:.2%}",
            "p_uc": round(k["p_uc"], 4),
            "p_ind": round(c["p_ind"], 4),
            "p_dq": round(dq["p_dq"], 4),
            "Z1": round(az["Z1"], 3),
            "p_Z1": round(az["p_Z1"], 3),
            "FZ0": round(float(fz0_loss(f, alpha).mean()), 4),
            "Basel": f"{tl['zone']} {tl['multiplier']:.2f}x",
            "avgVaR": f"{f.var.mean():.2%}",
            "avgES": f"{f.es.mean():.2%}",
            "verdict": _verdict(k, c, az, dq),
        })
    tbl = pd.DataFrame(rows).set_index("model")
    return (tbl, fits) if with_fits else tbl


def _verdict(k, c, az, dq) -> str:
    bad = []
    if k["p_uc"] < 0.05:
        bad.append("coverage")
    if c["p_ind"] < 0.05:
        bad.append("clustering")
    if dq["p_dq"] < 0.05:
        bad.append("DQ")
    if not np.isnan(az["p_Z1"]) and az["p_Z1"] < 0.05:
        bad.append("ES too small")
    return "PASS" if not bad else "FAIL: " + ", ".join(bad)


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def _sim_gjr(n, om=2e-6, al=0.05, ga=0.10, be=0.88, nu=6.0, seed=0):
    """Simulate the DGP the models are supposed to handle."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))
    r = np.zeros(n)
    s2 = om / (1 - al - 0.5 * ga - be)
    for t in range(n):
        r[t] = np.sqrt(s2) * z[t]
        s2 = om + (al + (ga if r[t] < 0 else 0.0)) * r[t] ** 2 + be * s2
    return r


def demo():
    a = 0.01

    # 1. closed-form ES: standardised t -> normal as nu -> inf, fatter for low nu
    assert abs(std_t_es(a, 500) - stats.norm.pdf(stats.norm.ppf(a)) / a) < 1e-2
    assert std_t_es(a, 5) > std_t_es(a, 500)

    # 2. analytic ES for t(6) matches a large sample
    zs = stats.t.rvs(6, size=400_000, random_state=1) / np.sqrt(6 / 4)
    assert abs(empirical_es(zs, a)[1] / std_t_es(a, 6) - 1) < 0.03

    # 3. Kupiec: exact coverage -> LR ~ 0; 5x the rate -> reject hard
    b = np.zeros(1000, bool)
    b[:10] = True
    assert kupiec(b, a)["LR_uc"] < 1e-9
    b5 = np.zeros(1000, bool)
    b5[::20] = True
    assert kupiec(b5, a)["p_uc"] < 1e-6

    # 4. Christoffersen: identical breach count, clustered vs spread
    spread = np.zeros(1000, bool)
    spread[::100] = True
    clust = np.zeros(1000, bool)
    clust[500:510] = True
    assert clust.sum() == spread.sum()
    assert christoffersen(clust, a)["p_ind"] < 0.01
    assert christoffersen(spread, a)["p_ind"] > 0.10
    assert christoffersen(spread, a)["LR_cc"] >= christoffersen(spread, a)["LR_ind"]

    # 5. Basel zones and FRTB desk thresholds
    green = np.zeros(300, bool)
    green[-3:] = True
    assert basel_traffic_light(green)["zone"] == "GREEN"
    z12 = np.zeros(300, bool)
    z12[-12:] = True
    assert basel_traffic_light(z12)["zone"] == "RED"
    assert frtb_desk_test(z12, z12)["ima_eligible"] is True     # 12<=12, 12<=30
    z13 = np.zeros(300, bool)
    z13[-13:] = True
    assert frtb_desk_test(z13, z13)["ima_eligible"] is False

    # 6. Gaussian model on genuinely iid normal data must PASS everything.
    #    If this fails, the tests are broken, not the model.
    rng = np.random.default_rng(5)
    riid = rng.standard_normal(4000) * 0.01
    fi = gaussian(riid, a, 500)
    assert kupiec(fi.breach, a)["p_uc"] > 0.05
    assert christoffersen(fi.breach, a)["p_ind"] > 0.05
    assert acerbi_szekely(fi, a, n_sim=400)["p_Z1"] > 0.02

    # 7. GJR filter + MLE recover the DGP on simulated GARCH data
    rg = _sim_gjr(4000)
    th = fit_gjr(rg[:2000])
    assert th[3] > 0.6, th                          # beta persistence found
    assert th[1] + 0.5 * th[2] + th[3] < 0.999      # stationary
    assert 2.1 < th[4] < 25, th                     # fat tails detected

    # 8. the headline claim: on fat-tailed clustered returns the Gaussian
    #    model under-states the tail and GJR-GARCH-FHS does not
    fg = gaussian(rg, a, 750)
    fh = gjr_garch(rg, a, 750, refit_every=63)
    assert fg.breach.sum() > fh.breach.sum(), (fg.breach.sum(), fh.breach.sum())
    assert kupiec(fg.breach, a)["p_uc"] < kupiec(fh.breach, a)["p_uc"]
    assert fh.var.std() > fg.var.std()              # GARCH VaR is responsive

    # 9. component ES is a true Euler decomposition: parts sum to the whole
    R = rng.standard_normal((5000, 3)) * np.array([0.01, 0.02, 0.005])
    w = np.array([0.5, 0.3, 0.2])
    ce = component_es(R, w, 0.05)
    _, es_p = empirical_es(R @ w, 0.05)
    assert abs(ce.sum() - es_p) < 1e-12, (ce.sum(), es_p)

    # 10. GPD tail recovers the analytic quantile of a known fat-tailed law.
    #     t(4) has xi = 1/nu = 0.25, so the fit must find a positive shape.
    zt = stats.t.rvs(4, size=60_000, random_state=3) / np.sqrt(4 / 2)
    ft = _gpd_fit(zt, 0.90)
    assert 0.10 < ft[1] < 0.45, ft                  # xi ~ 1/4
    v_evt, e_evt = _gpd_var_es(ft, 0.005)
    assert abs(v_evt / std_t_ppf(0.005, 4) + 1) < 0.10, (v_evt,)
    assert abs(e_evt / std_t_es(0.005, 4) - 1) < 0.15, (e_evt,)
    assert e_evt > v_evt                             # ES beyond VaR, always
    #     and the payoff: EVT extrapolates past the sample, FHS cannot
    assert _gpd_var_es(ft, 1e-4)[0] > -np.quantile(zt, 1e-4) * 0.7

    # 11. FZ0 is minimised at the true (VaR, ES) -- the property that makes it
    #     a legitimate ranking criterion. Grid search around the truth.
    rn = rng.standard_normal(200_000)
    a5 = 0.05
    v_true, e_true = -stats.norm.ppf(a5), stats.norm.pdf(stats.norm.ppf(a5)) / a5

    def _fz(vv, ee):
        v, e = -vv, -ee
        return float(np.mean((-1 / (a5 * e)) * (rn <= v) * (v - rn)
                             + v / e + np.log(-e) - 1))
    base = _fz(v_true, e_true)
    for dv, de in [(0.15, 0), (-0.15, 0), (0, 0.15), (0, -0.15), (0.1, 0.1)]:
        assert _fz(v_true + dv, e_true + de) > base, (dv, de)

    # 12. Diebold-Mariano: identical losses -> no difference; a shifted loss
    #     series -> detected, and the sign points at the better model.
    l1 = rng.standard_normal(2000)
    assert diebold_mariano(l1, l1.copy())["p_dm"] > 0.99
    dm = diebold_mariano(l1 + 0.2, l1)
    assert dm["p_dm"] < 0.01 and dm["DM"] > 0        # model 1 is worse

    # 13. DQ test: iid hits pass, clustered hits fail
    fq = gaussian(riid, a, 500)
    assert dq_test(fq, a)["p_dq"] > 0.05
    fbad = Forecast("bad", fq.idx, fq.r, fq.var * 0.55, fq.es * 0.55,
                    fq.mu, fq.sigma, fq.draw)
    assert dq_test(fbad, a)["p_dq"] < 0.05

    # 14. FRTB liquidity aggregation: one horizon is a no-op; adding horizons
    #     can only increase the charge
    assert abs(frtb_liquidity_adjusted_es({10: 0.02}) - 0.02) < 1e-12
    agg = frtb_liquidity_adjusted_es({10: 0.02, 20: 0.02, 60: 0.01})
    assert agg > 0.02
    assert abs(agg - np.sqrt(0.02**2 + (0.02 * np.sqrt(1)) ** 2
                             + (0.01 * np.sqrt(4)) ** 2)) < 1e-12

    # 15. stressed ES must find a worse window than the recent one on data
    #     that contains a genuine vol cluster
    sev = stressed_es(rg, 0.025)
    assert sev["stressed_es"] >= sev["current_es"] * 0.99

    print("self-check OK\n")
    print(f"Horse race on simulated GJR-GARCH-t returns, {1 - a:.0%} VaR, "
          f"{len(rg) - 750} out-of-sample days:")
    print(report(rg, alpha=a, window=750, es_sim=400).to_string())


if __name__ == "__main__":
    demo()
