"""CAPM, portfolio construction and walk-forward evaluation.

Every optimizer here is PyPortfolioOpt. Nothing in this file solves anything
itself; it only assembles inputs, picks the objective, and reports results in a
shape the pages can render.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pypfopt import EfficientFrontier, black_litterman, expected_returns, risk_models
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.hierarchical_portfolio import HRPOpt

TRADING_DAYS = 252
METHODS = ["equal_weight", "max_sharpe", "min_volatility", "black_litterman", "hrp"]


# --------------------------------------------------------------------------- CAPM

def capm(prices: pd.DataFrame, bench: pd.Series, rf: float = 0.05) -> pd.DataFrame:
    """Per-asset OLS of excess return on benchmark excess return.

    Returns beta, annualised alpha, R^2 and the standard error of beta. The
    standard error is the point of the table: a beta of 1.4 with se 0.6 is not a
    beta of 1.4.
    """
    r = prices.pct_change().dropna()
    rb = bench.pct_change().reindex(r.index).dropna()
    r = r.loc[rb.index]
    rf_d = rf / TRADING_DAYS
    x = (rb - rf_d).to_numpy()

    out = []
    for col in r.columns:
        y = (r[col] - rf_d).to_numpy()
        fit = stats.linregress(x, y)
        out.append({
            "asset": col,
            "beta": fit.slope,
            "se_beta": fit.stderr,
            "alpha_annual": fit.intercept * TRADING_DAYS,
            "r_squared": fit.rvalue ** 2,
            "mean_return_annual": r[col].mean() * TRADING_DAYS,
            "volatility_annual": r[col].std() * np.sqrt(TRADING_DAYS),
        })
    df = pd.DataFrame(out).set_index("asset")
    # Expected return under CAPM, for the SML plot
    mkt_prem = rb.mean() * TRADING_DAYS - rf
    df["capm_expected_return"] = rf + df["beta"] * mkt_prem
    df["sml_gap"] = df["mean_return_annual"] - df["capm_expected_return"]
    return df.sort_values("beta")


# ------------------------------------------------------------------- optimization

def _inputs(prices: pd.DataFrame):
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    return mu, S


def optimize(
    prices: pd.DataFrame,
    method: str = "max_sharpe",
    rf: float = 0.05,
    views: dict[str, float] | None = None,
    view_confidences: list[float] | None = None,
    market_caps: dict[str, float] | None = None,
    market_prices: pd.Series | None = None,
) -> pd.Series:
    """Weights for one objective. Long-only, fully invested."""
    assets = list(prices.columns)
    if method == "equal_weight":
        return pd.Series(1.0 / len(assets), index=assets)

    mu, S = _inputs(prices)

    if method == "hrp":
        w = HRPOpt(prices.pct_change().dropna()).optimize()
        return pd.Series(w).reindex(assets).fillna(0.0)

    if method == "black_litterman":
        prior = bl_prior(S, market_caps, market_prices, assets, rf)
        if not views:
            # No views => posterior collapses to the prior. Optimise on the prior
            # directly so the "zero views == prior" invariant is exact.
            mu = prior
        else:
            kwargs = {}
            if view_confidences:
                kwargs = {"omega": "idzorek", "view_confidences": view_confidences}
            bl = BlackLittermanModel(S, pi=prior, absolute_views=views, **kwargs)
            mu = bl.bl_returns()
            S = bl.bl_cov()
        method = "max_sharpe"

    ef = EfficientFrontier(mu, S)
    try:
        if method == "min_volatility":
            ef.min_volatility()
        else:
            ef.max_sharpe(risk_free_rate=rf)
    except Exception:
        # max_sharpe is infeasible when every expected return sits below rf.
        ef = EfficientFrontier(mu, S)
        ef.min_volatility()
    return pd.Series(ef.clean_weights()).reindex(assets).fillna(0.0)


def bl_prior(S, market_caps, market_prices, assets, rf) -> pd.Series:
    """Market-implied prior returns, or an equal prior if caps are unavailable.

    The caller is expected to surface which one was used — a Black-Litterman
    result built on an equal prior is not a market-equilibrium result.
    """
    if market_caps and market_prices is not None and len(market_caps) == len(assets):
        try:
            delta = black_litterman.market_implied_risk_aversion(market_prices)
            return black_litterman.market_implied_prior_returns(market_caps, delta, S)
        except Exception:
            pass
    return pd.Series(0.0, index=assets)


def performance(weights: pd.Series, prices: pd.DataFrame, rf: float = 0.05) -> dict:
    """In-sample expected return / volatility / Sharpe for a weight vector."""
    mu, S = _inputs(prices)
    w = weights.reindex(mu.index).fillna(0.0).to_numpy()
    ret = float(w @ mu.to_numpy())
    vol = float(np.sqrt(w @ S.to_numpy() @ w))
    return {"return": ret, "volatility": vol, "sharpe": (ret - rf) / vol if vol else np.nan}


def allocate(weights: pd.Series, prices: pd.DataFrame, capital: float) -> tuple[dict, float]:
    """Whole-share allocation for a given capital amount."""
    latest = get_latest_prices(prices)
    da = DiscreteAllocation(weights[weights > 0].to_dict(), latest, total_portfolio_value=capital)
    return da.greedy_portfolio()


# ---------------------------------------------------------------------- backtest

def metrics(returns: pd.Series, rf: float = 0.05) -> dict:
    r = returns.dropna()
    if r.empty:
        return dict.fromkeys(["cagr", "volatility", "sharpe", "max_drawdown", "cvar_95"], np.nan)
    # Seed the curve at 1.0 so a drawdown that starts on day one is counted;
    # without it, cummax() starts below par and the first loss is invisible.
    curve = pd.concat([pd.Series([1.0]), (1 + r).cumprod()], ignore_index=True)
    years = len(r) / TRADING_DAYS
    cagr = curve.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(TRADING_DAYS)
    dd = (curve / curve.cummax() - 1).min()
    tail = r[r <= r.quantile(0.05)]
    return {
        "cagr": float(cagr),
        "volatility": float(vol),
        "sharpe": float((cagr - rf) / vol) if vol else np.nan,
        "max_drawdown": float(dd),
        "cvar_95": float(tail.mean()) if len(tail) else np.nan,
    }


def walk_forward(
    prices: pd.DataFrame,
    method: str,
    fit_days: int = 504,
    hold_days: int = 63,
    rf: float = 0.05,
    **opt_kwargs,
) -> pd.Series:
    """Out-of-sample daily returns: fit on a trailing window, hold, roll forward.

    The only thing this function must never do is let the fit window see the hold
    window. The assert below is the guard, and `test_portfolio.py` checks it.
    """
    rets = prices.pct_change().dropna()
    out = []
    start = fit_days
    while start + hold_days <= len(rets):
        fit_slice = rets.index[start - fit_days: start]
        hold_slice = rets.index[start: start + hold_days]
        assert fit_slice[-1] < hold_slice[0], "look-ahead: fit window overlaps hold window"

        w = optimize(prices.loc[fit_slice], method=method, rf=rf, **opt_kwargs)
        held = rets.loc[hold_slice, w.index] @ w.to_numpy()
        out.append(held)
        start += hold_days

    return pd.concat(out) if out else pd.Series(dtype=float)


def compare(prices: pd.DataFrame, methods=None, rf=0.05, **kw) -> pd.DataFrame:
    methods = methods or METHODS
    rows = {}
    for m in methods:
        try:
            rows[m] = metrics(walk_forward(prices, m, rf=rf, **kw), rf=rf)
        except Exception as exc:
            rows[m] = {"cagr": np.nan, "volatility": np.nan, "sharpe": np.nan,
                       "max_drawdown": np.nan, "cvar_95": np.nan, "error": str(exc)[:80]}
    return pd.DataFrame(rows).T
