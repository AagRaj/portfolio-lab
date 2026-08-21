"""Conditional tail-risk forecasting and ES-targeted sizing for a portfolio.

The rest of this repo answers "how should capital be allocated". This module
answers the question that comes next on a real desk: **given the resulting
return stream, can its tail risk actually be forecast, and does that forecast
survive contact with the data?**

Those are different questions and they have different answers. A portfolio can
have the lowest realised volatility in the table and still have a tail nobody
can predict, which makes it impossible to size or set a limit on.

The forecasting models and backtests come from `var_engine.py` (see the
companion VaR/ES engine repo): GJR-GARCH volatility with Filtered Historical
Simulation or a Generalised Pareto tail, backtested with Kupiec,
Christoffersen, Engle-Manganelli DQ and Acerbi-Szekely.

The payoff is `es_target`. Sizing to a risk target is only as good as the risk
forecast: understate ES and you over-lever, and the realised risk lands above
the target you set. That turns an abstract backtest p-value into basis points
of overshoot, which is the form the question takes when someone senior asks it.

Run `python tailrisk.py` for the self-check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import var_engine as ve

# Ordered worst-to-best on the evidence in the companion repo, so the table
# reads as an argument rather than an alphabetical list.
MODELS = {
    "Gaussian": lambda r, a, w: ve.gaussian(r, a, w),
    "Historical": lambda r, a, w: ve.historical(r, a, w),
    "EWMA (RiskMetrics)": lambda r, a, w: ve.ewma(r, a, w),
    "GJR-GARCH + FHS": lambda r, a, w: ve.gjr_garch(r, a, w),
    "GJR-GARCH + EVT": lambda r, a, w: ve.evt_pot(r, a, w),
}


def forecast(returns: pd.Series, model: str = "GJR-GARCH + EVT",
             alpha: float = 0.025, window: int = 500) -> ve.Forecast:
    """One-day-ahead VaR/ES forecasts for a portfolio return series.

    Every value is out-of-sample: the forecast for day t uses returns up to
    t-1 only. That is enforced inside var_engine, not here.
    """
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; have {list(MODELS)}")
    r = np.asarray(returns, float)
    if len(r) < window + 60:
        raise ValueError(f"need > {window + 60} observations, got {len(r)}")
    return MODELS[model](r, alpha, window)


def backtest(returns: pd.Series, alpha: float = 0.025, window: int = 500,
             models=None, es_sim: int = 600) -> pd.DataFrame:
    """Run every model over the same series and score them all."""
    names = list(models or MODELS)
    return ve.report(np.asarray(returns, float), alpha=alpha, window=window,
                     models=[MODELS[n] for n in names], es_sim=es_sim)


def es_target(returns: pd.Series, fc: ve.Forecast, target: float,
              max_leverage: float = 3.0) -> pd.DataFrame:
    """Scale exposure so that FORECAST ES equals `target` (daily, percent).

        leverage_t = clip(target / ES_t, 0, max_leverage)
        return_t   = leverage_t * r_t

    ES_t is the forecast made at t-1, so the leverage applied on day t is
    knowable at t-1. No look-ahead, asserted in demo().

    Read the output like this: if the ES model is calibrated, the REALISED ES
    of the scaled series lands on the target. If the model understates ES, the
    leverage is too high and realised risk overshoots -- the backtest failure
    shows up as a number in basis points rather than a p-value.
    """
    if target <= 0:
        raise ValueError("target must be positive")
    idx = pd.Index(returns.index[fc.idx]) if hasattr(returns, "index") else None
    lev = np.clip(target / (fc.es * 100.0), 0.0, max_leverage)
    scaled = lev * fc.r
    out = pd.DataFrame({"raw": fc.r, "leverage": lev, "scaled": scaled,
                        "es_forecast_pct": fc.es * 100.0})
    if idx is not None:
        out.index = idx
    return out


def realised_es(r, alpha: float = 0.025) -> float:
    """Realised ES of a return series, as a positive percent magnitude."""
    r = np.asarray(r, float)
    return float(ve.empirical_es(r, alpha)[1] * 100.0)


def summary(df: pd.DataFrame, target: float, alpha: float = 0.025,
            trading_days: int = 252) -> pd.DataFrame:
    """Before/after table for an ES-targeted overlay."""
    rows = {}
    for name, col in [("unscaled", "raw"), ("ES-targeted", "scaled")]:
        r = df[col].to_numpy()
        curve = np.concatenate([[1.0], np.cumprod(1 + r)])
        yrs = len(r) / trading_days
        es = realised_es(r, alpha)
        rows[name] = {
            "CAGR": curve[-1] ** (1 / yrs) - 1,
            "vol": r.std(ddof=1) * np.sqrt(trading_days),
            "max_dd": (curve / np.maximum.accumulate(curve) - 1).min(),
            f"realised ES {1 - alpha:.1%}": es / 100.0,
            "vs target": es / target - 1.0,
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------

def demo():
    rng = np.random.default_rng(0)
    # A GARCH-like series with real vol clustering, so the models differ.
    n = 3000
    r = np.zeros(n)
    s2 = 1e-4
    z = rng.standard_t(6, n) / np.sqrt(6 / 4)
    for t in range(n):
        r[t] = np.sqrt(s2) * z[t]
        s2 = 2e-6 + (0.05 + (0.10 if r[t] < 0 else 0)) * r[t] ** 2 + 0.88 * s2
    s = pd.Series(r, index=pd.bdate_range("2012-01-02", periods=n))

    # 1. every model produces aligned, finite, positive forecasts
    for m in MODELS:
        f = forecast(s, m, alpha=0.025, window=500)
        assert len(f) == n - 500
        assert np.all(np.isfinite(f.var)) and np.all(f.var > 0)
        assert np.all(f.es >= f.var), m          # ES is never below VaR

    # 2. too-short input is refused rather than silently returning nonsense
    try:
        forecast(s.iloc[:400], window=500)
        raise SystemExit("should have refused a short sample")
    except ValueError:
        pass

    f = forecast(s, "GJR-GARCH + EVT", 0.025, 500)

    # 3. the overlay uses no future information: leverage on day t is a
    #    function of the ES forecast made at t-1, so truncating the input
    #    must not change any leverage value that survives.
    d_full = es_target(s, f, target=1.5)
    f_cut = forecast(s.iloc[:-200], "GJR-GARCH + EVT", 0.025, 500)
    d_cut = es_target(s.iloc[:-200], f_cut, target=1.5)
    k = len(d_cut)
    assert np.allclose(d_full["leverage"].to_numpy()[:k],
                       d_cut["leverage"].to_numpy(), atol=1e-12)

    # 4. a higher target must mean more leverage and more realised risk
    lo = es_target(s, f, target=1.0)
    hi = es_target(s, f, target=2.0)
    assert hi["leverage"].mean() > lo["leverage"].mean()
    assert realised_es(hi["scaled"]) > realised_es(lo["scaled"])

    # 5. the leverage cap binds and is never exceeded
    cap = es_target(s, f, target=99.0, max_leverage=2.0)
    assert cap["leverage"].max() <= 2.0 + 1e-12
    assert np.isclose(cap["leverage"].min(), 2.0)      # target so high it pins

    # 6. targeting works: realised ES lands near the target for a calibrated
    #    model, and the scaled series is less volatile in vol-clustered
    #    periods than the raw one is.
    d = es_target(s, f, target=1.5, max_leverage=5.0)
    hit = realised_es(d["scaled"])
    assert abs(hit / 1.5 - 1) < 0.35, hit
    #    ES targeting must reduce the DISPERSION of rolling risk -- that is
    #    the entire point, and it is what a flat leverage cannot do.
    roll_raw = pd.Series(d["raw"]).rolling(60).std().dropna()
    roll_scl = pd.Series(d["scaled"]).rolling(60).std().dropna()
    assert roll_scl.std() / roll_scl.mean() < roll_raw.std() / roll_raw.mean()

    # 7. an ES model that understates risk over-levers and overshoots the
    #    target by more than a calibrated one. This is the whole argument for
    #    backtesting the risk model before sizing on it.
    f_bad = forecast(s, "Gaussian", 0.025, 500)
    d_bad = es_target(s, f_bad, target=1.5, max_leverage=5.0)
    over_bad = abs(realised_es(d_bad["scaled"]) / 1.5 - 1)
    over_good = abs(hit / 1.5 - 1)
    assert over_bad > over_good, (over_bad, over_good)

    # 8. summary is internally consistent
    sm = summary(d, target=1.5)
    assert list(sm.index) == ["unscaled", "ES-targeted"]
    assert abs(sm.loc["ES-targeted", "realised ES 97.5%"] * 100 / 1.5 - 1
               - sm.loc["ES-targeted", "vs target"]) < 1e-9

    print("tailrisk self-check OK")
    print(f"  calibrated (EVT) overshoot {over_good:+.1%}   "
          f"Gaussian overshoot {over_bad:+.1%}")


if __name__ == "__main__":
    demo()
