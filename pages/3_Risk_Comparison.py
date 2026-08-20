"""Efficient frontier, and the walk-forward test that decides which method survives."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import common
import data
import portfolio as pf

st.set_page_config(page_title="Risk Comparison", layout="wide")
st.title("Risk comparison — in sample vs out of sample")

prices, bench, _, rf = common.sidebar()

st.sidebar.header("Backtest")
fit_days = st.sidebar.slider("Fit window (days)", 126, 1008, 504, 63)
hold_days = st.sidebar.slider("Hold window (days)", 21, 252, 63, 21)

left, right = st.columns([2, 3])

with left:
    st.subheader("Efficient frontier")
    mu, S = pf._inputs(prices)
    from pypfopt import EfficientFrontier

    vols, rets = [], []
    lo = pf.performance(pf.optimize(prices, "min_volatility", rf), prices, rf)["return"]
    for target in np.linspace(lo, mu.max() * 0.99, 25):
        try:
            ef = EfficientFrontier(mu, S)
            ef.efficient_return(target_return=target)
            r, v, _ = ef.portfolio_performance(risk_free_rate=rf)
            rets.append(r); vols.append(v)
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.plot(vols, rets, "b-", lw=1.5, label="efficient frontier")
    ax.scatter(prices.pct_change().std() * np.sqrt(252),
               prices.pct_change().mean() * 252, s=16, c="grey", label="single assets")
    for m, c in [("max_sharpe", "red"), ("min_volatility", "green"), ("equal_weight", "orange")]:
        p = pf.performance(pf.optimize(prices, m, rf), prices, rf)
        ax.scatter(p["volatility"], p["return"], marker="*", s=180, c=c, label=m, zorder=5)
    ax.set_xlabel("annual volatility"); ax.set_ylabel("annual return")
    ax.legend(fontsize=7)
    st.pyplot(fig)

with right:
    st.subheader("Walk-forward, out of sample")
    st.caption(
        f"Fit on a trailing {fit_days}-day window, hold {hold_days} days, roll forward. "
        "The fit window never overlaps the hold window — `walk_forward` asserts it and "
        "`test_portfolio.py` checks the assert is reachable."
    )
    caps = data.get_market_caps(list(prices.columns),
                                 allow_network=not st.session_state.get("offline"))
    if not caps:
        st.info(
            "Market caps unavailable, so the Black-Litterman arm runs on a flat prior "
            "and collapses onto min-volatility. Identical rows below are that, not a bug."
        )
    with st.spinner("Rolling…"):
        table = pf.compare(prices, rf=rf, fit_days=fit_days, hold_days=hold_days,
                           market_caps=caps, market_prices=bench)

    st.dataframe(
        table.style.format({"cagr": "{:.2%}", "volatility": "{:.2%}", "sharpe": "{:.2f}",
                            "max_drawdown": "{:.2%}", "cvar_95": "{:.2%}"})
             .background_gradient(subset=["sharpe"], cmap="RdYlGn"),
        use_container_width=True,
    )

    if "sharpe" in table and table["sharpe"].notna().any():
        best = table["sharpe"].idxmax()
        ms = table["sharpe"].get("max_sharpe", float("nan"))
        ew = table["sharpe"].get("equal_weight", float("nan"))
        st.markdown(
            f"**Best out-of-sample Sharpe: `{best}`.** In-sample max-Sharpe scored "
            f"{ms:.2f} here versus {ew:.2f} for naive equal weighting."
            + ("  Equal weighting won, which is the usual result and the reason "
               "estimation error, not optimisation, is the binding constraint in "
               "mean-variance investing." if ew >= ms else
               "  Optimisation beat equal weighting on this sample — worth checking "
               "whether it survives a different fit window before believing it.")
        )

st.subheader("Equity curves")
fig, ax = plt.subplots(figsize=(11, 3.6))
for m in pf.METHODS:
    try:
        r = pf.walk_forward(prices, m, fit_days=fit_days, hold_days=hold_days, rf=rf,
                            market_caps=caps, market_prices=bench)
        if len(r):
            (1 + r).cumprod().plot(ax=ax, lw=1.2, label=m)
    except Exception:
        continue
ax.set_ylabel("growth of 1 (out of sample)"); ax.legend(fontsize=8)
st.pyplot(fig)
