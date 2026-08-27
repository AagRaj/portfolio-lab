"""What to buy, and how much of it: max-Sharpe, min-vol, and Black-Litterman views."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import common
import data
import portfolio as pf

st.set_page_config(page_title="Allocation", layout="wide")
st.title("Optimal allocation")

prices, bench, _, rf = common.sidebar()
assets = list(prices.columns)

tab_mv, tab_bl = st.tabs(["Mean-variance", "Black-Litterman (your views)"])

with tab_mv:
    weights = {m: pf.optimize(prices, m, rf) for m in ["max_sharpe", "min_volatility", "equal_weight"]}
    wdf = pd.DataFrame(weights)

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Weights")
        st.dataframe(wdf.style.format("{:.2%}").background_gradient(cmap="Blues"),
                     use_container_width=True)
    with right:
        st.subheader("In-sample performance")
        perf = pd.DataFrame({m: pf.performance(w, prices, rf) for m, w in weights.items()}).T
        st.dataframe(perf.style.format({"return": "{:.2%}", "volatility": "{:.2%}",
                                        "sharpe": "{:.2f}"}), use_container_width=True)
        st.warning(
            "These are **in-sample** numbers: the same prices produced both the "
            "expected returns and the weights. Max-Sharpe will always look best here "
            "and frequently is not. The Risk Comparison page runs the out-of-sample "
            "version, which is the one to quote."
        )

        fig, ax = plt.subplots(figsize=(7, 3.2))
        wdf.plot.bar(ax=ax, width=0.8)
        ax.set_ylabel("weight"); ax.tick_params(axis="x", labelsize=7, rotation=90)
        ax.legend(fontsize=7)
        st.pyplot(fig)

    st.subheader("Whole-share allocation")
    c1, c2 = st.columns([1, 3])
    capital = c1.number_input("Capital", 10_000, 100_000_000, 500_000, step=10_000)
    method = c1.selectbox("Portfolio", list(weights))
    alloc, leftover = pf.allocate(weights[method], prices, capital)
    c2.dataframe(pd.Series(alloc, name="shares").to_frame(), use_container_width=True)
    c2.metric("Uninvested cash", f"{leftover:,.2f}")

with tab_bl:
    st.caption(
        "Black-Litterman starts from the returns the market's own capitalisation "
        "weights imply, then tilts toward your views in proportion to how confident "
        "you say you are. It exists because plain Markowitz on historical means "
        "produces extreme, unstable weights, which is the limitation this page demonstrates."
    )

    caps = data.get_market_caps(assets, allow_network=not st.session_state.get("offline"))
    if caps:
        st.success("Prior: market-capitalisation implied equilibrium returns.")
    else:
        st.warning(
            "Market caps unavailable from the data provider, so the prior is flat "
            "(zeros). The result below is still Black-Litterman machinery but it is "
            "**not** a market-equilibrium prior. Do not present it as one."
        )

    picks = st.multiselect("Assets you have a view on", assets, max_selections=4)
    views, confs = {}, []
    if picks:
        cols = st.columns(len(picks))
        for col, a in zip(cols, picks):
            views[a] = col.number_input(f"{a}\nexpected annual return", -0.5, 1.0, 0.15, 0.01,
                                        key=f"v_{a}")
            confs.append(col.slider("confidence", 0.05, 0.95, 0.50, 0.05, key=f"c_{a}"))

    prior_w = pf.optimize(prices, "black_litterman", rf, views=None,
                          market_caps=caps, market_prices=bench)
    post_w = pf.optimize(prices, "black_litterman", rf, views=views or None,
                         view_confidences=confs or None,
                         market_caps=caps, market_prices=bench)

    cmp = pd.DataFrame({"prior (no views)": prior_w, "posterior (your views)": post_w})
    cmp["change"] = cmp["posterior (your views)"] - cmp["prior (no views)"]

    left, right = st.columns([2, 3])
    left.dataframe(cmp.style.format("{:+.2%}").background_gradient(subset=["change"], cmap="RdYlGn"),
                   use_container_width=True)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    cmp[["prior (no views)", "posterior (your views)"]].plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("weight"); ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.legend(fontsize=8)
    right.pyplot(fig)

    if not picks:
        st.info("With no views entered, the posterior is the prior. That identity is "
                "asserted in `test_portfolio.py`.")
