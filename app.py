"""Portfolio Lab — universe and data quality. Other pages are in pages/."""

import matplotlib.pyplot as plt
import streamlit as st

import common

st.set_page_config(page_title="Portfolio Lab", layout="wide")
st.title("Portfolio Lab")
st.caption(
    "Mean-variance, CAPM, Black-Litterman and hierarchical risk parity on live "
    "market data. Pages in the left rail; the sidebar controls every page."
)

prices, bench, report, rf = common.sidebar()

st.subheader("Data quality")
st.caption(
    f"{len(prices.columns)} assets kept, {len(prices)} aligned trading days. "
    "`source` shows what came from the parquet cache versus the network."
)
st.dataframe(report, use_container_width=True, hide_index=True)

left, right = st.columns(2)

with left:
    st.subheader("Normalised price history")
    fig, ax = plt.subplots(figsize=(6, 4))
    (prices / prices.iloc[0]).plot(ax=ax, linewidth=0.9)
    (bench / bench.iloc[0]).plot(ax=ax, linewidth=2.0, color="black", label="benchmark")
    ax.set_ylabel("growth of 1")
    ax.legend(fontsize=6, ncol=2)
    st.pyplot(fig)

    st.subheader("Annualised return vs volatility")
    r = prices.pct_change().dropna()
    summary = (r.mean() * 252).to_frame("return").join((r.std() * (252 ** 0.5)).to_frame("volatility"))
    st.dataframe(summary.style.format("{:.2%}"), use_container_width=True)

with right:
    st.subheader("Correlation")
    r = prices.pct_change().dropna()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(r.corr(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(r.columns)), r.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(r.columns)), r.columns, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig)
    st.caption(
        "Mean pairwise correlation "
        f"{r.corr().values[~__import__('numpy').eye(len(r.columns), dtype=bool)].mean():.2f}. "
        "Diversification benefit shrinks as this rises — the number to watch before "
        "trusting any efficient-frontier result below."
    )
