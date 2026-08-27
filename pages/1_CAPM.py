"""CAPM: beta, alpha, and the security market line."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import common
import portfolio as pf

st.set_page_config(page_title="CAPM", layout="wide")
st.title("CAPM: betas and the security market line")

prices, bench, _, rf = common.sidebar()
tbl = pf.capm(prices, bench, rf)

st.caption(
    "Each row is an OLS regression of the asset's daily excess return on the "
    "benchmark's. **`se_beta` is the column that matters**: a beta of 1.4 with a "
    "standard error of 0.6 is not a beta of 1.4."
)

show = tbl[["beta", "se_beta", "r_squared", "alpha_annual",
            "mean_return_annual", "capm_expected_return", "sml_gap"]]
st.dataframe(
    show.style.format({
        "beta": "{:.3f}", "se_beta": "{:.3f}", "r_squared": "{:.2f}",
        "alpha_annual": "{:.2%}", "mean_return_annual": "{:.2%}",
        "capm_expected_return": "{:.2%}", "sml_gap": "{:+.2%}",
    }).background_gradient(subset=["sml_gap"], cmap="RdYlGn"),
    use_container_width=True,
)

left, right = st.columns([3, 2])

with left:
    st.subheader("Security market line")
    fig, ax = plt.subplots(figsize=(7, 5))
    mkt_prem = bench.pct_change().mean() * 252 - rf
    xs = np.linspace(0, max(1.6, tbl["beta"].max() * 1.15), 50)
    ax.plot(xs, rf + xs * mkt_prem, "k--", label="SML")
    ax.errorbar(tbl["beta"], tbl["mean_return_annual"], xerr=tbl["se_beta"],
                fmt="o", capsize=3, alpha=0.8)
    for name, row in tbl.iterrows():
        ax.annotate(name.replace(".NS", ""), (row["beta"], row["mean_return_annual"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("beta"); ax.set_ylabel("realised annual return")
    ax.axhline(rf, color="grey", lw=0.5); ax.legend()
    st.pyplot(fig)

with right:
    st.subheader("Reading this honestly")
    above = tbl[tbl["sml_gap"] > 0].index.tolist()
    below = tbl[tbl["sml_gap"] <= 0].index.tolist()
    st.markdown(
        f"- **Above the line** ({len(above)}): {', '.join(a.replace('.NS','') for a in above) or 'none'}\n"
        f"- **Below the line** ({len(below)}): {', '.join(b.replace('.NS','') for b in below) or 'none'}\n\n"
        "Sitting above the SML is **not** a buy signal. This is an in-sample fit on "
        "the same history used to compute the average return, so a positive gap is "
        "partly the estimation error of that same sample. Single-factor CAPM also "
        f"leaves most variance unexplained here. Median $R^2$ is "
        f"**{tbl['r_squared'].median():.2f}**, so {1 - tbl['r_squared'].median():.0%} "
        "of each asset's movement is idiosyncratic and invisible to this model.\n\n"
        "The defensible use of this page is the **beta** column as a covariance input, "
        "not the alpha column as a stock pick."
    )
    st.metric("Benchmark equity risk premium (annualised)", f"{mkt_prem:.2%}")
    st.metric("Risk-free rate used", f"{rf:.2%}")
