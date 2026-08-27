"""Can this portfolio's tail risk be forecast, and what does the answer cost?

Every other page decides how to allocate capital. This one takes the resulting
return stream and asks the question a risk desk asks next: is the tail
predictable, and if you size on that prediction, do you land on the risk you
aimed at?
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import common
import data
import portfolio as pf
import tailrisk as tr

st.set_page_config(page_title="Tail Risk", layout="wide")
st.title("Tail risk: forecasting it, backtesting it, sizing on it")

prices, bench, _, rf = common.sidebar()

st.sidebar.header("Tail risk")
method = st.sidebar.selectbox("Portfolio", pf.METHODS,
                              index=pf.METHODS.index("equal_weight"))
conf = st.sidebar.select_slider("Confidence", [0.95, 0.975, 0.99], value=0.975,
                                format_func=lambda x: f"{x:.1%}")
window = st.sidebar.slider("ES estimation window (days)", 250, 1000, 500, 50)
target = st.sidebar.slider("Daily ES target (%)", 0.5, 4.0, 1.6, 0.1)
max_lev = st.sidebar.slider("Max leverage", 1.0, 3.0, 3.0, 0.25)
alpha = 1 - conf


@st.cache_data(show_spinner=False)
def _stream(_prices, _bench, method, rf, caps):
    return pf.walk_forward(_prices, method, rf=rf, market_caps=caps,
                           market_prices=_bench)


@st.cache_data(show_spinner=False)
def _backtest(vals, alpha, window):
    return tr.backtest(pd.Series(vals), alpha=alpha, window=window, es_sim=500)


@st.cache_data(show_spinner=False)
def _fc(vals, model, alpha, window):
    f = tr.forecast(pd.Series(vals), model, alpha, window)
    return f.idx, f.r, f.var, f.es


caps = data.get_market_caps(list(prices.columns),
                            allow_network=not st.session_state.get("offline"))
with st.spinner("Rolling the portfolio forward…"):
    stream = _stream(prices, bench, method, rf, caps)

if len(stream) < window + 60:
    st.error(
        f"`{method}` produced {len(stream)} out-of-sample days, but an ES model "
        f"with a {window}-day window needs more than {window + 60}. Widen the date "
        "range in the sidebar or shorten the estimation window."
    )
    st.stop()

st.caption(
    f"`{method}` walk-forward stream: {len(stream)} out-of-sample days, "
    f"{stream.index[0]:%Y-%m-%d} to {stream.index[-1]:%Y-%m-%d}. Every VaR/ES "
    f"number below is a one-day-ahead forecast made without seeing the day it "
    f"predicts."
)

# ---------------------------------------------------------------- backtests
st.subheader(f"Which risk model actually works on this stream? ({conf:.1%})")
st.caption(
    "Kupiec tests the breach COUNT, Christoffersen tests whether breaches "
    "cluster, DQ tests whether they are predictable from lagged breaches or "
    "from the level of VaR itself, and Acerbi-Szekely tests Expected Shortfall "
    "directly, the only one of the four that looks past the threshold at how "
    "bad the losses beyond it are."
)
with st.spinner("Backtesting five risk models…"):
    bt = _backtest(stream.to_numpy(), alpha, window)

st.dataframe(
    bt[["breach", "exp", "rate", "p_uc", "p_ind", "p_dq", "p_Z1", "Basel",
        "avgVaR", "avgES", "verdict"]],
    use_container_width=True,
)
passed = [i for i in bt.index if bt.loc[i, "verdict"] == "PASS"]
st.markdown(
    f"**{len(passed)} of {len(bt)} models pass every test:** "
    + (", ".join(f"`{p}`" for p in passed) if passed else "_none_.")
    + "  A model that fails here is not usable for limits or sizing, however "
      "good its average VaR looks."
)

# ------------------------------------------------------------ ES targeting
st.subheader("Sizing on the forecast")
st.caption(
    f"Leverage each day = clip(target / forecast ES, 0, {max_lev:g}), using the "
    "forecast made the day before. If the ES model is calibrated the realised "
    "ES lands on the target; if it understates risk, the book is over-levered "
    "and realised risk overshoots. That is what the last column measures."
)

rows, curves = {}, {}
for m in tr.MODELS:
    idx, r, var, es = _fc(stream.to_numpy(), m, alpha, window)
    f = tr.ve.Forecast(m, idx, r, var, es, np.zeros(len(idx)),
                       np.zeros(len(idx)), lambda: np.zeros(len(idx)))
    d = tr.es_target(stream, f, target, max_leverage=max_lev)
    s = tr.summary(d, target, alpha=alpha)
    rows[m] = {"avg leverage": d["leverage"].mean(),
               **s.loc["ES-targeted"].to_dict()}
    curves[m] = d["scaled"]
    unscaled = s.loc["unscaled"]

rows["(unscaled)"] = {"avg leverage": 1.0, **unscaled.to_dict()}
tbl = pd.DataFrame(rows).T
es_col = [c for c in tbl.columns if c.startswith("realised ES")][0]

st.dataframe(
    tbl.style.format({"avg leverage": "{:.2f}", "CAGR": "{:.2%}", "vol": "{:.2%}",
                      "max_dd": "{:.2%}", es_col: "{:.3%}", "vs target": "{:+.1%}"})
       .background_gradient(subset=["vs target"], cmap="RdYlGn_r"),
    use_container_width=True,
)

best = tbl.drop(index="(unscaled)")["vs target"].abs().idxmin()
worst = tbl.drop(index="(unscaled)")["vs target"].abs().idxmax()
st.markdown(
    f"**`{best}` lands closest to the target "
    f"({tbl.loc[best, 'vs target']:+.1%}); `{worst}` is furthest "
    f"({tbl.loc[worst, 'vs target']:+.1%}).** The ordering here tracks the "
    "backtest table above. Models that fail the ES test under-forecast the "
    "tail, so sizing on them over-levers the book. A p-value became basis "
    "points."
)

# ------------------------------------------------------------------- charts
c1, c2 = st.columns(2)

model = st.sidebar.selectbox("Chart this model", list(tr.MODELS),
                             index=list(tr.MODELS).index("GJR-GARCH + FHS"))
idx, r, var, es = _fc(stream.to_numpy(), model, alpha, window)
dates = stream.index[idx]
breach = r < -var

with c1:
    st.markdown(f"**{model}, {conf:.1%} VaR vs realised returns**")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(dates, r * 100, lw=0.4, color="0.75", label="daily return")
    ax.plot(dates, -var * 100, lw=1.0, color="#1f77b4", label="VaR forecast")
    ax.plot(dates, -es * 100, lw=0.9, ls="--", color="#c0392b", label="ES forecast")
    ax.scatter(dates[breach], r[breach] * 100, s=10, color="#c0392b", zorder=3,
               label=f"{int(breach.sum())} breaches")
    ax.set_ylabel("%")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    st.pyplot(fig)

with c2:
    st.markdown("**Growth of 1, ES-targeted vs unscaled**")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    (1 + pd.Series(r, index=dates)).cumprod().plot(
        ax=ax, lw=1.4, color="0.4", label="unscaled")
    for m in ("Gaussian", model):
        cv = curves[m]
        (1 + pd.Series(np.asarray(cv), index=dates)).cumprod().plot(
            ax=ax, lw=1.2, label=f"ES-targeted ({m})")
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 (log)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    st.pyplot(fig)

st.caption(
    "Targeting trades return for control: leverage falls when forecast risk "
    "rises, so the drawdown shrinks by more than the CAGR does. The gap "
    "between the two targeted curves is the cost of using a risk model that "
    "fails its backtest."
)
