"""Shared sidebar so every page can be opened directly without session plumbing."""

import streamlit as st

import data

DEFAULT_UNIVERSE = (
    "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS, HINDUNILVR.NS, "
    "ITC.NS, LT.NS, SBIN.NS, BHARTIARTL.NS, ASIANPAINT.NS, MARUTI.NS, "
    "SUNPHARMA.NS, TITAN.NS, NESTLEIND.NS"
)
DEFAULT_BENCH = "^NSEI"


@st.cache_data(show_spinner="Loading prices…")
def _load(tickers, bench, start, end, allow_network):
    px, rep = data.get_prices(list(tickers) + [bench], start, end, allow_network)
    return px, rep


def sidebar():
    """Returns (prices, benchmark_series, quality_report, risk_free_rate)."""
    st.sidebar.header("Universe")
    raw = st.sidebar.text_area("Tickers (Yahoo symbols)", DEFAULT_UNIVERSE, height=110)
    bench = st.sidebar.text_input("Benchmark", DEFAULT_BENCH)
    c1, c2 = st.sidebar.columns(2)
    start = c1.date_input("Start", value=__import__("datetime").date(2019, 1, 1))
    end = c2.date_input("End", value=__import__("datetime").date.today())
    rf = st.sidebar.slider("Risk-free rate (annual)", 0.0, 0.12, 0.065, 0.005)
    offline = st.sidebar.checkbox("Offline (cache only)", value=False)
    st.session_state["offline"] = offline

    tickers = tuple(t.strip().upper() for t in raw.replace("\n", ",").split(",") if t.strip())
    if not tickers:
        st.stop()

    px, rep = _load(tickers, bench.strip().upper(), start, end, not offline)
    err = rep.attrs.get("network_error")
    if err:
        st.sidebar.warning(f"Network unavailable, served from cache.\n\n`{err[:120]}`")

    b = bench.strip().upper()
    if px.empty or b not in px.columns:
        st.error(
            f"No usable price history for the benchmark `{b}`. "
            "Check the symbol, widen the date range, or untick Offline."
        )
        st.dataframe(rep, use_container_width=True)
        st.stop()

    dropped = rep.loc[~rep["kept"], "ticker"].tolist()
    if dropped:
        st.sidebar.caption(f"Dropped (insufficient history): {', '.join(dropped)}")

    return px.drop(columns=[b]), px[b], rep, rf
