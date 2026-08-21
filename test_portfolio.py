"""Correctness checks. Run: python -m pytest -q

Synthetic prices only — the suite must pass with the network unplugged.
The look-ahead test is the one that matters; everything else is a guardrail.
"""

import numpy as np
import pandas as pd
import pytest

import data
import portfolio as pf
import riskstats as rstats


@pytest.fixture(scope="module")
def prices():
    """Four assets with different betas against a common factor, plus the factor."""
    rng = np.random.default_rng(0)
    n = 1500
    idx = pd.bdate_range("2019-01-01", periods=n)
    mkt = rng.normal(0.0004, 0.010, n)
    cols = {}
    for name, beta, idio in [("LOW", 0.4, 0.004), ("MID", 1.0, 0.006),
                             ("HIGH", 1.6, 0.010), ("FLAT", 0.1, 0.003)]:
        r = beta * mkt + rng.normal(0.0001, idio, n)
        cols[name] = 100 * np.exp(np.cumsum(r))
    bench = pd.Series(100 * np.exp(np.cumsum(mkt)), index=idx, name="BENCH")
    return pd.DataFrame(cols, index=idx), bench


# ------------------------------------------------------------------ weights

@pytest.mark.parametrize("method", ["equal_weight", "max_sharpe", "min_volatility",
                                    "hrp", "risk_parity"])
def test_weights_are_a_long_only_portfolio(prices, method):
    w = pf.optimize(prices[0], method=method)
    assert np.isclose(w.sum(), 1.0, atol=1e-6), f"{method} weights sum to {w.sum()}"
    assert (w >= -1e-9).all(), f"{method} produced a short position"
    assert list(w.index) == list(prices[0].columns)


def test_min_volatility_is_not_riskier_than_max_sharpe(prices):
    px = prices[0]
    lo = pf.performance(pf.optimize(px, "min_volatility"), px)["volatility"]
    hi = pf.performance(pf.optimize(px, "max_sharpe"), px)["volatility"]
    assert lo <= hi + 1e-8, f"min-vol {lo:.6f} exceeded max-sharpe {hi:.6f}"


def test_black_litterman_with_no_views_reproduces_the_prior(prices):
    px, bench = prices
    caps = {c: 1e9 * (i + 1) for i, c in enumerate(px.columns)}
    _, S = pf._inputs(px)
    prior = pf.bl_prior(S, caps, bench, list(px.columns), rf=0.05)
    assert not np.allclose(prior.to_numpy(), 0.0), "market caps should give a non-trivial prior"

    from pypfopt import EfficientFrontier
    ef = EfficientFrontier(prior, S)
    ef.max_sharpe(risk_free_rate=0.05)
    direct = pd.Series(ef.clean_weights())

    bl = pf.optimize(px, "black_litterman", rf=0.05, views=None,
                     market_caps=caps, market_prices=bench)
    assert np.allclose(bl.to_numpy(), direct.reindex(bl.index).to_numpy(), atol=1e-6)


def test_views_move_the_posterior(prices):
    px, bench = prices
    caps = {c: 1e9 * (i + 1) for i, c in enumerate(px.columns)}
    base = pf.optimize(px, "black_litterman", views=None, market_caps=caps, market_prices=bench)
    bull = pf.optimize(px, "black_litterman", views={"FLAT": 0.40},
                       market_caps=caps, market_prices=bench)
    assert bull["FLAT"] > base["FLAT"], "a strong bullish view did not raise that weight"


# --------------------------------------------------------------------- CAPM

def test_benchmark_beta_against_itself_is_one(prices):
    bench = prices[1]
    out = pf.capm(bench.to_frame("BENCH"), bench, rf=0.05)
    assert np.isclose(out.loc["BENCH", "beta"], 1.0, atol=1e-9)
    assert np.isclose(out.loc["BENCH", "alpha_annual"], 0.0, atol=1e-9)
    assert np.isclose(out.loc["BENCH", "r_squared"], 1.0, atol=1e-9)


def test_capm_recovers_the_planted_betas(prices):
    out = pf.capm(*prices, rf=0.05)
    for asset, true_beta in [("LOW", 0.4), ("MID", 1.0), ("HIGH", 1.6)]:
        est = out.loc[asset, "beta"]
        se = out.loc[asset, "se_beta"]
        assert abs(est - true_beta) < 4 * se, f"{asset}: {est:.3f} vs {true_beta} (se {se:.3f})"
    assert (out["se_beta"] > 0).all()


# ----------------------------------------------------------------- backtest

def test_walk_forward_never_sees_the_future(prices):
    """The guard inside walk_forward must hold, and must actually be reachable."""
    px = prices[0]
    r = pf.walk_forward(px, "equal_weight", fit_days=252, hold_days=63)
    assert len(r) > 0

    rets = px.pct_change().dropna()
    n_windows = (len(rets) - 252) // 63
    assert len(r) == n_windows * 63, "hold periods do not tile the sample cleanly"
    assert r.index.is_monotonic_increasing and not r.index.duplicated().any()

    # every return must post-date the first window it could have been fitted on
    assert r.index[0] > rets.index[251]


def test_equal_weight_backtest_matches_a_hand_computed_average(prices):
    px = prices[0]
    r = pf.walk_forward(px, "equal_weight", fit_days=252, hold_days=63)
    manual = px.pct_change().dropna().mean(axis=1).loc[r.index]
    assert np.allclose(r.to_numpy(), manual.to_numpy(), atol=1e-12)


def test_metrics_on_a_known_series():
    flat = pd.Series([0.0] * 252)
    m = pf.metrics(flat, rf=0.0)
    assert np.isclose(m["cagr"], 0.0) and np.isclose(m["volatility"], 0.0)
    assert np.isclose(m["max_drawdown"], 0.0)

    down = pd.Series([-0.10] + [0.0] * 100)
    assert pf.metrics(down)["max_drawdown"] <= -0.0999


# -------------------------------------------------------------------- cache

def test_cache_serves_the_same_data_with_the_network_off(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE", tmp_path)
    idx = pd.bdate_range("2022-01-03", periods=300)
    s = pd.Series(np.linspace(100, 150, 300), index=idx)
    data._write_cache("FAKE", s)

    first, rep1 = data.get_prices(["FAKE"], "2022-01-03", idx[-1], allow_network=False)
    second, rep2 = data.get_prices(["FAKE"], "2022-01-03", idx[-1], allow_network=False)
    assert not first.empty
    assert first.equals(second)
    assert rep1.loc[0, "source"] == "cache" and rep1.loc[0, "kept"]


def test_short_history_tickers_are_dropped_but_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE", tmp_path)
    idx = pd.bdate_range("2022-01-03", periods=300)
    data._write_cache("GOOD", pd.Series(np.linspace(100, 150, 300), index=idx))
    data._write_cache("SHORT", pd.Series(np.linspace(10, 12, 30), index=idx[:30]))

    px, rep = data.get_prices(["GOOD", "SHORT"], idx[0], idx[-1], allow_network=False)
    assert list(px.columns) == ["GOOD"]
    assert set(rep["ticker"]) == {"GOOD", "SHORT"}
    assert not rep.set_index("ticker").loc["SHORT", "kept"]


# ------------------------------------------------- risk parity & significance

def test_risk_parity_equalises_risk_contributions(prices):
    """The defining property: every asset contributes the same share of vol.

    Equal WEIGHT does not do this, which is the whole reason ERC exists -- so
    the test also asserts the two portfolios are actually different.
    """
    px = prices[0]
    _, S = pf._inputs(px)
    w = pf.optimize(px, "risk_parity").to_numpy()
    rc = rstats.risk_contributions(w, S.to_numpy())

    assert np.isclose(w.sum(), 1.0) and (w > 0).all()
    assert rc.std() / rc.mean() < 1e-5, f"risk contributions not equal: {rc}"
    assert np.isclose(rc.sum(), np.sqrt(w @ S.to_numpy() @ w))   # Euler identity

    eq = np.full(len(w), 1 / len(w))
    rc_eq = rstats.risk_contributions(eq, S.to_numpy())
    assert rc_eq.std() / rc_eq.mean() > rc.std() / rc.mean()
    # HIGH beta / high vol asset must get less capital than the low-vol one
    assert w[list(px.columns).index("HIGH")] < w[list(px.columns).index("FLAT")]


def test_uncorrelated_risk_parity_is_inverse_volatility():
    """With a diagonal covariance ERC has a closed form: w_i proportional to
    1/sigma_i. Anything else means the solver is wrong."""
    S = np.diag([0.04, 0.01, 0.0025, 0.16])
    w = rstats.erc_weights(S)
    inv = 1 / np.sqrt(np.diag(S))
    assert np.allclose(w, inv / inv.sum(), atol=1e-6)


def test_deflation_never_raises_the_sharpe_verdict(prices):
    """DSR <= PSR always: pricing in the number of trials can only make a
    result less significant, never more."""
    rng = np.random.default_rng(1)
    trials = [rng.normal(0, 0.03) for _ in range(6)]
    r = rng.standard_normal(1200) * 0.01 + 0.0005
    d = rstats.deflated_sharpe(r, trials)
    assert d["SR0"] > 0
    assert d["DSR"] <= d["PSR"] + 1e-12


def test_best_of_many_random_strategies_fails_deflation():
    """The property the whole thing exists for: pick the best of 40 coin
    flips and it looks significant on its own, but must not survive DSR."""
    rng = np.random.default_rng(7)
    cands = [rng.standard_normal(1500) * 0.01 for _ in range(40)]
    srs = [rstats.per_period_sharpe(c) for c in cands]
    best = cands[int(np.argmax(srs))]
    d = rstats.deflated_sharpe(best, srs)
    assert d["PSR"] > 0.90 and d["DSR"] < 0.95


def test_sharpe_diff_test_identifies_no_difference(prices):
    a = np.asarray(prices[0]["MID"].pct_change().dropna())
    t = rstats.sharpe_diff_test(a, a.copy(), n_boot=400)
    assert t["p_boot"] > 0.9 and abs(t["diff"]) < 1e-12


def test_compare_reports_significance_columns(prices):
    """compare() must carry the significance layer, not just the metrics --
    a table of Sharpes with no p-values is what this repo is arguing against."""
    out = pf.compare(prices[0], methods=["equal_weight", "risk_parity"],
                     fit_days=250, hold_days=63, n_boot=200)
    for col in ("PSR", "DSR", "p_vs_equal_weight"):
        assert col in out.columns, out.columns.tolist()
    assert out.loc["equal_weight", "p_vs_equal_weight"] > 0.9   # vs itself
    assert (out["DSR"] <= out["PSR"] + 1e-12).all()
