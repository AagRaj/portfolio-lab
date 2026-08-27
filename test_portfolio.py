"""Correctness checks. Run: python -m pytest -q

Synthetic prices only, so the suite must pass with the network unplugged.
The look-ahead test is the one that matters; everything else is a guardrail.
"""

import numpy as np
import pandas as pd
import pytest

import data
import portfolio as pf
import riskstats as rstats
import tailrisk as tr


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


# ---------------------------------------------------------------- tail risk

@pytest.fixture(scope="module")
def garch_stream():
    """Returns with genuine volatility clustering, so risk models differ."""
    rng = np.random.default_rng(3)
    n = 2200
    r = np.zeros(n)
    s2 = 1e-4
    z = rng.standard_t(6, n) / np.sqrt(6 / 4)
    for t in range(n):
        r[t] = np.sqrt(s2) * z[t]
        s2 = 2e-6 + (0.05 + (0.10 if r[t] < 0 else 0)) * r[t] ** 2 + 0.88 * s2
    return pd.Series(r, index=pd.bdate_range("2014-01-01", periods=n))


def test_es_forecast_is_out_of_sample(garch_stream):
    """Truncating the input must not change any forecast that survives.

    This is the tail-risk equivalent of the walk-forward look-ahead test: if a
    forecast for day t changed when data after t was removed, it had been
    using it.
    """
    full = tr.forecast(garch_stream, "GJR-GARCH + FHS", 0.025, 500)
    cut = tr.forecast(garch_stream.iloc[:-150], "GJR-GARCH + FHS", 0.025, 500)
    k = len(cut)
    assert np.allclose(full.var[:k], cut.var[:k], atol=1e-12)
    assert np.allclose(full.es[:k], cut.es[:k], atol=1e-12)


def test_es_is_never_below_var(garch_stream):
    for m in tr.MODELS:
        f = tr.forecast(garch_stream, m, 0.025, 500)
        assert (f.es >= f.var).all(), m
        assert np.isfinite(f.es).all() and (f.var > 0).all(), m


def test_short_sample_is_refused(garch_stream):
    with pytest.raises(ValueError):
        tr.forecast(garch_stream.iloc[:300], window=500)


def test_es_target_respects_leverage_cap_and_monotonicity(garch_stream):
    f = tr.forecast(garch_stream, "GJR-GARCH + FHS", 0.025, 500)
    lo = tr.es_target(garch_stream, f, 1.0, max_leverage=2.0)
    hi = tr.es_target(garch_stream, f, 2.0, max_leverage=2.0)
    assert lo["leverage"].max() <= 2.0 + 1e-12
    assert hi["leverage"].mean() > lo["leverage"].mean()
    assert tr.realised_es(hi["scaled"]) > tr.realised_es(lo["scaled"])


def test_a_failing_risk_model_overshoots_the_target_by_more(garch_stream):
    """The point of the whole integration: a risk model that understates the
    tail over-levers the book, so its realised risk misses the target by more
    than a calibrated model's does."""
    good = tr.forecast(garch_stream, "GJR-GARCH + FHS", 0.025, 500)
    bad = tr.forecast(garch_stream, "Gaussian", 0.025, 500)
    dg = tr.es_target(garch_stream, good, 1.5, max_leverage=5.0)
    db = tr.es_target(garch_stream, bad, 1.5, max_leverage=5.0)
    miss_good = abs(tr.realised_es(dg["scaled"]) / 1.5 - 1)
    miss_bad = abs(tr.realised_es(db["scaled"]) / 1.5 - 1)
    assert miss_bad > miss_good, (miss_bad, miss_good)


def test_targeting_stabilises_rolling_risk(garch_stream):
    """ES targeting must reduce the DISPERSION of realised risk over time --
    that is what a static leverage cannot do."""
    f = tr.forecast(garch_stream, "GJR-GARCH + FHS", 0.025, 500)
    d = tr.es_target(garch_stream, f, 1.5, max_leverage=5.0)
    cv = lambda x: x.std() / x.mean()
    raw = pd.Series(d["raw"].to_numpy()).rolling(60).std().dropna()
    scl = pd.Series(d["scaled"].to_numpy()).rolling(60).std().dropna()
    assert cv(scl) < cv(raw)


def test_every_ui_option_changes_the_result(garch_stream):
    """The app's controls must actually control something.

    A dashboard whose sliders do not move the numbers is a screenshot. Each
    block below is one sidebar control on the Tail Risk page.
    """
    s = garch_stream

    # confidence level: a deeper tail means a bigger ES and less leverage
    es = {a: tr.forecast(s, "GJR-GARCH + FHS", a, 500).es.mean()
          for a in (0.05, 0.025, 0.01)}
    assert es[0.01] > es[0.025] > es[0.05]

    # estimation window
    a250 = tr.forecast(s, "GJR-GARCH + FHS", 0.025, 250)
    a500 = tr.forecast(s, "GJR-GARCH + FHS", 0.025, 500)
    assert not np.allclose(a250.es[-100:], a500.es[-100:])

    # risk model
    per_model = {m: tr.forecast(s, m, 0.025, 500).es.mean() for m in tr.MODELS}
    assert len(set(np.round(list(per_model.values()), 8))) == len(tr.MODELS)

    # ES target and leverage cap
    f = tr.forecast(s, "GJR-GARCH + FHS", 0.025, 500)
    levs = [tr.es_target(s, f, t, max_leverage=3.0)["leverage"].mean()
            for t in (1.0, 1.6, 2.5)]
    assert levs[0] < levs[1] < levs[2]
    capped = tr.es_target(s, f, 2.5, max_leverage=1.0)["leverage"]
    assert capped.max() <= 1.0 + 1e-12


# ------------------------------------------------------------------- verdict

def test_verdict_names_the_method_that_actually_won():
    """The bug this guards: the page compared only max_sharpe against
    equal_weight, so after min_volatility and risk_parity joined METHODS it
    announced 'Best Sharpe: min_volatility' and then said 'Equal weighting won'
    in the same breath."""
    s = pd.Series({"equal_weight": 0.86, "max_sharpe": 0.72,
                   "min_volatility": 0.94, "risk_parity": 0.90})
    out = pf.backtest_verdict(s)
    assert "min_volatility" in out
    assert "Equal weighting beat every optimiser" not in out, \
        "claimed equal weight won while min_volatility scored higher"
    assert "beat max-Sharpe" in out


def test_verdict_credits_equal_weight_only_when_it_beats_everything():
    s = pd.Series({"equal_weight": 0.90, "max_sharpe": 0.40, "hrp": 0.55})
    out = pf.backtest_verdict(s)
    assert "Equal weighting beat every optimiser" in out


def test_verdict_when_optimisation_wins_outright():
    s = pd.Series({"equal_weight": 0.30, "max_sharpe": 0.80, "hrp": 0.50})
    out = pf.backtest_verdict(s)
    assert "Optimisation beat equal weighting" in out


def test_verdict_survives_an_all_nan_table():
    assert "No method" in pf.backtest_verdict(
        pd.Series({"equal_weight": np.nan, "max_sharpe": np.nan}))
