# Portfolio Lab

A four-page web app for building and stress-testing equity portfolios: CAPM betas,
mean-variance optimisation, Black-Litterman views, and hierarchical risk parity —
all evaluated **out of sample**, because in sample they all look good.

Every optimiser is [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt).
Nothing here re-implements a solver. What this repo adds is the data layer, the
walk-forward harness, and the tests that keep both honest.

## The result the app exists to show

Fifteen NSE large caps plus `^NSEI`, 2019-01-01 to 2026-08-19, fit on a trailing
504-day window, held 63 days, rolled forward:

| Method | CAGR | Volatility | Sharpe | Max drawdown | CVaR 95% |
|---|---|---|---|---|---|
| equal weight | 13.9% | 12.9% | **0.57** | **-16.5%** | -1.80% |
| Black-Litterman (market prior) | **17.2%** | 18.8% | 0.57 | -20.7% | -2.60% |
| HRP | 12.9% | 12.5% | 0.52 | -18.0% | -1.71% |
| min volatility | 11.6% | 12.3% | 0.42 | -21.0% | **-1.65%** |
| max Sharpe | 10.5% | 16.4% | 0.24 | -22.5% | -2.30% |

Optimising for Sharpe produced the *worst* out-of-sample Sharpe, the highest
volatility of the four optimisers, and the deepest drawdown — beaten by dividing
capital equally and doing nothing. The binding constraint in mean-variance
investing is estimation error in the expected-return vector, not the
optimisation.

That is exactly what the other two rows are for. Black-Litterman replaces the
historical mean with a market-capitalisation-implied prior and recovers the
return that max-Sharpe threw away (17.2% vs 10.5%) at the same risk-adjusted
Sharpe as equal weighting. HRP never inverts the covariance matrix at all and
lands between the two with the second-shallowest drawdown. Both are answers to
"the limitations of the Markowitz model", which is why they sit on the same page
as the naive benchmark rather than in a section of their own.

One caveat worth stating before quoting any of this: the market-cap prior uses
*today's* capitalisations for every historical rebalance, so the Black-Litterman
row carries a mild look-ahead in the prior even though the price data does not.
Fixing it needs a historical market-cap series, which the free API does not
provide.

## Pages

| Page | What it answers |
|---|---|
| `app.py` — **Universe & Data** | What did we actually load? Rows, NaNs, date coverage, cache-vs-network per ticker, correlation structure. |
| `1_CAPM` | Beta, alpha, R² and **the standard error of beta** per asset, plotted against the security market line. |
| `2_Allocation` | Max-Sharpe / min-vol weights, a Black-Litterman tab where you enter views with confidences and see prior vs posterior, and whole-share allocation for a given capital. |
| `3_Risk_Comparison` | Efficient frontier, then the walk-forward table above and out-of-sample equity curves. |

## Data layer

`data.py` caches adjusted closes to `cache/{ticker}.parquet`. The cache is
authoritative; the network is only touched when the cache cannot cover the
requested window, and if the fetch fails the app serves cached data with a visible
warning rather than crashing. Tickers with under 120 observations are dropped from
the optimiser but stay in the quality report marked `kept=False`, so nothing
disappears silently. There is an **Offline** switch in the sidebar that forbids
network access entirely — that is also how the test suite runs.

## Tests

```bash
python -m pytest -q        # 14 tests, no network required
```

The suite runs on synthetic prices with planted betas. It checks that weights form
a long-only fully-invested portfolio, that min-volatility is never riskier than
max-Sharpe in sample, that Black-Litterman with zero views reproduces the prior
exactly, that a bullish view raises that asset's weight, that CAPM recovers planted
betas within 4 standard errors and returns beta = 1.000000000 for the benchmark
against itself, and that the cache round-trips with the network off.

The one that matters is `test_walk_forward_never_sees_the_future`. `walk_forward`
asserts `fit_window[-1] < hold_window[0]` on every roll; the test confirms the
holds tile the sample without overlap or duplication and that no return predates
the first fitted window. A backtest without that check is a number generator.

Writing the tests first also paid for itself once: `metrics()` computed the equity
curve as `(1 + r).cumprod()`, so `cummax()` started below par and a drawdown
beginning on day one was invisible. Caught by the known-series test, fixed by
seeding the curve at 1.0.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Scope

Not built, deliberately: user accounts, a database, a REST backend, live streaming
quotes, options pricing. The syllabus units this covers are Markowitz, CAPM, the
limitations of the Markowitz model, and modern risk measures. Binomial/CRR/
Black-Scholes belong to a separate options project and would dilute this one.
