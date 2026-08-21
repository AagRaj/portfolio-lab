# Portfolio Lab

A four-page web app for building and stress-testing equity portfolios: CAPM betas,
mean-variance optimisation, Black-Litterman views, hierarchical risk parity and
equal-risk-contribution risk parity — all evaluated **out of sample**, because in
sample they all look good, and all reported **with significance tests**, because a
column of Sharpe ratios with no standard errors is not a result.

The mean-variance optimisers are
[PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt). What this repo
adds is the data layer, the walk-forward harness, the ERC risk-parity solver, the
significance layer in `riskstats.py` (Probabilistic and Deflated Sharpe, block
bootstrap), and the tests that keep all of it honest.

## The result the app exists to show

Fifteen NSE large caps plus `^NSEI`, 2019-01-01 to 2026-08-19, fit on a trailing
504-day window, held 63 days, rolled forward. `rf` = 6.5%.

| Method | CAGR | Vol | Sharpe | Max DD | CVaR 95% | PSR | **DSR** | p vs equal-weight |
|---|---|---|---|---|---|---|---|---|
| equal weight | 13.9% | 12.9% | **0.572** | **-16.5%** | -1.80% | 0.993 | **0.978** | — |
| Black-Litterman | **17.2%** | 18.8% | 0.569 | -20.7% | -2.60% | 0.984 | 0.957 | 0.649 |
| risk parity (ERC) | 13.2% | 12.6% | 0.533 | -17.2% | -1.71% | 0.991 | 0.975 | 0.526 |
| HRP | 12.9% | 12.5% | 0.516 | -17.9% | -1.71% | 0.991 | 0.974 | 0.636 |
| min volatility | 11.6% | 12.3% | 0.418 | -21.0% | **-1.65%** | 0.986 | 0.961 | 0.515 |
| max Sharpe | 10.5% | 16.4% | 0.240 | -22.5% | -2.30% | 0.941 | **0.873** | 0.191 |

Optimising for Sharpe produced the *worst* out-of-sample Sharpe, the highest
volatility of the five optimisers and the deepest drawdown — beaten by dividing
capital equally and doing nothing.

**But that gap is not statistically significant, and the table says so.** A
block-bootstrap test of the Sharpe difference against equal weighting gives
**p = 0.19** for max Sharpe. Over 7.5 years a Sharpe difference of 0.33 sits inside
the noise. Anyone quoting "equal weight beats mean-variance" off this table as a
*proven* result is over-reading it — and that includes an earlier version of this
README.

What the data does support, stated precisely:

1. **No optimiser significantly beats equal weighting.** Every `p vs equal-weight`
   is above 0.5 except max Sharpe at 0.19. That is the DeMiguel–Garlappi–Uppal
   (2009) result reproduced on Indian equities, and it is a statement about the
   *absence* of evidence, which is what these numbers actually license.
2. **Max Sharpe is the only method that fails deflation.** Its DSR of 0.873 sits
   below the 0.95 threshold: once you price in having tried six methods, its
   performance is not distinguishable from the luckiest of six coin flips. Every
   other method clears 0.95. This is the sharper claim, and it survives the
   multiple-testing objection that the raw Sharpe column does not.
3. **The binding constraint is the expected-return vector, not the optimiser.**
   The three methods that never estimate expected returns from history — equal
   weight, ERC, HRP — take three of the top four rows. Black-Litterman is the
   exception that proves it: it recovers the return max Sharpe threw away (17.2%
   vs 10.5%) by replacing the historical mean with a market-cap-implied prior.
4. **Plain ERC beats HRP here** on Sharpe, drawdown and CAGR. HRP's pitch is
   robustness through clustering instead of matrix inversion; on this universe the
   simpler equal-risk-contribution solution does better. Worth knowing before
   reaching for the more complicated one.

One caveat before quoting any of this: the market-cap prior uses *today's*
capitalisations for every historical rebalance, so the Black-Litterman row carries
a mild look-ahead in the prior even though the price data does not. Fixing it needs
a historical market-cap series the free API does not provide.

## Why there are p-values in a portfolio table

Most portfolio backtests report a column of Sharpe ratios and stop. Two things are
wrong with that, and `riskstats.py` fixes both.

**A Sharpe ratio is an estimate with a standard error**, and over ~1,400 daily
observations that error is large — large enough that a 0.33 gap fails to reach 5%
significance. The **Probabilistic Sharpe Ratio** (Bailey & Lopez de Prado, 2012)
gives P(true SR > 0) while correcting for the two ways returns break the normal-iid
assumption behind the usual formula:

```
PSR = Phi[ (SR - SR*) * sqrt(T-1) / sqrt(1 - g3*SR + (g4-1)/4 * SR^2) ]
```

Negative skew (g3) and fat tails (g4) both inflate the denominator, so a strategy
earning a good Sharpe from many small gains and rare large losses is correctly
penalised.

**Trying six methods and reporting the best inflates the winner** even if none of
them has skill. The **Deflated Sharpe Ratio** measures PSR not against zero but
against the expected maximum Sharpe of N no-skill trials:

```
SR0 = sqrt(V[SR]) * [ (1-g)*PhiInv(1 - 1/N) + g*PhiInv(1 - 1/(N*e)) ]
```

Here SR0 = 0.0113 per day across the six methods. That is the hurdle the winner has
to clear, and max Sharpe does not.

The `p vs equal-weight` column is a **circular block bootstrap** on the Sharpe
difference — blocks rather than iid resampling because returns are autocorrelated,
and the same resample index applied to both series so the strong cross-correlation
between two portfolios holding overlapping assets is preserved rather than assumed
away. The closed-form Jobson-Korkie statistic would be shorter, but it assumes iid
normal returns, which is precisely the assumption PSR exists to avoid.

## Pages

| Page | What it answers |
|---|---|
| `app.py` — **Universe & Data** | What did we actually load? Rows, NaNs, date coverage, cache-vs-network per ticker, correlation structure. |
| `1_CAPM` | Beta, alpha, R² and **the standard error of beta** per asset, plotted against the security market line. |
| `2_Allocation` | Max-Sharpe / min-vol weights, a Black-Litterman tab where you enter views with confidences and see prior vs posterior, and whole-share allocation for a given capital. |
| `3_Risk_Comparison` | Efficient frontier, then the walk-forward table above with DSR and bootstrap p-values, and out-of-sample equity curves. |

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
python -m pytest -q        # 21 tests, no network required
```

The suite runs on synthetic prices with planted betas. It checks that weights form
a long-only fully-invested portfolio, that min-volatility is never riskier than
max-Sharpe in sample, that Black-Litterman with zero views reproduces the prior
exactly, that a bullish view raises that asset's weight, that CAPM recovers planted
betas within 4 standard errors and returns beta = 1.000000000 for the benchmark
against itself, and that the cache round-trips with the network off.

Three matter most.

`test_walk_forward_never_sees_the_future` — `walk_forward` asserts
`fit_window[-1] < hold_window[0]` on every roll; the test confirms the holds tile
the sample without overlap or duplication and that no return predates the first
fitted window. A backtest without that check is a number generator.

`test_uncorrelated_risk_parity_is_inverse_volatility` — pins the ERC solver to its
closed form. With a diagonal covariance, equal-risk-contribution weights must be
proportional to 1/sigma_i. Anything else means the optimiser is wrong, and a
risk-parity solver that is quietly wrong still returns plausible-looking weights.

`test_best_of_many_random_strategies_fails_deflation` — builds forty pure-noise
strategies, picks the best, and asserts it looks significant on its own
(PSR > 0.90) and does *not* survive deflation (DSR < 0.95). That is the exact
failure mode the Deflated Sharpe Ratio exists to catch, verified rather than
asserted.

Writing the tests first also paid for itself once: `metrics()` computed the equity
curve as `(1 + r).cumprod()`, so `cummax()` started below par and a drawdown
beginning on day one was invisible. Caught by the known-series test, fixed by
seeding the curve at 1.0.

## Run

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

```bash
python riskstats.py    # significance-layer self-check
```

## Related

The risk *measurement* side — VaR/Expected Shortfall models, Kupiec /
Christoffersen / Engle-Manganelli / Acerbi-Szekely backtesting, and the FRTB
capital charge — lives in a separate repo. This one builds portfolios; that one
validates the risk model you would monitor them with.

## Scope

Not built, deliberately: user accounts, a database, a REST backend, live streaming
quotes, options pricing. The syllabus units this covers are Markowitz, CAPM, the
limitations of the Markowitz model, and modern risk measures. Binomial/CRR/
Black-Scholes belong to a separate options project and would dilute this one.
