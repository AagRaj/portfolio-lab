# Portfolio Lab

A five-page web app that allocates capital, then asks whether the resulting risk
can actually be forecast — and what it costs when it can't.

Everything is evaluated **out of sample**, because in sample everything looks good,
and reported **with significance tests**, because a column of Sharpe ratios with no
standard errors is not a result.

The mean-variance optimisers are
[PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt). What this repo
adds is the data layer, the walk-forward harness, an ERC risk-parity solver, the
significance layer in `riskstats.py` (Probabilistic and Deflated Sharpe, block
bootstrap), the conditional tail-risk engine in `var_engine.py` / `tailrisk.py`, and
the 28 tests that keep all of it honest.

Fifteen NSE large caps plus `^NSEI`, **2010-01-04 to 2026-08-18** (4,107 days), fit
on a trailing 504-day window, held 63 days, rolled forward. `rf` = 6.5%.

## 1. Allocation: nothing beats equal weighting, and one thing loses to it

| Method | CAGR | Vol | Sharpe | Max DD | CVaR 95% | DSR | p vs equal-weight |
|---|---|---|---|---|---|---|---|
| min volatility | 19.2% | 13.8% | **0.915** | -29.8% | -1.9% | 1.000 | 0.474 |
| HRP | 19.3% | 14.0% | 0.914 | -30.8% | -1.9% | 1.000 | 0.271 |
| risk parity (ERC) | 19.4% | 14.4% | 0.895 | -31.3% | -2.0% | 1.000 | 0.145 |
| equal weight | 19.4% | 15.0% | 0.861 | -32.4% | -2.1% | 1.000 | — |
| max Sharpe | 19.4% | 17.9% | 0.721 | -29.6% | -2.5% | 0.999 | 0.380 |
| Black-Litterman | 16.8% | 23.4% | 0.439 | -39.2% | -3.1% | 0.970 | **0.006** |

The three risk-based methods that never touch the expected-return vector — min-vol,
HRP, ERC — take the top three places, and all five optimisers deliver within 0.3
percentage points of the same CAGR at visibly different volatility. But **no method
significantly beats equal weighting** (every p > 0.14). The one significant result
in the table is negative: Black-Litterman is significantly *worse*, at p = 0.006.

### An earlier version of this README was wrong, and that is worth stating

On a 2019-2026 sample this table looked completely different: max Sharpe came last
at 0.240 with a DSR of 0.873, and the headline was "optimising for Sharpe produces
the worst out-of-sample Sharpe." **That conclusion did not survive extending the
sample to 2010.** Over 16 years max Sharpe sits mid-table at 0.721 and clears
deflation comfortably.

The 7.5-year result was a small-sample artifact. It was reported with the correct
caveat at the time (the Sharpe gap was not significant, p = 0.19) — which is exactly
why the caveat was there. A backtest conclusion that flips when you double the
sample was never a finding, and the significance column is what tells you that
before someone else does.

## 2. Tail risk: can this portfolio's risk be forecast?

Allocation decides how capital is split. The next question on a real desk is whether
the resulting return stream has a **predictable tail** — because a portfolio whose
risk you cannot forecast cannot be sized or limited, however low its realised
volatility.

Five conditional VaR/ES models are fitted to the walk-forward stream (3,591
out-of-sample days) and backtested at 97.5%:

| Risk model | Breaches (exp 77.3) | Kupiec | Christoffersen | DQ | Acerbi-Székely | Verdict |
|---|---|---|---|---|---|---|
| Gaussian | 83 | 0.515 | **0.030** | **0.000** | **0.000** | clustering, DQ, ES too small |
| Historical | 87 | 0.272 | **0.014** | **0.000** | 0.070 | clustering, DQ |
| EWMA (RiskMetrics) | 85 | 0.381 | 0.312 | **0.001** | **0.000** | DQ, ES too small |
| **GJR-GARCH + FHS** | **82** | **0.590** | **0.362** | **0.295** | **0.062** | **PASS** |
| GJR-GARCH + EVT | 80 | 0.755 | **0.039** | 0.319 | 0.082 | clustering |

**One model out of five is usable.** Note that Kupiec passes for *every* model — the
breach counts are all close to expected. Only the tests that look at *when* breaches
happen (Christoffersen, DQ) and *how bad* they are (Acerbi-Székely) separate them.
Counting breaches is not backtesting.

Note also that EVT loses to FHS here, the reverse of the result on a US cross-asset
book. The fitted GPD shape is **ξ = −0.06** — a *bounded* tail — so extrapolation
past the sample buys nothing and the extra estimation noise costs a little. EVT's
edge is conditional on ξ > 0, not automatic.

## 3. The payoff: a p-value becomes basis points

Size the book so that forecast ES equals a target: `leverage = clip(target / ES, 0,
max)`, using the forecast made the day before. If the risk model is calibrated,
realised ES lands on the target. If it understates the tail, the book is over-levered
and realised risk overshoots.

Target = 1.6% daily ES at 97.5%:

| Risk model used for sizing | Avg leverage | CAGR | Vol | Max DD | Realised ES | **vs target** |
|---|---|---|---|---|---|---|
| Gaussian | 0.81 | 12.97% | 11.48% | -25.05% | 1.983% | **+23.9%** |
| EWMA | 0.90 | 15.14% | 11.39% | -16.55% | 1.871% | +17.0% |
| Historical | 0.71 | 10.89% | 9.99% | -22.36% | 1.749% | +9.3% |
| GJR-GARCH + FHS | 0.82 | 13.79% | 10.45% | -15.98% | 1.706% | +6.6% |
| GJR-GARCH + EVT | 0.81 | 13.73% | 10.32% | **-15.18%** | 1.683% | **+5.2%** |
| *(unscaled)* | 1.00 | 18.49% | 14.89% | -32.37% | 2.614% | +63.4% |

**The overshoot column reproduces the backtest ranking.** Models that fail the ES
test under-forecast the tail, so sizing on them over-levers the book — Gaussian
misses its own risk target by 24%, the model that passes misses by 5%. That is the
argument for backtesting a risk model before sizing on it, in the units the question
gets asked in.

Targeting also converts return into control: max drawdown falls from **-32.4% to
-16.0%** while CAGR falls from 18.5% to 13.8%. Risk-adjusted, the targeted book is
slightly ahead *and* takes half the drawdown.

## 4. Every control changes the answer

The Tail Risk page is parameterised, and the parameters matter. Measured, not
asserted (`test_every_ui_option_changes_the_result` pins this):

| Control | Effect |
|---|---|
| **Confidence** 95% → 97.5% → 99% | models passing 2 → 1 → 1; leverage 0.96 → 0.82 → 0.68; CAGR 16.3% → 13.7% → 11.3% |
| **ES window** 250 → 500 → 1000 | breaches 103 → 85 → 60; max DD -21.5% → -15.6% → -13.8% |
| **Portfolio** equal-wt / min-vol / ERC | models passing 1 / 1 / **2** — the ERC stream is the most forecastable |
| **ES target** 1.0% → 4.0% | leverage 0.51 → 2.05; CAGR 8.5% → 34.8%; max DD -10.0% → -35.3% |

Deeper confidence levels are harder to forecast, so the overshoot grows (+4.6% at
95%, +14.2% at 99%). Longer windows react more slowly and breach less. And the risk
target traces out a clean risk/return frontier — 4× the leverage for 4× the CAGR and
3.5× the drawdown, which is what the trade-off is supposed to look like.

## Pages

| Page | What it answers |
|---|---|
| `app.py` — **Universe & Data** | What did we actually load? Rows, NaNs, date coverage, cache-vs-network per ticker, correlation structure. |
| `1_CAPM` | Beta, alpha, R² and **the standard error of beta** per asset, against the security market line. |
| `2_Allocation` | Max-Sharpe / min-vol weights, Black-Litterman views with confidences showing prior vs posterior, whole-share allocation. |
| `3_Risk_Comparison` | Efficient frontier, the walk-forward table with DSR and bootstrap p-values, out-of-sample equity curves. |
| `4_Tail_Risk` | Conditional VaR/ES forecasts, four backtests, and ES-targeted sizing. |

## Why there are p-values in a portfolio table

A Sharpe ratio is an estimate with a standard error. The **Probabilistic Sharpe
Ratio** (Bailey & Lopez de Prado, 2012) gives P(true SR > 0) while correcting for
skew and kurtosis:

```
PSR = Phi[ (SR - SR*) * sqrt(T-1) / sqrt(1 - g3*SR + (g4-1)/4 * SR^2) ]
```

Trying six methods and reporting the best inflates the winner even if none has skill.
The **Deflated Sharpe Ratio** measures PSR against the expected maximum Sharpe of N
no-skill trials rather than against zero. `p vs equal-weight` is a **circular block
bootstrap** — blocks because returns are autocorrelated, and the same resample index
applied to both series so the cross-correlation between two portfolios holding
overlapping assets is preserved rather than assumed away.

## Data layer

`data.py` caches adjusted closes to `cache/{ticker}.parquet`. The cache is
authoritative; the network is touched only when the cache cannot cover the requested
window, and if the fetch fails the app serves cached data with a visible warning
rather than crashing. Tickers with under 120 observations are dropped from the
optimiser but stay in the quality report marked `kept=False`. An **Offline** switch
in the sidebar forbids network access entirely — that is also how the tests run.

## Tests

```bash
python -m pytest -q        # 28 tests, no network required
```

Synthetic prices with planted betas and a simulated GARCH stream. Four matter most:

`test_walk_forward_never_sees_the_future` — `walk_forward` asserts
`fit_window[-1] < hold_window[0]` on every roll; the test confirms the holds tile the
sample without overlap and that no return predates the first fitted window.

`test_es_forecast_is_out_of_sample` — the tail-risk equivalent. Truncating the input
must not change any forecast that survives. If a forecast for day *t* changed when
data after *t* was removed, it had been using it.

`test_a_failing_risk_model_overshoots_the_target_by_more` — the integration in one
assert: a risk model that understates the tail over-levers, so its realised risk
misses the target by more than a calibrated model's.

`test_best_of_many_random_strategies_fails_deflation` — forty pure-noise strategies,
best one picked; it must look significant alone (PSR > 0.90) and fail deflation
(DSR < 0.95).

Writing tests first paid for itself once already: `metrics()` computed the equity
curve as `(1 + r).cumprod()`, so `cummax()` started below par and a drawdown
beginning on day one was invisible. Caught by the known-series test, fixed by seeding
the curve at 1.0.

## Run

```bash
pip install -r requirements.txt
```

```bash
python -m streamlit run app.py
```

Use `python -m streamlit`, not bare `streamlit`. If you have more than one Python on
PATH, the bare launcher can resolve to a different interpreter than the one holding
your packages — the app then loses `yfinance`, silently falls back to cached prices,
and shows a warning most people will not read. `python -m` pins it to the
interpreter you meant.

```bash
python riskstats.py     # significance-layer self-check
```

```bash
python tailrisk.py      # tail-risk self-check
```

## Related

`var_engine.py` is vendored from a companion VaR/ES engine repo — 6 models, Kupiec /
Christoffersen / Engle-Manganelli DQ / Acerbi-Székely backtests, Basel and FRTB
capital. It is copied rather than imported so this app clones and runs standalone.

## Scope

Not built, deliberately: user accounts, a database, a REST backend, live streaming
quotes, options pricing.

Known limitation, disclosed rather than hidden: the Black-Litterman market-cap prior
uses *today's* capitalisations for every historical rebalance, so that row carries a
mild look-ahead in the prior even though the price data does not. Fixing it needs a
historical market-cap series the free API does not provide. Given BL is the one
method that significantly *underperforms* here, the look-ahead is not flattering it.
