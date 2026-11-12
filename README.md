# Volatility Regime Monitor

An interactive dashboard that classifies the daily volatility of FX pairs, gold and US equity indices (EUR/USD by default) into three hidden regimes (**Low**, **Normal** and **High** volatility) using a Gaussian Hidden Markov Model (HMM). It then tests whether those regimes are **useful**, using two walk-forward experiments:

1. **Forecasting:** does the HMM forecast future volatility better than the industry benchmarks, EWMA (RiskMetrics) and GARCH(1,1)? Answered with QLIKE loss and Diebold–Mariano tests.
2. **Allocation:** does sizing a position with HMM forecasts give better risk control than buy-and-hold or vol targeting with the benchmark models, **after transaction costs**?
3. **Signals:** does conditioning two classic trading signals (an EMA trend crossover and Bollinger-band mean reversion) on the HMM regime improve them? The hypotheses are fixed before seeing results, and Sharpe differences are tested with a block bootstrap.

Every result is strictly out-of-sample. Models are re-estimated on an expanding window using only data before each refit date, and every forecast or position uses only information available at that day's close.

**Stack:** Python · Streamlit · yfinance · hmmlearn · Plotly · NumPy / pandas / SciPy

---

## 1. Quick start

```bash
# 1. Create and activate a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard (opens http://localhost:8501)
streamlit run app.py
```

For the full dark theme (including native widgets), put `config.toml` in a `.streamlit/` folder next to `app.py`:

```
volatility-regime-monitor/
├── app.py
├── requirements.txt
├── README.md
├── data/                 # optional price snapshots (see Section 6)
└── .streamlit/
    └── config.toml
```

You can also pass the theme on the command line: `streamlit run app.py --theme.base dark`.

### Using the dashboard

| Control | Purpose |
|---|---|
| Instrument / custom symbol | One of the built-in instruments (see below), or any Yahoo Finance symbol (`NOK=X`, `^GDAXI`, `SI=F`, …) |
| Start / End | Calendar range requested from the vendor |
| In-sample share | Chronological train/test split (default 70 / 30) |
| Realized-vol window | Look-back *w* for rolling volatility (default 21 ≈ 1 month) |
| Estimation settings | Number of EM random restarts and base random seed |
| Walk-forward forecasting | Refit frequency (21/63/126/252 days), forecast horizon *h* (1/5/10/21 days), EM restarts per refit |
| Vol-target backtest | Target volatility, leverage cap, transaction cost (bp per 1× traded), rebalance band. Also used by the Signal Lab |
| Signal lab | EMA preset (50/200 or 20/100). Bollinger is fixed at 20 days, ±2σ |

### Supported instruments

| Label | Market | Asset class | Yahoo symbol(s) |
|---|---|---|---|
| EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD, EUR/GBP, EUR/JPY | Major and cross FX pairs | FX | `EURUSD=X`, `JPY=X`, `CHF=X`, … |
| XAUUSD | Gold | Commodity | `XAUUSD=X` (spot), falling back to `GC=F` (COMEX front-month futures) |
| US500 | S&P 500 | Equity index | `^GSPC` |
| USTEC | Nasdaq 100 | Equity index | `^NDX` |
| US30 | Dow Jones Industrial Average | Equity index | `^DJI` |

The labels US500, USTEC and US30 follow common CFD naming. The data comes from the underlying cash indices, which a CFD tracks closely. Yahoo's spot-gold feed has historically been unreliable, so the app downloads every candidate symbol for an instrument and uses the one with the most valid daily bars. If it falls back to `GC=F`, the dashboard says so: futures prices sit slightly above spot and can jump on contract-roll dates.

Settings take effect when you press **Run model**. Results are cached, so returning to a previous configuration is instant. The first walk-forward run re-estimates every model at each refit date and takes roughly 30–90 seconds with default settings; a progress bar shows how far it has got.

**Dashboard tabs**

*Regime Monitor* (a single fit on the in-sample block, evaluated out of sample)
- **Snapshot KPIs:** current regime, its filtered probability, current realized vol compared with the out-of-sample average, the one-day-ahead probability of the High-Vol regime, and the length of the current spell.
- **Main chart:** out-of-sample candlesticks on a background shaded by the filtered regime; realized volatility with each regime's in-sample "typical" level; and stacked filtered regime probabilities.
- **Transition matrix and regime statistics** for both the in-sample and out-of-sample blocks.
- **Diagnostics:** in-sample and out-of-sample predictive log-likelihood, BIC, EM convergence, emission parameters in market units, and a CSV export.

*Volatility Forecasts* (walk-forward)
- Realized *h*-day volatility plotted against the Historical, EWMA, GARCH and HMM forecasts, with refit dates marked.
- A loss table (QLIKE, RMSE, Mincer–Zarnowitz slope and R²) and Diebold–Mariano tests against a benchmark you choose, plus a plain-language verdict.
- A refit log with HMM and GARCH parameters at every re-estimation.

*Vol-Target Backtest* (walk-forward, net of costs)
- Equity curves, drawdowns and held leverage for buy-and-hold, vol targeting with each of the four forecasters, and an HMM de-risk overlay.
- Rolling 63-day realized volatility of each strategy against the target.
- A strategy table: CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, leverage, turnover, cost drag and risk stability.

*Signal Lab* (walk-forward regimes, net of costs)
- Price with EMAs and Bollinger bands over regime shading, plus the signals' positions.
- Equity, drawdown and position curves for five strategies: long (vol-targeted), EMA trend, EMA trend + regime, Bollinger reversion, Bollinger reversion + regime.
- Block-bootstrap tests of four pre-specified Sharpe comparisons, and a descriptive breakdown of each signal's performance by regime.

*Methodology* gives a compact mathematical summary of every component.

---

## 2. Methodology

### 2.1 Features

For daily closes $P_t$:

$$
r_t = \ln\frac{P_t}{P_{t-1}}, \qquad
\sigma_t = \sqrt{\frac{252}{w}\sum_{i=0}^{w-1} r_{t-i}^2}, \qquad
x_t = \big(r_t,\ \ln\sigma_t\big)^\top .
$$

- Realized volatility uses the **zero-mean** estimator. At a daily horizon the drift of FX, gold and index returns is much smaller than its own estimation error, so subtracting a sample mean adds noise without reducing bias.
- Volatility enters in **logs** because realized variance is approximately log-normal. That makes $\ln\sigma_t$ much closer to the Gaussian emission assumption than $\sigma_t$.
- Features are standardized, $z_t = (x_t - \hat\mu)/\hat s$, with $\hat\mu$ and $\hat s$ estimated **on the training block only**.

### 2.2 The Gaussian HMM

A latent state $S_t \in \{1,2,3\}$ follows a first-order, time-homogeneous Markov chain, and observations are Gaussian conditional on the state:

$$
P(S_t=j \mid S_{t-1}=i) = A_{ij}, \qquad
z_t \mid S_t = k \;\sim\; \mathcal{N}(\mu_k, \Sigma_k).
$$

The unconditional distribution of $z_t$ is a Gaussian mixture whose weights move persistently over time through $A$. This reproduces two stylized facts of daily asset returns: **volatility clustering** and **fat-tailed unconditional returns**.

Parameters $\theta = (\pi, A, \{\mu_k\}, \{\Sigma_k\})$ are estimated by maximum likelihood with the **EM (Baum–Welch)** algorithm and full covariance matrices. EM only finds local optima, so the fit is repeated from several random initializations and the restart with the highest training log-likelihood is kept.

**Label identification.** HMM states are only identified up to a permutation. After fitting, states are sorted by the emission mean of $\ln\sigma_t$, which maps them to Low / Normal / High. This ordering also comes from training parameters only.

### 2.3 Out-of-sample validation without look-ahead bias

| Source of look-ahead | How it is prevented |
|---|---|
| Random train/test shuffling | Strict chronological split; the test block starts after the last training day |
| Full-sample feature scaling | Scaler moments estimated on the training block only |
| Model parameters | EM sees training observations only; parameters are frozen afterwards |
| State labelling | Low/Normal/High ordering taken from training emission means |
| **Smoothed state inference** | **Out-of-sample labels come from the forward filter, not Viterbi or forward–backward** |
| Rolling-window features | Strictly backward-looking: $\sigma_t$ uses $r_{t-w+1},\dots,r_t$ |

The most common mistake in HMM regime studies is calling `model.predict()` on the test set. That method runs the Viterbi algorithm, which picks the most likely *whole path* given **all** observations, including future ones. `predict_proba()` has the same problem because it returns smoothed probabilities $P(S_t \mid z_{1:T})$. This app instead implements the **Hamilton (forward) filter** directly:

$$
\alpha_t(j) = P(S_t=j \mid z_{1:t})
= \frac{\mathcal{N}(z_t;\mu_j,\Sigma_j)\,\sum_i \alpha_{t-1}(i)\,A_{ij}}{c_t},
\qquad c_t = p(z_t \mid z_{1:t-1}).
$$

The regime call on day $t$ is $\arg\max_j \alpha_t(j)$, and it uses only data up to that day's close. Any trading rule built on it should act from $t+1$. The normalizers $c_t$ are one-step-ahead predictive densities, so the mean of $\ln c_t$ over the test block is a genuine **out-of-sample predictive log-likelihood**. The dashboard compares it with the in-sample value as a check for overfitting.

The one-day-ahead regime forecast is $P(S_{t+1} \mid z_{1:t}) = \alpha_t^\top A$.

### 2.4 Walk-forward re-estimation

The *Regime Monitor* tab uses a single fit, which is the clearest way to show an out-of-sample classification. A real investor, though, would retrain the model as new data arrives. The forecasting and backtest tabs therefore use an **expanding-window walk-forward** protocol:

```
refit k:  |------ estimation data [0, t_k) ------|-- forecasts on days [t_k, t_k+1) --|
```

At each refit date $t_k$ (every 63 trading days by default, starting on the first out-of-sample day), **everything that is estimated** is re-estimated on observations strictly before $t_k$: scaler moments, HMM parameters and state ordering, and GARCH parameters. Parameters are then frozen until the next refit. Each EM run is warm-started from the previous refit's parameters, with extra random restarts to avoid getting stuck in one local optimum.

### 2.5 Volatility forecasts and their evaluation

Each model, at the close of day $t$, forecasts the same target: the average annualized variance over the next $h$ days,

$$
y_t = \frac{252}{h}\sum_{i=1}^{h} r_{t+i}^2 .
$$

| Model | Forecast of $y_t$ |
|---|---|
| **Historical** (random-walk benchmark) | current rolling realized variance $\sigma_t^2$ |
| **EWMA** (RiskMetrics, $\lambda = 0.94$) | $252\,\hat\sigma^2_{t+1}$ with $\hat\sigma^2_{t+1} = \lambda\hat\sigma^2_t + (1-\lambda)r_t^2$ (flat across horizons) |
| **GARCH(1,1)** (Gaussian QMLE, zero mean) | $252 \cdot \frac{1}{h}\sum_{i=1}^{h}\big[\bar\sigma^2 + (\alpha+\beta)^{i-1}(\sigma^2_{t+1} - \bar\sigma^2)\big]$ |
| **HMM** | $252\,\alpha_t^\top\big(\tfrac1h\sum_{i=1}^{h}A^i\big)\,m$, where $m_k = E[r^2 \mid S = k]$ |

The HMM forecast follows directly from the model. The filtered distribution $\alpha_t$ is propagated $1, \dots, h$ steps through the transition matrix, and each future state contributes its own return second moment, $m_k = s_r^2\Sigma_{k,00} + (\bar r + s_r\mu_{k,0})^2$. GARCH is implemented from scratch: a constrained SLSQP optimizer with $\alpha + \beta < 1$, and the variance recursion evaluated as a linear IIR filter.

**Loss function.** Realized variance only approximates true variance, and the approximation is noisy. Patton (2011) shows that only certain loss functions rank forecasts consistently in that case. The primary loss here is the robust **QLIKE**,

$$
L(y, f) = \frac{y}{f} - \ln\frac{y}{f} - 1 ,
$$

which also penalizes under-forecasting risk more heavily than over-forecasting it. That is the right asymmetry for a risk manager. RMSE on volatility is reported for intuition only.

**Diebold–Mariano test.** For the loss differential $d_t = L^{\text{model}}_t - L^{\text{bench}}_t$, overlapping $h$-day targets make $d_t$ an MA($h-1$) process. The long-run variance therefore sums $h-1$ autocovariances (the original DM rectangular kernel, with a Newey–West fallback if the estimate is not positive). The Harvey–Leybourne–Newbold small-sample correction is applied, and p-values come from a $t_{n-1}$ distribution. In simulation under the null hypothesis, the test rejects 4.3–5.3% of the time at the 5% level for $h \in \{1, 5, 21\}$. A Newey–West-only version rejected 12.5% of the time, which is why the original kernel is used.

**Mincer–Zarnowitz regression.** $y_t = a + b f_t + e_t$. An unbiased forecast has $b \approx 1$, and $R^2$ measures how much of realized variance the forecast explains.

### 2.6 Volatility-targeting backtest

At close $t$, each vol-targeting strategy sets leverage on a long position in the instrument:

$$
w^*_t = \min\!\Big(\frac{\sigma^\star}{\hat\sigma_t},\ L_{\max}\Big),
\qquad
R^{\text{net}}_{t+1} = w_t R_{t+1} - c\,|w_t - w_{t-1}| ,
$$

- $\hat\sigma_t$ is that model's walk-forward forecast issued at $t$, and $R_{t+1}$ is the **next** day's simple price return.
- A **no-trade band** keeps $w_t = w_{t-1}$ unless $|w^*_t - w_{t-1}|$ exceeds the band, which cuts turnover caused by noise in the forecasts.
- **Costs** $c$ (default 1 bp per unit of notional traded) cover the half-spread and slippage, including the initial trade into the position.
- **HMM De-risk** uses the regimes directly instead of a volatility number: $w_t = 1 - P(S_{t+1} = \text{High} \mid \mathcal F_t)$.

**What counts as success** depends on the asset:

- **FX:** a long-only position in a currency pair has no reliable risk premium, so Sharpe ratios near zero are expected for **all** strategies.
- **Equity indices:** equities carry a risk premium, so positive Sharpe ratios are normal. Vol targeting has historically helped equities most (Moreira & Muir, 2017), because high-volatility periods tend to coincide with falling prices. This is the *leverage effect*, visible here as a negative within-regime correlation between returns and log-volatility in the emission table.
- **Gold:** in between. There is no cash flow, and the long-run return has historically come in long, irregular swings.

In every case, vol targeting is a risk-management tool, so it is judged on:
- whether realized volatility stays close to the target;
- **risk stability**, the standard deviation of rolling 63-day realized volatility;
- drawdowns;
- turnover and cost drag.

### 2.7 Signal Lab: regime-conditioned trading signals

The vol-target backtest asks *how much* to hold. The Signal Lab asks whether the regime model also helps decide *which direction* to trade. It does this by conditioning two textbook signals on the walk-forward regime probabilities.

**Signals** (computed from closes up to day $t$; positions taken at close $t$ earn day $t+1$):

- **EMA trend:** $s_t = \operatorname{sign}(EMA^{f}_t - EMA^{s}_t)$ with fixed spans 50/200 (or 20/100). This is a basic form of time-series momentum, which is well documented across asset classes (Moskowitz, Ooi & Pedersen, 2012). Simple technical rules in FX lost much of their profitability after the 1990s (Neely, Weller & Ulrich, 2009), so no edge is assumed.
- **Bollinger reversion:** go long below the lower band and short above the upper band (20-day, ±2σ), then exit when the close crosses back to the middle band.

**Pre-specified hypotheses and regime gates.** Each hypothesis was written down before running any test, and each gate is a smooth probability weight with no threshold to tune:

| Signal | Hypothesis | Gate $g_t$ |
|---|---|---|
| EMA trend | Trend rules suffer sharp reversals in turbulent markets ("momentum crashes", Daniel & Moskowitz, 2016) | $1 - P(S_{t+1} = \text{High} \mid \mathcal F_t)$ |
| Bollinger reversion | Mean reversion works in calm, range-bound markets and fails in strong trends | $P(S_{t+1} = \text{Low} \mid \mathcal F_t)$ |

**Identical sizing.** Every strategy uses the same EWMA vol target, leverage cap, rebalance band and costs:

$$
w_t = s_t \cdot g_t \cdot \min\!\big(\sigma^\star / \hat\sigma^{EWMA}_t,\ L_{\max}\big).
$$

The regime gate is therefore the *only* difference between a signal and its filtered version. EWMA rather than the HMM sets the size, so the regime enters only through the gate.

**Inference.** Sharpe-ratio differences are tested with a **circular block bootstrap** (Politis & Romano, 1992). Blocks of 20 days are resampled jointly for both strategies, which keeps the autocorrelation and the strong correlation between a signal and its filtered version. The two-sided p-value comes from the bootstrap distribution re-centred on zero. In simulation with autocorrelated returns and equal true Sharpe ratios, the test rejects 5.3% of the time at the 5% level.

**Regime breakdown.** The tab also reports each unfiltered signal's return, Sharpe and hit rate within each regime. This is **descriptive only**. A new filter chosen after reading it would be fitted to the same data it is tested on.

### 2.8 Verification

The core logic was checked numerically with simulated data:

- The forward filter matches brute-force enumeration over all state paths, and perturbing data after a date leaves every filtered probability before that date unchanged.
- The GARCH estimator recovers known parameters (true $\omega, \alpha, \beta = 0.010, 0.060, 0.920$; estimated $0.011, 0.063, 0.916$).
- The walk-forward engine passes the same perturbation test for all four forecasters and the regime probabilities: forecasts dated before a shock to the data are identical with and without the shock.
- The DM test is correctly sized (see 2.5), as is the Sharpe bootstrap (see 2.7).
- The Signal Lab passes the perturbation test too: positions dated before a price shock are identical with and without it.
- The Bollinger state machine enters only outside the bands and exits at the middle band. The EMA signal is zero until the slow EMA has warmed up.

---

## 3. Data assumptions: FX, gold and equity indices

Spot FX trades over the counter, roughly 24 hours a day and 5 days a week, with no single exchange and no official close. Fitting a discrete-time HMM to daily FX bars therefore relies on several assumptions:

The assumptions below are written for FX, the hardest case. Asset-specific differences follow the list.

1. **The daily close is a consistent snapshot.** Yahoo Finance FX bars are vendor-defined cuts of a continuous market (they are not the WM/Reuters 4pm London fix or the 5pm New York close). The model assumes the cut time is stable, so each $r_t$ covers a comparable 24-hour interval. If the vendor's cut time drifts or differs across pairs, returns are **non-synchronous**: they measure slightly different intervals, which adds noise and biases cross-pair comparisons.
2. **Equally spaced time steps.** The Markov chain moves one step per observation. A Friday-to-Monday return covers about 72 hours of calendar time but only a small amount of extra trading time (the market is mostly shut at weekends), so it is treated as one step. Holidays with thin liquidity are treated as ordinary days. Weekend prints are removed during cleaning.
3. **A time-homogeneous Markov chain.** $A$ is constant, so regime durations are geometric with mean $1/(1-A_{kk})$ and no memory of how long a regime has already lasted.
4. **Conditional Gaussianity and independence.** Given the state, $z_t$ is Gaussian and independent of past observations.
5. **Stationarity across the split.** The regime structure estimated in-sample (number of states, their volatility levels, their persistence) is assumed to carry over to the out-of-sample period.
6. **Quotes approximate transactable prices.** Mid/indicative quotes are used and bid–ask spreads are ignored. That is reasonable for G10 majors but not for illiquid crosses.

---

**Differences for the other asset classes**

- **Equity indices** have an official exchange close (4:00 pm New York time), so the timing problem in point 1 mostly disappears. Instead, the index is closed overnight and on exchange holidays. Overnight news therefore arrives as a single **gap** that the model treats like any other daily return. Returns also mix about 6.5 hours of trading with about 17.5 hours of closed market. Holidays are removed from the data and hidden on the charts.
- **Different closing times across assets.** FX bars, US index closes and gold futures settlement (about 1:30 pm New York time) are taken at different times of day. That doesn't matter for a single-instrument model. It does matter if you compare regimes across assets on the same date: a "High Vol" day in US500 and in EUR/USD can refer to slightly different 24-hour windows.
- **Gold.** Spot gold trades over the counter like FX, so the FX assumptions apply. If the futures fallback is used, the series is the continuous front-month contract, which is **not roll-adjusted**. Each roll (about every two months) adds a small artificial return.
- **Index levels are price indices**, so they exclude dividends. Volatility is essentially unaffected, but backtest returns understate a total-return position by roughly the dividend yield.

## 4. Limitations

- **Overlapping-window features break conditional independence.** Consecutive values of $\ln\sigma_t$ share $w-1$ of their $w$ returns, so the observations are strongly autocorrelated even within a regime. The HMM's likelihood treats them as independent given the state. That inflates the effective sample size, overstates confidence (filtered probabilities are often near 0 or 1), and makes regimes look more persistent than the underlying volatility process. The realized-vol window also makes regime detection **lag** by up to about *w* days.
- **Gaussian emissions are thin-tailed.** Within a regime, daily returns still show excess kurtosis and occasional jumps (central bank surprises, the 2015 SNB de-peg, the 2020 equity crash). Single outliers can trigger short, spurious switches to High Vol. Student-*t* emissions would be more robust.
- **Regime flicker.** The argmax of the filtered probabilities can switch back and forth when probabilities are close to each other. No minimum-duration filter or hysteresis is applied, so as not to hide the raw model output.
- **Fixed number of states.** Three regimes is a modelling choice, not a result. The reported BIC can be used to compare alternatives, but BIC is also distorted by the autocorrelated features.
- **The Regime Monitor uses a single fit.** That tab trains once and freezes the parameters, so over a long test window structural breaks (such as EUR/USD's post-2014 low-volatility period, or the 2022 rate-shock regime) can make them stale. The forecasting and backtest tabs refit walk-forward, but on an *expanding* window that weights old and new data equally. A rolling window adapts faster but gives noisier estimates.
- **GARCH persistence and regime shifts.** If the estimation sample contains regime shifts, GARCH estimates tend to show persistence $\alpha+\beta$ close to 1 (Lamoureux & Lastrapes, 1990). Its long-run variance is then poorly identified and its forecasts behave like EWMA. The refit log shows this when it happens. That is evidence the regime structure is real, not a bug.
- **The backtest omits carry, dividends and financing.** Strategy returns are price-only. A real FX position earns or pays the interest-rate differential through forward points, index positions earn dividends, and futures embed roll yield. Leverage above 1× also has a funding cost. All of these matter for long-horizon P&L.
- **GARCH(1,1) is symmetric.** Equity volatility rises more after falls than after rises (the leverage effect). Symmetric GARCH misses this, which partly handicaps it on US500, USTEC and US30. GJR-GARCH or EGARCH would be the fairer equity benchmark.
- **Multiple comparisons and settings.** The dashboard compares four forecasters, six vol-target strategies and four Signal Lab hypotheses across settings you can change. With enough combinations, something will look significant by chance. The honest approach is to fix settings before looking at results, or to report results across a range of them. DM and bootstrap p-values are not adjusted for multiple testing.
- **Statistical uncertainty in backtests.** Differences in Sharpe ratio or drawdown between strategies are not tested for significance. Over a few years of daily data, Sharpe differences of 0.2–0.3 are typically within noise.
- **Labels are relative to the training sample.** "High Vol" means high *relative to the training period*. If the test period is structurally calmer or more turbulent, one label may dominate.
- **EM finds local optima.** Random restarts reduce this risk but do not remove it. Results can depend on the seed, particularly for short samples.
- **Data quality.** Yahoo Finance data is free; its FX and spot-gold prices are indicative quotes. It can contain stale (repeated) prices, gaps and occasional bad ticks, and OHLC values for some dates are unreliable. Only structurally invalid rows are removed; no outlier filtering is applied.
- **Hyperparameter snooping.** The volatility window and split are user controls. Tuning them by looking at out-of-sample results turns the test block into a validation block. For honest reporting, choose them in advance or use a separate validation period.
- **Not a production trading system.** The backtest is a research tool. It has no slippage model beyond a fixed cost, no execution-time risk between the close and the trade, no margin constraints, and only one asset.

### Possible extensions

- Student-*t* or mixture emissions; Markov-switching GARCH
- Range-based volatility estimators (Parkinson, Garman–Klass, Yang–Zhang) to react faster and reduce feature autocorrelation
- Duration-dependent (semi-Markov) regimes
- A Model Confidence Set (Hansen, Lunde & Nason, 2011) instead of pairwise DM tests, to handle multiple comparisons
- Block-bootstrap confidence intervals for backtest Sharpe ratios and drawdowns
- An FX carry overlay with forward points and funding costs
- GJR-GARCH / EGARCH benchmarks for equity indices, and total-return index data
- A cross-asset regime heatmap (FX, gold, indices) to separate broad risk-off episodes from single-market stress

### References

- Daniel, K., & Moskowitz, T. J. (2016). Momentum crashes. *Journal of Financial Economics*.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*.
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*.
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*.
- Lamoureux, C. G., & Lastrapes, W. D. (1990). Persistence in variance, structural change, and the GARCH model. *Journal of Business & Economic Statistics*.
- Ledoit, O., & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *Journal of Empirical Finance*.
- Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*.
- Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*.
- Neely, C. J., Weller, P. A., & Ulrich, J. M. (2009). The adaptive markets hypothesis: evidence from the foreign exchange market. *Journal of Financial and Quantitative Analysis*.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*.
- Politis, D. N., & Romano, J. P. (1992). A circular block-resampling procedure for stationary data. In *Exploring the Limits of Bootstrap*. Wiley.

---

## 5. Code structure

`app.py` is a single script organized into classes with separate responsibilities:

| Component | Responsibility |
|---|---|
| `DataFetcher` | Downloads OHLC data from yfinance, normalizes the index, removes invalid rows |
| `FeatureEngineer` | Log returns, rolling realized volatility, chronological split, train-only scaler |
| `RegimeModeler` | Multi-restart EM fit, state canonicalization, log-space forward filter |
| `RegimeAnalytics` | Regime and spell statistics, stationary distribution, emission summary |
| `EwmaForecaster` | RiskMetrics EWMA variance recursion |
| `GarchForecaster` | GARCH(1,1) QMLE (implemented from scratch), variance path, multi-step forecasts |
| `WalkForwardEngine` | Expanding-window refits of the scaler, HMM and GARCH; out-of-sample forecasts from all four models |
| `ForecastEvaluator` | QLIKE, RMSE, Diebold–Mariano (HLN-corrected), Mincer–Zarnowitz |
| `VolTargetBacktester` | Leverage rules, rebalance band, transaction costs, net P&L |
| `PerformanceAnalytics` | CAGR, Sharpe, Sortino, drawdown, Calmar, turnover, cost drag, risk stability |
| `SignalGenerator` | EMA trend signal, Bollinger bands and the reversion state machine |
| `SharpeBootstrap` | Circular block bootstrap for Sharpe-ratio differences |
| `SignalLab` | Regime-gated signal strategies, hypothesis tests, regime breakdown |
| `load_price_history` / `save_snapshots` | Live Yahoo data with symbol fallback; bundled CSV snapshots when Yahoo is unreachable |
| `run_regime_pipeline` / `run_walk_forward` | Cached orchestration for the single-fit and walk-forward experiments |
| `ChartFactory` | Plotly figures: regime candlesticks, transition heatmap, forecasts, equity/drawdown/leverage, rolling risk |
| `DashboardUI` | Streamlit layout, theme, sidebar, tabs, KPIs, tables |

---

## 6. Deploying a public link (Streamlit Community Cloud)

Streamlit Community Cloud hosts public Streamlit apps for free, straight from a GitHub repository, and gives you a shareable `*.streamlit.app` URL.

**1. Create a price snapshot locally (recommended).** Yahoo Finance often rate-limits or blocks requests from shared cloud servers, so a hosted app can fail to download data even when the same code works on your laptop. The app tries live data first and falls back to CSV snapshots in `data/`. To create them, run this on your own machine:

```bash
python app.py --snapshot      # downloads every built-in instrument into data/
```

When the app uses a snapshot, it shows a banner saying so, along with the snapshot's end date. Re-run the command and push again whenever you want fresher data.

**2. Push to GitHub.** Create a public repository containing `app.py`, `requirements.txt`, `README.md`, `.streamlit/config.toml` and `data/`.

**3. Deploy.** Sign in to [share.streamlit.io](https://share.streamlit.io) with GitHub, choose **Create app**, select the repository, branch and `app.py`, and deploy. In **Advanced settings**, choose Python 3.11 or 3.12.

**Things to know**
- Free apps go to sleep after a period with no visitors, and the first visitor has to wake them. Open your link shortly before sending it to anyone.
- The free tier has limited memory and CPU, and the first walk-forward run re-estimates every model at each refit date. To keep the first load short, consider defaulting to a later start date or refitting every 126 days. Results are cached once computed.
- The app can't run purely in the browser (for example via Pyodide/stlite), because `hmmlearn` is a compiled extension that isn't available there. A Python host is needed.

---

*Disclaimer: for educational and research purposes only. This is not investment advice.*
