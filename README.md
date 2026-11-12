# Volatility Regime Monitor

**Live app: [volatility-regime-monitor-bastianfh.streamlit.app](https://volatility-regime-monitor-bastianfh.streamlit.app)**

An interactive dashboard that detects whether a market is **calm, normal or turbulent**, and then tests whether knowing that actually helps an investor.

It covers major currency pairs (EUR/USD by default), gold, and the three main US stock indices (S&P 500, Nasdaq 100 and Dow Jones). It's built in Python with Streamlit.

---

## The idea in one paragraph

Markets don't move the same way all the time. They go through quiet stretches where prices barely move, and stormy stretches where they swing wildly, and those stretches tend to last a while. This project uses a statistical model to label every trading day as calm, normal or turbulent. It then asks three practical questions:

1. **Forecasting:** does the model predict upcoming volatility better than the standard methods banks and academics use?
2. **Risk management:** if you hold less when the market is turbulent and more when it's calm, do you get steadier risk and smaller losses, even after trading costs?
3. **Trading signals:** do two popular technical-analysis signals (a moving-average trend signal and Bollinger bands) work better if you only trust them in the right market conditions?

---

## How it works

**1. Measure how jumpy prices are.**
For every day, the app looks at the daily price changes over the past month (21 trading days) and calculates *volatility*: how much the price typically swings, scaled to a yearly number. 8% volatility means a typical one-year move of roughly ±8%.

**2. Assume the market has hidden "moods".**
The core model is a **Hidden Markov Model (HMM)**. It assumes the market is always in one of three moods (calm, normal or turbulent), and each mood has its own typical size of price swings. We can't see the mood directly; we only see prices. The model works backwards from the prices to estimate which mood the market is most likely in.

**3. Learn from history.**
From past data, the model learns two things: what each mood looks like (how big the swings usually are), and how often the market switches from one mood to another. In practice, moods turn out to be *sticky*: a turbulent day is usually followed by another turbulent day. That stickiness is what makes the model useful for looking ahead.

**4. Read the current mood, one day at a time.**
Each day, the model combines that day's price move with what it believed the day before, and updates its estimate. Crucially, it only ever uses information that was available on that day.

---

## Why the results can be trusted: no peeking at the future

The most common mistake in backtesting is letting a model "see" future data without realising it. That makes any strategy look brilliant on paper and useless in real life. This project is built to avoid it:

- **Learning period vs. test period.** The model learns from the first 70% of the data and is judged only on the remaining 30%, which it has never seen.
- **Regular re-training.** For the forecasting and strategy tests, every model is re-trained every 63 trading days (about once a quarter) using only the data up to that point, just as an analyst would update a model in real life.
- **Decisions use yesterday's information.** Every trade is decided at the close of one day and earns the next day's return.
- **Trading costs are included.** Every time a position changes, a small cost is subtracted (1 basis point, i.e. 0.01%, by default).
- **Statistical tests for luck.** When two methods are compared, the app checks whether the difference is big enough to be real or could just be chance. The dashboard reports this as "Is the difference real? Yes / Not proven".

The core math was also checked with simulated data. The test confirmed, for example, that changing future prices never changes any earlier result.

---

## What's in the dashboard

The app opens in **Simple view**, a plain-language version of every tab. Switch it off at the top of the sidebar for the **analyst view**, which has every statistic, chart and setting.

| Tab | What it shows |
|---|---|
| **Market State** | Today's market state, how confident the model is, and a price chart colored by state (green = calm, blue = normal, red = turbulent). Also how long each state typically lasts. |
| **Forecast Accuracy** | Four ways of forecasting volatility over the next 5 days, compared head to head: last month's volatility, **EWMA** (the weighted average banks use), **GARCH** (the classic academic model) and this project's regime model. |
| **Risk-Managed Strategy** | A strategy called *volatility targeting*: hold more when markets are calm and less when they're turbulent, aiming for a steady 8% yearly risk. Compared against simply buying and holding. |
| **Trading Signals** | A trend signal (buy when the 50-day average price is above the 200-day average) and a Bollinger-band signal (buy unusually low prices, sell unusually high ones), each tested with and without a market-state filter. |
| **How It Works** | A plain-English explanation and a glossary of the terms used. |

### Markets covered

| Name in app | Market |
|---|---|
| EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD, EUR/GBP, EUR/JPY | Major currency pairs |
| XAUUSD | Gold |
| US500 | S&P 500 |
| USTEC | Nasdaq 100 |
| US30 | Dow Jones Industrial Average |

In the analyst view you can also type in any symbol from Yahoo Finance.

---

## How to read the results

- **Different markets, different expectations.** Stocks tend to rise over time, so positive returns are normal for US500, USTEC and US30. Currency pairs have no built-in long-run return, so for FX the real question is whether risk stays steadier and losses stay smaller, not whether returns are high.
- **"Not proven" is a real answer.** If the regime model doesn't clearly beat the standard methods, that's a legitimate finding. Many published strategies disappear once costs and honest testing are applied.
- **The per-state breakdown describes the past.** It shows where a signal tended to work, not where it will work.

---

## Limitations

- **It describes risk, not direction.** The model tells you how turbulent the market is, not whether prices will go up or down.
- **Patterns can break.** A model trained on the past can go stale after a crisis or a big change in central-bank policy.
- **Simplifying assumptions.** The model assumes price moves within each state follow a bell curve, which underestimates rare, extreme days. The past-month volatility measure also reacts with a delay of up to a few weeks.
- **Free data.** Prices come from Yahoo Finance, which is free and can contain small gaps or errors. Yahoo's spot gold feed is unreliable, so the app falls back to gold futures prices when needed.
- **The backtests leave out some real-world costs and income.** They include trading costs but not interest on borrowed money, currency interest differences ("carry") or stock dividends.
- **Tuning settings to fit the results would be cheating.** The app lets you change settings, but adjusting them until the results look good would undo the "no peeking" protection. The signals use standard textbook settings for this reason.

---

## Run it yourself

You'll need **Python 3.12** ([download the 3.12.10 Windows installer](https://www.python.org/downloads/release/python-31210/); on Mac, any 3.12 installer works).

**Windows (PowerShell):**
```powershell
git clone https://github.com/bastianfh/volatility-regime-monitor.git
cd volatility-regime-monitor
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

**Mac (Terminal):**
```bash
git clone https://github.com/bastianfh/volatility-regime-monitor.git
cd volatility-regime-monitor
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. The first visit to the forecasting tabs takes a minute or two while the models are re-trained; after that, results load instantly.

**If live data won't download** (Yahoo sometimes blocks requests), the app automatically uses the saved prices in the `data/` folder. To refresh those saved prices, run `python app.py --snapshot`.

---

## Project files

| File | What it is |
|---|---|
| `app.py` | The whole app: data download, the regime model, the forecasting and strategy tests, and the dashboard |
| `requirements.txt` | The Python packages the app needs |
| `data/` | Saved price history, used when live data isn't available |
| `.streamlit/config.toml` | The dark color theme |

`app.py` is organized into classes, each with one job (for example `DataFetcher`, `RegimeModeler`, `GarchForecaster`, `VolTargetBacktester` and `DashboardUI`). Each one has comments explaining the math behind it.

---

## Further reading

The methods used here come from well-known research:

- **Hamilton (1989)** introduced regime-switching models to economics and finance.
- **Bollerslev (1986)** introduced the GARCH volatility model; the EWMA method comes from J.P. Morgan's RiskMetrics (1996).
- **Patton (2011)** shows how to fairly compare volatility forecasts.
- **Diebold & Mariano (1995)** created the standard test for whether one forecast is really better than another.
- **Moreira & Muir (2017)** show that volatility targeting has historically improved stock returns relative to risk.
- **Moskowitz, Ooi & Pedersen (2012)** document trend-following ("time-series momentum") across many markets.
- **Daniel & Moskowitz (2016)** show that trend strategies can crash in turbulent markets.

---

*For educational and research purposes only. Not investment advice.*
