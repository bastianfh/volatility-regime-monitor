"""
Volatility Regime Monitor
=========================

An interactive Streamlit application that classifies the daily volatility of
FX pairs, gold and US equity indices into three latent regimes
(Low / Normal / High) with a Gaussian Hidden Markov Model (HMM), evaluated
under a strictly causal out-of-sample protocol.

Pipeline
--------
1. ``DataFetcher``     Downloads and cleans daily OHLC bars from Yahoo Finance.
2. ``FeatureEngineer`` Builds log returns and rolling realized volatility, and
                       standardizes them with *training-sample* moments only.
3. ``RegimeModeler``   Fits a Gaussian HMM by EM (Baum-Welch) on the training
                       window only, then produces *filtered* (causal) regime
                       probabilities with the forward algorithm.
4. ``RegimeAnalytics`` Regime statistics, spell durations, emission summaries.
5. ``DashboardUI``     Streamlit + Plotly presentation layer.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from scipy.stats import t as student_t

FloatArray = npt.NDArray[np.float64]

logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE: Final[str] = "Volatility Regime Monitor"

#: Business-day annualization convention used for volatility quotes.
ANNUALIZATION_FACTOR: Final[int] = 252

N_REGIMES: Final[int] = 3
REGIME_LABELS: Final[Tuple[str, ...]] = ("Low Vol", "Normal Vol", "High Vol")

#: Column order of the HMM observation vector x_t.
FEATURE_COLUMNS: Final[Tuple[str, str]] = ("log_return", "log_realized_vol")
#: Index of the volatility feature, used to order (label) the hidden states.
VOL_FEATURE_INDEX: Final[int] = 1

MIN_TRAIN_OBS: Final[int] = 250
MIN_TEST_OBS: Final[int] = 20
EARLIEST_DATE: Final[date] = date(2003, 12, 1)



@dataclass(frozen=True)
class Instrument:
    """A market and the Yahoo Finance symbols that can supply its data.

    Attributes:
        label: Short display name, e.g. ``"US500"``.
        description: Long name, e.g. ``"S&P 500 index"``.
        asset_class: ``"FX"``, ``"Commodity"``, ``"Equity Index"`` or
            ``"Custom"``.
        symbols: Candidate Yahoo symbols in order of preference. Every
            candidate is downloaded and the one with the most valid daily
            bars is used, so a patchy primary feed falls back cleanly.
    """

    label: str
    description: str
    asset_class: str
    symbols: Tuple[str, ...]


INSTRUMENTS: Final[Tuple[Instrument, ...]] = (
    Instrument("EUR/USD", "Euro / US dollar", "FX", ("EURUSD=X",)),
    Instrument("GBP/USD", "British pound / US dollar", "FX", ("GBPUSD=X",)),
    Instrument("USD/JPY", "US dollar / Japanese yen", "FX", ("JPY=X",)),
    Instrument("USD/CHF", "US dollar / Swiss franc", "FX", ("CHF=X",)),
    Instrument("AUD/USD", "Australian dollar / US dollar", "FX",
               ("AUDUSD=X",)),
    Instrument("USD/CAD", "US dollar / Canadian dollar", "FX", ("CAD=X",)),
    Instrument("NZD/USD", "New Zealand dollar / US dollar", "FX",
               ("NZDUSD=X",)),
    Instrument("EUR/GBP", "Euro / British pound", "FX", ("EURGBP=X",)),
    Instrument("EUR/JPY", "Euro / Japanese yen", "FX", ("EURJPY=X",)),
    Instrument("XAUUSD", "Gold (spot; falls back to COMEX futures)",
               "Commodity", ("XAUUSD=X", "GC=F")),
    Instrument("US500", "S&P 500 index", "Equity Index", ("^GSPC",)),
    Instrument("USTEC", "Nasdaq 100 index", "Equity Index", ("^NDX",)),
    Instrument("US30", "Dow Jones Industrial Average", "Equity Index",
               ("^DJI",)),
)
INSTRUMENT_BY_LABEL: Final[Dict[str, Instrument]] = {
    i.label: i for i in INSTRUMENTS
}

#: Asset-class-specific guidance shown under the backtest.
BACKTEST_NOTES: Final[Dict[str, str]] = {
    "FX": (
        "A long-only position in a currency pair has no reliable risk "
        "premium, so Sharpe ratios near zero are expected for every "
        "strategy. Spot returns also exclude carry (the interest "
        "differential earned through forward points)."
    ),
    "Commodity": (
        "Gold pays no cash flows, so returns are pure price changes. If "
        "the futures fallback (GC=F) is in use, prices are the continuous "
        "front-month contract: roll dates can add small artificial jumps, "
        "and roll yield and financing are not modeled."
    ),
    "Equity Index": (
        "Returns are on the price index and exclude dividends (roughly "
        "1–2% a year for these indices). Equities carry a risk premium, "
        "so positive Sharpe ratios are normal. Vol targeting has "
        "historically helped equities most, because high-volatility "
        "periods tend to coincide with losses (Moreira & Muir, 2017)."
    ),
    "Custom": (
        "Check what the symbol represents (spot, futures, price index or "
        "total-return index) before interpreting returns."
    ),
}

FONT_URL: Final[str] = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
    "&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
)

THEME: Final[Dict[str, str]] = {
    "bg": "#0B0F17",
    "panel": "#111827",
    "border": "#1F2937",
    "grid": "#1A2333",
    "text": "#E5E7EB",
    "muted": "#94A3B8",
    "accent": "#38BDF8",
    "candle_up": "#E5E7EB",
    "candle_down": "#64748B",
    "font": "IBM Plex Sans, Inter, -apple-system, Segoe UI, sans-serif",
    "mono": "IBM Plex Mono, SFMono-Regular, Menlo, monospace",
}

REGIME_COLORS: Final[Dict[str, str]] = {
    "Low Vol": "#10B981",
    "Normal Vol": "#3B82F6",
    "High Vol": "#EF4444",
}

#: Folder for bundled price snapshots (used when Yahoo is unreachable).
SNAPSHOT_DIR: Final[Path] = Path(__file__).resolve().parent / "data"

LAB_LONG: Final[str] = "Long (vol-targeted)"
LAB_TREND: Final[str] = "EMA Trend"
LAB_TREND_REGIME: Final[str] = "EMA Trend + Regime"
LAB_REVERSION: Final[str] = "BB Reversion"
LAB_REVERSION_REGIME: Final[str] = "BB Reversion + Regime"
LAB_COLORS: Final[Dict[str, str]] = {
    LAB_LONG: "#E5E7EB",
    LAB_TREND: "#F59E0B",
    LAB_TREND_REGIME: "#FB7185",
    LAB_REVERSION: "#A78BFA",
    LAB_REVERSION_REGIME: "#2DD4BF",
}

#: RiskMetrics daily decay factor for the EWMA variance benchmark.
EWMA_DECAY: Final[float] = 0.94
#: Window (days) for rolling strategy-volatility diagnostics.
ROLLING_RISK_WINDOW: Final[int] = 63

FORECAST_MODELS: Final[Tuple[str, ...]] = (
    "Historical", "EWMA", "GARCH", "HMM",
)
MODEL_COLORS: Final[Dict[str, str]] = {
    "Realized": "#E5E7EB",
    "Historical": "#94A3B8",
    "EWMA": "#F59E0B",
    "GARCH": "#A78BFA",
    "HMM": "#38BDF8",
}
BUY_AND_HOLD: Final[str] = "Buy & Hold"
HMM_DERISK: Final[str] = "HMM De-risk"
STRATEGY_COLORS: Final[Dict[str, str]] = {
    BUY_AND_HOLD: "#E5E7EB",
    **{f"VolTgt · {m}": MODEL_COLORS[m] for m in FORECAST_MODELS},
    HMM_DERISK: "#10B981",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DataUnavailableError(RuntimeError):
    """Raised when the data vendor returns no usable price history."""


class InsufficientDataError(ValueError):
    """Raised when a sample is too short to fit or evaluate the model."""


class ModelFitError(RuntimeError):
    """Raised when every EM restart fails to produce a valid model."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """User-controlled settings for one end-to-end model run.

    Attributes:
        instrument: The market to model (see :class:`Instrument`).
        start: First calendar date requested from the vendor.
        end: Last calendar date requested (inclusive).
        train_fraction: Share of usable observations, in chronological
            order, assigned to the training sample (0 < f < 1).
        vol_window: Look-back length *w* (in trading days) of the rolling
            realized-volatility estimator.
        n_restarts: Number of EM random restarts; the restart with the
            highest training log-likelihood is retained.
        random_seed: Base seed; restart *k* uses ``random_seed + k``.
        max_iter: Maximum EM iterations per restart.
    """

    instrument: Instrument
    start: date
    end: date
    train_fraction: float
    vol_window: int
    n_restarts: int = 10
    random_seed: int = 42
    max_iter: int = 500


@dataclass(frozen=True)
class StandardScalerParams:
    """Feature-wise location/scale estimated on the training sample only."""

    mean: FloatArray
    std: FloatArray

    def transform(self, x: FloatArray) -> FloatArray:
        """Return ``(x - mean) / std`` using the frozen training moments."""
        return (x - self.mean) / self.std


@dataclass(frozen=True)
class HMMParameters:
    """Gaussian HMM parameters, with states ordered Low -> High volatility.

    Attributes:
        start_prob: Initial state distribution, shape ``(K,)``.
        trans_mat: Row-stochastic transition matrix ``A``, shape ``(K, K)``,
            where ``A[i, j] = P(S_t = j | S_{t-1} = i)``.
        means: Emission means in standardized feature space, ``(K, D)``.
        covars: Full emission covariance matrices, ``(K, D, D)``.
    """

    start_prob: FloatArray
    trans_mat: FloatArray
    means: FloatArray
    covars: FloatArray

    @property
    def n_states(self) -> int:
        """Number of hidden states K."""
        return int(self.trans_mat.shape[0])

    @property
    def n_features(self) -> int:
        """Dimension D of the observation vector."""
        return int(self.means.shape[1])

    @property
    def n_free_parameters(self) -> int:
        """Free parameter count used for the BIC penalty.

        ``(K-1)`` initial probabilities, ``K(K-1)`` transition probabilities,
        ``KD`` means and ``K D(D+1)/2`` covariance terms.
        """
        k, d = self.n_states, self.n_features
        return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2


@dataclass(frozen=True)
class FitDiagnostics:
    """Goodness-of-fit and optimisation diagnostics for the selected model."""

    train_log_likelihood: float
    train_ll_per_obs: float
    oos_ll_per_obs: float
    bic: float
    converged: bool
    n_iter: int
    n_restarts_ok: int
    n_restarts_converged: int


@dataclass(frozen=True)
class RegimeResult:
    """Everything the UI needs from one model run."""

    frame: pd.DataFrame
    params: HMMParameters
    scaler: StandardScalerParams
    diagnostics: FitDiagnostics
    config: ModelConfig
    n_train: int
    symbol: str
    data_source: str

    @property
    def train_frame(self) -> pd.DataFrame:
        """In-sample rows (used to estimate every model parameter)."""
        return self.frame.iloc[: self.n_train]

    @property
    def test_frame(self) -> pd.DataFrame:
        """Out-of-sample rows (never seen during estimation)."""
        return self.frame.iloc[self.n_train:]


@dataclass(frozen=True)
class WalkForwardConfig:
    """Settings for walk-forward re-estimation and forecasting.

    Attributes:
        refit_every: Trading days between re-estimations. Each refit uses
            an expanding window of all data before the refit date.
        horizon: Forecast horizon *h*. Each forecast targets the average
            annualized variance over days ``t+1 .. t+h``.
        n_restarts: EM restarts per refit; the first restart is
            warm-started from the previous refit's parameters.
    """

    refit_every: int = 63
    horizon: int = 5
    n_restarts: int = 3


@dataclass(frozen=True)
class BacktestConfig:
    """Volatility-targeting backtest settings.

    Attributes:
        target_vol: Annualized volatility target (decimal, 0.08 = 8%).
        max_leverage: Cap on gross exposure (multiple of capital).
        cost_bps: Cost in basis points per unit of notional traded
            (half bid-ask spread plus slippage).
        rebalance_band: No-trade band. The position is only changed when
            the target leverage moves by more than this amount.
    """

    target_vol: float = 0.08
    max_leverage: float = 3.0
    cost_bps: float = 1.0
    rebalance_band: float = 0.10


@dataclass(frozen=True)
class GarchParams:
    """Zero-mean GARCH(1,1) parameters for returns measured in percent."""

    omega: float
    alpha: float
    beta: float
    sigma2_0: float
    log_likelihood: float

    @property
    def persistence(self) -> float:
        """alpha + beta: the decay rate of variance shocks."""
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        """Unconditional daily variance omega / (1 - alpha - beta)."""
        return self.omega / max(1.0 - self.persistence, 1e-6)


@dataclass(frozen=True)
class WalkForwardResult:
    """Out-of-sample forecasts produced by :class:`WalkForwardEngine`.

    ``frame`` is indexed by forecast date *t* (the close at which the
    forecast is issued) and holds:

    * ``target_var``: realized average annualized variance over
      ``t+1 .. t+h`` (NaN for the last *h* rows).
    * ``fc_<model>``: each model's annualized variance forecast.
    * ``p_<regime>``: filtered ``P(S_t = regime | data up to t)``.
    * ``pnext_<regime>``: one-day-ahead ``P(S_{t+1} = regime | data to t)``.
    * ``p_high_next``: ``P(S_{t+1} = High | data up to t)``.
    * ``regime``: argmax of the filtered probabilities at *t*.
    * ``ret_next``: simple return from close *t* to close *t+1*.

    ``close_history`` holds the full close series (in-sample included) so
    that price-based indicators can be warmed up without look-ahead.
    """

    frame: pd.DataFrame
    refits: pd.DataFrame
    horizon: int
    refit_every: int
    close_history: pd.Series


@dataclass(frozen=True)
class BacktestResult:
    """Net daily strategy returns, positions and costs.

    All frames are indexed by the date on which the return is realized
    (day *t+1*); the position on that row was chosen at close *t*.
    """

    returns: pd.DataFrame
    weights: pd.DataFrame
    costs: pd.DataFrame
    metrics: pd.DataFrame
    config: BacktestConfig


@dataclass(frozen=True)
class SignalConfig:
    """Pre-specified parameters for the Signal Lab.

    Only textbook defaults are offered, and none are optimised. Tuning
    indicator lengths on the evaluation sample would be data snooping.

    Attributes:
        ema_fast: Fast EMA span (days).
        ema_slow: Slow EMA span (days).
        bb_window: Bollinger moving-average window (days).
        bb_width: Band half-width in rolling standard deviations.
        bootstrap_block: Block length for the Sharpe-difference bootstrap.
        n_bootstrap: Number of bootstrap resamples.
        seed: Bootstrap random seed.
    """

    ema_fast: int = 50
    ema_slow: int = 200
    bb_window: int = 20
    bb_width: float = 2.0
    bootstrap_block: int = 20
    n_bootstrap: int = 2000
    seed: int = 7


@dataclass(frozen=True)
class SignalLabResult:
    """Outputs of :class:`SignalLab`."""

    backtest: BacktestResult
    indicators: pd.DataFrame
    filter_tests: pd.DataFrame
    regime_breakdown: pd.DataFrame
    config: SignalConfig


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

class DataFetcher:
    """Download and clean daily OHLC bars for one Yahoo Finance symbol.

    Yahoo's FX and spot-metal bars are snapshots of a 24-hour
    over-the-counter market, so their "close" is a vendor-defined cut.
    Index closes are exchange closing levels, and futures (``GC=F``) are
    continuous front-month prices. Cleaning therefore focuses on removing
    structurally invalid rows (non-positive prices, weekend prints,
    duplicated timestamps) and enforcing ``Low <= {Open, Close} <= High``.
    """

    PRICE_COLUMNS: Final[Tuple[str, ...]] = ("Open", "High", "Low", "Close")

    def __init__(self, ticker: str, start: date, end: date) -> None:
        if start >= end:
            raise ValueError("Start date must be before end date.")
        self.ticker = ticker.strip().upper()
        self.start = start
        self.end = end

    def fetch(self) -> pd.DataFrame:
        """Return a cleaned OHLC frame indexed by naive daily timestamps."""
        return self._clean(self._download())

    def _download(self) -> pd.DataFrame:
        """Query yfinance. The vendor's ``end`` bound is exclusive."""
        try:
            history = yf.Ticker(self.ticker).history(
                start=self.start.isoformat(),
                end=(self.end + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=False,
            )
        except Exception as exc:  # vendor/network errors vary by version
            raise DataUnavailableError(
                f"Download failed for {self.ticker}: {exc}"
            ) from exc
        if history is None or history.empty:
            raise DataUnavailableError(
                f"No data returned for {self.ticker} between "
                f"{self.start} and {self.end}."
            )
        return history

    @classmethod
    def _clean(cls, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalise columns and index, then drop structurally bad rows."""
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        missing = [c for c in cls.PRICE_COLUMNS if c not in df.columns]
        if missing:
            raise DataUnavailableError(f"Vendor data missing {missing}.")

        df = df.loc[:, list(cls.PRICE_COLUMNS)].astype(float)
        index = pd.DatetimeIndex(df.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        df.index = index.normalize()
        df.index.name = "Date"

        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = df[(df > 0.0).all(axis=1)]
        df = df[df.index.dayofweek < 5]

        cols = list(cls.PRICE_COLUMNS)
        df["High"] = df[cols].max(axis=1)
        df["Low"] = df[cols].min(axis=1)

        if df.empty:
            raise DataUnavailableError("No valid rows remain after cleaning.")
        return df


def snapshot_path(symbol: str) -> Path:
    """File used to store a bundled snapshot for ``symbol``."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_")
    return SNAPSHOT_DIR / f"{safe}.csv"


def load_snapshot(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Read and clean a bundled CSV snapshot, restricted to [start, end]."""
    path = snapshot_path(symbol)
    if not path.exists():
        raise DataUnavailableError(f"No snapshot file for {symbol}.")
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = DataFetcher._clean(raw)
    prices = prices.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if prices.empty:
        raise DataUnavailableError(
            f"Snapshot for {symbol} has no data in the selected range."
        )
    return prices


def save_snapshots(start: date = EARLIEST_DATE) -> None:
    """Download every built-in instrument and save CSV snapshots.

    Run ``python app.py --snapshot`` locally, then commit the ``data/``
    folder so a hosted copy of the app still works when Yahoo blocks or
    rate-limits the host's IP address.
    """
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    for inst in INSTRUMENTS:
        for symbol in inst.symbols:
            try:
                prices = DataFetcher(symbol, start, date.today()).fetch()
            except DataUnavailableError as exc:
                print(f"skip {symbol}: {exc}")
                continue
            path = snapshot_path(symbol)
            prices.to_csv(path)
            print(f"saved {symbol}: {len(prices):,} rows -> {path}")


@st.cache_data(ttl=3600, show_spinner=False)
def load_price_history(
    symbols: Tuple[str, ...],
    start: date,
    end: date,
) -> Tuple[pd.DataFrame, str, str]:
    """Return ``(prices, symbol_used, source)`` for the best candidate.

    Live Yahoo data is tried first for every candidate symbol, and the
    best-covered series wins. Only if every live download fails are the
    bundled snapshots in ``data/`` used (``source == "snapshot"``).
    Cached for one hour.

    Raises:
        DataUnavailableError: If neither live data nor a snapshot exists.
    """
    errors: List[str] = []
    for source, loader in (
        ("live", lambda sym: DataFetcher(sym, start, end).fetch()),
        ("snapshot", lambda sym: load_snapshot(sym, start, end)),
    ):
        best: Optional[pd.DataFrame] = None
        best_symbol = ""
        for symbol in symbols:
            try:
                prices = loader(symbol)
            except DataUnavailableError as exc:
                errors.append(str(exc))
                continue
            if best is None or len(prices) > len(best):
                best, best_symbol = prices, symbol
        if best is not None:
            return best, best_symbol, source
    raise DataUnavailableError(" | ".join(errors))


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class FeatureEngineer:
    r"""Construct HMM observations from prices without look-ahead.

    For close prices :math:`P_t` the daily log return is

    .. math:: r_t = \ln(P_t / P_{t-1}).

    Rolling realized volatility over a window of *w* days is the
    zero-mean estimator

    .. math:: \sigma_t = \sqrt{\tfrac{252}{w} \sum_{i=0}^{w-1} r_{t-i}^2},

    which uses only returns observed up to and including day *t*. The
    zero-mean form is standard for daily returns of liquid markets, where
    the drift is an order of magnitude smaller than its own estimation
    error.

    The observation vector is :math:`x_t = (r_t, \ln \sigma_t)`. Taking
    logs makes the volatility feature far closer to Gaussian (realized
    variance is approximately log-normal), which matters because the HMM
    emission densities are Gaussian.
    """

    def __init__(
        self,
        vol_window: int,
        annualization: int = ANNUALIZATION_FACTOR,
    ) -> None:
        if vol_window < 2:
            raise ValueError("Volatility window must be at least 2 days.")
        self.vol_window = vol_window
        self.annualization = annualization

    def build(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Append return/volatility features and drop the warm-up rows."""
        out = prices.copy()
        out["log_return"] = np.log(out["Close"]).diff()
        mean_sq = (
            out["log_return"].pow(2)
            .rolling(self.vol_window, min_periods=self.vol_window)
            .mean()
        )
        out["realized_vol"] = np.sqrt(self.annualization * mean_sq)
        with np.errstate(divide="ignore"):
            out["log_realized_vol"] = np.log(out["realized_vol"])
        out = out.replace([np.inf, -np.inf], np.nan)
        return out.dropna(subset=list(FEATURE_COLUMNS))

    @staticmethod
    def chronological_split(n_obs: int, train_fraction: float) -> int:
        """Return the number of training rows for a time-ordered split.

        The first ``floor(f * N)`` rows form the training sample and the
        remainder is the out-of-sample (test) block. No shuffling is ever
        applied: shuffling would leak future information into training
        through the serial dependence of volatility.

        Raises:
            InsufficientDataError: If either block is below its minimum.
        """
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must lie strictly in (0, 1).")
        n_train = int(np.floor(train_fraction * n_obs))
        n_test = n_obs - n_train
        if n_train < MIN_TRAIN_OBS or n_test < MIN_TEST_OBS:
            raise InsufficientDataError(
                f"Need >= {MIN_TRAIN_OBS} training and >= {MIN_TEST_OBS} "
                f"test observations; got {n_train} and {n_test}. "
                "Widen the date range or change the split."
            )
        return n_train

    @staticmethod
    def fit_scaler(x_train: FloatArray) -> StandardScalerParams:
        """Estimate standardization moments on the training block only.

        Using full-sample moments would leak the out-of-sample mean and
        variance of volatility into the model's feature space.
        """
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0, ddof=1)
        std = np.where(std > 0.0, std, 1.0)
        return StandardScalerParams(mean=mean, std=std)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RegimeModeler:
    r"""Gaussian Hidden Markov Model for volatility regimes.

    Model
    -----
    A latent state :math:`S_t \in \{1, \dots, K\}` follows a first-order,
    time-homogeneous Markov chain

    .. math:: P(S_t = j \mid S_{t-1} = i) = A_{ij},
              \qquad P(S_1 = k) = \pi_k,

    and, conditional on the state, the standardized observation is
    Gaussian

    .. math:: z_t \mid S_t = k \sim \mathcal{N}(\mu_k, \Sigma_k),

    with observations conditionally independent given the state path.
    The marginal law of :math:`z_t` is therefore a *mixture* of Gaussians
    whose weights evolve persistently through :math:`A`, which reproduces
    volatility clustering and fat-tailed unconditional returns.

    Estimation
    ----------
    Parameters :math:`\theta = (\pi, A, \mu, \Sigma)` are estimated by
    maximum likelihood with the EM (Baum-Welch) algorithm, **using the
    training block only**. EM converges to a local optimum, so the fit is
    repeated from several random initialisations and the restart with the
    highest training log-likelihood is retained.

    Label identification
    --------------------
    HMM states are only identified up to permutation. After fitting,
    states are sorted by the emission mean of :math:`\ln\sigma_t`, which
    maps them deterministically to Low / Normal / High volatility. The
    ordering depends only on training-sample parameters.

    Out-of-sample inference (no look-ahead)
    ---------------------------------------
    The Viterbi path and the forward-backward ("smoothed") posteriors
    :math:`P(S_t \mid z_{1:T})` condition on *future* observations and so
    cannot be used to evaluate a model out of sample. This class instead
    runs the forward recursion (Hamilton filter) with parameters frozen at
    their training estimates:

    .. math::
        \alpha_t(j) = \frac{b_j(z_t) \sum_i \alpha_{t-1}(i) A_{ij}}
                           {c_t},
        \qquad c_t = p(z_t \mid z_{1:t-1}),

    where :math:`b_j(z) = \mathcal{N}(z; \mu_j, \Sigma_j)`. The filtered
    probability :math:`\alpha_t(j) = P(S_t = j \mid z_{1:t})` uses only
    information available at the close of day *t*, and the regime call is
    :math:`\arg\max_j \alpha_t(j)`. The normalisers :math:`c_t` are
    one-step-ahead predictive densities, so :math:`\sum_t \ln c_t` over
    the test block is a genuine out-of-sample log-likelihood.
    """

    def __init__(
        self,
        n_states: int = N_REGIMES,
        n_restarts: int = 10,
        max_iter: int = 500,
        tol: float = 1e-5,
        random_seed: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_restarts = max(1, n_restarts)
        self.max_iter = max_iter
        self.tol = tol
        self.random_seed = random_seed
        self._params: Optional[HMMParameters] = None
        self._model: Optional[GaussianHMM] = None
        self._n_ok: int = 0
        self._n_converged: int = 0

    # -- estimation ---------------------------------------------------------

    def fit(
        self,
        z_train: FloatArray,
        warm_start: Optional[HMMParameters] = None,
    ) -> "RegimeModeler":
        """Fit the HMM on the training block with multiple EM restarts.

        Args:
            z_train: Standardized training observations, shape ``(T, D)``.
            warm_start: Optional parameters from an earlier fit. If given,
                the first restart starts EM from them (the walk-forward
                engine uses this) and the remaining restarts are random.

        Returns:
            ``self``, for chaining.

        Raises:
            ModelFitError: If no restart yields a finite log-likelihood.
        """
        best_model: Optional[GaussianHMM] = None
        best_score = -np.inf
        self._n_ok = 0
        self._n_converged = 0

        for k in range(self.n_restarts):
            warm = warm_start is not None and k == 0
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.max_iter,
                tol=self.tol,
                min_covar=1e-4,
                random_state=self.random_seed + k,
                init_params="" if warm else "stmc",
                params="stmc",
            )
            if warm and warm_start is not None:
                model.startprob_ = warm_start.start_prob.copy()
                model.transmat_ = warm_start.trans_mat.copy()
                model.means_ = warm_start.means.copy()
                model.covars_ = warm_start.covars.copy()
            try:
                model.fit(z_train)
                score = float(model.score(z_train))
            except (ValueError, AttributeError, np.linalg.LinAlgError):
                continue
            if not np.isfinite(score):
                continue
            self._n_ok += 1
            self._n_converged += int(model.monitor_.converged)
            if score > best_score:
                best_score, best_model = score, model

        if best_model is None:
            raise ModelFitError("All EM restarts failed to converge.")
        self._model = best_model
        self._params = self._canonicalize(best_model)
        return self

    @staticmethod
    def _canonicalize(model: GaussianHMM) -> HMMParameters:
        """Permute states so they are ordered by mean log-volatility."""
        order = np.argsort(model.means_[:, VOL_FEATURE_INDEX])
        return HMMParameters(
            start_prob=np.asarray(model.startprob_[order], dtype=float),
            trans_mat=np.asarray(
                model.transmat_[np.ix_(order, order)], dtype=float
            ),
            means=np.asarray(model.means_[order], dtype=float),
            covars=np.asarray(model.covars_[order], dtype=float),
        )

    # -- inference ----------------------------------------------------------

    @property
    def params(self) -> HMMParameters:
        """Fitted parameters (states ordered Low -> High)."""
        if self._params is None:
            raise RuntimeError("Model has not been fitted.")
        return self._params

    def emission_log_likelihood(self, z: FloatArray) -> FloatArray:
        """Return ``log b_k(z_t)`` for every t and k, shape ``(T, K)``."""
        p = self.params
        return np.column_stack([
            multivariate_normal.logpdf(z, mean=p.means[k], cov=p.covars[k])
            for k in range(p.n_states)
        ]).reshape(len(z), p.n_states)

    def filter(self, z: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """Run the causal forward (Hamilton) filter in log space.

        Args:
            z: Standardized observations in time order, shape ``(T, D)``.

        Returns:
            ``(probs, log_c)`` where ``probs[t, k] = P(S_t = k | z_{1:t})``
            and ``log_c[t] = log p(z_t | z_{1:t-1})``.
        """
        p = self.params
        log_b = self.emission_log_likelihood(z)
        with np.errstate(divide="ignore"):
            log_a = np.log(p.trans_mat)
            log_prior = np.log(p.start_prob)

        n_obs = log_b.shape[0]
        log_alpha = np.empty_like(log_b)
        log_c = np.empty(n_obs)
        for t in range(n_obs):
            joint = log_prior + log_b[t]
            log_c[t] = logsumexp(joint)
            log_alpha[t] = joint - log_c[t]
            # One-step-ahead prior: P(S_{t+1} | z_{1:t}).
            log_prior = logsumexp(log_alpha[t][:, None] + log_a, axis=0)
        return np.exp(log_alpha), log_c

    @staticmethod
    def return_second_moments(
        params: HMMParameters,
        scaler: StandardScalerParams,
    ) -> FloatArray:
        r"""Daily :math:`m_k = E[r_t^2 \mid S_t = k]` in decimal units.

        The return feature is :math:`r_t = \bar r + s_r z_{t,0}`, so within
        state *k* it has mean :math:`\bar r + s_r\mu_{k,0}` and variance
        :math:`s_r^2\Sigma_{k,00}`. The second moment is variance plus
        squared mean, which matches the zero-mean realized-variance target.
        """
        s_r, m_r = scaler.std[0], scaler.mean[0]
        state_mean = m_r + s_r * params.means[:, 0]
        state_var = s_r ** 2 * params.covars[:, 0, 0]
        return state_var + state_mean ** 2

    @staticmethod
    def average_transition(trans_mat: FloatArray, horizon: int) -> FloatArray:
        r"""Return :math:`\tfrac1h\sum_{i=1}^{h} A^i`.

        Row *j* is the average distribution of the state over the next *h*
        days, given the current state is *j*.
        """
        k = trans_mat.shape[0]
        power = np.eye(k)
        acc = np.zeros((k, k))
        for _ in range(horizon):
            power = power @ trans_mat
            acc += power
        return acc / horizon

    @property
    def n_iter(self) -> int:
        """EM iterations used by the selected restart."""
        return int(self._model.monitor_.iter) if self._model else 0

    @property
    def converged(self) -> bool:
        """Whether the selected restart met the EM tolerance."""
        return bool(self._model.monitor_.converged) if self._model else False

    @property
    def restart_summary(self) -> Tuple[int, int]:
        """``(successful restarts, converged restarts)``."""
        return self._n_ok, self._n_converged


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class RegimeAnalytics:
    """Descriptive statistics for regime sequences and fitted parameters."""

    @staticmethod
    def runs(regimes: pd.Series) -> List[Tuple[int, int, str]]:
        """Return contiguous runs as ``(start_pos, end_pos_excl, label)``."""
        values = regimes.to_numpy()
        if values.size == 0:
            return []
        breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [values.size]))
        return [(int(s), int(e), str(values[s])) for s, e in zip(starts, ends)]

    @classmethod
    def current_spell(cls, regimes: pd.Series) -> int:
        """Length in days of the most recent regime spell."""
        spells = cls.runs(regimes)
        return spells[-1][1] - spells[-1][0] if spells else 0

    @staticmethod
    def stationary_distribution(trans_mat: FloatArray) -> FloatArray:
        """Solve ``pi A = pi`` with ``sum(pi) = 1`` by least squares."""
        k = trans_mat.shape[0]
        lhs = np.vstack([trans_mat.T - np.eye(k), np.ones(k)])
        rhs = np.zeros(k + 1)
        rhs[-1] = 1.0
        pi, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        pi = np.clip(pi, 0.0, None)
        return pi / pi.sum()

    @classmethod
    def regime_statistics(
        cls,
        frame: pd.DataFrame,
        trans_mat: FloatArray,
        labels: Sequence[str] = REGIME_LABELS,
    ) -> pd.DataFrame:
        """Per-regime summary of returns, volatility and persistence.

        Expected duration uses the geometric sojourn law of a Markov chain:
        ``E[duration_k] = 1 / (1 - A_kk)``.
        """
        spells = cls.runs(frame["regime"])
        n_total = max(len(frame), 1)
        rows: List[Dict[str, object]] = []
        for k, label in enumerate(labels):
            sub = frame.loc[frame["regime"] == label]
            lengths = [e - s for s, e, lab in spells if lab == label]
            p_kk = float(trans_mat[k, k])
            rets = sub["log_return"]
            rows.append({
                "Regime": label,
                "Days": int(len(sub)),
                "Time Share (%)": 100.0 * len(sub) / n_total,
                "Ann. Mean Return (%)": (
                    100.0 * ANNUALIZATION_FACTOR * rets.mean()
                    if len(sub) else np.nan
                ),
                "Ann. Return Vol (%)": (
                    100.0 * np.sqrt(ANNUALIZATION_FACTOR) * rets.std(ddof=1)
                    if len(sub) > 1 else np.nan
                ),
                "Avg Realized Vol (%)": (
                    100.0 * sub["realized_vol"].mean() if len(sub) else np.nan
                ),
                "Spells": len(lengths),
                "Avg Spell (days)": (
                    float(np.mean(lengths)) if lengths else np.nan
                ),
                "Persistence A_kk": p_kk,
                "Expected Duration (days)": (
                    1.0 / (1.0 - p_kk) if p_kk < 1.0 else np.inf
                ),
            })
        return pd.DataFrame(rows)

    @classmethod
    def emission_summary(
        cls,
        params: HMMParameters,
        scaler: StandardScalerParams,
        labels: Sequence[str] = REGIME_LABELS,
    ) -> pd.DataFrame:
        """Translate standardized emission parameters into market units."""
        stationary = cls.stationary_distribution(params.trans_mat)
        rows: List[Dict[str, object]] = []
        for k, label in enumerate(labels):
            mu_x = params.means[k] * scaler.std + scaler.mean
            cov_x = params.covars[k] * np.outer(scaler.std, scaler.std)
            corr = cov_x[0, 1] / np.sqrt(cov_x[0, 0] * cov_x[1, 1])
            rows.append({
                "Regime": label,
                "Mean Daily Return (bp)": 1e4 * mu_x[0],
                "State Return Vol, ann. (%)": (
                    100.0 * np.sqrt(cov_x[0, 0] * ANNUALIZATION_FACTOR)
                ),
                "Typical Realized Vol (%)": 100.0 * np.exp(mu_x[1]),
                "Corr(r, ln σ)": corr,
                "Stationary Prob.": stationary[k],
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Volatility forecasting models
# ---------------------------------------------------------------------------

class EwmaForecaster:
    r"""RiskMetrics exponentially weighted moving-average variance.

    .. math::
        \hat\sigma^2_{t+1} = \lambda\,\hat\sigma^2_t + (1-\lambda)\,r_t^2

    With the RiskMetrics daily decay :math:`\lambda = 0.94` there is no
    parameter to estimate, so the forecast is causal by construction.
    EWMA is an integrated GARCH(1,1) with no intercept: shocks never decay
    toward a long-run mean, so the *h*-step forecast is flat at
    :math:`\hat\sigma^2_{t+1}`.
    """

    def __init__(self, decay: float = EWMA_DECAY, init_window: int = 21):
        self.decay = decay
        self.init_window = init_window

    def variance_path(self, r: FloatArray) -> FloatArray:
        """Return ``s`` with ``s[t] = Var(r_t | F_{t-1})``, length ``T+1``.

        ``s[T]`` is the forecast for the day after the last observation.
        The seed is the mean squared return of the first ``init_window``
        days, which only affects a burn-in period long before any
        out-of-sample date.
        """
        lam = self.decay
        s0 = float(np.mean(r[: self.init_window] ** 2))
        rest, _ = lfilter([1.0], [1.0, -lam], (1.0 - lam) * r ** 2,
                          zi=[lam * s0])
        return np.concatenate(([s0], rest))


class GarchForecaster:
    r"""Zero-mean GARCH(1,1) estimated by Gaussian quasi-maximum likelihood.

    .. math::
        r_t = \sigma_t\varepsilon_t, \qquad
        \sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta\sigma_{t-1}^2,

    subject to :math:`\omega > 0`, :math:`\alpha, \beta \ge 0` and
    :math:`\alpha + \beta < 1` (covariance stationarity). The QMLE
    maximises :math:`-\tfrac12\sum_t(\ln\sigma_t^2 + r_t^2/\sigma_t^2)`
    and stays consistent when :math:`\varepsilon_t` is not Gaussian
    (Bollerslev & Wooldridge, 1992). Returns are scaled to percent for
    numerical conditioning, and the variance recursion is evaluated as a
    linear IIR filter.

    Multi-step forecasts revert geometrically to the long-run variance
    :math:`\bar\sigma^2 = \omega / (1 - \alpha - \beta)`:

    .. math::
        E_t[\sigma_{t+i}^2] = \bar\sigma^2
            + (\alpha+\beta)^{i-1}\,(\sigma_{t+1}^2 - \bar\sigma^2).
    """

    MAX_PERSISTENCE: Final[float] = 0.9995

    @staticmethod
    def variance_path(
        r_pct: FloatArray,
        omega: float,
        alpha: float,
        beta: float,
        sigma2_0: float,
    ) -> FloatArray:
        """Conditional variances ``s[t] = Var(r_t | F_{t-1})``, length T+1."""
        drive = omega + alpha * r_pct ** 2
        rest, _ = lfilter([1.0], [1.0, -beta], drive, zi=[beta * sigma2_0])
        return np.concatenate(([sigma2_0], rest))

    @classmethod
    def _neg_log_likelihood(
        cls,
        theta: FloatArray,
        r_pct: FloatArray,
        sigma2_0: float,
    ) -> float:
        omega, alpha, beta = theta
        sig2 = cls.variance_path(r_pct, omega, alpha, beta, sigma2_0)[:-1]
        if not np.all(np.isfinite(sig2)) or np.any(sig2 <= 0.0):
            return 1e12
        return 0.5 * float(np.sum(np.log(sig2) + r_pct ** 2 / sig2))

    def fit(
        self,
        r_pct: FloatArray,
        warm_start: Optional[GarchParams] = None,
    ) -> GarchParams:
        """Estimate (omega, alpha, beta) by constrained QMLE (SLSQP).

        Several starting points are tried (including the previous refit's
        estimate when given) and the best likelihood is kept.
        """
        var = float(np.var(r_pct))
        starts: List[FloatArray] = [
            np.array([0.05 * var, 0.05, 0.90]),
            np.array([0.02 * var, 0.03, 0.95]),
        ]
        if warm_start is not None:
            starts.insert(0, np.array(
                [warm_start.omega, warm_start.alpha, warm_start.beta]
            ))
        bounds = [(1e-8, 10.0 * var), (0.0, 1.0), (0.0, 1.0)]
        constraints = [{
            "type": "ineq",
            "fun": lambda th: self.MAX_PERSISTENCE - th[1] - th[2],
        }]

        best_x: Optional[FloatArray] = None
        best_f = np.inf
        for x0 in starts:
            res = minimize(
                self._neg_log_likelihood, x0, args=(r_pct, var),
                method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": 300, "ftol": 1e-10},
            )
            if np.isfinite(res.fun) and res.fun < best_f:
                best_f, best_x = float(res.fun), np.asarray(res.x)
        if best_x is None:
            raise ModelFitError("GARCH(1,1) estimation failed.")

        omega = float(max(best_x[0], 1e-8))
        alpha = float(np.clip(best_x[1], 0.0, 1.0))
        beta = float(np.clip(best_x[2], 0.0, self.MAX_PERSISTENCE - alpha))
        return GarchParams(omega, alpha, beta, var, -best_f)

    @staticmethod
    def average_forecast(
        sigma2_next: FloatArray,
        params: GarchParams,
        horizon: int,
    ) -> FloatArray:
        """Average of the 1..h step variance forecasts (percent^2 daily)."""
        decay = params.persistence ** np.arange(horizon)
        lr = params.long_run_variance
        return lr + (sigma2_next - lr) * decay.mean()


class WalkForwardEngine:
    r"""Expanding-window walk-forward re-estimation and forecasting.

    Protocol (refit dates :math:`t_0 < t_1 < \dots`, with :math:`t_0` the
    first out-of-sample day)::

        |------ estimation [0, t_k) ------|-- forecasts [t_k, t_k+1) --|

    At each refit date :math:`t_k` every estimated object (scaler
    moments, HMM parameters and state ordering, GARCH parameters) is
    re-estimated on all observations **strictly before** :math:`t_k`.
    Parameters are then frozen, and a forecast is issued at the close of
    each day :math:`t \in [t_k, t_{k+1})` using data up to day *t* only.
    No forecast ever depends on data or parameters from after its date.

    Every model forecasts the same target, the average annualized variance
    over the next *h* days:

    .. math:: y_t = \frac{252}{h}\sum_{i=1}^{h} r_{t+i}^2 .

    * **Historical**: the current rolling realized variance (random-walk
      benchmark).
    * **EWMA**: the flat RiskMetrics forecast.
    * **GARCH(1,1)**: the mean-reverting average of the *h* step forecasts.
    * **HMM**: the regime-mixture second moment

      .. math::
          \hat y_t = 252\;\alpha_t^\top
              \Big(\tfrac1h\textstyle\sum_{i=1}^{h} A^i\Big)\, m,
          \qquad m_k = E[r^2 \mid S = k],

      where :math:`\alpha_t` is the *filtered* state distribution.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        n_train: int,
        wf_config: WalkForwardConfig,
        random_seed: int = 42,
        max_iter: int = 300,
    ) -> None:
        self.features = features
        self.n_train = n_train
        self.cfg = wf_config
        self.random_seed = random_seed
        self.max_iter = max_iter

    def refit_points(self) -> List[int]:
        """Row positions at which all models are re-estimated."""
        return list(range(self.n_train, len(self.features),
                          self.cfg.refit_every))

    def run(
        self,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> WalkForwardResult:
        """Execute the walk-forward loop and assemble forecasts."""
        f = self.features
        n_obs, h = len(f), self.cfg.horizon
        r = f["log_return"].to_numpy(dtype=float)
        r_pct = 100.0 * r
        x = f.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)

        fc_hmm = np.full(n_obs, np.nan)
        fc_garch = np.full(n_obs, np.nan)
        p_filt = np.full((n_obs, N_REGIMES), np.nan)
        p_next = np.full((n_obs, N_REGIMES), np.nan)
        records: List[Dict[str, object]] = []
        hmm_prev: Optional[HMMParameters] = None
        garch_prev: Optional[GarchParams] = None
        garch = GarchForecaster()
        points = self.refit_points()

        for k, t0 in enumerate(points):
            t1 = min(t0 + self.cfg.refit_every, n_obs)

            # HMM: scaler and parameters from [0, t0); filter up to t1.
            scaler = FeatureEngineer.fit_scaler(x[:t0])
            z = scaler.transform(x[:t1])
            modeler = RegimeModeler(
                n_restarts=self.cfg.n_restarts,
                max_iter=self.max_iter,
                random_seed=self.random_seed + 1000 * k,
            ).fit(z[:t0], warm_start=hmm_prev)
            params = modeler.params
            probs, log_c = modeler.filter(z)
            seg = probs[t0:t1]
            m2 = RegimeModeler.return_second_moments(params, scaler)
            avg_a = RegimeModeler.average_transition(params.trans_mat, h)
            fc_hmm[t0:t1] = ANNUALIZATION_FACTOR * (seg @ avg_a @ m2)
            p_filt[t0:t1] = seg
            p_next[t0:t1] = seg @ params.trans_mat
            hmm_prev = params

            # GARCH: parameters from [0, t0); variance path up to t1.
            gp = garch.fit(r_pct[:t0], warm_start=garch_prev)
            path = garch.variance_path(
                r_pct[:t1], gp.omega, gp.alpha, gp.beta, gp.sigma2_0
            )
            fc_garch[t0:t1] = (
                GarchForecaster.average_forecast(path[t0 + 1:t1 + 1], gp, h)
                * ANNUALIZATION_FACTOR / 1e4
            )
            garch_prev = gp

            records.append({
                "Refit Date": f.index[t0],
                "Estimation Obs": t0,
                "HMM LL/obs": float(log_c[:t0].mean()),
                "HMM Converged": modeler.converged,
                "HMM P(stay High)": float(params.trans_mat[-1, -1]),
                "GARCH α": gp.alpha,
                "GARCH β": gp.beta,
                "GARCH α+β": gp.persistence,
                "GARCH LR Vol (%)": float(
                    np.sqrt(gp.long_run_variance * ANNUALIZATION_FACTOR)
                ),
            })
            if progress is not None:
                progress((k + 1) / len(points),
                         f"Refit {k + 1}/{len(points)} · "
                         f"{f.index[t0]:%Y-%m-%d}")

        ewma = EwmaForecaster().variance_path(r)[1:] * ANNUALIZATION_FACTOR
        target = (
            pd.Series(r ** 2, index=f.index).rolling(h).mean().shift(-h)
            * ANNUALIZATION_FACTOR
        )
        out = pd.DataFrame({
            "target_var": target,
            "fc_Historical": f["realized_vol"] ** 2,
            "fc_EWMA": ewma,
            "fc_GARCH": fc_garch,
            "fc_HMM": fc_hmm,
            "p_high_next": p_next[:, -1],
            "ret_next": f["Close"].pct_change().shift(-1),
            "Close": f["Close"],
        }, index=f.index)
        for k, label in enumerate(REGIME_LABELS):
            out[f"p_{label}"] = p_filt[:, k]
            out[f"pnext_{label}"] = p_next[:, k]
        out = out.iloc[self.n_train:].copy()
        out["regime"] = np.asarray(REGIME_LABELS)[
            np.nan_to_num(p_filt[self.n_train:]).argmax(axis=1)
        ]
        return WalkForwardResult(
            frame=out,
            refits=pd.DataFrame(records),
            horizon=h,
            refit_every=self.cfg.refit_every,
            close_history=f["Close"].copy(),
        )


# ---------------------------------------------------------------------------
# Forecast evaluation
# ---------------------------------------------------------------------------

class ForecastEvaluator:
    r"""Out-of-sample comparison of variance forecasts.

    **Loss.** Realized variance is a noisy proxy for true variance, so
    forecasts are ranked with QLIKE,

    .. math:: L(y, f) = \frac{y}{f} - \ln\frac{y}{f} - 1,

    which Patton (2011) shows is *robust*: it ranks forecasts in the same
    order as the true (unobservable) variance would. It also penalises
    under-prediction of risk more than over-prediction. RMSE on
    volatility is reported for intuition but is not robust in this sense.

    **Diebold–Mariano test.** For loss differential
    :math:`d_t = L_t^{\text{model}} - L_t^{\text{bench}}`,

    .. math:: DM = \frac{\bar d}{\sqrt{\widehat{\text{LRV}}(d)/n}},

    where the long-run variance sums :math:`h-1` autocovariances with a
    rectangular kernel (Diebold & Mariano, 1995), because *h*-day targets
    overlap and make :math:`d_t` an MA(h−1) process. If that estimate is
    not positive, a Bartlett (Newey–West) kernel is used instead. In
    simulation this test rejects close to 5% of the time at the 5% level
    for h = 1, 5 and 21. The Harvey–Leybourne–Newbold
    small-sample correction is applied and p-values use a
    :math:`t_{n-1}` distribution. Negative DM means the model has lower
    loss than the benchmark.

    **Mincer–Zarnowitz regression.** :math:`y_t = a + b f_t + e_t`. An
    unbiased, efficient forecast has :math:`b \approx 1`, and :math:`R^2`
    measures how much realized variance it explains.
    """

    @staticmethod
    def qlike(target: FloatArray, forecast: FloatArray) -> FloatArray:
        """Patton's robust QLIKE loss, elementwise."""
        ratio = np.maximum(target, 1e-12) / forecast
        return ratio - np.log(ratio) - 1.0

    @staticmethod
    def diebold_mariano(
        loss_model: FloatArray,
        loss_bench: FloatArray,
        horizon: int,
    ) -> Tuple[float, float]:
        """Return the HLN-corrected DM statistic and two-sided p-value."""
        d = np.asarray(loss_model - loss_bench, dtype=float)
        n = d.size
        if n < 10:
            return np.nan, np.nan
        dc = d - d.mean()
        lags = max(horizon - 1, 0)
        gammas = [float(dc @ dc) / n] + [
            float(dc[j:] @ dc[:-j]) / n for j in range(1, lags + 1)
        ]
        # Original DM: rectangular kernel, exact for an MA(h-1) process.
        lrv = gammas[0] + 2.0 * sum(gammas[1:])
        if lrv <= 0.0:
            # Fall back to the positive-definite Bartlett kernel.
            lrv = gammas[0] + 2.0 * sum(
                (1.0 - j / (lags + 1)) * g
                for j, g in enumerate(gammas[1:], start=1)
            )
        if lrv <= 0.0:
            return np.nan, np.nan
        dm = d.mean() / np.sqrt(lrv / n)
        hln = np.sqrt(
            max(n + 1 - 2 * horizon + horizon * (horizon - 1) / n, 1.0) / n
        )
        stat = float(dm * hln)
        p_value = float(2.0 * student_t.sf(abs(stat), df=n - 1))
        return stat, p_value

    @staticmethod
    def mincer_zarnowitz(
        target: FloatArray,
        forecast: FloatArray,
    ) -> Tuple[float, float, float]:
        """OLS of target on forecast: returns ``(a, b, R^2)``."""
        f_c = forecast - forecast.mean()
        b = float((f_c @ (target - target.mean())) / (f_c @ f_c))
        a = float(target.mean() - b * forecast.mean())
        r2 = float(np.corrcoef(target, forecast)[0, 1] ** 2)
        return a, b, r2

    @classmethod
    def evaluate(
        cls,
        wf: WalkForwardResult,
        benchmark: str = "EWMA",
    ) -> pd.DataFrame:
        """Loss table for all models against the chosen benchmark."""
        cols = ["target_var"] + [f"fc_{m}" for m in FORECAST_MODELS]
        df = wf.frame.dropna(subset=cols)
        y = df["target_var"].to_numpy(dtype=float)
        losses = {
            m: cls.qlike(y, df[f"fc_{m}"].to_numpy(dtype=float))
            for m in FORECAST_MODELS
        }
        bench_loss = losses[benchmark]
        rows: List[Dict[str, object]] = []
        for m in FORECAST_MODELS:
            fc = df[f"fc_{m}"].to_numpy(dtype=float)
            if m == benchmark:
                dm, p_value = np.nan, np.nan
            else:
                dm, p_value = cls.diebold_mariano(
                    losses[m], bench_loss, wf.horizon
                )
            _, slope, r2 = cls.mincer_zarnowitz(y, fc)
            rows.append({
                "Model": m,
                "QLIKE": float(losses[m].mean()),
                "Δ QLIKE vs Bench (%)": float(
                    100.0 * (losses[m].mean() / bench_loss.mean() - 1.0)
                ),
                "DM Stat": dm,
                "DM p-value": p_value,
                "RMSE (vol pts)": float(np.sqrt(np.mean(
                    (100.0 * (np.sqrt(y) - np.sqrt(fc))) ** 2
                ))),
                "MZ Slope b": slope,
                "MZ R²": r2,
                "Mean Fcst Vol (%)": float(100.0 * np.sqrt(fc).mean()),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

class PerformanceAnalytics:
    """Standard performance and risk statistics for daily returns."""

    @staticmethod
    def summary(
        name: str,
        returns: pd.Series,
        weights: FloatArray,
        costs: FloatArray,
    ) -> Dict[str, object]:
        """Compute CAGR, risk, drawdown, turnover and risk-stability stats.

        *Risk stability* is the standard deviation (in vol points) of the
        rolling 63-day realized volatility of the strategy. Lower values
        mean the strategy delivers a steadier risk level, which is what
        volatility targeting is for.
        """
        r = returns.to_numpy(dtype=float)
        years = len(r) / ANNUALIZATION_FACTOR
        equity = np.concatenate(([1.0], np.cumprod(1.0 + r)))
        drawdown = equity / np.maximum.accumulate(equity) - 1.0
        max_dd = float(drawdown.min())
        cagr = float(equity[-1] ** (1.0 / years) - 1.0)
        ann_vol = float(np.std(r, ddof=1) * np.sqrt(ANNUALIZATION_FACTOR))
        ann_mean = float(np.mean(r) * ANNUALIZATION_FACTOR)
        downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
                         * np.sqrt(ANNUALIZATION_FACTOR))
        rolling = (returns.rolling(ROLLING_RISK_WINDOW).std(ddof=1)
                   * np.sqrt(ANNUALIZATION_FACTOR))
        turnover = np.abs(np.diff(np.concatenate(([0.0], weights))))
        return {
            "Strategy": name,
            "CAGR (%)": 100.0 * cagr,
            "Ann. Vol (%)": 100.0 * ann_vol,
            "Sharpe": ann_mean / ann_vol if ann_vol > 0 else np.nan,
            "Sortino": ann_mean / downside if downside > 0 else np.nan,
            "Max DD (%)": 100.0 * max_dd,
            "Calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
            "Avg Leverage (×)": float(np.mean(np.abs(weights))),
            "Turnover (×/yr)": float(turnover.sum() / years),
            "Cost Drag (bp/yr)": float(1e4 * costs.sum() / years),
            "Risk Stability (pp)": float(100.0 * rolling.std(ddof=1)),
        }


class VolTargetBacktester:
    r"""Volatility-targeting backtest on a long position.

    At the close of day *t* the strategy picks leverage

    .. math::
        w_t^{*} = \min\!\Big(\frac{\sigma^{\star}}{\hat\sigma_{t}},\;
                  L_{\max}\Big),

    where :math:`\hat\sigma_t` is a model's *h*-day volatility forecast
    issued at *t* and :math:`\sigma^\star` is the target. A no-trade band
    keeps the held position :math:`w_t` unchanged unless
    :math:`|w_t^* - w_{t-1}| > \text{band}`. The net return realized on
    day *t+1* is

    .. math::
        R^{\text{net}}_{t+1} = w_t\,R_{t+1} - c\,|w_t - w_{t-1}|,

    with :math:`R_{t+1}` the simple price return and *c* the cost per unit
    of notional traded. Because :math:`w_t` uses forecasts issued at *t*,
    and those forecasts come from the walk-forward engine, the backtest
    has no look-ahead.

    Strategies compared: buy & hold (1×), vol targeting with each
    forecaster, and an **HMM de-risk** overlay
    :math:`w_t = 1 - P(S_{t+1} = \text{High} \mid \mathcal F_t)` that uses
    the regime probabilities directly rather than a volatility number.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.cfg = config

    def target_weights(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Desired (pre-band) leverage for every strategy at each close."""
        out = pd.DataFrame(index=frame.index)
        out[BUY_AND_HOLD] = 1.0
        for m in FORECAST_MODELS:
            sigma = np.sqrt(frame[f"fc_{m}"].to_numpy(dtype=float))
            out[f"VolTgt · {m}"] = np.minimum(
                self.cfg.target_vol / sigma, self.cfg.max_leverage
            )
        out[HMM_DERISK] = 1.0 - frame["p_high_next"].to_numpy(dtype=float)
        return out

    @staticmethod
    def apply_band(target: FloatArray, band: float) -> FloatArray:
        """Hold the position until the target drifts outside the band."""
        held = np.empty_like(target)
        current = target[0]
        for i, w in enumerate(target):
            if abs(w - current) > band:
                current = w
            held[i] = current
        return held

    @staticmethod
    def tradable_window(
        wf: WalkForwardResult,
    ) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        """Decision rows (with a next-day return) and realization dates."""
        frame = wf.frame.dropna(subset=["ret_next"])
        return frame, wf.frame.index[1:len(frame) + 1]

    def run(self, wf: WalkForwardResult) -> BacktestResult:
        """Simulate every strategy over the walk-forward test window."""
        frame, realized_dates = self.tradable_window(wf)
        targets = self.target_weights(frame)
        return self.simulate(
            {c: targets[c].to_numpy(dtype=float) for c in targets.columns},
            frame["ret_next"].to_numpy(dtype=float),
            realized_dates,
        )

    def simulate(
        self,
        targets: Dict[str, FloatArray],
        ret_next: FloatArray,
        realized_dates: pd.DatetimeIndex,
    ) -> BacktestResult:
        """Apply the band and costs to target positions and compute P&L.

        Args:
            targets: Desired signed position per strategy at each close.
            ret_next: Simple return from each close to the next.
            realized_dates: Date on which each of those returns accrues.
        """
        cost_rate = self.cfg.cost_bps / 1e4
        rets, weights, costs = {}, {}, {}
        metrics: List[Dict[str, object]] = []
        for name, target in targets.items():
            held = self.apply_band(target, self.cfg.rebalance_band)
            traded = np.abs(np.diff(np.concatenate(([0.0], held))))
            cost = cost_rate * traded
            net = pd.Series(held * ret_next - cost, index=realized_dates)
            rets[name], weights[name], costs[name] = net, held, cost
            metrics.append(
                PerformanceAnalytics.summary(name, net, held, cost)
            )
        return BacktestResult(
            returns=pd.DataFrame(rets),
            weights=pd.DataFrame(weights, index=realized_dates),
            costs=pd.DataFrame(costs, index=realized_dates),
            metrics=pd.DataFrame(metrics),
            config=self.cfg,
        )


# ---------------------------------------------------------------------------
# Signal Lab
# ---------------------------------------------------------------------------

class SignalGenerator:
    """Classic price-based trading signals, computed causally.

    Every value at day *t* uses closes up to and including *t* only. The
    position is taken at close *t* and earns the return of day *t+1*.
    """

    @staticmethod
    def ema_trend(close: pd.Series, fast: int, slow: int) -> pd.Series:
        r"""Trend signal :math:`s_t = \mathrm{sign}(EMA^f_t - EMA^s_t)`.

        +1 (long) when the fast EMA is above the slow EMA, −1 (short) when
        it is below, and 0 until the slow EMA has ``slow`` observations.
        This is a simple form of time-series momentum (Moskowitz, Ooi &
        Pedersen, 2012).
        """
        ema_f = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_s = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        return np.sign(ema_f - ema_s).fillna(0.0)

    @staticmethod
    def bollinger_bands(
        close: pd.Series,
        window: int,
        width: float,
    ) -> pd.DataFrame:
        """Middle band (SMA) and upper/lower bands at ``width`` std devs."""
        mid = close.rolling(window, min_periods=window).mean()
        sd = close.rolling(window, min_periods=window).std(ddof=0)
        return pd.DataFrame({
            "bb_mid": mid,
            "bb_upper": mid + width * sd,
            "bb_lower": mid - width * sd,
        })

    @classmethod
    def bollinger_reversion(
        cls,
        close: pd.Series,
        window: int,
        width: float,
    ) -> pd.Series:
        """Stateful mean-reversion signal.

        Go long when the close falls below the lower band, short when it
        rises above the upper band, and exit when the close crosses back
        to the middle band. Flat otherwise.
        """
        bands = cls.bollinger_bands(close, window, width)
        c = close.to_numpy(dtype=float)
        mid = bands["bb_mid"].to_numpy()
        upper = bands["bb_upper"].to_numpy()
        lower = bands["bb_lower"].to_numpy()
        pos = np.zeros(c.size)
        current = 0.0
        for i in range(c.size):
            if np.isnan(mid[i]):
                current = 0.0
            elif current == 0.0:
                if c[i] > upper[i]:
                    current = -1.0
                elif c[i] < lower[i]:
                    current = 1.0
            elif (current > 0 and c[i] >= mid[i]) or (
                    current < 0 and c[i] <= mid[i]):
                current = 0.0
            pos[i] = current
        return pd.Series(pos, index=close.index)


class SharpeBootstrap:
    r"""Circular block bootstrap for the difference of two Sharpe ratios.

    Daily strategy returns are autocorrelated and heteroskedastic, so
    i.i.d. resampling would understate uncertainty. Blocks of ``block``
    consecutive days are resampled jointly for both strategies, which
    keeps their serial dependence and cross-correlation (Politis & Romano,
    1992). The two-sided p-value for :math:`H_0: SR_A = SR_B` comes from
    the bootstrap distribution re-centred on zero. Ledoit & Wolf (2008)
    give a studentized refinement of the same idea.
    """

    @staticmethod
    def sharpe(r: FloatArray, axis: int = -1) -> FloatArray:
        """Annualized Sharpe ratio along ``axis`` (0 where vol is 0)."""
        sd = r.std(axis=axis, ddof=1)
        mean = r.mean(axis=axis)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(sd > 0, mean / sd, 0.0)
        return out * np.sqrt(ANNUALIZATION_FACTOR)

    @classmethod
    def difference(
        cls,
        r_a: FloatArray,
        r_b: FloatArray,
        block: int = 20,
        n_boot: int = 2000,
        seed: int = 7,
    ) -> Dict[str, float]:
        """Return Sharpe A, Sharpe B, the difference, a 95% CI and p."""
        n = r_a.size
        block = max(1, min(block, n))
        n_blocks = int(np.ceil(n / block))
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, n, size=(n_boot, n_blocks))
        idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(
            n_boot, -1)[:, :n]
        boot = cls.sharpe(r_a[idx]) - cls.sharpe(r_b[idx])
        sr_a = float(cls.sharpe(r_a))
        sr_b = float(cls.sharpe(r_b))
        observed = sr_a - sr_b
        p_value = float(np.mean(np.abs(boot - observed) >= abs(observed)))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {
            "Sharpe A": sr_a,
            "Sharpe B": sr_b,
            "Δ Sharpe": observed,
            "95% CI Low": float(lo),
            "95% CI High": float(hi),
            "p-value": p_value,
        }


class SignalLab:
    r"""Test whether HMM regimes improve two classic trading signals.

    Hypotheses are fixed *before* any result is seen:

    * **Trend (EMA crossover)** suffers from sharp reversals in turbulent
      markets ("momentum crashes", Daniel & Moskowitz, 2016), so its
      position is scaled by :math:`1 - P(S_{t+1} = \text{High})`.
    * **Mean reversion (Bollinger)** works in calm, range-bound markets
      and is run over by strong trends, so its position is scaled by
      :math:`P(S_{t+1} = \text{Low})`.

    The gates are smooth probability weights rather than thresholds, so
    there is no cut-off to tune. Every strategy is sized with the same
    EWMA volatility target,

    .. math::
        w_t = s_t \cdot g_t \cdot
              \min\!\big(\sigma^\star / \hat\sigma^{EWMA}_t,\ L_{\max}\big),

    so the *only* difference between a signal and its filtered version is
    the regime gate :math:`g_t`. Positions then go through the same band
    and cost model as the vol-target backtest. The regime probabilities
    come from the walk-forward engine and are therefore out of sample.
    """

    def __init__(self, signal_cfg: SignalConfig, bt_cfg: BacktestConfig):
        self.sig = signal_cfg
        self.bt_cfg = bt_cfg

    def run(self, wf: WalkForwardResult) -> SignalLabResult:
        """Build signals, simulate strategies and run the statistics."""
        c = self.sig
        close_all = wf.close_history
        trend_all = SignalGenerator.ema_trend(close_all, c.ema_fast,
                                              c.ema_slow)
        rev_all = SignalGenerator.bollinger_reversion(
            close_all, c.bb_window, c.bb_width
        )
        frame, realized_dates = VolTargetBacktester.tradable_window(wf)
        idx = frame.index

        size = np.minimum(
            self.bt_cfg.target_vol
            / np.sqrt(frame["fc_EWMA"].to_numpy(dtype=float)),
            self.bt_cfg.max_leverage,
        )
        trend = trend_all.reindex(idx).to_numpy(dtype=float)
        rev = rev_all.reindex(idx).to_numpy(dtype=float)
        p_low = frame[f"pnext_{REGIME_LABELS[0]}"].to_numpy(dtype=float)
        p_high = frame[f"pnext_{REGIME_LABELS[-1]}"].to_numpy(dtype=float)
        targets: Dict[str, FloatArray] = {
            LAB_LONG: size,
            LAB_TREND: trend * size,
            LAB_TREND_REGIME: trend * size * (1.0 - p_high),
            LAB_REVERSION: rev * size,
            LAB_REVERSION_REGIME: rev * size * p_low,
        }
        bt = VolTargetBacktester(self.bt_cfg).simulate(
            targets, frame["ret_next"].to_numpy(dtype=float), realized_dates
        )

        regime_next = np.asarray(REGIME_LABELS)[
            frame[[f"pnext_{lab}" for lab in REGIME_LABELS]]
            .to_numpy(dtype=float).argmax(axis=1)
        ]
        indicators = pd.concat([
            wf.frame[["Close", "regime"]],
            pd.DataFrame({
                "ema_fast": close_all.ewm(span=c.ema_fast,
                                          adjust=False).mean(),
                "ema_slow": close_all.ewm(span=c.ema_slow,
                                          adjust=False).mean(),
            }).reindex(wf.frame.index),
            SignalGenerator.bollinger_bands(
                close_all, c.bb_window, c.bb_width
            ).reindex(wf.frame.index),
        ], axis=1)
        return SignalLabResult(
            backtest=bt,
            indicators=indicators,
            filter_tests=self._filter_tests(bt),
            regime_breakdown=self._regime_breakdown(bt, regime_next),
            config=c,
        )

    def _filter_tests(self, bt: BacktestResult) -> pd.DataFrame:
        """Block-bootstrap tests of the pre-specified comparisons."""
        pairs = [
            (LAB_TREND_REGIME, LAB_TREND),
            (LAB_REVERSION_REGIME, LAB_REVERSION),
            (LAB_TREND, LAB_LONG),
            (LAB_REVERSION, LAB_LONG),
        ]
        rows: List[Dict[str, object]] = []
        for a, b in pairs:
            stats = SharpeBootstrap.difference(
                bt.returns[a].to_numpy(dtype=float),
                bt.returns[b].to_numpy(dtype=float),
                block=self.sig.bootstrap_block,
                n_boot=self.sig.n_bootstrap,
                seed=self.sig.seed,
            )
            rows.append({"Strategy A": a, "Strategy B": b, **stats})
        return pd.DataFrame(rows)

    @staticmethod
    def _regime_breakdown(
        bt: BacktestResult,
        regime_next: npt.NDArray[np.str_],
    ) -> pd.DataFrame:
        """Performance of each unfiltered strategy within each regime.

        Days are grouped by the regime forecast for the day the return is
        earned (the argmax of ``P(S_{t+1} | data to t)``), which is known
        when the position is taken.
        """
        rows: List[Dict[str, object]] = []
        for name in (LAB_LONG, LAB_TREND, LAB_REVERSION):
            r = bt.returns[name].to_numpy(dtype=float)
            w = bt.weights[name].to_numpy(dtype=float)
            for label in REGIME_LABELS:
                mask = regime_next == label
                n_days = int(mask.sum())
                active = mask & (np.abs(w) > 1e-12)
                sub = r[mask]
                rows.append({
                    "Strategy": name,
                    "Regime (t+1 forecast)": label,
                    "Days": n_days,
                    "Active (%)": (100.0 * active.sum() / n_days
                                   if n_days else np.nan),
                    "Ann. Return (%)": (100.0 * ANNUALIZATION_FACTOR
                                        * sub.mean() if n_days else np.nan),
                    "Sharpe (ann.)": (float(SharpeBootstrap.sharpe(sub))
                                      if n_days > 2 else np.nan),
                    "Hit Rate (%)": (100.0 * float(np.mean(r[active] > 0))
                                     if active.any() else np.nan),
                })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def prepare_features(
    symbols: Tuple[str, ...],
    start: date,
    end: date,
    vol_window: int,
    train_fraction: float,
) -> Tuple[pd.DataFrame, int, str, str]:
    """Load prices and build features.

    Returns ``(features, n_train, symbol_used, data_source)``.
    """
    prices, symbol, source = load_price_history(symbols, start, end)
    engineer = FeatureEngineer(vol_window)
    features = engineer.build(prices)
    n_train = engineer.chronological_split(len(features), train_fraction)
    return features, n_train, symbol, source


@st.cache_resource(max_entries=8, show_spinner=False)
def run_walk_forward(
    symbols: Tuple[str, ...],
    start: date,
    end: date,
    train_fraction: float,
    vol_window: int,
    refit_every: int,
    horizon: int,
    n_restarts: int,
    random_seed: int,
    _progress: Optional[Callable[[float, str], None]] = None,
) -> WalkForwardResult:
    """Cached walk-forward run. ``_progress`` is excluded from the key."""
    features, n_train, _, _ = prepare_features(
        symbols, start, end, vol_window, train_fraction
    )
    engine = WalkForwardEngine(
        features, n_train,
        WalkForwardConfig(refit_every, horizon, n_restarts),
        random_seed=random_seed,
    )
    return engine.run(progress=_progress)

@st.cache_resource(max_entries=16, show_spinner=False)
def run_regime_pipeline(
    instrument: Instrument,
    start: date,
    end: date,
    train_fraction: float,
    vol_window: int,
    n_restarts: int,
    random_seed: int,
) -> RegimeResult:
    """Execute data -> features -> fit (train only) -> causal filter.

    Every quantity that is *estimated* (scaler moments, HMM parameters,
    state ordering) is computed from the training block. The forward
    filter is then run over the full series with those frozen parameters;
    because it is causal, filtered probabilities in the test block depend
    on test data only up to the current day.
    """
    config = ModelConfig(
        instrument=instrument,
        start=start,
        end=end,
        train_fraction=train_fraction,
        vol_window=vol_window,
        n_restarts=n_restarts,
        random_seed=random_seed,
    )
    features, n_train, symbol, source = prepare_features(
        instrument.symbols, start, end, vol_window, train_fraction
    )
    x_all = features.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    scaler = FeatureEngineer.fit_scaler(x_all[:n_train])
    z_all = scaler.transform(x_all)

    modeler = RegimeModeler(
        n_restarts=n_restarts,
        max_iter=config.max_iter,
        random_seed=random_seed,
    ).fit(z_all[:n_train])
    probs, log_c = modeler.filter(z_all)

    frame = features.copy()
    for k, label in enumerate(REGIME_LABELS):
        frame[f"p_{label}"] = probs[:, k]
    regime_id = probs.argmax(axis=1)
    frame["regime_id"] = regime_id
    frame["regime"] = np.asarray(REGIME_LABELS)[regime_id]
    frame["sample"] = np.where(
        np.arange(len(frame)) < n_train, "In-Sample", "Out-of-Sample"
    )

    params = modeler.params
    train_ll = float(log_c[:n_train].sum())
    n_ok, n_conv = modeler.restart_summary
    diagnostics = FitDiagnostics(
        train_log_likelihood=train_ll,
        train_ll_per_obs=train_ll / n_train,
        oos_ll_per_obs=float(log_c[n_train:].mean()),
        bic=-2.0 * train_ll + params.n_free_parameters * np.log(n_train),
        converged=modeler.converged,
        n_iter=modeler.n_iter,
        n_restarts_ok=n_ok,
        n_restarts_converged=n_conv,
    )
    return RegimeResult(
        frame=frame,
        params=params,
        scaler=scaler,
        diagnostics=diagnostics,
        config=config,
        n_train=n_train,
        symbol=symbol,
        data_source=source,
    )


# ---------------------------------------------------------------------------
# Presentation layer
# ---------------------------------------------------------------------------

def price_decimals(level: float) -> int:
    """Display precision suited to a price level (FX vs JPY vs indices)."""
    if level < 20.0:
        return 5
    if level < 500.0:
        return 3
    return 2


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#RRGGBB`` to a CSS ``rgba()`` string."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


class ChartFactory:
    """Build Plotly figures in a consistent dark institutional style."""

    @staticmethod
    def _rangebreaks(index: pd.DatetimeIndex) -> List[Dict[str, object]]:
        """Hide weekends and weekdays with no bar (exchange holidays)."""
        missing = pd.bdate_range(index.min(), index.max()).difference(index)
        breaks: List[Dict[str, object]] = [dict(bounds=["sat", "mon"])]
        if len(missing):
            breaks.append(dict(values=[d.strftime("%Y-%m-%d")
                                       for d in missing]))
        return breaks

    @staticmethod
    def _base_layout(fig: go.Figure, height: int) -> None:
        fig.update_layout(
            template="plotly_dark",
            height=height,
            paper_bgcolor=THEME["panel"],
            plot_bgcolor=THEME["panel"],
            font=dict(family=THEME["font"], color=THEME["text"], size=12),
            margin=dict(l=60, r=24, t=48, b=36),
            hoverlabel=dict(
                bgcolor=THEME["bg"],
                bordercolor=THEME["border"],
                font=dict(family=THEME["mono"], size=12),
            ),
        )
        fig.update_xaxes(
            gridcolor=THEME["grid"], zeroline=False,
            linecolor=THEME["border"], showspikes=True,
            spikecolor=THEME["muted"], spikethickness=1, spikedash="dot",
        )
        fig.update_yaxes(
            gridcolor=THEME["grid"], zeroline=False,
            linecolor=THEME["border"],
        )

    @classmethod
    def regime_chart(cls, result: RegimeResult) -> go.Figure:
        """Out-of-sample candlesticks with regime-shaded background.

        Row 1: OHLC candles. Row 2: realized volatility with the training
        emission levels of each regime. Row 3: filtered regime
        probabilities ``P(S_t | z_{1:t})``.
        """
        df = result.test_frame
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.035,
            row_heights=[0.58, 0.21, 0.21],
        )
        # Set regime bands first: later add_hline calls append to shapes.
        fig.update_layout(shapes=cls._regime_shapes(df))

        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price",
            increasing=dict(
                line=dict(color=THEME["candle_up"], width=1),
                fillcolor=THEME["candle_up"],
            ),
            decreasing=dict(
                line=dict(color=THEME["candle_down"], width=1),
                fillcolor=THEME["candle_down"],
            ),
            text=[f"Regime: {r}" for r in df["regime"]],
            hoverinfo="x+y+text", showlegend=False,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=100.0 * df["realized_vol"], mode="lines",
            line=dict(color=THEME["accent"], width=1.6),
            name=f"Realized Vol ({result.config.vol_window}d)",
            hovertemplate="%{y:.2f}%<extra>Realized Vol</extra>",
            showlegend=False,
        ), row=2, col=1)

        emissions = RegimeAnalytics.emission_summary(
            result.params, result.scaler
        )
        for _, row in emissions.iterrows():
            fig.add_hline(
                y=row["Typical Realized Vol (%)"], row=2, col=1,
                line=dict(
                    color=REGIME_COLORS[row["Regime"]], width=1, dash="dot"
                ),
                opacity=0.8,
            )

        for label in REGIME_LABELS:
            color = REGIME_COLORS[label]
            fig.add_trace(go.Scatter(
                x=df.index, y=df[f"p_{label}"], mode="lines",
                stackgroup="probs", name=label, legendgroup=label,
                line=dict(width=0.4, color=color),
                fillcolor=hex_to_rgba(color, 0.55),
                hovertemplate="%{y:.1%}<extra>" + label + "</extra>",
            ), row=3, col=1)

        cls._base_layout(fig, height=780)
        fig.update_layout(
            hovermode="x unified",
            legend=dict(
                orientation="h", x=1.0, xanchor="right", y=1.045,
                bgcolor="rgba(0,0,0,0)", title_text="Filtered regime:  ",
            ),
        )
        fig.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=cls._rangebreaks(df.index),
        )
        fig.update_xaxes(
            row=1, col=1,
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month",
                         stepmode="backward"),
                    dict(count=6, label="6M", step="month",
                         stepmode="backward"),
                    dict(count=1, label="YTD", step="year",
                         stepmode="todate"),
                    dict(count=1, label="1Y", step="year",
                         stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor=THEME["bg"], activecolor=THEME["border"],
                bordercolor=THEME["border"], borderwidth=1,
                font=dict(color=THEME["text"], size=11),
                x=0.0, y=1.045,
            ),
        )
        decimals = price_decimals(float(df["Close"].median()))
        tick_fmt = {5: ",.4f", 3: ",.2f"}.get(decimals, ",.0f")
        fig.update_yaxes(title_text="Price", row=1, col=1,
                         tickformat=tick_fmt)
        fig.update_yaxes(title_text="RV (%)", row=2, col=1)
        fig.update_yaxes(
            title_text="P(state)", row=3, col=1, range=[0, 1],
            tickformat=".0%",
        )
        return fig

    @staticmethod
    def _regime_shapes(df: pd.DataFrame) -> List[Dict[str, object]]:
        """One full-height background rectangle per contiguous regime run.

        Rectangles are offset by half a day so each bar sits centred in
        its band; ``yref='paper'`` lets a single shape span all subplots.
        """
        half_day = pd.Timedelta(hours=12)
        idx = df.index
        shapes: List[Dict[str, object]] = []
        for start, end, label in RegimeAnalytics.runs(df["regime"]):
            x1 = idx[end] if end < len(idx) else idx[-1] + 2 * half_day
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=idx[start] - half_day, x1=x1 - half_day, y0=0, y1=1,
                fillcolor=hex_to_rgba(REGIME_COLORS[label], 0.13),
                line=dict(width=0), layer="below",
            ))
        return shapes

    @classmethod
    def transition_heatmap(cls, trans_mat: FloatArray) -> go.Figure:
        """Annotated heatmap of ``A[i, j] = P(S_t = j | S_{t-1} = i)``."""
        fig = go.Figure(go.Heatmap(
            z=trans_mat,
            x=[f"→ {lab}" for lab in REGIME_LABELS],
            y=list(REGIME_LABELS),
            zmin=0.0, zmax=1.0,
            colorscale=[[0.0, "#0F172A"], [0.5, "#0C4A6E"], [1.0, "#0284C7"]],
            texttemplate="%{z:.3f}",
            textfont=dict(family=THEME["mono"], size=15, color="#F8FAFC"),
            hovertemplate=(
                "From %{y} to %{x}<br>P = %{z:.4f}<extra></extra>"
            ),
            showscale=False, xgap=3, ygap=3,
        ))
        cls._base_layout(fig, height=340)
        fig.update_layout(margin=dict(l=90, r=16, t=56, b=16))
        fig.update_xaxes(side="top", showgrid=False, showspikes=False,
                         title_text="State at t")
        fig.update_yaxes(autorange="reversed", showgrid=False,
                         title_text="State at t−1")
        return fig

    @classmethod
    def forecast_chart(cls, wf: WalkForwardResult) -> go.Figure:
        """Realized h-day volatility against each model's ex-ante forecast."""
        df = wf.frame
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=100.0 * np.sqrt(df["target_var"]),
            name=f"Realized (next {wf.horizon}d)", mode="lines",
            line=dict(color=MODEL_COLORS["Realized"], width=1.0),
            opacity=0.55,
            hovertemplate="%{y:.2f}%<extra>Realized</extra>",
        ))
        for m in FORECAST_MODELS:
            fig.add_trace(go.Scatter(
                x=df.index, y=100.0 * np.sqrt(df[f"fc_{m}"]), name=m,
                mode="lines",
                line=dict(color=MODEL_COLORS[m],
                          width=2.0 if m == "HMM" else 1.3),
                hovertemplate="%{y:.2f}%<extra>" + m + "</extra>",
            ))
        for refit in wf.refits["Refit Date"]:
            fig.add_vline(x=refit, line=dict(color=THEME["border"],
                                             width=1, dash="dot"))
        cls._base_layout(fig, height=440)
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", x=1.0, xanchor="right", y=1.08,
                        bgcolor="rgba(0,0,0,0)"),
        )
        fig.update_xaxes(rangebreaks=cls._rangebreaks(df.index))
        fig.update_yaxes(title_text="Annualized vol (%)")
        return fig

    @classmethod
    def backtest_chart(
        cls,
        bt: BacktestResult,
        colors: Optional[Dict[str, str]] = None,
        emphasis: Sequence[str] = ("HMM",),
        position_title: str = "Leverage (×)",
    ) -> go.Figure:
        """Equity curves, drawdowns and held positions for all strategies.

        Args:
            bt: Backtest output.
            colors: Strategy -> colour map (defaults to the vol-target set).
            emphasis: Substrings of strategy names drawn with thicker lines.
            position_title: Axis title for the position panel.
        """
        palette = colors or STRATEGY_COLORS
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            row_heights=[0.52, 0.26, 0.22],
        )
        for name in bt.returns.columns:
            color = palette.get(name, THEME["muted"])
            width = 2.0 if any(e in name for e in emphasis) else 1.2
            equity = (1.0 + bt.returns[name]).cumprod()
            drawdown = equity / np.maximum(equity.cummax(), 1.0) - 1.0
            common = dict(legendgroup=name, mode="lines")
            fig.add_trace(go.Scatter(
                x=equity.index, y=equity, name=name,
                line=dict(color=color, width=width),
                hovertemplate="%{y:.3f}<extra>" + name + "</extra>",
                **common,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=drawdown.index, y=100.0 * drawdown, name=name,
                showlegend=False, line=dict(color=color, width=1.0),
                hovertemplate="%{y:.1f}%<extra>" + name + "</extra>",
                **common,
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=bt.weights.index, y=bt.weights[name], name=name,
                showlegend=False, line=dict(color=color, width=1.0,
                                            shape="hv"),
                hovertemplate="%{y:.2f}×<extra>" + name + "</extra>",
                **common,
            ), row=3, col=1)
        cls._base_layout(fig, height=720)
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", x=1.0, xanchor="right", y=1.06,
                        bgcolor="rgba(0,0,0,0)"),
        )
        fig.update_xaxes(rangebreaks=cls._rangebreaks(bt.returns.index))
        fig.update_yaxes(title_text="Growth of 1", type="log", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text=position_title, row=3, col=1)
        return fig

    @classmethod
    def signal_chart(cls, lab: SignalLabResult) -> go.Figure:
        """Price with EMAs and Bollinger bands over regime shading, plus
        the unfiltered signal positions."""
        ind = lab.indicators
        cfg = lab.config
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            row_heights=[0.72, 0.28],
        )
        fig.update_layout(shapes=cls._regime_shapes(ind))
        fig.add_trace(go.Scatter(
            x=ind.index, y=ind["bb_upper"], mode="lines",
            line=dict(color=hex_to_rgba(LAB_COLORS[LAB_REVERSION], 0.5),
                      width=0.8),
            name=f"Bollinger {cfg.bb_window}d ±{cfg.bb_width:g}σ",
            legendgroup="bb", hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=ind.index, y=ind["bb_lower"], mode="lines",
            line=dict(color=hex_to_rgba(LAB_COLORS[LAB_REVERSION], 0.5),
                      width=0.8),
            fill="tonexty",
            fillcolor=hex_to_rgba(LAB_COLORS[LAB_REVERSION], 0.08),
            name="Bollinger lower", legendgroup="bb", showlegend=False,
            hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=ind.index, y=ind["Close"], mode="lines", name="Close",
            line=dict(color=THEME["text"], width=1.2),
            hovertemplate="%{y:,.4f}<extra>Close</extra>",
        ), row=1, col=1)
        for col, span, dash in (("ema_fast", cfg.ema_fast, "solid"),
                                ("ema_slow", cfg.ema_slow, "dash")):
            fig.add_trace(go.Scatter(
                x=ind.index, y=ind[col], mode="lines",
                name=f"EMA {span}",
                line=dict(color=LAB_COLORS[LAB_TREND], width=1.3,
                          dash=dash),
                hovertemplate="%{y:,.4f}<extra>EMA " + str(span)
                + "</extra>",
            ), row=1, col=1)
        for name in (LAB_TREND, LAB_REVERSION):
            w = lab.backtest.weights[name]
            fig.add_trace(go.Scatter(
                x=w.index, y=w, mode="lines", name=f"{name} position",
                line=dict(color=LAB_COLORS[name], width=1.0, shape="hv"),
                hovertemplate="%{y:+.2f}×<extra>" + name + "</extra>",
            ), row=2, col=1)
        cls._base_layout(fig, height=620)
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", x=1.0, xanchor="right", y=1.08,
                        bgcolor="rgba(0,0,0,0)"),
        )
        fig.update_xaxes(rangebreaks=cls._rangebreaks(ind.index))
        decimals = price_decimals(float(ind["Close"].median()))
        fig.update_yaxes(
            title_text="Price", row=1, col=1,
            tickformat={5: ",.4f", 3: ",.2f"}.get(decimals, ",.0f"),
        )
        fig.update_yaxes(title_text="Position (×)", row=2, col=1)
        return fig

    @classmethod
    def rolling_risk_chart(cls, bt: BacktestResult) -> go.Figure:
        """Rolling 63-day realized volatility of each strategy vs target."""
        fig = go.Figure()
        rolling = (bt.returns.rolling(ROLLING_RISK_WINDOW).std(ddof=1)
                   * np.sqrt(ANNUALIZATION_FACTOR) * 100.0)
        for name in rolling.columns:
            fig.add_trace(go.Scatter(
                x=rolling.index, y=rolling[name], name=name, mode="lines",
                line=dict(color=STRATEGY_COLORS[name],
                          width=2.0 if "HMM" in name else 1.2),
                hovertemplate="%{y:.2f}%<extra>" + name + "</extra>",
            ))
        fig.add_hline(
            y=100.0 * bt.config.target_vol,
            line=dict(color=THEME["muted"], width=1, dash="dash"),
            annotation_text="target", annotation_position="top left",
            annotation_font_color=THEME["muted"],
        )
        cls._base_layout(fig, height=360)
        fig.update_layout(hovermode="x unified", showlegend=False)
        fig.update_xaxes(rangebreaks=cls._rangebreaks(rolling.index))
        fig.update_yaxes(title_text=f"{ROLLING_RISK_WINDOW}d vol (%)")
        return fig


class DashboardUI:
    """Streamlit front-end: sidebar controls, KPIs, charts and tables."""

    def __init__(self) -> None:
        st.set_page_config(
            page_title=APP_TITLE,
            page_icon="◆",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    # -- styling ------------------------------------------------------------

    @staticmethod
    def inject_theme() -> None:
        """Apply a dark institutional skin on top of Streamlit's theme."""
        t = THEME
        st.markdown(
            f"""
            <style>
            @import url('{FONT_URL}');
            html, body, .stApp, [class*="css"] {{
                font-family: {t['font']};
            }}
            .stApp {{ background-color: {t['bg']}; color: {t['text']}; }}
            [data-testid="stHeader"] {{ background: transparent; }}
            [data-testid="stSidebar"] {{
                background-color: {t['panel']};
                border-right: 1px solid {t['border']};
            }}
            .block-container {{ padding-top: 2.2rem; max-width: 1600px; }}
            [data-testid="stMetric"] {{
                background: {t['panel']};
                border: 1px solid {t['border']};
                border-radius: 6px;
                padding: 14px 18px;
            }}
            [data-testid="stMetricLabel"] p {{
                color: {t['muted']}; font-size: 0.74rem;
                text-transform: uppercase; letter-spacing: 0.08em;
            }}
            [data-testid="stMetricValue"] {{
                font-family: {t['mono']}; font-size: 1.55rem;
            }}
            .section-title {{
                color: {t['muted']}; font-size: 0.78rem; font-weight: 600;
                text-transform: uppercase; letter-spacing: 0.12em;
                border-bottom: 1px solid {t['border']};
                padding-bottom: 6px; margin: 18px 0 10px 0;
            }}
            .app-title {{
                font-size: 1.6rem; font-weight: 600; margin-bottom: 0;
                color: {t['text']};
            }}
            .app-subtitle {{
                color: {t['muted']}; font-family: {t['mono']};
                font-size: 0.82rem; margin-top: 2px;
            }}
            .pill {{
                display: inline-block; padding: 2px 10px; margin-right: 6px;
                border: 1px solid {t['border']}; border-radius: 999px;
                font-family: {t['mono']}; font-size: 0.74rem;
                color: {t['muted']};
            }}
            .regime-dot {{
                display: inline-block; width: 10px; height: 10px;
                border-radius: 2px; margin-right: 6px;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def section(title: str) -> None:
        """Render a small uppercase section header."""
        st.markdown(
            f'<div class="section-title">{title}</div>',
            unsafe_allow_html=True,
        )

    # -- sidebar ------------------------------------------------------------

    @staticmethod
    def render_sidebar() -> Optional[
        Tuple[ModelConfig, WalkForwardConfig, BacktestConfig, SignalConfig]
    ]:
        """Collect all settings; returns ``None`` if inputs are invalid."""
        sb = st.sidebar
        sb.markdown("### Model Controls")
        with sb.form("controls"):
            label = st.selectbox(
                "Instrument", list(INSTRUMENT_BY_LABEL.keys()), index=0,
                format_func=lambda lab: (
                    f"{lab} · {INSTRUMENT_BY_LABEL[lab].asset_class}"
                ),
                help="FX pairs, gold and US equity indices. Index labels "
                     "follow common CFD names (US500 = S&P 500, USTEC = "
                     "Nasdaq 100, US30 = Dow Jones).",
            )
            custom = st.text_input(
                "Custom Yahoo symbol (optional)", value="",
                placeholder="e.g. NOK=X, ^GDAXI, SI=F",
                help="Overrides the instrument above when filled in.",
            )
            col_a, col_b = st.columns(2)
            start = col_a.date_input(
                "Start", value=date(2010, 1, 1),
                min_value=EARLIEST_DATE, max_value=date.today(),
            )
            end = col_b.date_input(
                "End", value=date.today(),
                min_value=EARLIEST_DATE, max_value=date.today(),
            )
            train_pct = st.slider(
                "In-sample share (%)", min_value=50, max_value=90,
                value=70, step=5,
                help="Chronological split. The model is estimated on the "
                     "first part only and evaluated on the remainder.",
            )
            vol_window = st.slider(
                "Realized-vol window (days)", min_value=5, max_value=63,
                value=21, step=1,
                help="21 ≈ one month, 63 ≈ one quarter.",
            )
            with st.expander("Estimation settings"):
                n_restarts = st.slider(
                    "EM random restarts", min_value=1, max_value=25,
                    value=10,
                )
                seed = int(st.number_input(
                    "Random seed", min_value=0, max_value=10_000,
                    value=42, step=1,
                ))
            with st.expander("Walk-forward forecasting"):
                refit_every = st.select_slider(
                    "Refit every (trading days)",
                    options=[21, 63, 126, 252], value=63,
                    help="Models are re-estimated on an expanding window "
                         "at this frequency. Shorter means more refits "
                         "and a slower first run.",
                )
                horizon = st.select_slider(
                    "Forecast horizon h (days)", options=[1, 5, 10, 21],
                    value=5,
                )
                wf_restarts = st.slider(
                    "EM restarts per refit", min_value=1, max_value=10,
                    value=3,
                    help="The first restart is warm-started from the "
                         "previous refit's parameters.",
                )
            with st.expander("Vol-target backtest"):
                target_vol = st.slider(
                    "Target volatility (%)", min_value=2.0, max_value=20.0,
                    value=8.0, step=0.5,
                )
                max_leverage = st.slider(
                    "Max leverage (×)", min_value=1.0, max_value=5.0,
                    value=3.0, step=0.5,
                )
                cost_bps = st.slider(
                    "Cost (bp per 1× notional traded)", min_value=0.0,
                    max_value=10.0, value=1.0, step=0.25,
                    help="Half-spread plus slippage. Rough guide: "
                         "EUR/USD 0.1–0.5 bp institutional, 1–2 bp "
                         "retail; index futures 0.5–1 bp; gold 1–3 bp.",
                )
                band = st.slider(
                    "Rebalance band (Δ leverage)", min_value=0.0,
                    max_value=0.5, value=0.10, step=0.05,
                    help="Trade only when target leverage moves by more "
                         "than this amount. Reduces turnover.",
                )
            with st.expander("Signal lab"):
                ema_preset = st.selectbox(
                    "EMA trend (fast / slow)", ["50 / 200", "20 / 100"],
                    index=0,
                    help="Standard presets only. Optimising lengths on "
                         "this sample would be data snooping.",
                )
                st.caption(
                    "Bollinger reversion uses the textbook 20-day bands at "
                    "±2σ and exits at the middle band. These are fixed."
                )
            st.form_submit_button("Run model", type="primary")

        sb.caption(
            "Regimes are inferred with a causal forward filter: each day's "
            "label uses only data available at that day's close."
        )

        symbol = custom.strip().upper()
        instrument = (
            Instrument(symbol, "Custom Yahoo symbol", "Custom", (symbol,))
            if symbol else INSTRUMENT_BY_LABEL[label]
        )
        if not isinstance(start, date) or not isinstance(end, date):
            st.error("Please select a valid start and end date.")
            return None
        if start >= end:
            st.error("Start date must be before end date.")
            return None
        model_cfg = ModelConfig(
            instrument=instrument, start=start, end=end,
            train_fraction=train_pct / 100.0, vol_window=vol_window,
            n_restarts=n_restarts, random_seed=seed,
        )
        wf_cfg = WalkForwardConfig(
            refit_every=int(refit_every), horizon=int(horizon),
            n_restarts=wf_restarts,
        )
        bt_cfg = BacktestConfig(
            target_vol=target_vol / 100.0, max_leverage=max_leverage,
            cost_bps=cost_bps, rebalance_band=band,
        )
        fast, slow = (int(x) for x in ema_preset.split("/"))
        signal_cfg = SignalConfig(ema_fast=fast, ema_slow=slow)
        return model_cfg, wf_cfg, bt_cfg, signal_cfg

    # -- main panels --------------------------------------------------------

    @staticmethod
    def render_header(result: RegimeResult) -> None:
        """Title bar with sample boundaries."""
        cfg = result.config
        inst = cfg.instrument
        tr, te = result.train_frame, result.test_frame
        st.markdown(
            f'<div class="app-title">{APP_TITLE}</div>'
            f'<div class="app-subtitle">{inst.label} · {inst.description}'
            f' · data: {result.symbol} · Gaussian HMM, {N_REGIMES} states'
            f' · {cfg.vol_window}d realized vol</div>',
            unsafe_allow_html=True,
        )
        fmt = "%Y-%m-%d"
        st.markdown(
            f'<span class="pill">IN-SAMPLE {tr.index[0]:{fmt}} → '
            f'{tr.index[-1]:{fmt}} · {len(tr):,} obs</span>'
            f'<span class="pill">OUT-OF-SAMPLE {te.index[0]:{fmt}} → '
            f'{te.index[-1]:{fmt}} · {len(te):,} obs</span>'
            f'<span class="pill">{inst.asset_class.upper()}</span>',
            unsafe_allow_html=True,
        )
        if result.data_source == "snapshot":
            st.warning(
                "Live Yahoo Finance data is unavailable right now (Yahoo "
                "often rate-limits shared cloud hosts), so this run uses "
                "the bundled snapshot, with data up to "
                f"{te.index[-1]:%Y-%m-%d}. Run the app locally for live "
                "data."
            )
        if result.symbol != inst.symbols[0]:
            st.warning(
                f"Primary feed {inst.symbols[0]} was unavailable or less "
                f"complete for this range, so {result.symbol} is used "
                "instead."
                + (" GC=F is the continuous front-month COMEX futures "
                   "price, which sits slightly above spot and can jump "
                   "on roll dates." if result.symbol == "GC=F" else "")
            )

    @staticmethod
    def render_kpis(result: RegimeResult, next_day: FloatArray) -> None:
        """Top-row KPI tiles, all as of the last out-of-sample close."""
        te = result.test_frame
        last = te.iloc[-1]
        label = str(last["regime"])
        oos_mean_rv = float(te["realized_vol"].mean())
        rv_now = float(last["realized_vol"])
        spell = RegimeAnalytics.current_spell(te["regime"])

        cols = st.columns(5)
        cols[0].metric(
            "Current Regime", label,
            help=f"argmax of P(S_t | data up to {te.index[-1]:%Y-%m-%d}).",
        )
        cols[1].metric(
            "Regime Confidence", f"{last[f'p_{label}']:.1%}",
            help="Filtered posterior probability of the current regime.",
        )
        cols[2].metric(
            f"Realized Vol ({result.config.vol_window}d, ann.)",
            f"{rv_now:.2%}",
            delta=f"{100 * (rv_now - oos_mean_rv):+.2f} pp vs OOS avg",
            delta_color="inverse",
        )
        cols[3].metric(
            "P(High Vol, t+1)", f"{next_day[-1]:.1%}",
            help="One-day-ahead forecast: filtered probabilities at t "
                 "propagated through the transition matrix.",
        )
        cols[4].metric(
            "Current Spell", f"{spell} days",
            help="Consecutive days in the current regime.",
        )
        st.caption(
            f"As of close {te.index[-1]:%a %d %b %Y} · "
            f"last price {last['Close']:,.{price_decimals(last['Close'])}f}"
        )

    def render_main_chart(self, result: RegimeResult) -> None:
        """Out-of-sample candlestick chart with regime shading."""
        self.section("Out-of-sample price action · filtered regimes")
        st.plotly_chart(
            ChartFactory.regime_chart(result), width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
        )
        st.caption(
            "Background shading = argmax filtered regime. Dotted lines in "
            "the volatility panel mark each regime's typical realized vol "
            "as estimated in-sample."
        )

    def render_bottom_row(self, result: RegimeResult) -> None:
        """Transition matrix and regime statistics."""
        left, right = st.columns([0.38, 0.62], gap="large")

        with left:
            self.section("Transition matrix (estimated in-sample)")
            st.plotly_chart(
                ChartFactory.transition_heatmap(result.params.trans_mat),
                width="stretch", config={"displayModeBar": False},
            )
            durations = 1.0 / (1.0 - np.diag(result.params.trans_mat))
            st.caption(
                "Expected durations 1/(1−Aₖₖ): " + " · ".join(
                    f"{lab} {d:.1f}d"
                    for lab, d in zip(REGIME_LABELS, durations)
                )
            )

        with right:
            self.section("Regime statistics")
            tab_oos, tab_is = st.tabs(["Out-of-sample", "In-sample"])
            trans = result.params.trans_mat
            with tab_oos:
                self._stats_table(RegimeAnalytics.regime_statistics(
                    result.test_frame, trans
                ))
            with tab_is:
                self._stats_table(RegimeAnalytics.regime_statistics(
                    result.train_frame, trans
                ))
            st.caption(
                "Return statistics use daily log returns; annualized with "
                f"{ANNUALIZATION_FACTOR} days. Persistence and expected "
                "duration come from the in-sample transition matrix."
            )

    @staticmethod
    def _stats_table(stats: pd.DataFrame) -> None:
        num = st.column_config.NumberColumn
        st.dataframe(
            stats, hide_index=True, width="stretch",
            column_config={
                "Days": num(format="%d"),
                "Time Share (%)": num(format="%.1f"),
                "Ann. Mean Return (%)": num(format="%+.2f"),
                "Ann. Return Vol (%)": num(format="%.2f"),
                "Avg Realized Vol (%)": num(format="%.2f"),
                "Spells": num(format="%d"),
                "Avg Spell (days)": num(format="%.1f"),
                "Persistence A_kk": num(format="%.3f"),
                "Expected Duration (days)": num(format="%.1f"),
            },
        )

    def render_diagnostics(self, result: RegimeResult) -> None:
        """Fit diagnostics, emission parameters and data export."""
        d = result.diagnostics
        with st.expander("Model diagnostics & emission parameters"):
            cols = st.columns(4)
            cols[0].metric("In-sample LL / obs", f"{d.train_ll_per_obs:.3f}")
            cols[1].metric(
                "OOS predictive LL / obs", f"{d.oos_ll_per_obs:.3f}",
                delta=f"{d.oos_ll_per_obs - d.train_ll_per_obs:+.3f}",
                help="Mean of log p(z_t | z_1:t-1) over the test block. "
                     "A large drop vs in-sample signals overfitting or a "
                     "structural break.",
            )
            cols[2].metric("BIC (in-sample)", f"{d.bic:,.0f}")
            cols[3].metric(
                "EM restarts converged",
                f"{d.n_restarts_converged}/{result.config.n_restarts}",
                help=f"Selected restart: {d.n_iter} iterations, "
                     f"converged={d.converged}.",
            )
            num = st.column_config.NumberColumn
            st.dataframe(
                RegimeAnalytics.emission_summary(
                    result.params, result.scaler
                ),
                hide_index=True, width="stretch",
                column_config={
                    "Mean Daily Return (bp)": num(format="%+.2f"),
                    "State Return Vol, ann. (%)": num(format="%.2f"),
                    "Typical Realized Vol (%)": num(format="%.2f"),
                    "Corr(r, ln σ)": num(format="%+.2f"),
                    "Stationary Prob.": num(format="%.3f"),
                },
            )
            export_cols = [
                "Open", "High", "Low", "Close", "log_return",
                "realized_vol", "regime",
            ] + [f"p_{lab}" for lab in REGIME_LABELS]
            st.download_button(
                "Download out-of-sample regime series (CSV)",
                data=result.test_frame[export_cols].to_csv().encode(),
                file_name=f"{result.symbol}_oos_regimes.csv",
                mime="text/csv",
            )

    # -- walk-forward tabs --------------------------------------------------

    @staticmethod
    def _walk_forward(
        config: ModelConfig,
        wf_cfg: WalkForwardConfig,
    ) -> WalkForwardResult:
        """Run (or fetch from cache) the walk-forward engine with progress."""
        bar = st.progress(0.0, text="Walk-forward: re-estimating models…")

        def update(frac: float, text: str) -> None:
            bar.progress(min(max(frac, 0.0), 1.0),
                         text=f"Walk-forward: {text}")

        try:
            return run_walk_forward(
                config.instrument.symbols, config.start, config.end,
                config.train_fraction, config.vol_window,
                wf_cfg.refit_every, wf_cfg.horizon, wf_cfg.n_restarts,
                config.random_seed, _progress=update,
            )
        finally:
            bar.empty()

    @staticmethod
    def _forecast_verdict(table: pd.DataFrame, benchmark: str) -> str:
        """Plain-language summary of the Diebold–Mariano results."""
        lines: List[str] = []
        for _, row in table.iterrows():
            if row["Model"] == benchmark:
                continue
            p_value = float(row["DM p-value"])
            delta = float(row["Δ QLIKE vs Bench (%)"])
            if np.isnan(p_value):
                verdict = "could not be tested"
            elif p_value < 0.05:
                verdict = ("significantly **better**" if delta < 0
                           else "significantly **worse**")
            else:
                verdict = "not significantly different"
            lines.append(
                f"- **{row['Model']}** vs {benchmark}: {delta:+.1f}% "
                f"QLIKE, {verdict} at the 5% level (p = {p_value:.3f})."
            )
        return "\n".join(lines)

    def render_forecast_tab(self, wf: WalkForwardResult) -> None:
        """Forecast comparison: KPIs, chart, loss table, refit log."""
        self.section(f"Walk-forward {wf.horizon}-day volatility forecasts")
        st.caption(
            f"Every model is re-estimated every {wf.refit_every} trading "
            "days on an expanding window of data before the refit date "
            "(dotted lines). It then forecasts the average annualized "
            f"variance over the next {wf.horizon} days, using data up to "
            "each close only."
        )
        bench = st.selectbox(
            "Benchmark for Diebold–Mariano tests", FORECAST_MODELS,
            index=FORECAST_MODELS.index("EWMA"), key="dm_benchmark",
        )
        table = ForecastEvaluator.evaluate(wf, bench)
        n_days = int(wf.frame["target_var"].notna().sum())
        best = str(table.loc[table["QLIKE"].idxmin(), "Model"])
        hmm = table.set_index("Model").loc["HMM"]

        cols = st.columns(5)
        cols[0].metric("Best model (QLIKE)", best,
                       help="Lowest mean QLIKE loss out of sample.")
        if bench == "HMM":
            cols[1].metric("HMM QLIKE vs benchmark", "—")
            cols[2].metric("DM p-value (HMM)", "—")
        else:
            cols[1].metric(
                f"HMM QLIKE vs {bench}",
                f"{hmm['Δ QLIKE vs Bench (%)']:+.1f}%",
                help="Negative means the HMM has lower loss.",
            )
            cols[2].metric(
                "DM p-value (HMM)", f"{hmm['DM p-value']:.3f}",
                help="Two-sided Diebold–Mariano test of equal QLIKE loss, "
                     "with the HLN small-sample correction.",
            )
        cols[3].metric("Forecast days", f"{n_days:,}")
        cols[4].metric("Refits", f"{len(wf.refits)}")

        st.plotly_chart(
            ChartFactory.forecast_chart(wf), width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
        )

        self.section("Out-of-sample loss comparison")
        num = st.column_config.NumberColumn
        st.dataframe(
            table, hide_index=True, width="stretch",
            column_config={
                "QLIKE": num(format="%.4f"),
                "Δ QLIKE vs Bench (%)": num(format="%+.2f"),
                "DM Stat": num(format="%+.2f"),
                "DM p-value": num(format="%.3f"),
                "RMSE (vol pts)": num(format="%.2f"),
                "MZ Slope b": num(format="%.2f"),
                "MZ R²": num(format="%.3f"),
                "Mean Fcst Vol (%)": num(format="%.2f"),
            },
        )
        st.markdown(self._forecast_verdict(table, bench))
        st.caption(
            "QLIKE is the primary loss (robust to noise in realized "
            "variance; Patton, 2011). RMSE is shown for intuition. MZ: "
            "regression of realized on forecast variance, where slope "
            f"b ≈ 1 means no bias. Mean realized vol: "
            f"{100 * np.sqrt(wf.frame['target_var']).mean():.2f}%."
        )

        with st.expander("Refit log"):
            st.dataframe(
                wf.refits, hide_index=True, width="stretch",
                column_config={
                    "Refit Date": st.column_config.DateColumn(
                        format="YYYY-MM-DD"),
                    "HMM LL/obs": num(format="%.3f"),
                    "HMM P(stay High)": num(format="%.3f"),
                    "GARCH α": num(format="%.3f"),
                    "GARCH β": num(format="%.3f"),
                    "GARCH α+β": num(format="%.4f"),
                    "GARCH LR Vol (%)": num(format="%.2f"),
                },
            )
            st.caption(
                "When GARCH α+β is close to 1, the long-run vol is poorly "
                "identified. Regime shifts in the sample are a known cause "
                "of spuriously high GARCH persistence (Lamoureux & "
                "Lastrapes, 1990)."
            )
            st.download_button(
                "Download forecasts (CSV)",
                data=wf.frame.to_csv().encode(),
                file_name="walk_forward_forecasts.csv", mime="text/csv",
            )

    def render_backtest_tab(
        self,
        wf: WalkForwardResult,
        bt_cfg: BacktestConfig,
        instrument: Instrument,
    ) -> None:
        """Vol-targeting backtest: KPIs, equity/drawdown/leverage, table."""
        bt = VolTargetBacktester(bt_cfg).run(wf)
        m = bt.metrics.set_index("Strategy")
        hmm, bh = m.loc["VolTgt · HMM"], m.loc[BUY_AND_HOLD]

        self.section("Volatility-targeting backtest · walk-forward, net of "
                     "costs")
        st.caption(
            f"Long {instrument.label}. Leverage is set at each close "
            "from "
            "forecasts issued at that close and earns the next day's "
            f"return. Target {100 * bt_cfg.target_vol:.1f}% · cap "
            f"{bt_cfg.max_leverage:.1f}× · cost {bt_cfg.cost_bps:.2f} bp "
            f"per 1× traded · band ±{bt_cfg.rebalance_band:.2f}×."
        )
        cols = st.columns(5)
        cols[0].metric(
            "Realized vol · HMM target", f"{hmm['Ann. Vol (%)']:.2f}%",
            delta=(f"{hmm['Ann. Vol (%)'] - 100 * bt_cfg.target_vol:+.2f}"
                   " pp vs target"),
            delta_color="off",
        )
        stability_gap = (hmm["Risk Stability (pp)"]
                         - bh["Risk Stability (pp)"])
        cols[1].metric(
            "Risk stability · HMM", f"{hmm['Risk Stability (pp)']:.2f} pp",
            delta=f"{stability_gap:+.2f} vs buy & hold",
            delta_color="inverse",
            help="Std. dev. of rolling 63-day realized vol. Lower means "
                 "steadier risk.",
        )
        cols[2].metric(
            "Max drawdown · HMM", f"{hmm['Max DD (%)']:.1f}%",
            delta=f"{hmm['Max DD (%)'] - bh['Max DD (%)']:+.1f} pp vs B&H",
        )
        cols[3].metric(
            "Sharpe · HMM", f"{hmm['Sharpe']:.2f}",
            delta=f"{hmm['Sharpe'] - bh['Sharpe']:+.2f} vs B&H",
        )
        cols[4].metric(
            "Cost drag · HMM", f"{hmm['Cost Drag (bp/yr)']:.1f} bp/yr",
            help=f"Turnover {hmm['Turnover (×/yr)']:.1f}× notional per "
                 "year.",
        )

        st.plotly_chart(
            ChartFactory.backtest_chart(bt), width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
        )
        self.section(f"Rolling {ROLLING_RISK_WINDOW}-day realized volatility "
                     "vs target")
        st.plotly_chart(
            ChartFactory.rolling_risk_chart(bt), width="stretch",
            config={"displaylogo": False},
        )

        self.section("Strategy statistics")
        num = st.column_config.NumberColumn
        st.dataframe(
            bt.metrics, hide_index=True, width="stretch",
            column_config={
                "CAGR (%)": num(format="%+.2f"),
                "Ann. Vol (%)": num(format="%.2f"),
                "Sharpe": num(format="%+.2f"),
                "Sortino": num(format="%+.2f"),
                "Max DD (%)": num(format="%.1f"),
                "Calmar": num(format="%+.2f"),
                "Avg Leverage (×)": num(format="%.2f"),
                "Turnover (×/yr)": num(format="%.1f"),
                "Cost Drag (bp/yr)": num(format="%.1f"),
                "Risk Stability (pp)": num(format="%.2f"),
            },
        )
        st.caption(
            "How to read this: "
            + BACKTEST_NOTES.get(instrument.asset_class,
                                 BACKTEST_NOTES["Custom"])
            + " Judge volatility targeting on risk control: realized vol "
            "close to target, low risk-stability numbers and shallower "
            "drawdowns, after costs. 'HMM De-risk' holds 1 − P(High Vol "
            "tomorrow) and uses the regime probabilities directly."
        )
        st.download_button(
            "Download strategy returns (CSV)",
            data=bt.returns.to_csv().encode(),
            file_name="vol_target_backtest_returns.csv", mime="text/csv",
        )

    def render_signal_tab(
        self,
        wf: WalkForwardResult,
        sig_cfg: SignalConfig,
        bt_cfg: BacktestConfig,
    ) -> None:
        """Signal Lab: regime-gated EMA trend and Bollinger reversion."""
        lab = SignalLab(sig_cfg, bt_cfg).run(wf)
        tests = lab.filter_tests
        metrics = lab.backtest.metrics.set_index("Strategy")

        self.section("Signal lab · do HMM regimes improve classic "
                     "trading signals?")
        st.caption(
            f"EMA {sig_cfg.ema_fast}/{sig_cfg.ema_slow} trend and "
            f"Bollinger {sig_cfg.bb_window}d ±{sig_cfg.bb_width:g}σ "
            "mean reversion, with fixed textbook parameters. The regime "
            "gates were chosen before looking at results: trend positions "
            "are scaled by 1 − P(High Vol tomorrow) (momentum-crash "
            "hypothesis) and reversion positions by P(Low Vol tomorrow) "
            "(range-bound hypothesis). Every strategy uses the same EWMA "
            f"vol target ({100 * bt_cfg.target_vol:.1f}%), the same band "
            "and the same costs, so the regime gate is the only "
            "difference between a signal and its filtered version."
        )

        cols = st.columns(4)
        for col, (a, b), title in zip(
            cols[:2],
            [(LAB_TREND_REGIME, LAB_TREND),
             (LAB_REVERSION_REGIME, LAB_REVERSION)],
            ["Trend: regime filter", "Reversion: regime filter"],
        ):
            row = tests[(tests["Strategy A"] == a)
                        & (tests["Strategy B"] == b)].iloc[0]
            col.metric(
                f"{title} ΔSharpe", f"{row['Δ Sharpe']:+.2f}",
                delta=f"p = {row['p-value']:.3f}", delta_color="off",
                help="Sharpe of the filtered strategy minus the unfiltered "
                     "one, with a block-bootstrap p-value.",
            )
        best = metrics["Sharpe"].idxmax()
        cols[2].metric("Best Sharpe", f"{metrics.loc[best, 'Sharpe']:+.2f}",
                       delta=best, delta_color="off")
        bb_active = float(
            (lab.backtest.weights[LAB_REVERSION].abs() > 1e-12).mean()
        )
        cols[3].metric("BB time in market", f"{bb_active:.0%}",
                       help="Share of days with an open Bollinger position.")

        st.plotly_chart(
            ChartFactory.signal_chart(lab), width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
        )
        self.section("Equity, drawdown and positions (net of costs)")
        st.plotly_chart(
            ChartFactory.backtest_chart(
                lab.backtest, colors=LAB_COLORS, emphasis=("Regime",),
                position_title="Position (×)",
            ),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
        )

        num = st.column_config.NumberColumn
        self.section("Strategy statistics")
        st.dataframe(
            lab.backtest.metrics, hide_index=True, width="stretch",
            column_config={
                "CAGR (%)": num(format="%+.2f"),
                "Ann. Vol (%)": num(format="%.2f"),
                "Sharpe": num(format="%+.2f"),
                "Sortino": num(format="%+.2f"),
                "Max DD (%)": num(format="%.1f"),
                "Calmar": num(format="%+.2f"),
                "Avg Leverage (×)": num(format="%.2f"),
                "Turnover (×/yr)": num(format="%.1f"),
                "Cost Drag (bp/yr)": num(format="%.1f"),
                "Risk Stability (pp)": num(format="%.2f"),
            },
        )

        self.section("Hypothesis tests · Sharpe differences (block "
                     "bootstrap)")
        st.dataframe(
            tests, hide_index=True, width="stretch",
            column_config={
                "Sharpe A": num(format="%+.2f"),
                "Sharpe B": num(format="%+.2f"),
                "Δ Sharpe": num(format="%+.2f"),
                "95% CI Low": num(format="%+.2f"),
                "95% CI High": num(format="%+.2f"),
                "p-value": num(format="%.3f"),
            },
        )
        lines = []
        for _, row in tests.iterrows():
            significant = row["p-value"] < 0.05
            if abs(row["Δ Sharpe"]) < 0.005:
                direction = "matches"
            else:
                direction = "improves on" if row["Δ Sharpe"] > 0 else "lags"
            lines.append(
                f"- **{row['Strategy A']}** {direction} "
                f"**{row['Strategy B']}** by {row['Δ Sharpe']:+.2f} Sharpe "
                f"(95% CI {row['95% CI Low']:+.2f} to "
                f"{row['95% CI High']:+.2f}): "
                + ("**significant** at the 5% level." if significant
                   else "not significant at the 5% level.")
            )
        st.markdown("\n".join(lines))

        self.section("Where does each signal work? Performance by regime")
        st.dataframe(
            lab.regime_breakdown, hide_index=True, width="stretch",
            column_config={
                "Days": num(format="%d"),
                "Active (%)": num(format="%.0f"),
                "Ann. Return (%)": num(format="%+.2f"),
                "Sharpe (ann.)": num(format="%+.2f"),
                "Hit Rate (%)": num(format="%.1f"),
            },
        )
        st.caption(
            "The regime breakdown is descriptive. Use it to understand the "
            "signals, not to choose a new filter. A filter picked after "
            "reading this table would be fitted to the same data it is "
            "tested on. The bootstrap resamples 20-day blocks of both "
            "strategies jointly, which preserves their serial dependence "
            "and correlation. The four tests are not adjusted for multiple "
            "comparisons."
        )
        st.download_button(
            "Download signal-lab returns (CSV)",
            data=lab.backtest.returns.to_csv().encode(),
            file_name="signal_lab_returns.csv", mime="text/csv",
        )

    @staticmethod
    def render_methodology() -> None:
        """Concise mathematical summary for reviewers."""
        st.markdown("#### Regime model")
        st.markdown("**Observations** (all backward-looking):")
        st.latex(
            r"r_t=\ln\frac{P_t}{P_{t-1}},\quad "
            r"\sigma_t=\sqrt{\frac{252}{w}\sum_{i=0}^{w-1}r_{t-i}^2},"
            r"\quad x_t=\big(r_t,\ \ln\sigma_t\big)"
        )
        st.markdown(
            "**Model:** hidden Markov chain with Gaussian emissions on "
            "training-standardized features $z_t$:"
        )
        st.latex(
            r"P(S_t=j\mid S_{t-1}=i)=A_{ij},\qquad "
            r"z_t\mid S_t=k\sim\mathcal{N}(\mu_k,\Sigma_k)"
        )
        st.markdown("**Causal inference** (Hamilton forward filter):")
        st.latex(
            r"P(S_t=j\mid z_{1:t})\propto "
            r"\mathcal{N}(z_t;\mu_j,\Sigma_j)"
            r"\sum_i P(S_{t-1}=i\mid z_{1:t-1})\,A_{ij}"
        )
        st.markdown(
            "- The **Regime Monitor** tab uses a single fit: scaler, EM "
            "parameters and state ordering are estimated on the in-sample "
            "block only, then frozen.\n"
            "- Out-of-sample labels use the **filtered** probability "
            "$P(S_t\\mid z_{1:t})$, never Viterbi or forward-backward "
            "smoothing, both of which condition on future data.\n"
            "- A label at day $t$ uses the close of day $t$; any "
            "trading rule built on it acts from $t+1$."
        )

        st.markdown("#### Walk-forward forecasting")
        st.markdown(
            "At each refit date $t_k$, every model (scaler, HMM, GARCH) "
            "is re-estimated on data in $[0, t_k)$ and frozen until "
            "$t_{k+1}$. Each forecast targets"
        )
        st.latex(r"y_t=\frac{252}{h}\sum_{i=1}^{h}r_{t+i}^2")
        st.latex(
            r"\hat y^{\text{HMM}}_t=252\,\alpha_t^\top"
            r"\Big(\tfrac1h\textstyle\sum_{i=1}^{h}A^i\Big)m,\quad "
            r"m_k=E[r^2\mid S=k]"
        )
        st.latex(
            r"\sigma^2_{t+1}=\omega+\alpha r_t^2+\beta\sigma_t^2"
            r"\ \ (\text{GARCH}),\qquad "
            r"\sigma^2_{t+1}=0.94\,\sigma_t^2+0.06\,r_t^2"
            r"\ \ (\text{EWMA})"
        )
        st.markdown(
            "Forecasts are ranked by QLIKE loss, "
            "$L=y/f-\\ln(y/f)-1$, and compared with Diebold–Mariano "
            "tests (HAC variance with $h-1$ lags, HLN correction)."
        )

        st.markdown("#### Vol-target backtest")
        st.latex(
            r"w_t=\min\!\Big(\frac{\sigma^\star}{\hat\sigma_t},L_{\max}"
            r"\Big),\qquad R^{\text{net}}_{t+1}=w_tR_{t+1}-c\,"
            r"|w_t-w_{t-1}|"
        )
        st.markdown("#### Signal lab")
        st.latex(
            r"s^{\text{trend}}_t=\operatorname{sign}"
            r"(EMA^{f}_t-EMA^{s}_t),\qquad "
            r"g^{\text{trend}}_t=1-P(S_{t+1}=\text{High}\mid\mathcal F_t)"
        )
        st.latex(
            r"s^{\text{rev}}_t\in\{-1,0,+1\}\ (\text{band entry, mid exit}),"
            r"\qquad g^{\text{rev}}_t=P(S_{t+1}=\text{Low}\mid\mathcal F_t)"
        )
        st.markdown(
            r"Positions are $w_t=s_t\,g_t\,\min(\sigma^\star/"
            r"\hat\sigma^{EWMA}_t,L_{\max})$. Sharpe differences are "
            "tested with a circular block bootstrap."
        )
        st.markdown(
            "A no-trade band holds $w_t$ fixed until the target moves by "
            "more than the band. Choosing settings by looking at "
            "out-of-sample results is itself data snooping. Fix them in "
            "advance, or report results across a range of settings."
        )

    # -- entry point --------------------------------------------------------

    def run(self) -> None:
        """Render the full dashboard."""
        self.inject_theme()
        settings = self.render_sidebar()
        if settings is None:
            st.stop()
            return
        config, wf_cfg, bt_cfg, sig_cfg = settings

        try:
            with st.spinner("Fetching data and fitting the HMM…"):
                result = run_regime_pipeline(
                    config.instrument, config.start, config.end,
                    config.train_fraction, config.vol_window,
                    config.n_restarts, config.random_seed,
                )
        except (DataUnavailableError, InsufficientDataError,
                ModelFitError, ValueError) as exc:
            st.markdown(
                f'<div class="app-title">{APP_TITLE}</div>',
                unsafe_allow_html=True,
            )
            st.error(str(exc))
            st.stop()
            return

        filtered_last = result.test_frame[
            [f"p_{lab}" for lab in REGIME_LABELS]
        ].to_numpy()[-1]
        next_day = filtered_last @ result.params.trans_mat

        self.render_header(result)
        tabs = st.tabs([
            "Regime Monitor", "Volatility Forecasts",
            "Vol-Target Backtest", "Signal Lab", "Methodology",
        ])
        with tabs[0]:
            self.section("Snapshot")
            self.render_kpis(result, next_day)
            self.render_main_chart(result)
            self.render_bottom_row(result)
            self.render_diagnostics(result)
        with tabs[4]:
            self.render_methodology()

        wf: Optional[WalkForwardResult] = None
        with tabs[1]:
            try:
                wf = self._walk_forward(config, wf_cfg)
            except (ModelFitError, ValueError,
                    np.linalg.LinAlgError) as exc:
                st.error(f"Walk-forward run failed: {exc}")
            if wf is not None:
                self.render_forecast_tab(wf)
        with tabs[2]:
            if wf is None:
                st.info("The backtest needs the walk-forward forecasts, "
                        "which failed to run.")
            else:
                self.render_backtest_tab(wf, bt_cfg, config.instrument)
        with tabs[3]:
            if wf is None:
                st.info("The signal lab needs the walk-forward regime "
                        "probabilities, which failed to run.")
            else:
                self.render_signal_tab(wf, sig_cfg, bt_cfg)


def main() -> None:
    """Entry point.

    ``streamlit run app.py`` starts the dashboard. ``python app.py
    --snapshot`` downloads every built-in instrument into ``data/`` so a
    hosted copy keeps working when Yahoo is unreachable.
    """
    if "--snapshot" in sys.argv:
        save_snapshots()
        return
    DashboardUI().run()


if __name__ == "__main__":
    main()
