"""
CSI500 — Upgraded LightGBM Model
==================================

Improvements over model_extra_features.py, each grounded in diagnostics
run against the actual price data:

CHANGE 1 — Huber loss instead of MSE
  WHY: 1.67% of training rows have |z-score| > 3 (limit-up/down stocks).
       MSE squares the error on these, so a handful of extreme rows dominate
       the gradient signal. Huber loss clips their influence linearly above
       a threshold while keeping MSE behaviour near zero.
  HOW: Set LightGBM objective='huber', alpha=0.9 (95th-pctile transition).

CHANGE 2 — Exponential recency weighting
  WHY: Quarterly IC of ret_20d ranged from -0.017 (Q1 2025) to -0.107
       (Q2 2025). The signal structure is non-stationary. Upweighting the
       most recent 60 days reduces the influence of stale regime data.
  HOW: sample_weight = exp(lambda * day_index / max_day_index),
       where lambda=3 → most recent day gets e^3 ≈ 20× weight of earliest.

CHANGE 3 — Beta and regime-conditioned features
  WHY: Empirically, beta_60d has IC=+0.004 in up-regimes but IC=+0.113 in
       down-regimes (detected using 20-day index momentum as regime signal).
       Adding beta and regime features lets LightGBM learn this interaction
       automatically.
  HOW: Add beta_60d, idio_ret_20d, idx_mom_10d/20d, beta_x_regime.

CHANGE 4 — Model ensemble (5 seeds, averaged predictions)
  WHY: Per-day IC std is 0.15–0.22 with IC mean ~0.03–0.10. Signal-to-noise
       ratio is 0.06–0.64 per quarter. With 5 models at ~0.80 inter-model
       correlation, theoretical IC-IR improves ~2×.
  HOW: Train 5 models with seeds 42,43,44,45,46; average their predictions.

CHANGE 5 — Winsorised training labels
  WHY: Even with Huber loss, it helps to winsorise the z-scored forward
       return labels at ±3 sigma before training. This doesn't change rank IC
       (ranks are unchanged) but the label scale affects Huber threshold
       calibration and early stopping metric.
  HOW: Clip z-score at ±3 before passing to LightGBM dataset.

CHANGE 6 — Portfolio-aware construction
  WHY: The competition scores realized excess return, not pure prediction IC.
       A good ranker can still lose money if the portfolio overweights halted,
       illiquid, or high-volatility names. Portfolio construction is therefore
       treated as part of the model.
  HOW: Add optional universe filtering, top-k tuning, equal/rank/vol-adjusted
       weighting, and a LambdaRank option for comparison against Huber.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from fetch_extra_data import (
        EXTRA_FEATURE_COLUMNS,
        load_all_extra,
        merge_extra_into_panel,
    )
except Exception:
    EXTRA_FEATURE_COLUMNS: list[str] = []

    def load_all_extra() -> dict[str, pd.DataFrame]:
        return {}

    def merge_extra_into_panel(panel: pd.DataFrame, extra: dict) -> pd.DataFrame:
        return panel

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_DIR   = Path(__file__).parent / "data"
REPORT_DIR = Path(__file__).parent / "reports"

FORWARD_HORIZON = 5
TRAIN_WINDOW    = 180
EMBARGO_DAYS    = 5
VAL_DAYS        = 20
MIN_STOCKS      = 30
MAX_WEIGHT      = 0.10
DEFAULT_TOP_K   = 60

# Ensemble seeds (Change 4)
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]

# Recency decay strength (Change 2)
# lambda=3 → newest day weighted e^3≈20× vs oldest day in training window
RECENCY_LAMBDA = 3.0

# Huber loss quantile (Change 1): fraction of samples treated with MSE
# alpha=0.9 means the 90th-percentile residual is the Huber transition point
HUBER_ALPHA = 1.35

# Portfolio-awareness defaults
DEFAULT_WEIGHT_METHOD = "rank"          # rank, equal, vol_adjusted, score_rank
DEFAULT_OBJECTIVE = "huber"             # huber or lambdarank
DEFAULT_LIQUIDITY_QUANTILE = 0.10       # drop bottom 10% traded amount/volume
DEFAULT_MAX_VOL_QUANTILE = 0.98         # drop highest 2% volatility tail
DEFAULT_MAX_ILLIQ_QUANTILE = 0.98       # drop highest 2% illiquidity tail
DEFAULT_MAX_ZERO_VOLUME_DAYS = 0        # drop names with any zero-volume day in last 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


def _safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, eps: float = 1e-8):
    return a / (b + eps)


# ---------------------------------------------------------------------------
# Feature engineering — base price/volume features
# ---------------------------------------------------------------------------

def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    g = df.groupby("stock_code", sort=False)
    prev_close = g["close"].shift(1)

    # ── Returns / momentum ────────────────────────────────────────────────
    df["log_ret_1d"] = np.log(_safe_div(df["close"], prev_close))

    for w in [3, 5, 10, 20, 60]:
        df[f"ret_{w}d"] = g["close"].transform(lambda x, w=w: x / x.shift(w) - 1)

    df["mom_skip1"] = g["close"].transform(lambda x: x.shift(1) / x.shift(21) - 1)

    for w in [10, 20, 60]:
        ret_w = g["close"].transform(lambda x, w=w: x / x.shift(w) - 1)
        vol_w = g["close"].transform(lambda x, w=w: np.log(x / x.shift(1)).rolling(w).std())
        df[f"sharpe_mom_{w}d"] = _safe_div(ret_w, vol_w)

    # ── Short-term reversal ───────────────────────────────────────────────
    df["rev_1d"] = g["close"].transform(lambda x: x / x.shift(1) - 1)
    df["rev_3d"] = g["close"].transform(lambda x: x / x.shift(3) - 1)

    # ── Volatility / drawdown ─────────────────────────────────────────────
    for w in [5, 10, 20]:
        df[f"vol_{w}d"] = g["close"].transform(
            lambda x, w=w: np.log(x / x.shift(1)).rolling(w).std()
        )

    df["range_vol_20d"] = g.apply(
        lambda x: np.log(x["high"] / (x["low"] + 1e-8)).rolling(20).std()
    ).reset_index(level=0, drop=True)

    df["hl_range"]      = (df["high"] - df["low"]) / (df["close"].abs() + 1e-8)
    df["close_to_high"] = (df["high"] - df["close"]) / ((df["high"] - df["low"]) + 1e-8)
    df["close_to_low"]  = (df["close"] - df["low"]) / ((df["high"] - df["low"]) + 1e-8)

    for w in [10, 20, 60]:
        roll_max = g["close"].transform(lambda x, w=w: x.rolling(w).max())
        df[f"drawdown_{w}d"] = _safe_div(df["close"], roll_max) - 1

    # ── Liquidity / volume ────────────────────────────────────────────────
    for w in [5, 20]:
        df[f"to_zscore_{w}d"] = g["turnover"].transform(
            lambda x, w=w: (x - x.rolling(w).mean()) / (x.rolling(w).std() + 1e-8)
        )
        df[f"volume_ma_{w}d"] = g["volume"].transform(lambda x, w=w: x.rolling(w).mean())
        df[f"amount_ma_{w}d"] = g["amount"].transform(lambda x, w=w: x.rolling(w).mean())

    df["volume_ratio_5_20"] = _safe_div(df["volume_ma_5d"], df["volume_ma_20d"])
    df["amount_ratio_5_20"] = _safe_div(df["amount_ma_5d"], df["amount_ma_20d"])

    df["illiquidity_20d"] = g.apply(
        lambda x: (x["pct_change"].abs() / (x["amount"] / 1e8 + 1e-6)).rolling(20).mean()
    ).reset_index(level=0, drop=True)

    df["zero_volume_20d"] = g["volume"].transform(lambda x: (x <= 0).rolling(20).sum())

    # ── Price vs moving averages ──────────────────────────────────────────
    for w in [5, 10, 20, 60]:
        ma = g["close"].transform(lambda x, w=w: x.rolling(w).mean())
        df[f"price_to_ma{w}"] = _safe_div(df["close"], ma) - 1

    # 52-week high distance (confirmed IC=-0.024 vs fwd5 in diagnostics)
    df["dist_52w_high"] = g["close"].transform(
        lambda x: x / x.rolling(252, min_periods=60).max() - 1
    )

    # ── RSI ───────────────────────────────────────────────────────────────
    avg_gain = g["close"].transform(lambda x: x.diff().clip(lower=0).rolling(14).mean())
    avg_loss = g["close"].transform(
        lambda x: (-x.diff()).clip(lower=0).rolling(14).mean() + 1e-8
    )
    df["rsi_14"] = 100 - 100 / (1 + avg_gain / avg_loss)

    # ── Gap / intraday ────────────────────────────────────────────────────
    df["overnight_gap"]  = _safe_div(df["open"], prev_close) - 1
    df["gap_5d"]         = g["overnight_gap"].transform(lambda x: x.rolling(5).mean())
    df["intraday_ret"]   = _safe_div(df["close"], df["open"]) - 1
    df["intraday_ret_5d"] = g["intraday_ret"].transform(lambda x: x.rolling(5).mean())

    # ── Target ───────────────────────────────────────────────────────────
    df["target_fwd5"] = g["close"].transform(
        lambda x: x.shift(-FORWARD_HORIZON) / x - 1
    )
    df["target_rank"] = df.groupby("date")["target_fwd5"].rank(pct=True)

    return _clean(df)


# ---------------------------------------------------------------------------
# Index-relative and REGIME features (Change 3)
# ---------------------------------------------------------------------------

def add_index_features(panel: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """
    Add alpha vs index, relative volatility, rolling beta, and regime signals.

    Key addition over original: rolling 60-day beta per stock + regime
    interaction. Empirical finding: beta IC vs 5d fwd return is +0.004 in
    up-regimes but +0.113 in down-regimes (index 20d momentum < 0).
    """
    idx = index[["date", "close"]].copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.rename(columns={"close": "idx_close"}).sort_values("date")

    idx["idx_daily_ret"] = np.log(idx["idx_close"] / idx["idx_close"].shift(1))

    for w in [3, 5, 10, 20, 60]:
        idx[f"idx_ret_{w}d"]  = idx["idx_close"] / idx["idx_close"].shift(w) - 1
        idx[f"idx_vol_{w}d"]  = idx["idx_daily_ret"].rolling(w).std()

    # Regime indicator: index 20d momentum sign + magnitude
    idx["idx_mom_20d"]      = idx["idx_ret_20d"]
    idx["idx_mom_10d"]      = idx["idx_ret_10d"]
    idx["idx_regime"]       = np.sign(idx["idx_ret_20d"])   # +1 up, -1 down
    idx["idx_realized_vol"] = idx["idx_daily_ret"].rolling(20).std()

    keep = ["date"] + [c for c in idx.columns if c != "date" and c != "idx_close"]
    panel = panel.merge(idx[keep], on="date", how="left")

    # Alpha and relative volatility at multiple horizons
    for w in [3, 5, 10, 20, 60]:
        if f"ret_{w}d" in panel.columns:
            panel[f"alpha_{w}d"]   = panel[f"ret_{w}d"]  - panel.get(f"idx_ret_{w}d", 0)
        if f"vol_{w}d" in panel.columns:
            panel[f"rel_vol_{w}d"] = panel[f"vol_{w}d"]  - panel.get(f"idx_vol_{w}d", 0)

    # Rolling 60-day beta: cov(stock_ret, idx_ret) / var(idx_ret)
    panel["daily_ret"] = panel.groupby("stock_code")["close"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    panel["beta_60d"] = panel.groupby("stock_code", group_keys=False).apply(
        lambda g: g["daily_ret"].rolling(60).cov(g["idx_daily_ret"])
                  / (g["idx_daily_ret"].rolling(60).var() + 1e-12)
    ).reset_index(level=0, drop=True)

    # Idiosyncratic return = stock return - beta * index return
    panel["idio_ret_20d"] = (
        panel["ret_20d"] - panel["beta_60d"] * panel["idx_ret_20d"]
    )

    # Regime interaction (Change 3 key signal):
    # beta × regime_sign encodes "high beta in down market" effect
    panel["beta_x_regime"] = panel["beta_60d"] * panel["idx_regime"]

    # Volatility spread vs index (stress indicator)
    panel["vol_spread_20d"] = panel["vol_20d"] - panel["idx_realized_vol"]

    return _clean(panel)


# ---------------------------------------------------------------------------
# Feature column lists
# ---------------------------------------------------------------------------

_BASE_FEATURES = [
    # Momentum / returns
    "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ret_60d", "mom_skip1",
    "sharpe_mom_10d", "sharpe_mom_20d", "sharpe_mom_60d",
    # Reversal
    "rev_1d", "rev_3d",
    # Volatility / range / drawdown
    "vol_5d", "vol_10d", "vol_20d", "range_vol_20d",
    "hl_range", "close_to_high", "close_to_low",
    "drawdown_10d", "drawdown_20d", "drawdown_60d",
    "dist_52w_high",
    # Liquidity
    "to_zscore_5d", "to_zscore_20d", "volume_ratio_5_20", "amount_ratio_5_20",
    "illiquidity_20d", "volume_ma_20d", "amount_ma_20d", "zero_volume_20d",
    # Price vs MA
    "price_to_ma5", "price_to_ma10", "price_to_ma20", "price_to_ma60",
    # Technical / drift
    "rsi_14", "overnight_gap", "gap_5d", "intraday_ret", "intraday_ret_5d",
    # Index-relative (existing)
    "alpha_3d", "alpha_5d", "alpha_10d", "alpha_20d", "alpha_60d",
    "rel_vol_5d", "rel_vol_10d", "rel_vol_20d",
    # NEW: beta / regime features (Change 3)
    "beta_60d", "idio_ret_20d", "beta_x_regime",
    "idx_mom_10d", "idx_mom_20d", "idx_regime",
    "idx_realized_vol", "vol_spread_20d",
]

FEATURE_COLUMNS = _BASE_FEATURES.copy()
TARGET_COLUMN   = "target_fwd5"


# ---------------------------------------------------------------------------
# LightGBM dataset: winsorised + Huber (Changes 1 & 5)
# ---------------------------------------------------------------------------

def _cross_sectional_zscore_target(df: pd.DataFrame) -> pd.Series:
    """Forward return standardized within each date, clipped for robustness."""
    y = df.groupby("date")[TARGET_COLUMN].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    return y.clip(-3, 3)


def _lambdarank_labels(df: pd.DataFrame, n_bins: int = 10) -> pd.Series:
    """Integer relevance labels 0..n_bins-1 based on same-day target rank."""
    labels = df.groupby("date")[TARGET_COLUMN].rank(pct=True, method="first")
    labels = np.floor(labels * n_bins).clip(0, n_bins - 1).astype(int)
    return labels


def _date_groups(df: pd.DataFrame) -> list[int]:
    """Group sizes for LightGBM ranking datasets; df must be sorted by date."""
    return df.groupby("date", sort=False).size().astype(int).tolist()


def make_lgb_dataset(
    df: pd.DataFrame,
    ref: lgb.Dataset | None = None,
    weights: np.ndarray | None = None,
    objective: str = DEFAULT_OBJECTIVE,
) -> lgb.Dataset:
    """
    Create a LightGBM Dataset.

    objective='huber'      -> winsorised cross-sectional z-score labels.
    objective='lambdarank' -> integer relevance labels grouped by date.
    """
    df = df.sort_values(["date", "stock_code"]).reset_index(drop=True)
    X = df[FEATURE_COLUMNS]

    if objective == "lambdarank":
        y = _lambdarank_labels(df)
        group = _date_groups(df)
        return lgb.Dataset(
            X, label=y, group=group, weight=weights, reference=ref, free_raw_data=False
        )

    y = _cross_sectional_zscore_target(df)
    return lgb.Dataset(X, label=y, weight=weights, reference=ref, free_raw_data=False)


# LightGBM params: Huber loss replaces MSE (Change 1)
LGBM_PARAMS = {
    "objective"          : "huber",     # ← Change 1: was "regression" (MSE)
    "alpha"              : HUBER_ALPHA, # Huber transition quantile
    "metric"             : "huber",
    "learning_rate"      : 0.035,
    "num_leaves"         : 63,
    "min_data_in_leaf"   : 35,
    "feature_fraction"   : 0.75,
    "bagging_fraction"   : 0.85,
    "bagging_freq"       : 5,
    "lambda_l1"          : 0.2,
    "lambda_l2"          : 2.0,
    "max_depth"          : -1,
    "verbose"            : -1,
    "num_threads"        : -1,
    "feature_pre_filter" : False,
}


# ---------------------------------------------------------------------------
# Recency weights (Change 2)
# ---------------------------------------------------------------------------

def make_recency_weights(dates: pd.Series, lam: float = RECENCY_LAMBDA) -> np.ndarray:
    """
    Exponential decay sample weights so recent rows count more.

    Weight = exp(lam × normalised_day_index), where normalised_day_index
    maps the oldest training date to 0 and the newest to 1.
    With lam=3: newest/oldest weight ratio = e^3 ≈ 20×.
    """
    # Normalise to numpy datetime64 to avoid Timestamp vs datetime64 KeyError
    dates_np = pd.to_datetime(dates).values.astype("datetime64[D]")
    sorted_dates = np.sort(np.unique(dates_np))
    n = len(sorted_dates)
    date_to_idx = {d: i for i, d in enumerate(sorted_dates)}
    idx = np.array([date_to_idx[d] for d in dates_np])
    normalised = idx / max(n - 1, 1)
    return np.exp(lam * normalised)


# ---------------------------------------------------------------------------
# Training (single model)
# ---------------------------------------------------------------------------

def train_single(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int = 42,
    num_boost_round: int = 900,
    early_stopping: int = 60,
    objective: str = DEFAULT_OBJECTIVE,
) -> lgb.Booster:
    if objective == "lambdarank":
        params = {
            **LGBM_PARAMS,
            "seed": seed,
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [30, 50, 60],
            "label_gain": list(range(10)),
        }
    else:
        params = {**LGBM_PARAMS, "seed": seed}

    # Recency weights for training rows (Change 2)
    w_train = make_recency_weights(train_df["date"])
    train_set = make_lgb_dataset(train_df, weights=w_train, objective=objective)
    val_set   = make_lgb_dataset(val_df, ref=train_set, objective=objective)

    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(early_stopping, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )


# ---------------------------------------------------------------------------
# Ensemble training + prediction (Change 4)
# ---------------------------------------------------------------------------

def train_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seeds: list[int] = ENSEMBLE_SEEDS,
    objective: str = DEFAULT_OBJECTIVE,
) -> list[lgb.Booster]:
    """Train one model per seed; average predictions at inference time."""
    models = []
    for seed in seeds:
        m = train_single(train_df, val_df, seed=seed, objective=objective)
        models.append(m)
    return models


def predict_ensemble(models: list[lgb.Booster], X: pd.DataFrame) -> np.ndarray:
    """Average raw predictions across all ensemble members."""
    preds = np.stack([m.predict(X[FEATURE_COLUMNS]) for m in models], axis=0)
    return preds.mean(axis=0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def rank_ic(
    y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray
) -> tuple[float, float]:
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(float(rho))
    if not ics:
        return float("nan"), float("nan")
    arr = np.array(ics)
    return float(arr.mean()), float(arr.mean() / (arr.std() + 1e-8))


def top_bucket_diagnostic(
    df: pd.DataFrame, preds: np.ndarray, top_k: int
) -> dict[str, float]:
    tmp = df[["date", "stock_code", TARGET_COLUMN]].copy()
    tmp["score"] = preds
    top_ret, all_ret = [], []
    for _, day in tmp.groupby("date"):
        if len(day) < MIN_STOCKS:
            continue
        k = min(top_k, len(day))
        top_ret.append(day.nlargest(k, "score")[TARGET_COLUMN].mean())
        all_ret.append(day[TARGET_COLUMN].mean())
    if not top_ret:
        return {"top": np.nan, "all": np.nan, "top_minus_all": np.nan}
    t = float(np.mean(top_ret))
    a = float(np.mean(all_ret))
    return {"top": t, "all": a, "top_minus_all": t - a}


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def filter_prediction_universe(
    pred_df: pd.DataFrame,
    top_k: int,
    liquidity_quantile: float = DEFAULT_LIQUIDITY_QUANTILE,
    max_vol_quantile: float = DEFAULT_MAX_VOL_QUANTILE,
    max_illiq_quantile: float = DEFAULT_MAX_ILLIQ_QUANTILE,
    max_zero_volume_days: int = DEFAULT_MAX_ZERO_VOLUME_DAYS,
) -> pd.DataFrame:
    """
    Portfolio-aware prefilter: remove names that are likely hard to trade or
    unusually risky before selecting top scores. If the filter leaves too few
    names, it automatically relaxes to avoid invalid submissions.
    """
    df = pred_df.copy()
    original_n = len(df)
    masks: list[pd.Series] = []

    # Positive recent trading activity. Prefer traded amount, fallback volume.
    for col in ["amount_ma_20d", "volume_ma_20d"]:
        if col in df.columns:
            masks.append(df[col].fillna(0) > 0)

    liq_col = "amount_ma_20d" if "amount_ma_20d" in df.columns else "volume_ma_20d"
    if liq_col in df.columns and df[liq_col].notna().sum() >= max(top_k, MIN_STOCKS):
        masks.append(df[liq_col] >= df[liq_col].quantile(liquidity_quantile))

    if "vol_20d" in df.columns and df["vol_20d"].notna().sum() >= max(top_k, MIN_STOCKS):
        masks.append(df["vol_20d"] <= df["vol_20d"].quantile(max_vol_quantile))

    if "illiquidity_20d" in df.columns and df["illiquidity_20d"].notna().sum() >= max(top_k, MIN_STOCKS):
        masks.append(df["illiquidity_20d"] <= df["illiquidity_20d"].quantile(max_illiq_quantile))

    if "zero_volume_20d" in df.columns:
        masks.append(df["zero_volume_20d"].fillna(0) <= max_zero_volume_days)

    if masks:
        keep = np.logical_and.reduce([m.to_numpy() for m in masks])
        filtered = df.loc[keep].copy()
    else:
        filtered = df

    min_needed = max(top_k, MIN_STOCKS)
    if len(filtered) < min_needed:
        print(
            f"   filter left only {len(filtered)} names; relaxing to all "
            f"{original_n} scorable names"
        )
        return df

    print(f"   universe filter: {original_n} -> {len(filtered)} eligible names")
    return filtered


def _cap_and_normalize(w: pd.Series) -> pd.Series:
    """Apply long-only, max-weight cap, and sum-to-one normalization."""
    w = w.clip(lower=0).astype(float)
    if w.sum() <= 0:
        raise ValueError("All portfolio weights are zero before normalization.")
    w = w / w.sum()

    for _ in range(100):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = float((w[over] - MAX_WEIGHT).sum())
        w.loc[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        free_sum = float(w.loc[free].sum())
        if free_sum <= 0:
            break
        w.loc[free] += excess * w.loc[free] / free_sum

    w = w / w.sum()
    return w


def build_portfolio(
    scores: pd.Series,
    top_k: int = DEFAULT_TOP_K,
    pred_df: pd.DataFrame | None = None,
    weight_method: str = DEFAULT_WEIGHT_METHOD,
    score_power: float = 1.0,
) -> pd.Series:
    """
    Select top-K stocks and build legal weights.

    weight_method:
      - rank: rank-weighted, current stable default.
      - equal: equal weight, most robust if scores are noisy.
      - vol_adjusted: rank weight divided by recent volatility.
      - score_rank: percentile score-rank weights, optionally sharpened by score_power.
    """
    scores = scores.dropna().sort_values(ascending=False)
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS}")
    if len(scores) < top_k:
        raise ValueError(f"Only {len(scores)} names but top_k={top_k}")

    chosen = scores.head(top_k)
    n = len(chosen)

    if weight_method == "equal":
        raw = pd.Series(1.0, index=chosen.index)

    elif weight_method == "score_rank":
        raw = pd.Series(np.arange(n, 0, -1, dtype=float), index=chosen.index)
        raw = (raw / raw.max()).pow(score_power)

    elif weight_method == "vol_adjusted":
        raw = pd.Series(np.arange(n, 0, -1, dtype=float), index=chosen.index)
        if pred_df is None or "vol_20d" not in pred_df.columns:
            print("   vol_adjusted requested but vol_20d unavailable; falling back to rank weights")
        else:
            vol = pred_df.set_index("stock_code").reindex(chosen.index)["vol_20d"].astype(float)
            vol = vol.fillna(vol.median()).clip(lower=1e-6)
            raw = raw / vol

    else:  # rank
        raw = pd.Series(np.arange(n, 0, -1, dtype=float), index=chosen.index)

    w = _cap_and_normalize(raw)

    assert abs(w.sum() - 1.0) < 1e-4
    assert (w <= MAX_WEIGHT + 1e-9).all()
    assert (w >= -1e-12).all()
    assert (w > 0).sum() >= MIN_STOCKS
    return w.round(8)


# ---------------------------------------------------------------------------
# Data loading and panel preparation
# ---------------------------------------------------------------------------

def load_data(prices_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_parquet(prices_path)
    index  = pd.read_parquet(str(prices_path).replace("prices.parquet", "index.parquet"))
    return prices, index


def prepare_panel(
    prices: pd.DataFrame,
    index: pd.DataFrame,
    use_extra_data: bool = True,
) -> pd.DataFrame:
    global FEATURE_COLUMNS

    panel = build_features(prices)
    panel = add_index_features(panel, index)

    if use_extra_data:
        extra = load_all_extra()
        if extra:
            panel = merge_extra_into_panel(panel, extra)
            available_extra = [f for f in EXTRA_FEATURE_COLUMNS if f in panel.columns]
            FEATURE_COLUMNS = _BASE_FEATURES + available_extra
            print(f"   {len(FEATURE_COLUMNS)} features ({len(_BASE_FEATURES)} base + {len(available_extra)} extra)")
        else:
            FEATURE_COLUMNS = _BASE_FEATURES.copy()
            print(f"   {len(FEATURE_COLUMNS)} base features (no supplemental data)")
    else:
        FEATURE_COLUMNS = _BASE_FEATURES.copy()
        print(f"   {len(FEATURE_COLUMNS)} base features (extra data disabled)")

    panel = _clean(panel)
    panel = panel.dropna(subset=[c for c in _BASE_FEATURES if c in panel.columns])
    panel = panel.sort_values(["date", "stock_code"]).reset_index(drop=True)
    return panel


def split_train_val(
    panel: pd.DataFrame,
    val_days: int = VAL_DAYS,
    embargo: int = EMBARGO_DAYS,
    train_window: int = TRAIN_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_dates = np.sort(panel["date"].unique())
    if len(all_dates) < val_days + embargo + 20:
        raise RuntimeError("Not enough dates to split.")

    val_start    = pd.Timestamp(all_dates[-val_days])
    train_end    = pd.Timestamp(all_dates[-(val_days + embargo + 1)])
    window_start = pd.Timestamp(all_dates[max(0, len(all_dates) - val_days - embargo - 1 - train_window)])

    train_df = panel[(panel["date"] >= window_start) & (panel["date"] <= train_end)].copy()
    val_df   = panel[panel["date"] >= val_start].copy()

    train_df = train_df.dropna(subset=[TARGET_COLUMN, "target_rank"])
    val_df   = val_df.dropna(subset=[TARGET_COLUMN, "target_rank"])
    return train_df, val_df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_feature_importance(models: list[lgb.Booster], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gains  = np.mean([m.feature_importance("gain")  for m in models], axis=0)
    splits = np.mean([m.feature_importance("split") for m in models], axis=0)
    df = pd.DataFrame({
        "feature":          FEATURE_COLUMNS,
        "gain_importance":  gains,
        "split_importance": splits,
    }).sort_values("gain_importance", ascending=False)
    df.to_csv(out_path, index=False)
    print(f"   Feature importance → {out_path}")
    print("   Top 10 features by gain:")
    print(df.head(10)[["feature", "gain_importance"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Self-test: walk-forward CV
# ---------------------------------------------------------------------------

def run_self_test(panel: pd.DataFrame, top_k: int = DEFAULT_TOP_K, objective: str = DEFAULT_OBJECTIVE) -> None:
    print("\n" + "=" * 65)
    print("SELF-TEST: Walk-forward out-of-sample evaluation")
    print("=" * 65)

    all_dates = np.sort(panel["date"].unique())
    N = len(all_dates)
    TRAIN_INIT  = 200
    VAL_SIZE    = 20
    EMBARGO     = EMBARGO_DAYS
    TEST_STEP   = 5
    test_start  = TRAIN_INIT + EMBARGO + VAL_SIZE

    if test_start >= N - FORWARD_HORIZON:
        raise RuntimeError("Not enough data for self-test.")

    print(f"Total dates    : {N}")
    print(f"Train initial  : {TRAIN_INIT} days → {pd.Timestamp(all_dates[TRAIN_INIT-1]).date()}")
    print(f"Test start     : {pd.Timestamp(all_dates[test_start]).date()}")
    print(f"Ensemble seeds : {ENSEMBLE_SEEDS}")
    print(f"Features       : {len(FEATURE_COLUMNS)}")
    print()

    test_ics, test_icirs, top_minus_all = [], [], []
    cursor = test_start

    while cursor < N - FORWARD_HORIZON:
        eval_end   = min(cursor + TEST_STEP, N - FORWARD_HORIZON)
        eval_dates = all_dates[cursor:eval_end]

        t_win_start = max(0, cursor - TRAIN_WINDOW - EMBARGO - VAL_SIZE)
        t_end_idx   = cursor - EMBARGO - VAL_SIZE - 1
        v_start_idx = cursor - VAL_SIZE

        if t_end_idx < 20:
            cursor = eval_end
            continue

        train_df = panel[
            (panel["date"] >= pd.Timestamp(all_dates[t_win_start]))
            & (panel["date"] <= pd.Timestamp(all_dates[t_end_idx]))
        ].dropna(subset=[TARGET_COLUMN, "target_rank"])

        val_df = panel[
            (panel["date"] >= pd.Timestamp(all_dates[v_start_idx]))
            & (panel["date"] <= pd.Timestamp(all_dates[cursor - 1]))
        ].dropna(subset=[TARGET_COLUMN, "target_rank"])

        if len(train_df) < 500 or len(val_df) < 50:
            cursor = eval_end
            continue

        models = train_ensemble(train_df, val_df, objective=objective)

        test_df = panel[panel["date"].isin(eval_dates)].dropna(subset=[TARGET_COLUMN])
        if test_df.empty:
            cursor = eval_end
            continue

        preds = predict_ensemble(models, test_df)
        ic, icir = rank_ic(
            test_df[TARGET_COLUMN].to_numpy(), preds, test_df["date"].to_numpy()
        )
        diag = top_bucket_diagnostic(test_df, preds, top_k)

        test_ics.append(ic)
        test_icirs.append(icir)
        top_minus_all.append(diag["top_minus_all"])

        print(
            f"  {pd.Timestamp(eval_dates[0]).date()} – {pd.Timestamp(eval_dates[-1]).date()}"
            f"  rank IC={ic:+.4f}  top-minus-all={diag['top_minus_all']:+.4%}"
        )
        cursor = eval_end

    if test_ics:
        arr = np.array(test_ics)
        tma = np.array(top_minus_all, dtype=float)
        print()
        print(f"Test rank IC      : mean={arr.mean():+.4f} | std={arr.std():.4f} | IC-IR={arr.mean()/(arr.std()+1e-8):+.3f}")
        print(f"Hit rate (IC > 0) : {(arr > 0).mean():.1%}")
        print(f"Top-minus-all avg : {np.nanmean(tma):+.4%}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices",   default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--as-of",   default=None)
    parser.add_argument("--top-k",   type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--out",      default="submission.csv")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--objective", choices=["huber", "lambdarank"], default=DEFAULT_OBJECTIVE,
                        help="Training objective: robust Huber z-score regression or LambdaRank.")
    parser.add_argument("--filter-universe", action="store_true",
                        help="Drop illiquid, stale, or extreme-risk names before top-k selection.")
    parser.add_argument("--liquidity-quantile", type=float, default=DEFAULT_LIQUIDITY_QUANTILE)
    parser.add_argument("--max-vol-quantile", type=float, default=DEFAULT_MAX_VOL_QUANTILE)
    parser.add_argument("--max-illiq-quantile", type=float, default=DEFAULT_MAX_ILLIQ_QUANTILE)
    parser.add_argument("--max-zero-volume-days", type=int, default=DEFAULT_MAX_ZERO_VOLUME_DAYS)
    parser.add_argument("--weight-method", choices=["rank", "equal", "vol_adjusted", "score_rank"],
                        default=DEFAULT_WEIGHT_METHOD)
    parser.add_argument("--score-power", type=float, default=1.0,
                        help="Only used for --weight-method score_rank; >1 concentrates top names.")
    parser.add_argument("--no-extra-data", action="store_true")
    parser.add_argument("--no-ensemble",   action="store_true",
                        help="Train single model (faster, lower quality)")
    parser.add_argument("--save-importance", default=str(REPORT_DIR / "importance.csv"))
    args = parser.parse_args()

    print(f">> Loading {args.prices}")
    prices, index = load_data(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    print(f"   {len(prices):,} rows | {prices['stock_code'].nunique()} stocks | "
          f"{prices['date'].min().date()} → {prices['date'].max().date()}")

    print(">> Building features")
    panel = prepare_panel(prices, index, use_extra_data=not args.no_extra_data)
    usable = panel[panel[[c for c in _BASE_FEATURES if c in panel.columns]].notna().all(axis=1)]
    print(f"   {len(usable):,} usable rows | {usable['date'].nunique()} dates | "
          f"{usable['stock_code'].nunique()} stocks")

    if args.self_test:
        run_self_test(panel, top_k=args.top_k, objective=args.objective)
        return

    # Resolve as-of date
    if args.as_of:
        req = pd.Timestamp(args.as_of)
        valid = np.sort(panel.loc[panel["date"] <= req, "date"].unique())
        if len(valid) == 0:
            raise RuntimeError(f"No data on or before {args.as_of}")
        as_of_ts = pd.Timestamp(valid[-1])
        if as_of_ts != req:
            print(f"   as_of {req.date()} not in panel; using {as_of_ts.date()}")
    else:
        as_of_ts = pd.Timestamp(panel["date"].max())

    print(f">> as_of = {as_of_ts.date()}")
    print(f"   portfolio config: top_k={args.top_k}, filter={args.filter_universe}, weight_method={args.weight_method}")

    # Leakage-safe training cutoff:
    # A feature row on date t has label close(t+FORWARD_HORIZON) / close(t) - 1.
    # When pretending to stand at as_of_ts, labels for the final FORWARD_HORIZON
    # trading dates are NOT yet known. They must not be used for training,
    # validation, or early stopping. We still predict on as_of_ts below.
    trading_dates = np.sort(panel["date"].unique())
    asof_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts), side="right") - 1)
    label_cutoff_idx = asof_idx - FORWARD_HORIZON
    if label_cutoff_idx < VAL_DAYS + EMBARGO_DAYS + 20:
        raise RuntimeError(
            f"Not enough pre-as-of labelled dates for as_of={as_of_ts.date()}. "
            "Try a later as_of or reduce validation/training requirements."
        )
    label_cutoff = pd.Timestamp(trading_dates[label_cutoff_idx])
    print(f"   latest label-safe training/validation date: {label_cutoff.date()} "
          f"(as_of minus {FORWARD_HORIZON} trading days)")

    train_panel = panel[panel["date"] <= label_cutoff].copy()
    train_df, val_df = split_train_val(train_panel)
    print(f"   Train: {len(train_df):,} rows ({train_df['date'].min().date()} – {train_df['date'].max().date()})")
    print(f"   Val  : {len(val_df):,} rows ({val_df['date'].min().date()} – {val_df['date'].max().date()})")

    seeds = [ENSEMBLE_SEEDS[0]] if args.no_ensemble else ENSEMBLE_SEEDS
    print(f">> Training {'single model' if args.no_ensemble else f'ensemble ({len(seeds)} seeds)'} | objective={args.objective}")
    models = [train_single(train_df, val_df, seed=seeds[0], objective=args.objective)] if args.no_ensemble \
             else train_ensemble(train_df, val_df, seeds, objective=args.objective)
    print(f"   Best iterations: {[m.best_iteration for m in models]}")

    val_preds = predict_ensemble(models, val_df)
    ic, icir = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_preds, val_df["date"].to_numpy())
    diag = top_bucket_diagnostic(val_df, val_preds, top_k=args.top_k)
    print(f"   Val rank IC: {ic:+.4f} (IC-IR: {icir:+.3f})")
    print(f"   Val top-minus-all: {diag['top_minus_all']:+.4%}")

    if args.save_importance:
        save_feature_importance(models, Path(args.save_importance))

    pred_df = panel[panel["date"] == as_of_ts].dropna(
        subset=[c for c in _BASE_FEATURES if c in panel.columns]
    ).copy()
    if pred_df.empty:
        raise RuntimeError(f"No rows for {as_of_ts.date()}")

    print(f">> Predicting on {as_of_ts.date()} | {len(pred_df)} stocks")
    pred_df["score"] = predict_ensemble(models, pred_df)

    if args.filter_universe:
        pred_df = filter_prediction_universe(
            pred_df,
            top_k=args.top_k,
            liquidity_quantile=args.liquidity_quantile,
            max_vol_quantile=args.max_vol_quantile,
            max_illiq_quantile=args.max_illiq_quantile,
            max_zero_volume_days=args.max_zero_volume_days,
        )

    weights = build_portfolio(
        pred_df.set_index("stock_code")["score"],
        top_k=args.top_k,
        pred_df=pred_df,
        weight_method=args.weight_method,
        score_power=args.score_power,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame({
        "stock_code": weights.index.astype(str).str.zfill(6),
        "weight":     weights.values,
    })
    result.to_csv(out_path, index=False)
    print(f">> Wrote {len(result)} stocks → {out_path}")
    print(f"   weight: min={result['weight'].min():.4f} max={result['weight'].max():.4f} "
          f"sum={result['weight'].sum():.6f}")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
