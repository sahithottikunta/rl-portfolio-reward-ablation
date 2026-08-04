"""
EXP-05: Real Market Data Fetch & Preprocessing
===============================================
Fetches adjusted close prices for 10 liquid S&P 500 constituents via yfinance.
Assets chosen for: long history, no survivorship bias issues (all were in S&P 500
throughout the window), sector diversity, high liquidity.

Assets: AAPL, MSFT, JPM, JNJ, XOM, AMZN, PG, UNH, NVDA, HD
Period: 2015-01-01 to 2024-12-31
Train: 2015-01-01 – 2021-12-31 (~1762 trading days)
Test:  2022-01-01 – 2024-12-31 (~754 trading days)
  → Test window includes: 2022 bear (-18% S&P), 2023 bull (+24%), 2024 bull (+23%)
  → Natural regime variation without synthetic labeling

TC: 0.001 (10 bps, same as synthetic experiment for apples-to-apples)
"""
import os as _os
# Repo-relative root; override with the RESEARCH_ROOT environment variable.
_ROOT = _os.environ.get("RESEARCH_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import json, os, sys

try:
    import yfinance as yf
except ImportError:
    os.system("pip install yfinance -q")
    import yfinance as yf

RESULTS_DIR = _ROOT + "/results"
DATA_DIR    = _ROOT + "/data"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── Asset universe ─────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "AMZN", "PG", "UNH", "NVDA", "HD"]
N_ASSETS = len(TICKERS)

TRAIN_START = "2015-01-01"
TRAIN_END   = "2021-12-31"
TEST_START  = "2022-01-01"
TEST_END    = "2024-12-31"

WINDOW = 20   # lookback window (same as synthetic)
TC     = 0.001  # 10 bps

print(f"Fetching {N_ASSETS} tickers: {TICKERS}")
print(f"Train: {TRAIN_START} to {TRAIN_END}")
print(f"Test:  {TEST_START}  to {TEST_END}")

# ── Download adjusted close prices ─────────────────────────────────────────────
raw = yf.download(
    TICKERS,
    start=TRAIN_START,
    end=TEST_END,
    auto_adjust=True,   # dividend/split adjusted
    progress=False,
)["Close"]

# Drop any days where ALL tickers are NaN (market holidays)
raw = raw.dropna(how="all")

# Forward-fill any remaining NaN (rare trading halts), then drop leading NaN rows
raw = raw.ffill().dropna()

print(f"Total trading days downloaded: {len(raw)}")
print(f"Columns: {list(raw.columns)}")
print(f"Date range: {raw.index[0].date()} → {raw.index[-1].date()}")

# ── Compute daily log returns ──────────────────────────────────────────────────
log_returns = np.log(raw / raw.shift(1)).dropna()  # shape (T-1, N)

# Split on date
train_mask = (log_returns.index >= TRAIN_START) & (log_returns.index <= TRAIN_END)
test_mask  = (log_returns.index >= TEST_START)  & (log_returns.index <= TEST_END)

train_returns = log_returns[train_mask].values.astype(np.float32)  # (T_train, N)
test_returns  = log_returns[test_mask].values.astype(np.float32)   # (T_test, N)

print(f"\nTrain returns shape: {train_returns.shape}")
print(f"Test returns shape:  {test_returns.shape}")
print(f"Train mean ann. return (AAPL): {train_returns[:,0].mean()*252*100:.1f}%")
print(f"Test  mean ann. return (AAPL): {test_returns[:,0].mean()*252*100:.1f}%")

# ── NBER-aligned recession indicator for test period ─────────────────────────
# 2022: predominantly bear (S&P -18.1%); we label months by SPY drawdown
# Simple rule: month is "bear" if SPY (AAPL proxy: index 0 = AAPL, but use mean)
# Actual NBER: no official recession in 2022-2024 (closest was 2022 Q1-Q2 technical)
# We use a rolling 60-day drawdown signal as a data-driven regime label:
#   bear = rolling 60-day cumulative return < -5%
spy_test = test_returns[:, 0]  # Use AAPL as SPY proxy (or average)
market_cum = np.cumprod(1 + spy_test)

# Use market cumulative return to define bear: rolling 60-day is below -5%
# More principled: use the cross-sectional mean return
mean_test_ret = test_returns.mean(axis=1)
roll_60 = pd.Series(mean_test_ret).rolling(60).sum().fillna(0).values
test_regimes_real = (roll_60 < -0.05).astype(int)  # 1 = bear

bull_days = int((test_regimes_real == 0).sum())
bear_days = int((test_regimes_real == 1).sum())
print(f"\nTest regime distribution: {bull_days} bull days, {bear_days} bear days")

# ── Save arrays ───────────────────────────────────────────────────────────────
np.save(f"{DATA_DIR}/real_train_returns.npy", train_returns)
np.save(f"{DATA_DIR}/real_test_returns.npy",  test_returns)
np.save(f"{DATA_DIR}/real_test_regimes.npy",  test_regimes_real)

# Save stats
train_ann_mean = (train_returns.mean(axis=0) * 252).tolist()
train_ann_vol  = (train_returns.std(axis=0) * np.sqrt(252)).tolist()
test_ann_mean  = (test_returns.mean(axis=0) * 252).tolist()
test_ann_vol   = (test_returns.std(axis=0) * np.sqrt(252)).tolist()

exp05_results = {
    "tickers":        TICKERS,
    "n_assets":       N_ASSETS,
    "train_days":     int(train_returns.shape[0]),
    "test_days":      int(test_returns.shape[0]),
    "window":         WINDOW,
    "tc":             TC,
    "train_start":    TRAIN_START,
    "train_end":      TRAIN_END,
    "test_start":     TEST_START,
    "test_end":       TEST_END,
    "train_ann_mean": train_ann_mean,
    "train_ann_vol":  train_ann_vol,
    "test_ann_mean":  test_ann_mean,
    "test_ann_vol":   test_ann_vol,
    "test_bull_days": bull_days,
    "test_bear_days": bear_days,
}

with open(f"{RESULTS_DIR}/exp_05_results.json", "w") as f:
    json.dump(exp05_results, f, indent=2)

print("\nEXP-05 complete. Results saved.")
print(f"  Train: {train_returns.shape[0]} days, Test: {test_returns.shape[0]} days")
