"""
EXP-19 Data Preparation: Download 2025-01-01 to 2026-04-04 S&P 500 data.
Saves:
  data/exp19_test_returns.npy    — (T_new, 10) daily log returns
  data/exp19_test_prices.npy     — (T_new, 10) adjusted close prices
  data/exp19_test_dates.npy      — (T_new,) string dates
  data/exp19_metadata.json       — date range, regime stats, VIX proxy
"""
import numpy as np
import json
import yfinance as yf
from datetime import datetime

TICKERS = ['AAPL', 'MSFT', 'JPM', 'JNJ', 'XOM', 'AMZN', 'PG', 'UNH', 'NVDA', 'HD']
DATA_DIR = '/home/user/workspace/research_output/data'

START = '2025-01-01'
END   = '2026-04-04'

print(f"Downloading {len(TICKERS)} tickers: {START} to {END}")
df = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)['Close']
df = df[TICKERS]  # ensure consistent column order

print(f"Downloaded {len(df)} trading days  ({df.index[0].date()} to {df.index[-1].date()})")
print(f"NaN counts: {df.isna().sum().to_dict()}")

# Forward-fill any NaNs then drop remaining
df = df.ffill().dropna()
print(f"After cleaning: {len(df)} days")

# Daily simple returns (consistent with existing data)
prices = df.values.astype(np.float32)
returns = np.diff(prices, axis=0) / prices[:-1]   # (T-1, 10)
dates_all = [str(d.date()) for d in df.index]
dates = dates_all[1:]  # aligned with returns

print(f"Returns shape: {returns.shape}")
print(f"Date range: {dates[0]} to {dates[-1]}")

# ── Regime detection: bull vs bear using 60-day rolling return of equal-weight ──
ew_cumret = (1 + returns.mean(axis=1)).cumprod() - 1
window = 60
regimes = np.zeros(len(returns), dtype=np.int32)  # 0=bear, 1=bull
for t in range(len(returns)):
    start_t = max(0, t - window + 1)
    seg = returns[start_t:t+1].mean(axis=1)
    annualized = seg.mean() * 252
    regimes[t] = 1 if annualized > 0 else 0

bull_days = int(regimes.sum())
bear_days = int((regimes == 0).sum())
print(f"Regimes: bull={bull_days} days, bear={bear_days} days")

# ── Per-asset annualised stats ──
ann_mean = float(returns.mean(axis=0).mean() * 252)
ann_vol  = float(returns.std(axis=0).mean() * np.sqrt(252))
print(f"Ann. mean return (EW): {ann_mean:.3f}")
print(f"Ann. volatility  (EW): {ann_vol:.3f}")

# ── Save ──
np.save(f'{DATA_DIR}/exp19_test_returns.npy', returns)
np.save(f'{DATA_DIR}/exp19_test_prices.npy', prices[1:])
np.save(f'{DATA_DIR}/exp19_test_dates.npy', np.array(dates))
np.save(f'{DATA_DIR}/exp19_test_regimes.npy', regimes)

# Equal-weight benchmark Sharpe on this period
ew_dr = returns.mean(axis=1)
ew_sr = float(ew_dr.mean() * 252 / (ew_dr.std() * np.sqrt(252) + 1e-9))

metadata = {
    'tickers': TICKERS,
    'n_assets': len(TICKERS),
    'test_start': dates[0],
    'test_end': dates[-1],
    'test_days': len(returns),
    'bull_days': bull_days,
    'bear_days': bear_days,
    'ann_mean_ew': ann_mean,
    'ann_vol_ew': ann_vol,
    'ew_sharpe': ew_sr,
    'window': 20,
    'download_date': str(datetime.now().date()),
}
with open(f'{DATA_DIR}/exp19_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved:")
print(f"  exp19_test_returns.npy  {returns.shape}")
print(f"  exp19_test_regimes.npy  {regimes.shape}")
print(f"  exp19_metadata.json")
print(f"\nEW benchmark Sharpe (2025-2026): {ew_sr:.3f}")
print("Data prep complete.")
