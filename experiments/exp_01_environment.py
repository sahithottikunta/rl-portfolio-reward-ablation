"""
EXP-01: Synthetic Market Environment and Baseline Portfolios
Purpose: Generate synthetic multi-asset return data using GBM with regime switching,
         and compute traditional portfolio baselines (Equal-Weight, Min-Variance,
         Max-Sharpe) as benchmarks.
Random seed: 42
"""
import os as _os
# Repo-relative root; override with the RESEARCH_ROOT environment variable.
_ROOT = _os.environ.get("RESEARCH_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json
import os

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION — Two-regime GBM
# ─────────────────────────────────────────────────────────────────────────────
N_ASSETS    = 5          # number of risky assets
N_DAYS_TRAIN = 1000      # training days  (~4 years)
N_DAYS_TEST  = 252       # test days      (~1 year)
N_DAYS_TOTAL = N_DAYS_TRAIN + N_DAYS_TEST

TRANSACTION_COST = 0.001  # 10 bps per trade (proportional to |Δw|)

# Regime 0 — Bull market
MU_BULL    = np.array([0.10, 0.12, 0.08, 0.11, 0.09]) / 252   # daily drift
SIGMA_BULL = np.array([0.15, 0.18, 0.12, 0.16, 0.14]) / np.sqrt(252)

# Regime 1 — Bear / High-volatility market
MU_BEAR    = np.array([-0.05, -0.08, -0.03, -0.06, -0.04]) / 252
SIGMA_BEAR = np.array([0.30, 0.35, 0.25, 0.32, 0.28]) / np.sqrt(252)

# Correlation structure (shared across regimes, realistic)
CORR = np.array([
    [1.00, 0.65, 0.45, 0.70, 0.55],
    [0.65, 1.00, 0.40, 0.75, 0.60],
    [0.45, 0.40, 1.00, 0.38, 0.42],
    [0.70, 0.75, 0.38, 1.00, 0.68],
    [0.55, 0.60, 0.42, 0.68, 1.00],
])

# Markov regime transition probabilities
P_BULL_TO_BEAR = 0.02   # probability of switching from bull to bear per day
P_BEAR_TO_BULL = 0.05   # probability of switching from bear to bull per day

def generate_returns(n_days, seed=SEED):
    rng = np.random.default_rng(seed)
    regime = 0   # start in bull
    regimes = np.zeros(n_days, dtype=int)
    returns = np.zeros((n_days, N_ASSETS))

    for t in range(n_days):
        regimes[t] = regime
        if regime == 0:
            mu    = MU_BULL
            sigma = SIGMA_BULL
            if rng.random() < P_BULL_TO_BEAR:
                regime = 1
        else:
            mu    = MU_BEAR
            sigma = SIGMA_BEAR
            if rng.random() < P_BEAR_TO_BULL:
                regime = 0

        # Cholesky decomposition of correlation matrix
        std_diag = np.diag(sigma)
        cov = std_diag @ CORR @ std_diag
        L   = np.linalg.cholesky(cov)
        z   = rng.standard_normal(N_ASSETS)
        returns[t] = mu + L @ z

    return returns, regimes

all_returns, all_regimes = generate_returns(N_DAYS_TOTAL)
train_returns = all_returns[:N_DAYS_TRAIN]
test_returns  = all_returns[N_DAYS_TRAIN:]
train_regimes = all_regimes[:N_DAYS_TRAIN]
test_regimes  = all_regimes[N_DAYS_TRAIN:]

print(f"Generated {N_DAYS_TOTAL} days of returns for {N_ASSETS} assets")
print(f"Train: {N_DAYS_TRAIN} days | Test: {N_DAYS_TEST} days")
print(f"Bull days (train): {(train_regimes==0).sum()}  Bear days (train): {(train_regimes==1).sum()}")
print(f"Bull days (test):  {(test_regimes==0).sum()}   Bear days (test):  {(test_regimes==1).sum()}")
print(f"Train mean returns (annualized): {train_returns.mean(0)*252}")
print(f"Train std returns  (annualized): {train_returns.std(0)*np.sqrt(252)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PORTFOLIO PERFORMANCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def portfolio_performance(weights_series, returns, tc=TRANSACTION_COST):
    """
    Given a T×N weights matrix and T×N returns matrix,
    compute cumulative return, annualized Sharpe, max drawdown.
    weights_series[t] is the portfolio at beginning of day t.
    Returns after transaction costs.
    """
    T = len(returns)
    portfolio_values = [1.0]
    daily_rets = []
    prev_w = np.zeros(len(weights_series[0]))

    for t in range(T):
        w = weights_series[t]
        # transaction cost: 0.1% of absolute change in weights
        cost = tc * np.sum(np.abs(w - prev_w))
        r_gross = np.dot(w, returns[t])
        r_net   = r_gross - cost
        daily_rets.append(r_net)
        portfolio_values.append(portfolio_values[-1] * (1 + r_net))
        prev_w = w.copy()

    daily_rets = np.array(daily_rets)
    pv = np.array(portfolio_values)

    ann_return = np.mean(daily_rets) * 252
    ann_vol    = np.std(daily_rets, ddof=1) * np.sqrt(252)
    sharpe     = ann_return / ann_vol if ann_vol > 0 else 0.0

    rolling_max = np.maximum.accumulate(pv)
    drawdowns   = (pv - rolling_max) / rolling_max
    max_dd      = drawdowns.min()

    cum_return  = pv[-1] - 1.0

    return {
        "cumulative_return": float(cum_return),
        "annualized_return": float(ann_return),
        "annualized_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "final_portfolio_value": float(pv[-1])
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. BASELINE STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

# --- (a) Equal-Weight (1/N) ---
ew_weights_test = np.full((N_DAYS_TEST, N_ASSETS), 1.0 / N_ASSETS)
ew_perf = portfolio_performance(ew_weights_test, test_returns)
print("\n--- Equal-Weight (1/N) ---")
print(ew_perf)

# --- (b) Min-Variance ---
cov_train = np.cov(train_returns.T)

def neg_sharpe(w):
    ret = np.dot(w, train_returns.mean(0)) * 252
    vol = np.sqrt(w @ cov_train @ w * 252)
    return -ret / vol if vol > 0 else 0.0

def min_var(w):
    return float(w @ cov_train @ w)

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
bounds = [(0.0, 1.0)] * N_ASSETS
w0 = np.ones(N_ASSETS) / N_ASSETS

res_mv = minimize(min_var, w0, method='SLSQP',
                  bounds=bounds, constraints=constraints,
                  options={'ftol': 1e-9, 'maxiter': 1000})
w_mv = res_mv.x
mv_weights_test = np.tile(w_mv, (N_DAYS_TEST, 1))
mv_perf = portfolio_performance(mv_weights_test, test_returns)
print("\n--- Min-Variance ---")
print("Weights:", np.round(w_mv, 4))
print(mv_perf)

# --- (c) Max-Sharpe ---
res_ms = minimize(neg_sharpe, w0, method='SLSQP',
                  bounds=bounds, constraints=constraints,
                  options={'ftol': 1e-9, 'maxiter': 1000})
w_ms = res_ms.x
ms_weights_test = np.tile(w_ms, (N_DAYS_TEST, 1))
ms_perf = portfolio_performance(ms_weights_test, test_returns)
print("\n--- Max-Sharpe (Markowitz) ---")
print("Weights:", np.round(w_ms, 4))
print(ms_perf)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE RESULTS & DATA
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(_ROOT + "/results", exist_ok=True)
os.makedirs(_ROOT + "/data",    exist_ok=True)

np.save(_ROOT + "/data/train_returns.npy", train_returns)
np.save(_ROOT + "/data/test_returns.npy",  test_returns)
np.save(_ROOT + "/data/train_regimes.npy", train_regimes)
np.save(_ROOT + "/data/test_regimes.npy",  test_regimes)

results = {
    "experiment": "EXP-01",
    "seed": SEED,
    "n_assets": N_ASSETS,
    "n_days_train": N_DAYS_TRAIN,
    "n_days_test":  N_DAYS_TEST,
    "transaction_cost": TRANSACTION_COST,
    "train_bull_days": int((train_regimes==0).sum()),
    "train_bear_days": int((train_regimes==1).sum()),
    "test_bull_days":  int((test_regimes==0).sum()),
    "test_bear_days":  int((test_regimes==1).sum()),
    "baselines": {
        "equal_weight":  ew_perf,
        "min_variance":  mv_perf,
        "max_sharpe":    ms_perf,
    },
    "min_variance_weights": w_mv.tolist(),
    "max_sharpe_weights":   w_ms.tolist(),
}

with open(_ROOT + "/results/exp_01_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nEXP-01 complete. Results saved.")
