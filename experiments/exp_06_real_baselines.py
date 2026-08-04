"""
EXP-06: Classical Portfolio Baselines on Real S&P 500 Data
===========================================================
Computes Equal-Weight, Min-Variance, and Max-Sharpe (MVO) portfolios
on the real market data fetched in EXP-05.
All optimization uses training-period statistics; evaluation on test period.
"""
import os as _os
# Repo-relative root; override with the RESEARCH_ROOT environment variable.
_ROOT = _os.environ.get("RESEARCH_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import json, os
from scipy.optimize import minimize

RESULTS_DIR = _ROOT + "/results"
DATA_DIR    = _ROOT + "/data"

# ── Load data ─────────────────────────────────────────────────────────────────
train_returns = np.load(f"{DATA_DIR}/real_train_returns.npy")
test_returns  = np.load(f"{DATA_DIR}/real_test_returns.npy")

with open(f"{RESULTS_DIR}/exp_05_results.json") as f:
    meta = json.load(f)

N   = meta["n_assets"]
TC  = meta["tc"]

print(f"Train: {train_returns.shape}, Test: {test_returns.shape}")
print(f"Assets: {meta['tickers']}")

# ── Portfolio evaluation function ─────────────────────────────────────────────
def evaluate_portfolio(weights, returns, tc=TC):
    """
    Evaluate a static-weight portfolio on a returns matrix.
    weights: (N,) array, sums to 1
    returns: (T, N) log-return array
    Returns dict of performance metrics.
    """
    w = np.array(weights)
    T = returns.shape[0]

    daily_rets = []
    w_prev = w.copy()
    for t in range(T):
        # Gross return
        gross = w.dot(returns[t])
        # TC on rebalancing (rebalance back to fixed w each day)
        turnover_t = np.sum(np.abs(w - w_prev))
        net = gross - tc * turnover_t
        daily_rets.append(net)
        # After market move, weights drift; rebalance back to w
        w_prev = w.copy()

    daily_rets = np.array(daily_rets)
    ann_ret    = daily_rets.mean() * 252
    ann_vol    = daily_rets.std()  * np.sqrt(252)
    sharpe     = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum_ret    = float(np.prod(1 + daily_rets) - 1)
    # Max drawdown
    pv = np.cumprod(1 + daily_rets)
    roll_max = np.maximum.accumulate(pv)
    dd = (pv - roll_max) / roll_max
    max_dd = float(dd.min())

    return {
        "weights":              w.tolist(),
        "sharpe_ratio":         float(sharpe),
        "annualized_return":    float(ann_ret),
        "annualized_volatility":float(ann_vol),
        "cumulative_return":    float(cum_ret),
        "max_drawdown":         float(max_dd),
        "daily_returns":        daily_rets.tolist(),
    }

# ── 1. Equal-Weight ───────────────────────────────────────────────────────────
ew_weights = np.ones(N) / N
ew_result  = evaluate_portfolio(ew_weights, test_returns)
print(f"\nEqual-Weight: Sharpe={ew_result['sharpe_ratio']:.3f}, "
      f"CumRet={ew_result['cumulative_return']:.3f}, "
      f"MaxDD={ew_result['max_drawdown']:.3f}")

# ── 2. Min-Variance ───────────────────────────────────────────────────────────
mu_train  = train_returns.mean(axis=0)
cov_train = np.cov(train_returns.T) * 252

def neg_sharpe(w, mu, cov, rf=0.0):
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    return -(ret - rf) / (vol + 1e-9)

def portfolio_vol(w, cov):
    return np.sqrt(w @ cov @ w)

constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
bounds      = [(0, 1)] * N
w0          = np.ones(N) / N

mv_res = minimize(
    portfolio_vol, w0, args=(cov_train,),
    method="SLSQP", bounds=bounds, constraints=constraints,
    options={"ftol": 1e-9, "maxiter": 1000}
)
mv_weights = mv_res.x
mv_result  = evaluate_portfolio(mv_weights, test_returns)
print(f"Min-Variance: Sharpe={mv_result['sharpe_ratio']:.3f}, "
      f"CumRet={mv_result['cumulative_return']:.3f}, "
      f"MaxDD={mv_result['max_drawdown']:.3f}")

# ── 3. Max-Sharpe (MVO) ───────────────────────────────────────────────────────
ms_res = minimize(
    neg_sharpe, w0, args=(mu_train * 252, cov_train),
    method="SLSQP", bounds=bounds, constraints=constraints,
    options={"ftol": 1e-9, "maxiter": 1000}
)
ms_weights = ms_res.x
ms_result  = evaluate_portfolio(ms_weights, test_returns)
print(f"Max-Sharpe:   Sharpe={ms_result['sharpe_ratio']:.3f}, "
      f"CumRet={ms_result['cumulative_return']:.3f}, "
      f"MaxDD={ms_result['max_drawdown']:.3f}")

print(f"\nMax-Sharpe weights: { {meta['tickers'][i]: f'{ms_weights[i]:.3f}' for i in range(N)} }")

# ── Save ──────────────────────────────────────────────────────────────────────
exp06_results = {
    "baselines": {
        "equal_weight": {k: v for k, v in ew_result.items() if k != "daily_returns"},
        "min_variance":  {k: v for k, v in mv_result.items() if k != "daily_returns"},
        "max_sharpe":    {k: v for k, v in ms_result.items() if k != "daily_returns"},
    },
    "daily_returns": {
        "equal_weight": ew_result["daily_returns"],
        "min_variance":  mv_result["daily_returns"],
        "max_sharpe":    ms_result["daily_returns"],
    }
}

with open(f"{RESULTS_DIR}/exp_06_results.json", "w") as f:
    json.dump(exp06_results, f, indent=2)

print("\nEXP-06 complete.")
