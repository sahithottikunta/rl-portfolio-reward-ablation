"""
EXP-08: Regime Analysis on Real S&P 500 Data
=============================================
Computes regime-conditional Sharpe ratios, turnover, and statistical
significance tests for all four reward formulations on real market data.

Uses the rolling 60-day mean return threshold from EXP-05 for regime labels.
"""
import os as _os
# Repo-relative root; override with the RESEARCH_ROOT environment variable.
_ROOT = _os.environ.get("RESEARCH_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import json, os
from scipy import stats

RESULTS_DIR = _ROOT + "/results"
DATA_DIR    = _ROOT + "/data"

# ── Load data ─────────────────────────────────────────────────────────────────
test_returns   = np.load(f"{DATA_DIR}/real_test_returns.npy")
test_regimes   = np.load(f"{DATA_DIR}/real_test_regimes.npy")

with open(f"{RESULTS_DIR}/exp_05_results.json") as f:
    meta = json.load(f)
with open(f"{RESULTS_DIR}/exp_07_results.json") as f:
    exp07 = json.load(f)
with open(f"{RESULTS_DIR}/exp_06_results.json") as f:
    exp06 = json.load(f)

N   = meta["n_assets"]
TC  = meta["tc"]

REWARD_TYPES = ["log_return", "dsr", "log_return_tc", "dsr_tc"]

def sharpe_on_days(daily_rets, day_mask):
    r = np.array(daily_rets)[day_mask]
    if len(r) < 5:
        return 0.0
    mu = r.mean() * 252
    sd = r.std() * np.sqrt(252)
    return float(mu / (sd + 1e-9))

def turnover_from_daily(daily_rets_series):
    """Estimate turnover from saved seed-42 returns (approximate)."""
    return float(exp07["results"][rt]["turnover_mean"])

# ── Align regime labels with daily return length ───────────────────────────────
# Daily returns from EXP-07 may be shorter than test_returns due to window alignment
# We use the first len(dr) elements of test_regimes

regime_results = {}
turnover_real  = {}
sharpe_samples = {}

bull_mask_full = (test_regimes == 0)
bear_mask_full = (test_regimes == 1)

print(f"Test period: {int(bull_mask_full.sum())} bull, {int(bear_mask_full.sum())} bear days")

for rt in REWARD_TYPES:
    dr = np.array(exp07["daily_returns"][rt])
    L  = len(dr)
    # Trim regime to dr length
    bull_mask = bull_mask_full[:L]
    bear_mask = bear_mask_full[:L]

    bull_sr = sharpe_on_days(dr, bull_mask)
    bear_sr = sharpe_on_days(dr, bear_mask)
    to      = exp07["results"][rt]["turnover_mean"]

    regime_results[rt] = {
        "bull": {"sharpe_ratio": bull_sr, "n_days": int(bull_mask.sum())},
        "bear": {"sharpe_ratio": bear_sr, "n_days": int(bear_mask.sum())},
    }
    turnover_real[rt]  = to
    sharpe_samples[rt] = exp07["results"][rt]["sharpe_samples"]

    print(f"{rt}: Bull SR={bull_sr:.3f}, Bear SR={bear_sr:.3f}, TO={to:.4f}")

# ── Paired t-tests: DSR+TC vs others ─────────────────────────────────────────
print("\nPaired t-tests (DSR+TC vs. competitors, n=3):")
ttest_results = {}
for rt in ["log_return", "dsr", "log_return_tc"]:
    s1 = sharpe_samples["dsr_tc"]
    s2 = sharpe_samples[rt]
    t_stat, p_val = stats.ttest_rel(s1, s2)
    ttest_results[f"dsr_tc_vs_{rt}"] = {
        "t_statistic": float(t_stat),
        "p_value":     float(p_val),
    }
    print(f"  DSR+TC vs {rt}: t={t_stat:.3f}, p={p_val:.3f}")

# ── Stability comparison (std of Sharpe across seeds) ────────────────────────
print("\nStability (Sharpe std across 3 seeds):")
stab = {}
for rt in REWARD_TYPES:
    std_s = float(np.std(sharpe_samples[rt], ddof=0))
    stab[rt] = std_s
    print(f"  {rt}: {std_s:.4f}")

# ── Save seed-42 daily returns for plotting ───────────────────────────────────
# Also grab EW daily returns
ew_dr = exp06["daily_returns"]["equal_weight"]

exp08_results = {
    "regime_results":      regime_results,
    "turnover":            turnover_real,
    "sharpe_samples":      sharpe_samples,
    "ttest_results":       ttest_results,
    "stability_std":       stab,
    "daily_return_series": {
        **{rt: exp07["daily_returns"][rt] for rt in REWARD_TYPES},
        "equal_weight": ew_dr,
    },
    "test_regimes":  test_regimes.tolist(),
}

with open(f"{RESULTS_DIR}/exp_08_results.json", "w") as f:
    json.dump(exp08_results, f, indent=2)

print("\nEXP-08 complete.")
