"""
EXP-12 Aggregator: combines partial results with existing n=3 seeds,
computes statistics, saves final exp_12_results.json
"""
import json, numpy as np
from scipy import stats

RESULTS_DIR = "/home/user/workspace/research_output/results"

# Load existing n=3 seed samples
with open(f"{RESULTS_DIR}/exp_07_results.json") as f: r7  = json.load(f)
with open(f"{RESULTS_DIR}/exp_10_results.json") as f: r10 = json.load(f)

old_fixed = r7["results"]["dsr_tc"]["sharpe_samples"]
old_k30   = r10["results"]["dsr_tc_k30"]["sharpe_samples"]
old_k50   = r10["results"]["dsr_tc_k50"]["sharpe_samples"]

# Load new partial results
with open(f"{RESULTS_DIR}/exp_12_fixed_dsr_tc_partial.json") as f: p_fixed = json.load(f)
with open(f"{RESULTS_DIR}/exp_12_adaptive_k30_partial.json") as f: p_k30   = json.load(f)
with open(f"{RESULTS_DIR}/exp_12_adaptive_k50_partial.json") as f: p_k50   = json.load(f)

all_fixed = old_fixed + p_fixed["sharpes"]
all_k30   = old_k30   + p_k30["sharpes"]
all_k50   = old_k50   + p_k50["sharpes"]

print("=== Seed samples (n=10) ===")
print(f"  fixed_dsr_tc: {[round(x,3) for x in all_fixed]}")
print(f"  adaptive_k30: {[round(x,3) for x in all_k30]}")
print(f"  adaptive_k50: {[round(x,3) for x in all_k50]}")

def summary_stats(samples, label):
    arr = np.array(samples)
    return {
        "label":   label,
        "n":       len(arr),
        "mean":    float(arr.mean()),
        "std":     float(arr.std(ddof=1)),
        "median":  float(np.median(arr)),
        "min":     float(arr.min()),
        "max":     float(arr.max()),
        "samples": samples,
    }

stats_fixed = summary_stats(all_fixed, "fixed_dsr_tc")
stats_k30   = summary_stats(all_k30,   "adaptive_k30")
stats_k50   = summary_stats(all_k50,   "adaptive_k50")

print(f"\n{'Config':<18} {'Mean':>7} {'Std':>7} {'Median':>8} {'Min':>7} {'Max':>7}")
print("-"*60)
for s in [stats_fixed, stats_k30, stats_k50]:
    print(f"{s['label']:<18} {s['mean']:>7.3f} {s['std']:>7.3f} "
          f"{s['median']:>8.3f} {s['min']:>7.3f} {s['max']:>7.3f}")

# Relative changes vs fixed
mean_diff_k30 = (stats_k30["mean"] - stats_fixed["mean"]) / (abs(stats_fixed["mean"]) + 1e-9) * 100
mean_diff_k50 = (stats_k50["mean"] - stats_fixed["mean"]) / (abs(stats_fixed["mean"]) + 1e-9) * 100
std_diff_k30  = (stats_k30["std"]  - stats_fixed["std"])  / (stats_fixed["std"]  + 1e-9) * 100
std_diff_k50  = (stats_k50["std"]  - stats_fixed["std"])  / (stats_fixed["std"]  + 1e-9) * 100

print(f"\nRelative changes vs fixed DSR+TC (n=10):")
print(f"  k30: mean {mean_diff_k30:+.1f}%,  std {std_diff_k30:+.1f}%")
print(f"  k50: mean {mean_diff_k50:+.1f}%,  std {std_diff_k50:+.1f}%")

# Statistical tests
def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.var(a,ddof=1) + (nb-1)*np.var(b,ddof=1)) / (na+nb-2))
    return float((np.mean(a) - np.mean(b)) / (pooled + 1e-9))

def welch_t(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)

def power_posthoc(a, b, n_bootstrap=10_000):
    rng = np.random.default_rng(0)
    rejections = sum(
        1 for _ in range(n_bootstrap)
        if stats.ttest_ind(rng.choice(a, size=len(a), replace=True),
                           rng.choice(b, size=len(b), replace=True),
                           equal_var=False)[1] < 0.05
    )
    return rejections / n_bootstrap

# Welch-Satterthwaite df
def welch_df(a, b):
    sa2, sb2 = np.var(a,ddof=1), np.var(b,ddof=1)
    na, nb   = len(a), len(b)
    num   = (sa2/na + sb2/nb)**2
    denom = (sa2/na)**2/(na-1) + (sb2/nb)**2/(nb-1)
    return num/denom

test_results = {}
print("\n=== STATISTICAL TESTS ===")
for label, adap_samples in [("k30", all_k30), ("k50", all_k50)]:
    a = np.array(adap_samples)
    b = np.array(all_fixed)
    d    = cohen_d(adap_samples, all_fixed)
    t, p = welch_t(adap_samples, all_fixed)
    df   = welch_df(adap_samples, all_fixed)
    pwr  = power_posthoc(a, b)
    test_results[label] = {
        "cohen_d":   round(d, 4),
        "t_stat":    round(t, 4),
        "p_value":   round(p, 4),
        "df_welch":  round(df, 1),
        "power":     round(pwr, 4),
        "n":         10,
    }
    sig    = "p<0.05 *" if p < 0.05 else ("p<0.10" if p < 0.10 else "n.s.")
    effect = "large" if abs(d)>0.8 else ("medium" if abs(d)>0.5 else "small")
    print(f"\n  Adaptive {label} vs. Fixed DSR+TC (n=10 vs n=10):")
    print(f"    Cohen d = {d:+.4f}  ({effect} effect)")
    print(f"    Welch t = {t:.4f},  df={df:.1f},  p = {p:.4f}  [{sig}]")
    print(f"    Post-hoc bootstrap power = {pwr:.4f}")

# Also compute std-only Cohen d (to characterize variance reduction)
std_cohen_k30 = (stats_k30["std"] - stats_fixed["std"]) / stats_fixed["std"]
std_cohen_k50 = (stats_k50["std"] - stats_fixed["std"]) / stats_fixed["std"]
print(f"\n  Std reduction (relative): k30={std_cohen_k30:+.3f}, k50={std_cohen_k50:+.3f}")

# Save full results
exp12_out = {
    "new_seeds":  list(range(45, 52)),
    "new_sharpes": {
        "fixed_dsr_tc": p_fixed["sharpes"],
        "adaptive_k30": p_k30["sharpes"],
        "adaptive_k50": p_k50["sharpes"],
    },
    "aggregated_n10": {
        "fixed_dsr_tc": stats_fixed,
        "adaptive_k30": stats_k30,
        "adaptive_k50": stats_k50,
    },
    "relative_changes": {
        "k30_mean_pct": round(mean_diff_k30, 2),
        "k30_std_pct":  round(std_diff_k30,  2),
        "k50_mean_pct": round(mean_diff_k50, 2),
        "k50_std_pct":  round(std_diff_k50,  2),
    },
    "statistical_tests": test_results,
    "timesteps":   r10["timesteps"],
    "total_seeds": 10,
}

with open(f"{RESULTS_DIR}/exp_12_results.json", "w") as f:
    json.dump(exp12_out, f, indent=2)

print("\nEXP-12 aggregation complete. Saved to exp_12_results.json")
print(f"\nFinal n=10 Summary:")
print(f"  fixed_dsr_tc:  SR = {stats_fixed['mean']:.3f} ± {stats_fixed['std']:.3f}")
print(f"  adaptive_k30:  SR = {stats_k30['mean']:.3f} ± {stats_k30['std']:.3f}")
print(f"  adaptive_k50:  SR = {stats_k50['mean']:.3f} ± {stats_k50['std']:.3f}")
