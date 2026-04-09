"""
EXP-13 Aggregator: combines n=10 (exp_12) + 30 new seeds (exp_13) = n=40 per config.
Computes Welch t-test, Cohen d, bootstrap power, and saves exp_13_results.json.
"""
import json, numpy as np
from scipy import stats

RESULTS_DIR = "/home/user/workspace/research_output/results"

# ── Load existing n=10 samples (from EXP-12) ─────────────────────────────────
with open(f"{RESULTS_DIR}/exp_12_results.json") as f: r12 = json.load(f)

n10_fixed = r12["aggregated_n10"]["fixed_dsr_tc"]["samples"]   # seeds 42–51
n10_k30   = r12["aggregated_n10"]["adaptive_k30"]["samples"]
n10_k50   = r12["aggregated_n10"]["adaptive_k50"]["samples"]

# ── Load new batches (seeds 52–81) ────────────────────────────────────────────
def load_batches(cfg_name, ranges):
    sharpes = []
    for (s, e) in ranges:
        path = f"{RESULTS_DIR}/exp_13_{cfg_name}_{s}_{e}.json"
        with open(path) as f:
            data = json.load(f)
        sharpes.extend(data["sharpes"])
        print(f"  Loaded {cfg_name} seeds {s}–{e}: {[round(x,3) for x in data['sharpes']]}")
    return sharpes

RANGES = [(52, 61), (62, 71), (72, 81)]

print("=== Loading EXP-13 batches ===")
new_fixed = load_batches("fixed_dsr_tc", RANGES)
new_k30   = load_batches("adaptive_k30",  RANGES)
new_k50   = load_batches("adaptive_k50",  RANGES)

# ── Combine n=40 ──────────────────────────────────────────────────────────────
all_fixed = n10_fixed + new_fixed
all_k30   = n10_k30   + new_k30
all_k50   = n10_k50   + new_k50

assert len(all_fixed) == 40, f"Expected 40 fixed samples, got {len(all_fixed)}"
assert len(all_k30)   == 40, f"Expected 40 k30 samples,   got {len(all_k30)}"
assert len(all_k50)   == 40, f"Expected 40 k50 samples,   got {len(all_k50)}"

print(f"\nSeed ranges: original 42–51 (n=10) + new 52–81 (n=30) = n=40 per config")

# ── Summary stats ─────────────────────────────────────────────────────────────
def summary(samples, label):
    a = np.array(samples)
    return {
        "label":   label,
        "n":       len(a),
        "mean":    float(a.mean()),
        "std":     float(a.std(ddof=1)),
        "median":  float(np.median(a)),
        "min":     float(a.min()),
        "max":     float(a.max()),
        "samples": samples,
    }

sf = summary(all_fixed, "fixed_dsr_tc")
sk30 = summary(all_k30,   "adaptive_k30")
sk50 = summary(all_k50,   "adaptive_k50")

print(f"\n{'Config':<18} {'n':>4} {'Mean':>7} {'Std':>7} {'Median':>8} {'Min':>7} {'Max':>7}")
print("-"*65)
for s in [sf, sk30, sk50]:
    print(f"{s['label']:<18} {s['n']:>4} {s['mean']:>7.3f} {s['std']:>7.3f} "
          f"{s['median']:>8.3f} {s['min']:>7.3f} {s['max']:>7.3f}")

mean_diff_k30 = (sk30["mean"] - sf["mean"]) / (abs(sf["mean"]) + 1e-9) * 100
mean_diff_k50 = (sk50["mean"] - sf["mean"]) / (abs(sf["mean"]) + 1e-9) * 100
std_diff_k30  = (sk30["std"]  - sf["std"])  / (sf["std"]  + 1e-9) * 100
std_diff_k50  = (sk50["std"]  - sf["std"])  / (sf["std"]  + 1e-9) * 100
print(f"\nRelative changes vs fixed (n=40):")
print(f"  k30: mean {mean_diff_k30:+.1f}%,  std {std_diff_k30:+.1f}%")
print(f"  k50: mean {mean_diff_k50:+.1f}%,  std {std_diff_k50:+.1f}%")

# ── Statistical tests ─────────────────────────────────────────────────────────
def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.var(a,ddof=1) + (nb-1)*np.var(b,ddof=1)) / (na+nb-2))
    return float((np.mean(a) - np.mean(b)) / (pooled + 1e-9))

def welch_df(a, b):
    sa2, sb2 = np.var(a,ddof=1), np.var(b,ddof=1)
    na, nb   = len(a), len(b)
    num   = (sa2/na + sb2/nb)**2
    denom = (sa2/na)**2/(na-1) + (sb2/nb)**2/(nb-1)
    return float(num/denom)

def bootstrap_power(a, b, n_boot=10_000, alpha=0.05):
    rng = np.random.default_rng(0)
    reject = sum(
        1 for _ in range(n_boot)
        if stats.ttest_ind(rng.choice(a, size=len(a), replace=True),
                           rng.choice(b, size=len(b), replace=True),
                           equal_var=False)[1] < alpha
    )
    return reject / n_boot

print("\n=== STATISTICAL TESTS (n=40 vs n=40) ===")
test_results = {}
for label, adap in [("k30", np.array(all_k30)), ("k50", np.array(all_k50))]:
    b   = np.array(all_fixed)
    d   = cohen_d(adap, b)
    t, p= stats.ttest_ind(adap, b, equal_var=False)
    df  = welch_df(adap, b)
    pwr = bootstrap_power(adap, b)
    test_results[label] = {
        "cohen_d":  round(float(d), 4),
        "t_stat":   round(float(t), 4),
        "p_value":  round(float(p), 4),
        "df_welch": round(df, 1),
        "power":    round(pwr, 4),
        "n":        40,
    }
    sig    = "*** p<0.001" if p<0.001 else ("** p<0.01" if p<0.01 else
             ("* p<0.05" if p<0.05 else ("~ p<0.10" if p<0.10 else "n.s.")))
    effect = "large" if abs(d)>0.8 else ("medium" if abs(d)>0.5 else "small")
    print(f"\n  Adaptive {label} vs Fixed DSR+TC (n=40 each):")
    print(f"    Cohen d  = {d:+.4f}  [{effect} effect]")
    print(f"    Welch t  = {float(t):.4f},  df={df:.1f},  p = {float(p):.4f}  {sig}")
    print(f"    Bootstrap power = {pwr:.4f}")
    print(f"    |d|>0.8 threshold met: {'YES' if abs(d)>0.8 else 'NO'}")

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    "total_seeds":       40,
    "seed_breakdown":    "42–51 (EXP-12) + 52–81 (EXP-13)",
    "new_seeds_exp13":   list(range(52, 82)),
    "new_sharpes_exp13": {
        "fixed_dsr_tc": new_fixed,
        "adaptive_k30": new_k30,
        "adaptive_k50": new_k50,
    },
    "aggregated_n40": {
        "fixed_dsr_tc": sf,
        "adaptive_k30": sk30,
        "adaptive_k50": sk50,
    },
    "relative_changes": {
        "k30_mean_pct": round(mean_diff_k30, 2),
        "k30_std_pct":  round(std_diff_k30,  2),
        "k50_mean_pct": round(mean_diff_k50, 2),
        "k50_std_pct":  round(std_diff_k50,  2),
    },
    "statistical_tests": test_results,
}

with open(f"{RESULTS_DIR}/exp_13_results.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n=== FINAL n=40 SUMMARY ===")
print(f"  fixed_dsr_tc:  SR = {sf['mean']:.3f} ± {sf['std']:.3f}")
print(f"  adaptive_k30:  SR = {sk30['mean']:.3f} ± {sk30['std']:.3f}")
print(f"  adaptive_k50:  SR = {sk50['mean']:.3f} ± {sk50['std']:.3f}")
print(f"\nSaved → exp_13_results.json")
