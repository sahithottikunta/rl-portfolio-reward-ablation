"""
EXP-14 Aggregator: combines all TC-sweep batches into n=40 per lambda,
runs Welch t-tests, Cohen d, bootstrap power, saves exp_14_results.json.
"""
import json, numpy as np, os
from scipy import stats

RESULTS_DIR = "/home/user/workspace/research_output/results"

# ── λ=0.001 already complete from EXP-13 ─────────────────────────────────────
with open(f"{RESULTS_DIR}/exp_13_results.json") as f: r13 = json.load(f)
samples_0001 = r13["aggregated_n40"]["fixed_dsr_tc"]["samples"]   # n=40, seeds 42–81
assert len(samples_0001) == 40

# Seed batches for new lambdas
BATCHES  = [(42, 51), (52, 61), (62, 71), (72, 81)]  # inclusive end
TC_STRS  = ["0.005", "0.010", "0.050"]

def load_lambda(tc_str):
    """Load and concatenate all four seed batches for one lambda."""
    sharpes = []
    for (s, e) in BATCHES:
        path = f"{RESULTS_DIR}/exp_14_tc{tc_str}_{s}_{e}.json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        with open(path) as f:
            d = json.load(f)
        sharpes.extend(d["sharpes"])
        print(f"  Loaded tc={tc_str} seeds {s}–{e}: n={len(d['sharpes'])}", flush=True)
    assert len(sharpes) == 40, f"Expected 40, got {len(sharpes)} for tc={tc_str}"
    return sharpes

print("=== Loading EXP-14 batches ===")
samples = {"0.001": samples_0001}
for tc_str in TC_STRS:
    samples[tc_str] = load_lambda(tc_str)

# ── Summary stats ─────────────────────────────────────────────────────────────
def summary(arr_list, label):
    a = np.array(arr_list)
    return {
        "label":   label,
        "n":       len(a),
        "mean":    float(a.mean()),
        "std":     float(a.std(ddof=1)),
        "median":  float(np.median(a)),
        "min":     float(a.min()),
        "max":     float(a.max()),
        "samples": arr_list,
    }

TC_LABELS = ["0.001", "0.005", "0.010", "0.050"]
stats_map = {tc: summary(samples[tc], f"DSR+TC λ={tc}") for tc in TC_LABELS}

print(f"\n{'Config':<20} {'n':>4} {'Mean':>7} {'Std':>7} {'Median':>8} {'Min':>7} {'Max':>7}")
print("-"*68)
for tc in TC_LABELS:
    s = stats_map[tc]
    print(f"DSR+TC λ={tc:<6}      {s['n']:>4} {s['mean']:>7.3f} {s['std']:>7.3f} "
          f"{s['median']:>8.3f} {s['min']:>7.3f} {s['max']:>7.3f}")

# ── Statistical tests vs λ=0.001 (baseline) ──────────────────────────────────
def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return float((np.mean(a)-np.mean(b))/(pooled+1e-9))

def welch_df(a, b):
    sa2, sb2 = np.var(a,ddof=1), np.var(b,ddof=1)
    na, nb   = len(a), len(b)
    return (sa2/na+sb2/nb)**2 / ((sa2/na)**2/(na-1)+(sb2/nb)**2/(nb-1))

def bootstrap_power(a, b, n_boot=10_000, alpha=0.05):
    rng = np.random.default_rng(42)
    return sum(
        1 for _ in range(n_boot)
        if stats.ttest_ind(rng.choice(a,size=len(a),replace=True),
                           rng.choice(b,size=len(b),replace=True),
                           equal_var=False)[1] < alpha
    ) / n_boot

print("\n=== STATISTICAL TESTS vs λ=0.001 (n=40 each) ===")
baseline = np.array(samples["0.001"])
test_results = {}
for tc_str in TC_STRS:
    a   = np.array(samples[tc_str])
    d   = cohen_d(a, baseline)
    t, p= stats.ttest_ind(a, baseline, equal_var=False)
    df  = welch_df(a, baseline)
    pwr = bootstrap_power(a, baseline)
    test_results[tc_str] = {
        "cohen_d":  round(float(d), 4),
        "t_stat":   round(float(t), 4),
        "p_value":  round(float(p), 4),
        "df_welch": round(float(df), 1),
        "power":    round(float(pwr), 4),
        "n": 40,
    }
    sig    = ("*** p<0.001" if p<0.001 else "** p<0.01" if p<0.01
              else "* p<0.05" if p<0.05 else "~ p<0.10" if p<0.10 else "n.s.")
    effect = "large" if abs(d)>0.8 else "medium" if abs(d)>0.5 else "small"
    print(f"\n  λ={tc_str} vs λ=0.001:")
    print(f"    Cohen d = {d:+.4f}  [{effect}]")
    print(f"    Welch t = {float(t):.4f},  df={float(df):.1f},  p = {float(p):.4f}  {sig}")
    print(f"    Bootstrap power = {pwr:.4f}")

# ── Also compute pairwise std comparisons (Levene test for homogeneity) ───────
print("\n=== VARIANCE TESTS (Levene) vs λ=0.001 ===")
levene_results = {}
for tc_str in TC_STRS:
    a = np.array(samples[tc_str])
    lev_stat, lev_p = stats.levene(a, baseline)
    std_diff = (a.std(ddof=1) - baseline.std(ddof=1)) / baseline.std(ddof=1) * 100
    levene_results[tc_str] = {
        "levene_stat": round(float(lev_stat), 4),
        "levene_p":    round(float(lev_p), 4),
        "std_rel_diff_pct": round(float(std_diff), 2),
    }
    sig = "* p<0.05" if lev_p < 0.05 else "n.s."
    print(f"  λ={tc_str}: std_diff={std_diff:+.1f}%  Levene F={lev_stat:.3f}  p={lev_p:.4f}  {sig}")

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    "tc_sweep":       TC_LABELS,
    "n_per_lambda":   40,
    "seed_range":     "42–81",
    "aggregated_n40": {tc: stats_map[tc] for tc in TC_LABELS},
    "statistical_tests_vs_baseline": test_results,
    "variance_tests_levene": levene_results,
}
with open(f"{RESULTS_DIR}/exp_14_results.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n=== SUMMARY ===")
for tc in TC_LABELS:
    s = stats_map[tc]
    print(f"  λ={tc}: SR {s['mean']:.3f} ± {s['std']:.3f}")
print("\nSaved → exp_14_results.json")
