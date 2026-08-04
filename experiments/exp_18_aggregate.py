"""
EXP-18 Aggregator: Combine EXP-17 (seeds 42-116) + EXP-18 (seeds 117-191)
to produce n=150 final comparison for both configs.

Target : lambda=0.0005, eta0=0.020
Baseline: lambda=0.001,  eta0=0.010
"""
import os as _os
# Repo-relative root; override with the RESEARCH_ROOT environment variable.
_ROOT = _os.environ.get("RESEARCH_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json, glob, os
import numpy as np
from scipy import stats

RESULTS_DIR = _ROOT + "/results"

def load_sharpes_from_batches(prefix):
    """Load and concatenate sharpes from all batch files matching prefix."""
    files = sorted(glob.glob(f"{RESULTS_DIR}/{prefix}_*.json"))
    sharpes = []
    seeds_seen = set()
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        for s, sr in zip(d["seeds"], d["sharpes"]):
            if s not in seeds_seen:
                seeds_seen.add(s)
                sharpes.append(sr)
    return np.array(sharpes), sorted(seeds_seen)

def bootstrap_power(a, b, n_boot=10000, alpha=0.05, rng_seed=0):
    """Bootstrap estimate of achieved power via permutation-style resampling."""
    rng = np.random.default_rng(rng_seed)
    observed_t = abs(stats.ttest_ind(a, b, equal_var=False).statistic)
    count = 0
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        t = abs(stats.ttest_ind(sa, sb, equal_var=False).statistic)
        if t >= stats.t.ppf(1 - alpha/2, df=len(a)+len(b)-2):
            count += 1
    return count / n_boot

print("=" * 60)
print("EXP-18 AGGREGATION: n=150 per group")
print("=" * 60)

# ── Target cell: EXP-17 batches + EXP-18 batches ──────────────
tgt17, tgt17_seeds = load_sharpes_from_batches("exp_17_tc0.0005_eta0.020")
tgt18, tgt18_seeds = load_sharpes_from_batches("exp_18_tc0.0005_eta0.020")
target_all = np.concatenate([tgt17, tgt18])
target_seeds = tgt17_seeds + tgt18_seeds
print(f"\nTarget (λ=0.0005, η₀=0.020):")
print(f"  EXP-17 batches: {len(tgt17)} seeds ({tgt17_seeds[0]}–{tgt17_seeds[-1]})")
print(f"  EXP-18 batches: {len(tgt18)} seeds ({tgt18_seeds[0] if tgt18_seeds else 'none'}–{tgt18_seeds[-1] if tgt18_seeds else 'none'})")
print(f"  Total n = {len(target_all)}")
print(f"  mean={target_all.mean():.4f}  std={target_all.std(ddof=1):.4f}")
print(f"  median={np.median(target_all):.4f}")

# ── Baseline: EXP-16 (seeds 42-91) + EXP-17 batches + EXP-18 batches ──
with open(f"{RESULTS_DIR}/exp_16_results.json") as f:
    e16 = json.load(f)
base16 = np.array(e16["baseline"]["sharpes"])          # 50 seeds (42-91)
base16_seeds = list(range(42, 92))

base17, base17_seeds = load_sharpes_from_batches("exp_17_tc0.001_eta0.010")
base18, base18_seeds = load_sharpes_from_batches("exp_18_tc0.001_eta0.010")
baseline_all = np.concatenate([base16, base17, base18])
baseline_seeds = base16_seeds + base17_seeds + base18_seeds

print(f"\nBaseline (λ=0.001, η₀=0.010):")
print(f"  EXP-16 (seeds 42-91): {len(base16)} seeds")
print(f"  EXP-17 batches: {len(base17)} seeds")
print(f"  EXP-18 batches: {len(base18)} seeds ({base18_seeds[0] if base18_seeds else 'none'}–{base18_seeds[-1] if base18_seeds else 'none'})")
print(f"  Total n = {len(baseline_all)}")
print(f"  mean={baseline_all.mean():.4f}  std={baseline_all.std(ddof=1):.4f}")
print(f"  median={np.median(baseline_all):.4f}")

# ── Guard: need exactly n=150 for each ────────────────────────
if len(target_all) < 150 or len(baseline_all) < 150:
    print(f"\n[WARNING] Not at n=150 yet — target={len(target_all)}, baseline={len(baseline_all)}")
    print("         Run remaining EXP-18 batches and re-aggregate.")

# Trim to 150 if we somehow have more (shouldn't happen with seeds 42-191)
target_all   = target_all[:150]
baseline_all = baseline_all[:150]

# ── Statistical tests ──────────────────────────────────────────
print("\n" + "─" * 60)
print("STATISTICAL TESTS  (n=150 per group)")
print("─" * 60)

t_stat, p_val = stats.ttest_ind(target_all, baseline_all, equal_var=False)
n1, n2 = len(target_all), len(baseline_all)
pooled_std = np.sqrt((target_all.std(ddof=1)**2 + baseline_all.std(ddof=1)**2) / 2)
cohens_d = (target_all.mean() - baseline_all.mean()) / pooled_std

lev_f, lev_p = stats.levene(target_all, baseline_all)
ci_diff = stats.t.interval(0.95, df=n1+n2-2,
                            loc=target_all.mean() - baseline_all.mean(),
                            scale=np.sqrt(target_all.std(ddof=1)**2/n1 + baseline_all.std(ddof=1)**2/n2))

print(f"Welch t-test : t={t_stat:.4f}, p={p_val:.4f}  ({'SIGNIFICANT p<0.05' if p_val < 0.05 else 'NOT significant'})")
print(f"Cohen d      : {cohens_d:+.4f}")
print(f"95% CI (diff): [{ci_diff[0]:.4f}, {ci_diff[1]:.4f}]")
print(f"Levene test  : F={lev_f:.4f}, p={lev_p:.4f}")

print("\nBootstrap power estimate (n_boot=10,000) ...")
power = bootstrap_power(target_all, baseline_all)
print(f"Bootstrap power: {power*100:.1f}%")

# ── 95% CIs per group ─────────────────────────────────────────
se_t = target_all.std(ddof=1) / np.sqrt(n1)
se_b = baseline_all.std(ddof=1) / np.sqrt(n2)
ci_t = stats.t.interval(0.95, df=n1-1, loc=target_all.mean(), scale=se_t)
ci_b = stats.t.interval(0.95, df=n2-1, loc=baseline_all.mean(), scale=se_b)

print(f"\nTarget   95% CI: [{ci_t[0]:.4f}, {ci_t[1]:.4f}]")
print(f"Baseline 95% CI: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")

# ── Save results ───────────────────────────────────────────────
results = {
    "experiment": "EXP-18",
    "description": "n=150 confirmatory run for mean-optimal cell vs baseline",
    "target": {
        "tc": 0.0005, "eta0": 0.020,
        "n": int(n1),
        "seeds_range": [42, 191],
        "mean": float(target_all.mean()),
        "std": float(target_all.std(ddof=1)),
        "median": float(np.median(target_all)),
        "ci_95": [float(ci_t[0]), float(ci_t[1])],
        "sharpes": target_all.tolist(),
    },
    "baseline": {
        "tc": 0.001, "eta0": 0.010,
        "n": int(n2),
        "seeds_range": [42, 191],
        "mean": float(baseline_all.mean()),
        "std": float(baseline_all.std(ddof=1)),
        "median": float(np.median(baseline_all)),
        "ci_95": [float(ci_b[0]), float(ci_b[1])],
        "sharpes": baseline_all.tolist(),
    },
    "welch_t": float(t_stat),
    "p_value": float(p_val),
    "cohens_d": float(cohens_d),
    "ci_diff_95": [float(ci_diff[0]), float(ci_diff[1])],
    "levene_f": float(lev_f),
    "levene_p": float(lev_p),
    "bootstrap_power": float(power),
    "significant": bool(p_val < 0.05),
}

outpath = f"{RESULTS_DIR}/exp_18_results.json"
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {outpath}")
print("=" * 60)
