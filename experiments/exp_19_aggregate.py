"""
EXP-19 Aggregator: Combine batch files, compute statistics, produce figures.
"""
import json, glob, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = "/home/user/workspace/research_output/results"
FIGURES_DIR = "/home/user/workspace/research_output/figures"
DATA_DIR    = "/home/user/workspace/research_output/data"

CONFIGS = ['dsr_tc', 'dsr_only', 'lr_tc', 'lr_only']
LABELS  = {
    'dsr_tc':   'R1: DSR+TC (baseline)',
    'dsr_only': 'R2: DSR-only',
    'lr_tc':    'R3: LR+TC',
    'lr_only':  'R4: LR-only',
}
COLORS = {
    'dsr_tc':   '#1565C0',
    'dsr_only': '#E65100',
    'lr_tc':    '#2E7D32',
    'lr_only':  '#6A1B9A',
}

def load_config(config):
    files = sorted(glob.glob(f"{RESULTS_DIR}/exp_19_{config}_*.json"))
    sharpes, bull_sharpes, bear_sharpes, cumrets, tos = [], [], [], [], []
    seeds_seen = set()
    for fp in files:
        with open(fp) as f: d = json.load(f)
        for i, s in enumerate(d['seeds']):
            if s not in seeds_seen:
                seeds_seen.add(s)
                sharpes.append(d['sharpes'][i])
                bull_sharpes.append(d['bull_sharpes'][i])
                bear_sharpes.append(d['bear_sharpes'][i])
                cumrets.append(d['cumrets'][i])
                tos.append(d['tos'][i])
    return {
        'sharpes':      np.array(sharpes),
        'bull_sharpes': np.array(bull_sharpes),
        'bear_sharpes': np.array(bear_sharpes),
        'cumrets':      np.array(cumrets),
        'tos':          np.array(tos),
        'n': len(sharpes),
        'seeds': sorted(seeds_seen),
    }

# Load metadata
with open(f"{DATA_DIR}/exp19_metadata.json") as f:
    meta19 = json.load(f)

# Load 2022-2024 DSR+TC results for direct comparison
with open(f"{RESULTS_DIR}/exp_18_results.json") as f:
    e18 = json.load(f)
old_dsr_tc = np.array(e18['baseline']['sharpes'])   # baseline IS dsr_tc with lambda=0.001

print("=" * 65)
print("EXP-19 AGGREGATION: Post-2024 Out-of-Sample (2025-01-03 to 2026-04-02)")
print("=" * 65)
print(f"Test period : {meta19['test_start']} to {meta19['test_end']}")
print(f"Days        : {meta19['test_days']} ({meta19['bull_days']} bull, {meta19['bear_days']} bear)")
print(f"EW benchmark Sharpe: {meta19['ew_sharpe']:.3f}")
print()

data = {c: load_config(c) for c in CONFIGS}

# ── Per-config summary ──────────────────────────────────────────
print(f"{'Config':<22} {'n':>4} {'Mean SR':>8} {'Std SR':>8} {'Median':>8} "
      f"{'Bull SR':>8} {'Bear SR':>8} {'Mean CR':>8}")
print("-" * 80)
for cfg in CONFIGS:
    d = data[cfg]
    sr = d['sharpes']
    bull = d['bull_sharpes']
    bear = d['bear_sharpes']
    cr = d['cumrets']
    bull_m = np.nanmean(bull); bear_m = np.nanmean(bear)
    print(f"{LABELS[cfg]:<22} {d['n']:>4} {sr.mean():>8.3f} {sr.std(ddof=1):>8.3f} "
          f"{np.median(sr):>8.3f} {bull_m:>8.3f} {bear_m:>8.3f} {cr.mean():>8.3f}")

# ── Statistical tests vs DSR+TC ─────────────────────────────────
print()
print("STATISTICAL TESTS vs R1 (DSR+TC):")
print("-" * 60)
base = data['dsr_tc']['sharpes']
for cfg in ['dsr_only', 'lr_tc', 'lr_only']:
    d = data[cfg]['sharpes']
    t, p = stats.ttest_ind(d, base, equal_var=False)
    pooled = np.sqrt((base.std(ddof=1)**2 + d.std(ddof=1)**2) / 2)
    cohen_d = (d.mean() - base.mean()) / pooled
    print(f"  {LABELS[cfg]:<22} vs DSR+TC: t={t:.3f}, p={p:.3f}, d={cohen_d:+.3f}")

# ── Comparison: 2025-2026 vs 2022-2024 for DSR+TC ───────────────
print()
print("DSR+TC: 2022-2024 (EXP-18 baseline, n=150) vs 2025-2026 (EXP-19, n=40):")
t, p = stats.ttest_ind(base, old_dsr_tc, equal_var=False)
pooled = np.sqrt((base.std(ddof=1)**2 + old_dsr_tc.std(ddof=1)**2) / 2)
cohen_d = (base.mean() - old_dsr_tc.mean()) / pooled
print(f"  2025-2026 mean={base.mean():.3f} std={base.std(ddof=1):.3f}")
print(f"  2022-2024 mean={old_dsr_tc.mean():.3f} std={old_dsr_tc.std(ddof=1):.3f}")
print(f"  t={t:.3f}, p={p:.3f}, Cohen d={cohen_d:+.3f}")

# ── 95% CIs ─────────────────────────────────────────────────────
print()
print("95% CIs (mean SR) per config:")
for cfg in CONFIGS:
    d = data[cfg]
    sr = d['sharpes']
    n = len(sr); se = sr.std(ddof=1) / np.sqrt(n)
    ci = stats.t.interval(0.95, df=n-1, loc=sr.mean(), scale=se)
    print(f"  {LABELS[cfg]:<22}: [{ci[0]:.3f}, {ci[1]:.3f}]")

# ── Save results JSON ────────────────────────────────────────────
results = {}
for cfg in CONFIGS:
    d = data[cfg]
    sr = d['sharpes']; n = len(sr)
    se = sr.std(ddof=1) / np.sqrt(n)
    ci = stats.t.interval(0.95, df=n-1, loc=sr.mean(), scale=se)
    results[cfg] = {
        'n': n,
        'mean': float(sr.mean()), 'std': float(sr.std(ddof=1)),
        'median': float(np.median(sr)),
        'ci_95': [float(ci[0]), float(ci[1])],
        'mean_cr': float(d['cumrets'].mean()),
        'mean_bull_sr': float(np.nanmean(d['bull_sharpes'])),
        'mean_bear_sr': float(np.nanmean(d['bear_sharpes'])),
        'sharpes': sr.tolist(),
        'cumrets': d['cumrets'].tolist(),
        'bull_sharpes': [float(x) for x in d['bull_sharpes']],
        'bear_sharpes': [float(x) for x in d['bear_sharpes']],
    }

# Cross-period comparison
t_cross, p_cross = stats.ttest_ind(base, old_dsr_tc, equal_var=False)
pooled_cross = np.sqrt((base.std(ddof=1)**2 + old_dsr_tc.std(ddof=1)**2) / 2)
d_cross = (base.mean() - old_dsr_tc.mean()) / pooled_cross

out = {
    'experiment': 'EXP-19',
    'description': 'Post-2024 out-of-sample evaluation (2025-2026)',
    'test_period': f"{meta19['test_start']} to {meta19['test_end']}",
    'test_days': meta19['test_days'],
    'bull_days': meta19['bull_days'],
    'bear_days': meta19['bear_days'],
    'ew_sharpe': meta19['ew_sharpe'],
    'configs': results,
    'cross_period_dsr_tc': {
        'period_2025_2026_mean': float(base.mean()),
        'period_2022_2024_mean': float(old_dsr_tc.mean()),
        'welch_t': float(t_cross), 'p_value': float(p_cross),
        'cohens_d': float(d_cross),
    },
}
with open(f"{RESULTS_DIR}/exp_19_results.json", 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved -> {RESULTS_DIR}/exp_19_results.json")

# ── Figure: Violin/box plot of Sharpe distributions ─────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Sharpe distributions (box + swarm)
ax = axes[0]
positions = [1, 2, 3, 4]
sharpe_data = [data[c]['sharpes'] for c in CONFIGS]
vp = ax.violinplot(sharpe_data, positions=positions, showmedians=True, showextrema=False)
for i, (body, cfg) in enumerate(zip(vp['bodies'], CONFIGS)):
    body.set_facecolor(COLORS[cfg])
    body.set_alpha(0.6)
vp['cmedians'].set_color('black'); vp['cmedians'].set_linewidth(2)
# Overlay individual points
for i, (cfg, pos) in enumerate(zip(CONFIGS, positions)):
    sr = data[cfg]['sharpes']
    jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(sr))
    ax.scatter([pos + j for j in jitter], sr, s=18, color=COLORS[cfg],
               alpha=0.5, zorder=3)
    ax.scatter([pos], [sr.mean()], s=80, color=COLORS[cfg],
               marker='D', zorder=5, edgecolor='black', linewidth=0.8)
# EW benchmark line
ax.axhline(meta19['ew_sharpe'], color='#B71C1C', lw=1.5, linestyle='--',
           label=f"EW benchmark SR={meta19['ew_sharpe']:.2f}")
ax.axhline(0, color='gray', lw=0.8, linestyle=':')
ax.set_xticks(positions)
ax.set_xticklabels([LABELS[c].split(' (')[0] for c in CONFIGS], fontsize=9)
ax.set_ylabel('Test Sharpe Ratio (2025-2026)', fontsize=10)
ax.set_title('Post-2024 Out-of-Sample\nSharpe Distributions (n=40)', fontsize=10.5)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# Right: Bull/bear breakdown
ax2 = axes[1]
x = np.arange(len(CONFIGS))
width = 0.35
bull_means = [np.nanmean(data[c]['bull_sharpes']) for c in CONFIGS]
bear_means = [np.nanmean(data[c]['bear_sharpes']) for c in CONFIGS]
bull_stds  = [np.nanstd(data[c]['bull_sharpes']) for c in CONFIGS]
bear_stds  = [np.nanstd(data[c]['bear_sharpes']) for c in CONFIGS]
bars_bull = ax2.bar(x - width/2, bull_means, width, yerr=bull_stds,
                    color='#43A047', alpha=0.8, label='Bull regime', capsize=4)
bars_bear = ax2.bar(x + width/2, bear_means, width, yerr=bear_stds,
                    color='#E53935', alpha=0.8, label='Bear regime', capsize=4)
ax2.axhline(0, color='gray', lw=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels([LABELS[c].split(' (')[0] for c in CONFIGS], fontsize=9)
ax2.set_ylabel('Mean Sharpe Ratio by Regime', fontsize=10)
ax2.set_title(f'Bull ({meta19["bull_days"]} days) vs Bear ({meta19["bear_days"]} days)\nRegime Breakdown', fontsize=10.5)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('EXP-19: Post-2024 Out-of-Sample Performance\n(Trained 2015-2021, Tested 2025-2026)',
             fontsize=11, fontweight='bold')
fig.tight_layout()
path20 = f'{FIGURES_DIR}/fig_20_post2024.png'
fig.savefig(path20, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved {path20}")

# ── Figure: Cross-period comparison for DSR+TC ──────────────────
fig2, ax3 = plt.subplots(figsize=(7, 4.2))
data_22_24 = old_dsr_tc
data_25_26 = base
for arr, label, color, pos in [
    (data_22_24, '2022-2024\n(EXP-18, n=150)', '#1565C0', 1),
    (data_25_26, '2025-2026\n(EXP-19, n=40)',  '#E65100', 2),
]:
    vp2 = ax3.violinplot([arr], positions=[pos], showmedians=True, showextrema=False)
    for body in vp2['bodies']:
        body.set_facecolor(color); body.set_alpha(0.6)
    vp2['cmedians'].set_color('black'); vp2['cmedians'].set_linewidth(2)
    jitter = np.random.default_rng(1).uniform(-0.08, 0.08, len(arr))
    ax3.scatter([pos + j for j in jitter], arr, s=18, color=color, alpha=0.45, zorder=3)
    ax3.scatter([pos], [arr.mean()], s=100, color=color, marker='D',
                zorder=5, edgecolor='black', linewidth=0.8,
                label=f'{label}  mean={arr.mean():.3f}±{arr.std(ddof=1):.3f}')
ax3.axhline(0, color='gray', lw=0.8, linestyle=':')
ax3.set_xticks([1, 2])
ax3.set_xticklabels(['2022-2024\n(EXP-18)', '2025-2026\n(EXP-19)'], fontsize=10)
ax3.set_ylabel('Sharpe Ratio (DSR+TC, R1)', fontsize=10)
ax3.set_title(f'DSR+TC Cross-Period Comparison\nt={t_cross:.2f}, p={p_cross:.3f}, d={d_cross:+.2f}',
              fontsize=10.5)
ax3.legend(fontsize=8.5, loc='upper right')
ax3.grid(True, alpha=0.3, axis='y')
fig2.tight_layout()
path21 = f'{FIGURES_DIR}/fig_21_cross_period.png'
fig2.savefig(path21, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"Saved {path21}")
print("\nEXP-19 aggregation complete.")
