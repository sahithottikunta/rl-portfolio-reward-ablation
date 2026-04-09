"""
EXP-20 Aggregator: Regime-Switching Multi-Objective (RSMO) Reward
Compares RSMO (k in {2,5,10}) vs DSR+TC and LR-only baselines on both test periods.

Baselines (from prior experiments):
  2022 period: DSR+TC SR = 0.391  (EXP-18 target config, n=150)
               LR-only SR = from exp_19_results.json R4 but using 2022 period
               We re-use the EXP-07 results for 2022 LR-only

  2025 period: DSR+TC SR = -0.115 (EXP-19 R1, n=60)
               LR-only SR =  0.523 (EXP-19 R4, n=60)

All Welch t-tests are two-tailed.
Cohen's d = (mean_A - mean_B) / pooled_std.
"""

import json, glob, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = "/home/user/workspace/research_output/results"
FIGURES_DIR = "/home/user/workspace/research_output/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load all EXP-20 batch files ────────────────────────────────────────────
def load_rsmo(k, period):
    files = sorted(glob.glob(f"{RESULTS_DIR}/exp_20_k{k}_{period}_*.json"))
    sharpes, bull_srs, bear_srs, alphas = [], [], [], []
    for f in files:
        d = json.load(open(f))
        sharpes.extend(d['sharpes'])
        bull_srs.extend(d['bull_sharpes'])
        bear_srs.extend(d['bear_sharpes'])
        alphas.append(d['mean_alpha'])
    return np.array(sharpes), np.array(bull_srs), np.array(bear_srs), float(np.mean(alphas))

rsmo = {}
for k in [2, 5, 10]:
    for period in ['2022', '2025']:
        srs, bull, bear, alpha = load_rsmo(k, period)
        rsmo[(k, period)] = {'srs': srs, 'bull': bull, 'bear': bear, 'alpha': alpha, 'n': len(srs)}

# ── Load baseline arrays from EXP-19 results (2025 period) ────────────────
with open(f"{RESULTS_DIR}/exp_19_results.json") as f:
    exp19 = json.load(f)

# EXP-19 uses configs dict with keys dsr_tc / lr_only etc.
dsr_tc_2025  = np.array(exp19['configs']['dsr_tc']['sharpes'])   # DSR+TC, 2025, n=60
lr_only_2025 = np.array(exp19['configs']['lr_only']['sharpes'])  # LR-only, 2025, n=60

# ── Load baseline arrays for 2022 period from EXP-18 ─────────────────────
with open(f"{RESULTS_DIR}/exp_18_results.json") as f:
    exp18 = json.load(f)

dsr_tc_2022 = np.array(exp18['baseline']['sharpes'])  # DSR+TC η=0.010 λ=0.001, 2022, n=150

# LR-only 2022: load from dedicated EXP-20 baseline runs (40 seeds)
def load_lronly_2022():
    files = sorted(glob.glob(f"{RESULTS_DIR}/exp_20_lronly_2022_*.json"))
    srs = []
    for f in files:
        d = json.load(open(f))
        srs.extend(d['sharpes'])
    return np.array(srs)

lr_only_2022 = load_lronly_2022()

print("=== EXP-20 RSMO Aggregation ===\n")
print(f"Baselines:")
print(f"  2022 DSR+TC  : n={len(dsr_tc_2022)}, mean={np.mean(dsr_tc_2022):.4f}, std={np.std(dsr_tc_2022, ddof=1):.4f}")
print(f"  2022 LR-only : n={len(lr_only_2022)}, mean={np.mean(lr_only_2022):.4f}, std={np.std(lr_only_2022, ddof=1):.4f}")
print(f"  2025 DSR+TC  : n={len(dsr_tc_2025)}, mean={np.mean(dsr_tc_2025):.4f}, std={np.std(dsr_tc_2025, ddof=1):.4f}")
print(f"  2025 LR-only : n={len(lr_only_2025)}, mean={np.mean(lr_only_2025):.4f}, std={np.std(lr_only_2025, ddof=1):.4f}")
print()

# ── Welch t-test helper ──────────────────────────────────────────────────
def welch(a, b, label_a, label_b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pool_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    d = (np.mean(a) - np.mean(b)) / pool_std if pool_std > 0 else 0.0
    diff = np.mean(a) - np.mean(b)
    se = np.sqrt(np.var(a, ddof=1)/len(a) + np.var(b, ddof=1)/len(b))
    ci_lo, ci_hi = diff - 1.96*se, diff + 1.96*se
    return {'t': t, 'p': p, 'd': d, 'diff': diff, 'ci': (ci_lo, ci_hi)}

# ── Main results table ────────────────────────────────────────────────────
print("=" * 100)
print(f"{'Config':<22} {'Period':<8} {'n':<6} {'Mean SR':<10} {'Std SR':<10} "
      f"{'vs DSR+TC t':<13} {'p':<8} {'d':<7} {'vs LR-only t':<13} {'p':<8} {'d':<7}")
print("=" * 100)

results_table = []

for period in ['2022', '2025']:
    bl_dsr = dsr_tc_2022 if period == '2022' else dsr_tc_2025
    bl_lr  = lr_only_2022 if period == '2022' else lr_only_2025
    for k in [2, 5, 10]:
        cell = rsmo[(k, period)]
        srs = cell['srs']
        n   = cell['n']
        m   = np.mean(srs)
        s   = np.std(srs, ddof=1)
        vs_dsr = welch(srs, bl_dsr, f'RSMO-k{k}', 'DSR+TC')
        vs_lr  = welch(srs, bl_lr,  f'RSMO-k{k}', 'LR-only')
        row = {
            'k': k, 'period': period, 'n': n,
            'mean_sr': m, 'std_sr': s,
            'vs_dsr': vs_dsr, 'vs_lr': vs_lr,
            'mean_alpha': cell['alpha'],
            'mean_bull': float(np.mean(cell['bull'])),
            'mean_bear': float(np.mean(cell['bear'])),
        }
        results_table.append(row)
        print(f"RSMO k={k:<17} {period:<8} {n:<6} {m:<10.4f} {s:<10.4f} "
              f"{vs_dsr['t']:<13.3f} {vs_dsr['p']:<8.4f} {vs_dsr['d']:<7.3f} "
              f"{vs_lr['t']:<13.3f} {vs_lr['p']:<8.4f} {vs_lr['d']:<7.3f}")

print("=" * 100)

# Print baselines for reference
for period in ['2022', '2025']:
    bl_dsr = dsr_tc_2022 if period == '2022' else dsr_tc_2025
    bl_lr  = lr_only_2022 if period == '2022' else lr_only_2025
    print(f"Baseline DSR+TC         {period:<8} {len(bl_dsr):<6} {np.mean(bl_dsr):<10.4f} {np.std(bl_dsr, ddof=1):<10.4f}")
    print(f"Baseline LR-only        {period:<8} {len(bl_lr):<6}  {np.mean(bl_lr):<10.4f} {np.std(bl_lr, ddof=1):<10.4f}")

# ── Save results JSON ──────────────────────────────────────────────────────
def arr_to_list(a):
    return [float(x) for x in a]

out = {
    'rsmo_cells': [
        {
            'k': r['k'], 'period': r['period'], 'n': r['n'],
            'mean_sr': r['mean_sr'], 'std_sr': r['std_sr'],
            'mean_alpha': r['mean_alpha'],
            'mean_bull_sr': r['mean_bull'], 'mean_bear_sr': r['mean_bear'],
            'vs_dsr_tc': {'t': r['vs_dsr']['t'], 'p': r['vs_dsr']['p'],
                          'd': r['vs_dsr']['d'], 'ci': list(r['vs_dsr']['ci'])},
            'vs_lr_only': {'t': r['vs_lr']['t'], 'p': r['vs_lr']['p'],
                           'd': r['vs_lr']['d'], 'ci': list(r['vs_lr']['ci'])},
        }
        for r in results_table
    ],
    'baselines': {
        '2022_dsr_tc': {'n': len(dsr_tc_2022), 'mean': float(np.mean(dsr_tc_2022)),
                        'std': float(np.std(dsr_tc_2022, ddof=1)),
                        'sharpes': arr_to_list(dsr_tc_2022)},
        '2022_lr_only': {'n': len(lr_only_2022), 'mean': float(np.mean(lr_only_2022)),
                         'std': float(np.std(lr_only_2022, ddof=1)),
                         'sharpes': arr_to_list(lr_only_2022)},
        '2025_dsr_tc': {'n': len(dsr_tc_2025), 'mean': float(np.mean(dsr_tc_2025)),
                        'std': float(np.std(dsr_tc_2025, ddof=1)),
                        'sharpes': arr_to_list(dsr_tc_2025)},
        '2025_lr_only': {'n': len(lr_only_2025), 'mean': float(np.mean(lr_only_2025)),
                         'std': float(np.std(lr_only_2025, ddof=1)),
                         'sharpes': arr_to_list(lr_only_2025)},
    }
}
with open(f"{RESULTS_DIR}/exp_20_results.json", 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved -> {RESULTS_DIR}/exp_20_results.json")

# ── Figure 1: Mean SR vs k for both periods ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, period, bl_dsr_arr, bl_lr_arr in [
    (ax1, '2022', dsr_tc_2022, lr_only_2022),
    (ax2, '2025', dsr_tc_2025, lr_only_2025),
]:
    ks = [2, 5, 10]
    means = [rsmo[(k, period)]['srs'].mean() for k in ks]
    stds  = [rsmo[(k, period)]['srs'].std(ddof=1) for k in ks]
    ns    = [rsmo[(k, period)]['n'] for k in ks]
    ses   = [s/np.sqrt(n) for s, n in zip(stds, ns)]

    ax.errorbar(ks, means, yerr=[1.96*se for se in ses],
                marker='o', color='steelblue', linewidth=2, markersize=7,
                capsize=5, label='RSMO (mean ± 95% CI)', zorder=3)
    ax.axhline(np.mean(bl_dsr_arr), color='crimson', linestyle='--', linewidth=1.8,
               label=f'DSR+TC (μ={np.mean(bl_dsr_arr):.3f})')
    ax.axhline(np.mean(bl_lr_arr), color='forestgreen', linestyle='-.', linewidth=1.8,
               label=f'LR-only (μ={np.mean(bl_lr_arr):.3f})')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)

    ax.set_xticks(ks)
    ax.set_xlabel('Blending Steepness $k$', fontsize=12)
    ax.set_ylabel('Annualised Sharpe Ratio', fontsize=12)
    period_label = '2022–2024 (Bear)' if period == '2022' else '2025–2026 (Bull)'
    ax.set_title(f'RSMO vs Baselines\n{period_label}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 11)

plt.tight_layout()
fig_path1 = f"{FIGURES_DIR}/fig_22_rsmo_sr_vs_k.png"
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved -> {fig_path1}")

# ── Figure 2: Bull/Bear breakdown heatmap ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, period in [(axes[0], '2022'), (axes[1], '2025')]:
    ks = [2, 5, 10]
    bull_means = [rsmo[(k, period)]['bull'].mean() for k in ks]
    bear_means = [rsmo[(k, period)]['bear'].mean() for k in ks]
    bl_dsr_bull = (np.mean(dsr_tc_2022) if period=='2022' else np.mean(dsr_tc_2025))
    bl_lr_bull  = (np.mean(lr_only_2022) if period=='2022' else np.mean(lr_only_2025))

    x = np.arange(len(ks))
    width = 0.3

    bars_bull = ax.bar(x - width/2, bull_means, width, label='Bull regime',
                       color='#2196F3', alpha=0.85, zorder=2)
    bars_bear = ax.bar(x + width/2, bear_means, width, label='Bear regime',
                       color='#F44336', alpha=0.85, zorder=2)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in ks])
    ax.set_xlabel('Blending Steepness $k$', fontsize=12)
    ax.set_ylabel('Mean Sharpe Ratio', fontsize=12)
    period_label = '2022–2024 (Bear)' if period == '2022' else '2025–2026 (Bull)'
    ax.set_title(f'Bull/Bear SR Decomposition\n{period_label}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate bars — place labels OUTSIDE bars in black for maximum contrast
    for bar in bars_bull:
        h = bar.get_height()
        # Position above positive bars, below negative bars
        if h >= 0:
            ypos, va = h + 0.05, 'bottom'
        else:
            ypos, va = h - 0.05, 'top'
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{h:.2f}',
                ha='center', va=va, fontsize=8, color='black', fontweight='bold')
    for bar in bars_bear:
        h = bar.get_height()
        if h >= 0:
            ypos, va = h + 0.05, 'bottom'
        else:
            ypos, va = h - 0.05, 'top'
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{h:.2f}',
                ha='center', va=va, fontsize=8, color='black', fontweight='bold')

plt.tight_layout()
fig_path2 = f"{FIGURES_DIR}/fig_23_rsmo_bull_bear.png"
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved -> {fig_path2}")

# ── Print summary for paper writing ──────────────────────────────────────
print("\n=== PAPER-READY SUMMARY ===\n")
print("RSMO vs DSR+TC (is RSMO better than DSR+TC?):")
for r in results_table:
    sig = "**" if r['vs_dsr']['p'] < 0.05 else ("*" if r['vs_dsr']['p'] < 0.10 else "ns")
    print(f"  k={r['k']}, {r['period']}: d={r['vs_dsr']['d']:+.3f}, "
          f"p={r['vs_dsr']['p']:.4f} [{sig}], CI=[{r['vs_dsr']['ci'][0]:.3f},{r['vs_dsr']['ci'][1]:.3f}]")

print("\nRSMO vs LR-only (is RSMO better than LR-only?):")
for r in results_table:
    sig = "**" if r['vs_lr']['p'] < 0.05 else ("*" if r['vs_lr']['p'] < 0.10 else "ns")
    print(f"  k={r['k']}, {r['period']}: d={r['vs_lr']['d']:+.3f}, "
          f"p={r['vs_lr']['p']:.4f} [{sig}], CI=[{r['vs_lr']['ci'][0]:.3f},{r['vs_lr']['ci'][1]:.3f}]")

print("\nMean alpha (fraction of LR-only weight):")
for r in results_table:
    print(f"  k={r['k']}, {r['period']}: mean_alpha={r['mean_alpha']:.4f}, "
          f"bull_SR={r['mean_bull']:+.3f}, bear_SR={r['mean_bear']:+.3f}")
