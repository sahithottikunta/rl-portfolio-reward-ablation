"""
EXP-04: Figure Generation (v2 — improved label placement)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os

FIGURES_DIR = "/home/user/workspace/research_output/figures"
RESULTS_DIR = "/home/user/workspace/research_output/results"
os.makedirs(FIGURES_DIR, exist_ok=True)

with open(f"{RESULTS_DIR}/exp_01_results.json") as f: r1 = json.load(f)
with open(f"{RESULTS_DIR}/exp_02_results.json") as f: r2 = json.load(f)
with open(f"{RESULTS_DIR}/exp_03_results.json") as f: r3 = json.load(f)

LABEL_MAP = {
    'log_return':    'LR (Baseline)',
    'dsr':           'DSR',
    'log_return_tc': 'LR+TC',
    'dsr_tc':        'DSR+TC\n(Proposed)',
    'equal_weight':  'Equal-Weight',
}
COLORS = {
    'log_return':    '#4C72B0',
    'dsr':           '#55A868',
    'log_return_tc': '#C44E52',
    'dsr_tc':        '#DD8452',
    'equal_weight':  '#8172B2',
}
REWARD_TYPES = ['log_return', 'dsr', 'log_return_tc', 'dsr_tc']

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1: Sharpe ratio comparison (mean ± std) — improved label placement
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))

all_strategies = list(REWARD_TYPES) + ['equal_weight', 'min_variance', 'max_sharpe']
sharpe_means, sharpe_stds = [], []

for s in all_strategies:
    if s in REWARD_TYPES:
        sharpe_means.append(r2['results'][s]['sharpe_mean'])
        sharpe_stds.append(r2['results'][s]['sharpe_std'])
    elif s == 'equal_weight':
        sharpe_means.append(r1['baselines']['equal_weight']['sharpe_ratio'])
        sharpe_stds.append(0.0)
    elif s == 'min_variance':
        sharpe_means.append(r1['baselines']['min_variance']['sharpe_ratio'])
        sharpe_stds.append(0.0)
    elif s == 'max_sharpe':
        sharpe_means.append(r1['baselines']['max_sharpe']['sharpe_ratio'])
        sharpe_stds.append(0.0)

labels_plot = [LABEL_MAP.get(s, s) for s in all_strategies]
labels_plot[-2] = 'Min-Variance'
labels_plot[-1] = 'Max-Sharpe\n(MVO)'
colors_plot = [COLORS.get(s, '#999999') for s in all_strategies]

x = np.arange(len(labels_plot))
bars = ax.bar(x, sharpe_means, yerr=sharpe_stds, capsize=5,
              color=colors_plot, alpha=0.85, edgecolor='black', linewidth=0.7,
              error_kw={'linewidth': 1.5})

ax.set_xticks(x)
ax.set_xticklabels(labels_plot, fontsize=10)
ax.set_ylabel('Annualized Sharpe Ratio', fontsize=11)
ax.set_title('Out-of-Sample Sharpe Ratio by Strategy (mean \u00b1 std, 3 runs)', fontsize=12)

# Place labels above the error bar cap rather than just above the bar
for bar, m, s in zip(bars, sharpe_means, sharpe_stds):
    label_y = m + s + 0.07
    ax.text(bar.get_x() + bar.get_width()/2, label_y,
            f'{m:.3f}', ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))

ax.set_ylim(0, max(sharpe_means) + max(sharpe_stds) + 0.35)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_01_sharpe_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("fig_01 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2: Cumulative wealth curves
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

daily_series = r3['daily_return_series']
test_regimes = np.array(r3['test_regimes'])

line_styles = {'log_return': '-', 'dsr': '--', 'log_return_tc': '-.', 'dsr_tc': '-', 'equal_weight': ':'}
line_widths  = {'log_return': 1.5, 'dsr': 1.5, 'log_return_tc': 1.5, 'dsr_tc': 2.5, 'equal_weight': 1.5}

for rtype in REWARD_TYPES + ['equal_weight']:
    if rtype in daily_series:
        dr = np.array(daily_series[rtype])
        pv = np.cumprod(1 + dr)
        t = np.arange(len(pv))
        ax.plot(t, pv, label=LABEL_MAP.get(rtype, rtype),
                color=COLORS.get(rtype, '#888888'),
                linewidth=line_widths.get(rtype, 1.5),
                linestyle=line_styles.get(rtype, '-'))

in_bear, bear_start = False, 0
T = len(test_regimes)
for i, reg in enumerate(test_regimes):
    if reg == 1 and not in_bear: bear_start = i; in_bear = True
    elif reg == 0 and in_bear:
        ax.axvspan(bear_start, i, alpha=0.10, color='red')
        in_bear = False
if in_bear: ax.axvspan(bear_start, T, alpha=0.10, color='red')

ax.set_xlabel('Trading Days (Test Period)', fontsize=11)
ax.set_ylabel('Portfolio Value (Initial = 1.0)', fontsize=11)
ax.set_title('Cumulative Wealth Curves — Test Period (252 days)', fontsize=12)
ax.grid(alpha=0.25)

bear_patch = mpatches.Patch(color='red', alpha=0.25, label='Bear Market')
handles, lbls = ax.get_legend_handles_labels()
ax.legend(handles + [bear_patch], lbls + ['Bear Market'], fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_02_cumulative_wealth.png", dpi=150, bbox_inches='tight')
plt.close()
print("fig_02 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3: Regime-split Sharpe
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax_idx, (regime_name, regime_key) in enumerate([('Bull Market', 'bull'), ('Bear Market', 'bear')]):
    ax = axes[ax_idx]
    vals   = [r3['regime_results'][rt][regime_key]['sharpe_ratio'] for rt in REWARD_TYPES]
    colors = [COLORS[rt] for rt in REWARD_TYPES]
    lbls   = [LABEL_MAP[rt].replace('\n', ' ') for rt in REWARD_TYPES]
    x = np.arange(len(lbls))
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(lbls, rotation=12, ha='right', fontsize=9)
    ax.set_ylabel('Sharpe Ratio', fontsize=10)
    ax.set_title(f'{regime_name} Sharpe Ratio', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f'{v:.3f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, max(vals)*1.30)
plt.suptitle('Regime-Conditional Sharpe Ratios', fontsize=13)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_03_regime_sharpe.png", dpi=150, bbox_inches='tight')
plt.close()
print("fig_03 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4: Portfolio Turnover
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
tover = r3['turnover']
lbls  = [LABEL_MAP[rt].replace('\n', ' ') for rt in REWARD_TYPES]
vals  = [tover[rt] for rt in REWARD_TYPES]
colors = [COLORS[rt] for rt in REWARD_TYPES]
x = np.arange(len(lbls))
bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels(lbls, rotation=12, ha='right', fontsize=10)
ax.set_ylabel('Mean Daily Turnover', fontsize=10)
ax.set_title('Portfolio Turnover by Reward Function', fontsize=11)
ax.grid(axis='y', alpha=0.3)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f'{v:.4f}',
            ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_04_turnover.png", dpi=150, bbox_inches='tight')
plt.close()
print("fig_04 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5: Training stability box plots
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
sharpe_samples = r3['sharpe_samples']
data = [sharpe_samples[rt] for rt in REWARD_TYPES]
bp = ax.boxplot(data, patch_artist=True, notch=False, widths=0.5)
for patch, rt in zip(bp['boxes'], REWARD_TYPES):
    patch.set_facecolor(COLORS[rt]); patch.set_alpha(0.7)
ax.set_xticklabels([LABEL_MAP[rt].replace('\n', ' ') for rt in REWARD_TYPES],
                    rotation=12, ha='right', fontsize=10)
ax.set_ylabel('Annualized Sharpe Ratio', fontsize=10)
ax.set_title('Sharpe Ratio Distribution Across Seeds (3 runs)', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_05_sharpe_stability.png", dpi=150, bbox_inches='tight')
plt.close()
print("fig_05 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6: Summary performance table
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))
ax.axis('off')

strategies_all = REWARD_TYPES + ['equal_weight', 'min_variance', 'max_sharpe']
row_labels_all = [LABEL_MAP.get(s, s).replace('\n', ' ') for s in strategies_all]
row_labels_all[-2] = 'Min-Variance'
row_labels_all[-1] = 'Max-Sharpe (MVO)'

def get_perf(s):
    if s in REWARD_TYPES:
        res = r2['results'][s]
        reg = r3['regime_results'][s]
        to  = r3['turnover'][s]
        return [f"{res['sharpe_mean']:.3f}\u00b1{res['sharpe_std']:.3f}",
                f"{res['cumret_mean']:.3f}",
                f"{res['vol_mean']:.3f}",
                f"{res['maxdd_mean']:.3f}",
                f"{reg['bull']['sharpe_ratio']:.3f}",
                f"{reg['bear']['sharpe_ratio']:.3f}",
                f"{to:.4f}"]
    elif s == 'equal_weight':
        b = r1['baselines']['equal_weight']
        return [f"{b['sharpe_ratio']:.3f}", f"{b['cumulative_return']:.3f}",
                f"{b['annualized_volatility']:.3f}", f"{b['max_drawdown']:.3f}", '-', '-', '-']
    elif s == 'min_variance':
        b = r1['baselines']['min_variance']
        return [f"{b['sharpe_ratio']:.3f}", f"{b['cumulative_return']:.3f}",
                f"{b['annualized_volatility']:.3f}", f"{b['max_drawdown']:.3f}", '-', '-', '-']
    elif s == 'max_sharpe':
        b = r1['baselines']['max_sharpe']
        return [f"{b['sharpe_ratio']:.3f}", f"{b['cumulative_return']:.3f}",
                f"{b['annualized_volatility']:.3f}", f"{b['max_drawdown']:.3f}", '-', '-', '-']

col_headers = ['Sharpe (mean\u00b1std)', 'Cum. Ret.', 'Ann. Vol', 'Max DD', 'Bull SR', 'Bear SR', 'Turnover']
table_data  = [get_perf(s) for s in strategies_all]

tbl = ax.table(cellText=table_data, rowLabels=row_labels_all, colLabels=col_headers,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.3, 1.8)

best_row_idx = np.argmin([float(row[0].split('\u00b1')[1]) for row in table_data[:4]])
for j in range(len(col_headers)):
    tbl[(best_row_idx+1, j)].set_facecolor('#FFFACD')

ax.set_title('Performance Summary — All Strategies (Test Period)', fontsize=11, pad=18)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_06_summary_table.png", dpi=150, bbox_inches='tight')
plt.close()
print("fig_06 saved")

print("\nAll figures regenerated.")
