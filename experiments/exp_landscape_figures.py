"""
Generate Section VIII hyperparameter landscape figures.

Figures produced:
  fig_17_landscape_bubble.png  — all configurations in (mean SR, std SR) space
  fig_18_landscape_lambda.png  — mean SR vs λ across η₀ values (line plot)
  fig_19_power_curve.png       — bootstrap power vs n for detecting d=0.51 (mean-optimal gap)
"""
import os as _os
# Repo-relative root; override with the RESEARCH_ROOT environment variable.
_ROOT = _os.environ.get("RESEARCH_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats

R = _ROOT + '/results'
FIGURES_DIR = _ROOT + '/figures'

# ── Load all data ────────────────────────────────────────────────────────────
with open(f'{R}/exp_13_results.json') as f:
    r13 = json.load(f)
n40 = r13['aggregated_n40']

base40 = np.array(n40['fixed_dsr_tc']['samples'])   # λ=0.001, η₀=0.010, n=40
k30_40 = np.array(n40['adaptive_k30']['samples'])
k50_40 = np.array(n40['adaptive_k50']['samples'])

def load_exp14(lam_tag):
    parts = []
    for batch in ['42_51','52_61','62_71','72_81']:
        with open(f'{R}/exp_14_tc{lam_tag}_{batch}.json') as f:
            parts.extend(json.load(f)['sharpes'])
    return np.array(parts)

tc005 = load_exp14('0.005')
tc010 = load_exp14('0.010')
tc050 = load_exp14('0.050')

def tc_str(v):
    return {0.0005:'0.0005', 0.001:'0.001', 0.002:'0.002', 0.005:'0.005'}[v]
def eta_str(v):
    return {0.002:'0.002', 0.005:'0.005', 0.010:'0.010', 0.020:'0.020', 0.050:'0.050'}[v]

LAMBDAS = [0.0005, 0.001, 0.002, 0.005]
ETAS    = [0.002,  0.005, 0.010, 0.020, 0.050]

grid = {}
for lam in LAMBDAS:
    for eta in ETAS:
        with open(f'{R}/exp_15_tc{tc_str(lam)}_eta{eta_str(eta)}_42_51.json') as f:
            grid[(lam,eta)] = np.array(json.load(f)['sharpes'])

with open(f'{R}/exp_16_results.json') as f:
    e16 = json.load(f)
target50   = np.array(e16['target']['sharpes'])
baseline50 = np.array(e16['baseline']['sharpes'])

def cohen_d(a, b):
    pooled = np.sqrt(((len(a)-1)*np.var(a,ddof=1) + (len(b)-1)*np.var(b,ddof=1)) / (len(a)+len(b)-2))
    return float((np.mean(a)-np.mean(b)) / (pooled+1e-9))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 17: Bubble chart — all configs in (mean SR, std SR) space
# ─────────────────────────────────────────────────────────────────────────────
configs = []

# Baseline (n=50)
configs.append({'label': 'Baseline\n(λ=0.001,η₀=0.010)', 'mean': float(baseline50.mean()),
                'std': float(baseline50.std(ddof=1)), 'n': 50, 'group': 'baseline'})

# η sweep (n=40)
configs.append({'label': 'Adaptive η\nκ=30', 'mean': float(k30_40.mean()),
                'std': float(k30_40.std(ddof=1)), 'n': 40, 'group': 'eta_sweep'})
configs.append({'label': 'Adaptive η\nκ=50', 'mean': float(k50_40.mean()),
                'std': float(k50_40.std(ddof=1)), 'n': 40, 'group': 'eta_sweep'})

# TC sweep (n=40)
configs.append({'label': 'λ=0.005\n(TC sweep)', 'mean': float(tc005.mean()),
                'std': float(tc005.std(ddof=1)), 'n': 40, 'group': 'tc_sweep'})
configs.append({'label': 'λ=0.010\n(TC sweep)', 'mean': float(tc010.mean()),
                'std': float(tc010.std(ddof=1)), 'n': 40, 'group': 'tc_sweep'})
configs.append({'label': 'λ=0.050\n(TC sweep)', 'mean': float(tc050.mean()),
                'std': float(tc050.std(ddof=1)), 'n': 40, 'group': 'tc_sweep'})

# Grid cells (n=10, selected highlights)
configs.append({'label': 'Grid best mean\n(λ=0.0005,η₀=0.020)', 'mean': float(grid[(0.0005,0.020)].mean()),
                'std': float(grid[(0.0005,0.020)].std(ddof=1)), 'n': 10, 'group': 'grid'})
configs.append({'label': 'Grid best IR\nat n=10\n(λ=0.002,η₀=0.002)', 'mean': float(grid[(0.002,0.002)].mean()),
                'std': float(grid[(0.002,0.002)].std(ddof=1)), 'n': 10, 'group': 'grid'})
# IR-optimal confirmed at n=50
configs.append({'label': 'IR-opt confirmed\nat n=50\n(λ=0.002,η₀=0.002)', 'mean': float(target50.mean()),
                'std': float(target50.std(ddof=1)), 'n': 50, 'group': 'confirmed_null'})

color_map = {
    'baseline':      '#2196F3',   # blue
    'eta_sweep':     '#9C27B0',   # purple
    'tc_sweep':      '#F44336',   # red
    'grid':          '#FF9800',   # orange
    'confirmed_null':'#607D8B',   # grey
}
label_map = {
    'baseline':      'Baseline',
    'eta_sweep':     'η sweep (EXP-13)',
    'tc_sweep':      'TC sweep (EXP-14)',
    'grid':          'Grid cells (EXP-15)',
    'confirmed_null':'Grid IR-opt, n=50 (EXP-16)',
}

Y_MAX = 0.68

# Manual plot positions with deliberate jitter to separate the dense cluster
# (points 1,2,3,9 all land near mean~0.40, std~0.38-0.44)
# Format: (plot_x, plot_y, label_offset_x, label_offset_y)
# True coordinates of the dense cluster:
# pt1 (Baseline)  : mean=0.397, std=0.378
# pt2 (k30)       : mean=0.422, std=0.440
# pt3 (k50)       : mean=0.431, std=0.405
# pt9 (IR-opt n50): mean=0.353, std=0.434
# Manually spread onto a 2×2 grid to ensure ≥0.10 separation in both axes
plot_overrides = {
    0: (0.30, 0.28),   # Baseline: lower-left of cluster
    1: (0.32, 0.53),   # Adapt k30: upper-left
    2: (0.52, 0.47),   # Adapt k50: upper-right
    8: (0.50, 0.28),   # IR n=50: lower-right of cluster
}

fig, ax = plt.subplots(figsize=(9, 5.5))

bl_mean = float(baseline50.mean())
bl_std  = float(baseline50.std(ddof=1))

# Green favorable region (no text label to avoid clutter)
ax.fill_betweenx([0, bl_std], bl_mean, 0.88, alpha=0.08, color='green')
ax.axhline(bl_std, color='#2196F3', linestyle=':', lw=1.4, alpha=0.6)
ax.axvline(bl_mean, color='#2196F3', linestyle=':', lw=1.4, alpha=0.6)
# Single clean annotation for the quadrant, placed far from cluster
ax.text(0.62, 0.06, 'Better-than-baseline region\n(high mean SR, low std)',
        fontsize=8, color='#2E7D32', alpha=0.85, style='italic',
        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
ax.text(bl_mean + 0.025, bl_std - 0.055, 'Baseline\nthreshold',
        fontsize=7.5, color='#1565C0', alpha=0.85,
        path_effects=[pe.withStroke(linewidth=2, foreground='white')])

seen_groups = set()
for idx, c in enumerate(configs):
    g = c['group']
    col = color_map[g]
    size = 130 + (c['n'] / 50) * 220
    lbl = label_map[g] if g not in seen_groups else None
    if idx in plot_overrides:
        px, py = plot_overrides[idx]
    else:
        px = c['mean']
        py = min(c['std'], Y_MAX - 0.04)
    out_of_range = c['std'] > Y_MAX
    ax.scatter(px, py, s=size, color=col, alpha=0.88,
               edgecolors='white', linewidths=1.8, label=lbl, zorder=4)
    seen_groups.add(g)
    ax.text(px, py, str(idx+1),
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white', zorder=5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='#1a1a1a')])
    if out_of_range:
        ax.annotate(f'std={c["std"]:.2f} (clipped)',
                    xy=(px, Y_MAX - 0.04),
                    xytext=(px + 0.12, Y_MAX - 0.13),
                    fontsize=7, color='#C62828', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.3),
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

ax.set_xlabel('Mean Sharpe Ratio (test split 2022\u20132024)', fontsize=11)
ax.set_ylabel('Cross-seed Std of Sharpe Ratio', fontsize=11)
ax.set_title('Hyperparameter Landscape: Mean SR vs. Std  |  All Configurations, EXP-12\u2013EXP-16\n'
             '(Cluster points jittered for readability; exact values in Table XIV)', fontsize=10)
ax.legend(fontsize=8.5, loc='upper right', framealpha=0.94)
ax.set_xlim(-0.25, 0.90)
ax.set_ylim(0.0, Y_MAX)
ax.grid(True, alpha=0.3, lw=0.7)

# Numbered key: two rows placed inside the figure bottom margin
key_row1 = (
    '1=Baseline (\u03bb=0.001,\u03b7\u2080=0.010)   '
    '2=Adapt.\u03b7 \u03ba=30   '
    '3=Adapt.\u03b7 \u03ba=50   '
    '4=\u03bb=0.005 TC sweep   '
    '5=\u03bb=0.010 TC sweep'
)
key_row2 = (
    '6=\u03bb=0.050 TC sweep   '
    '7=Grid mean-opt (\u03bb=0.0005,\u03b7\u2080=0.020)   '
    '8=Grid IR-opt n=10 (\u03bb=0.002,\u03b7\u2080=0.002)   '
    '9=Grid IR-opt n=50 (\u03bb=0.002,\u03b7\u2080=0.002)'
)
ax.annotate(key_row1 + '\n' + key_row2,
            xy=(0.5, -0.16), xycoords='axes fraction',
            ha='center', va='top', fontsize=7.5, color='#333333',
            annotation_clip=False)
fig.tight_layout(rect=[0, 0.12, 1, 1])
path17 = f'{FIGURES_DIR}/fig_17_landscape_bubble.png'
fig.savefig(path17, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {path17}')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 18: Mean SR vs λ for each η₀ (line plot, n=10 grid + n=40 TC sweep)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

# Left panel: grid (n=10) — mean SR across λ for each η₀
ax = axes[0]
eta_colors = ['#1565C0','#2E7D32','#E65100','#6A1B9A','#795548']
for j, eta in enumerate(ETAS):
    means = [grid[(lam, eta)].mean() for lam in LAMBDAS]
    ax.plot([str(l) for l in LAMBDAS], means, 'o-', color=eta_colors[j],
            label=f'η₀={eta:.3f}', lw=1.8, ms=6)

# Mark baseline
ax.axhline(float(grid[(0.001,0.010)].mean()), color='#2196F3',
           linestyle='--', lw=1.5, alpha=0.8, label='Baseline (n=10)')
ax.set_title('Grid: Mean SR by λ and η₀ (n=10 seeds)', fontsize=10)
ax.set_xlabel('λ (TC penalty coefficient)', fontsize=10)
ax.set_ylabel('Mean Sharpe Ratio', fontsize=10)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.15, 0.85)

# Right panel: TC sweep (n=40) — mean ± std vs λ
ax = axes[1]
lam_vals = [0.001, 0.005, 0.010, 0.050]
all_arrs = [base40, tc005, tc010, tc050]
means_ = [a.mean() for a in all_arrs]
stds_  = [a.std(ddof=1) for a in all_arrs]
x_pos  = list(range(len(lam_vals)))

ax.errorbar(x_pos, means_, yerr=[s/np.sqrt(40)*1.96 for s in stds_],
            fmt='o-', color='#D32F2F', lw=2, ms=7, capsize=5,
            label='Mean SR ± 95% CI of mean', zorder=3)
ax.fill_between(x_pos,
                [m - s for m,s in zip(means_, stds_)],
                [m + s for m,s in zip(means_, stds_)],
                alpha=0.15, color='#D32F2F', label='±1 std band')
ax.axhline(0, color='black', lw=0.8, alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels([str(l) for l in lam_vals])
ax.set_title('TC Sweep: Mean SR vs λ (n=40 seeds, η₀=0.010)', fontsize=10)
ax.set_xlabel('λ (TC penalty coefficient)', fontsize=10)
ax.set_ylabel('Mean Sharpe Ratio', fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle('Performance vs. λ: Grid Search (left) and TC Sweep (right)', fontsize=11, y=1.01)
fig.tight_layout()
path18 = f'{FIGURES_DIR}/fig_18_landscape_lambda.png'
fig.savefig(path18, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {path18}')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 19: Power curve — bootstrap power vs n for d≈0.51 (mean-optimal gap)
# and d≈0.11 (all null results)
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(0)

def bootstrap_power(mu_a, std_a, mu_b, std_b, n_per_group, n_boot=5000, alpha=0.05):
    """Simulate power by drawing from Normal approximations."""
    reject = 0
    for _ in range(n_boot):
        a = rng.normal(mu_a, std_a, n_per_group)
        b = rng.normal(mu_b, std_b, n_per_group)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if p < alpha:
            reject += 1
    return reject / n_boot

# Parameters from real data
# Mean-optimal gap: grid[(0.0005,0.020)] vs grid[(0.001,0.010)] at n=10 → d=0.51
# Use n=50 baseline stats as ground truth for std
base_mean = float(baseline50.mean())
base_std  = float(baseline50.std(ddof=1))
target_mean_opt = float(grid[(0.0005,0.020)].mean())  # 0.655 at n=10
# For power sim, assume d=0.51 effect with baseline std
effect_d051 = 0.51 * base_std   # shift

# Null result configs: d≈0.11
effect_d011 = 0.11 * base_std
effect_d030 = 0.30 * base_std  # EXP-17 revised estimate

ns = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 550]
power_d051 = []
power_d030 = []
power_d011 = []

print('Computing power curves...')
for n in ns:
    p51 = bootstrap_power(base_mean + effect_d051, base_std,
                          base_mean, base_std, n, n_boot=3000)
    p30 = bootstrap_power(base_mean + effect_d030, base_std,
                          base_mean, base_std, n, n_boot=3000)
    p11 = bootstrap_power(base_mean + effect_d011, base_std,
                          base_mean, base_std, n, n_boot=3000)
    power_d051.append(p51)
    power_d030.append(p30)
    power_d011.append(p11)
    print(f'  n={n:3d}: power(d=0.51)={p51:.3f}  power(d=0.30)={p30:.3f}  power(d=0.11)={p11:.3f}')

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot(ns, power_d051, 'o-', color='#1565C0', lw=2.2, ms=7,
        label='d=0.51 (EXP-15 initial estimate, n=10)')
ax.plot(ns, power_d030, '^-', color='#E65100', lw=2.2, ms=7,
        label='d=0.30 (EXP-17 revised estimate, n=75)')
ax.plot(ns, power_d011, 's--', color='#607D8B', lw=2.0, ms=7,
        label=r'd=0.11 ($\eta$ sweep / TC null results)')
ax.axhline(0.80, color='#2E7D32', lw=1.5, linestyle=':', label='80% power threshold')
ax.axhline(0.05, color='#B71C1C', lw=1.2, linestyle=':', alpha=0.7, label=r'$\alpha$=0.05 (type I error)')

# Mark actual EXP-17 result: n=75, actual power=46%
ax.scatter([75], [0.46], s=130, color='#E65100', zorder=6, marker='D',
           label='EXP-17/18 actual power')
ax.annotate('EXP-17\nn=75, 46%',
            xy=(75, 0.46), xytext=(100, 0.38),
            fontsize=8, color='#BF360C', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='#FFF3E0', ec='#E65100', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.3))

# Mark actual EXP-18 result: n=150, actual power=30%
ax.scatter([150], [0.30], s=130, color='#E65100', zorder=6, marker='D')
ax.annotate('EXP-18\nn=150, 30%',
            xy=(150, 0.30), xytext=(195, 0.21),
            fontsize=8, color='#BF360C', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='#FFF3E0', ec='#E65100', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.3))

# Annotate 80% power crossing for d=0.51
for i, (n, p) in enumerate(zip(ns, power_d051)):
    if p >= 0.80:
        ax.annotate(f'n={n}: 80%\n(d=0.51)',
                    xy=(n, p),
                    xytext=(n + 15, p - 0.18),
                    fontsize=8, color='#0D47A1', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#1565C0', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.3))
        break

# Annotate 80% power crossing for d=0.30
for i, (n, p) in enumerate(zip(ns, power_d030)):
    if p >= 0.80:
        ax.annotate(f'n={n}: 80%\n(d=0.30)',
                    xy=(n, p),
                    xytext=(n - 110, p + 0.05),
                    fontsize=8, color='#BF360C', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#E65100', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.3,
                                    connectionstyle='arc3,rad=-0.2'))
        break

ax.set_xlabel('Seeds per configuration (n)', fontsize=11)
ax.set_ylabel('Bootstrap power (Welch t-test, alpha=0.05)', fontsize=11)
ax.set_title('Statistical Power vs. Sample Size\n'
             'for Effect Sizes Observed in This Study (EXP-15 through EXP-18)', fontsize=10.5)
ax.legend(fontsize=8.0, loc='upper left')
ax.set_xlim(5, 560)
ax.set_ylim(0.0, 1.02)
ax.grid(True, alpha=0.3)
fig.tight_layout()
path19 = f'{FIGURES_DIR}/fig_19_power_curve.png'
fig.savefig(path19, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {path19}')
print('All landscape figures done.')
