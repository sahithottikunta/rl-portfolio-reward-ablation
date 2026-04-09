"""
EXP-11: Figures for Adaptive η Validation
==========================================
fig_11: Side-by-side grouped bar — Sharpe mean & std, all configs
         (fixed LR, fixed DSR, fixed DSR+TC, adaptive DSR+TC k=20/30/50,
          adaptive DSR k=30 as a representative)
fig_12: η_t time-series trace (seed 42, best config κ=30 DSR+TC) overlaid
         with annualized rolling vol of market, showing how η contracts
         during volatile periods (2022 bear) and relaxes in 2023-2024 bull
fig_13: Cumulative wealth — best adaptive DSR+TC (κ=30, seed 42) vs.
         fixed DSR+TC (seed 42) vs. EW baseline, with bear shading
"""

import numpy as np
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGURES_DIR = "/home/user/workspace/research_output/figures"
RESULTS_DIR = "/home/user/workspace/research_output/results"

with open(f"{RESULTS_DIR}/exp_07_results.json") as f: r7  = json.load(f)
with open(f"{RESULTS_DIR}/exp_10_results.json") as f: r10 = json.load(f)
with open(f"{RESULTS_DIR}/exp_06_results.json") as f: r6  = json.load(f)
with open(f"{RESULTS_DIR}/exp_08_results.json") as f: r8  = json.load(f)

test_regimes = np.array(r8["test_regimes"])

# ── Color palette ─────────────────────────────────────────────────────────────
C_FIXED_LR   = "#4C72B0"   # blue
C_FIXED_DSR  = "#937860"   # brown (poor performer)
C_FIXED_DSRTC= "#C44E52"   # red-orange (fixed baseline)
C_ADP_DSRTC  = "#20808D"   # teal (adaptive — highlight)
C_ADP_DSR    = "#BAB9B4"   # muted grey (adaptive DSR — still fails)
C_EW         = "#8172B2"   # purple

# ─────────────────────────────────────────────────────────────────────────────
# FIG 11: Grouped bar — Sharpe mean + std for all configs
# ─────────────────────────────────────────────────────────────────────────────
configs = [
    ("Fixed LR",           r7["results"]["log_return"],          C_FIXED_LR,    "//"),
    ("Fixed DSR",          r7["results"]["dsr"],                  C_FIXED_DSR,   "//"),
    ("Fixed DSR+TC",       r7["results"]["dsr_tc"],               C_FIXED_DSRTC, ""),
    ("Adap. DSR+TC κ=20",  r10["results"]["dsr_tc_k20"],          C_ADP_DSRTC,   ""),
    ("Adap. DSR+TC κ=30",  r10["results"]["dsr_tc_k30"],          C_ADP_DSRTC,   ""),
    ("Adap. DSR+TC κ=50",  r10["results"]["dsr_tc_k50"],          C_ADP_DSRTC,   ""),
    ("Adap. DSR κ=30",     r10["results"]["dsr_k30"],             C_ADP_DSR,     "xx"),
]

labels  = [c[0] for c in configs]
means   = [c[1]["sharpe_mean"] for c in configs]
stds    = [c[1]["sharpe_std"]  for c in configs]
colors  = [c[2] for c in configs]
hatches = [c[3] for c in configs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

x = np.arange(len(labels))
bar_w = 0.65

# Left: Sharpe mean
bars = ax1.bar(x, means, bar_w,
               color=colors, hatch=hatches, alpha=0.85,
               edgecolor="black", linewidth=0.7,
               yerr=stds, capsize=4,
               error_kw={"linewidth": 1.5, "ecolor": "black"})
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
ax1.set_ylabel("Annualized Sharpe Ratio", fontsize=10)
ax1.set_title("Sharpe Ratio: Fixed vs. Adaptive $\\eta$\n(mean ± std over 3 seeds)", fontsize=10)
ax1.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax1.grid(axis="y", alpha=0.3)
for bar, m, s in zip(bars, means, stds):
    ypos = m + s + 0.05 if m >= 0 else m - s - 0.15
    ax1.text(bar.get_x() + bar.get_width()/2, ypos,
             f"{m:.3f}", ha="center", va="bottom", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

# Right: Sharpe std (stability — lower is better)
bar2 = ax2.bar(x, stds, bar_w,
               color=colors, hatch=hatches, alpha=0.85,
               edgecolor="black", linewidth=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
ax2.set_ylabel("Cross-Seed Sharpe Std (lower = more stable)", fontsize=10)
ax2.set_title("Training Stability: Fixed vs. Adaptive $\\eta$", fontsize=10)
ax2.grid(axis="y", alpha=0.3)
for bar, s in zip(bar2, stds):
    ax2.text(bar.get_x() + bar.get_width()/2, s + 0.01,
             f"{s:.3f}", ha="center", va="bottom", fontsize=7.5)

# Legend patches
fixed_patch = mpatches.Patch(facecolor=C_FIXED_DSRTC, edgecolor="black",
                              label="Fixed $\\eta=0.01$")
adap_patch  = mpatches.Patch(facecolor=C_ADP_DSRTC,   edgecolor="black",
                              label="Adaptive $\\eta_t$ (proposed)")
dsr_patch   = mpatches.Patch(facecolor=C_ADP_DSR,  edgecolor="black",
                              hatch="xx", label="Adaptive DSR (no TC)")
for ax in [ax1, ax2]:
    ax.legend(handles=[fixed_patch, adap_patch, dsr_patch], fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_11_adaptive_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_11 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 12: η_t time-series trace + rolling market vol
# ─────────────────────────────────────────────────────────────────────────────
# Load eta trace for best config (DSR+TC, kappa=30, seed=42)
eta_trace = np.array(r10["eta_traces"]["dsr_tc_k30"])

# Also build market vol series from test returns
test_returns = np.load("/home/user/workspace/research_output/data/real_test_returns.npy")
mean_ret = test_returns.mean(axis=1)           # cross-asset mean daily return
VOLWIN = r10["volwin"]                         # 10
roll_vol = np.array([
    mean_ret[max(0, t-VOLWIN):t].std() if t >= 2 else 0.0
    for t in range(1, len(mean_ret)+1)
])
T_trace = min(len(eta_trace), len(roll_vol), test_returns.shape[0] - 1)
eta_trace = eta_trace[:T_trace]
roll_vol  = roll_vol[:T_trace]

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1]})

days = np.arange(T_trace)

# Top: η_t
ax_top.plot(days, eta_trace, color=C_ADP_DSRTC, linewidth=1.5, label="$\\eta_t$ (adaptive)")
ax_top.axhline(0.01, color=C_FIXED_DSRTC, linestyle="--", linewidth=1.2,
               label="$\\eta=0.01$ (fixed)")
ax_top.set_ylabel("EMA rate $\\eta_t$", fontsize=10)
ax_top.set_title(
    "Adaptive $\\eta_t$ Schedule — DSR+TC, $\\kappa=30$, Seed 42\n"
    "Test period: 2022–2024", fontsize=10)
ax_top.legend(fontsize=9, loc="upper right")
ax_top.grid(alpha=0.25)

# Bottom: rolling vol (annualised)
ax_bot.fill_between(days, roll_vol * np.sqrt(252), alpha=0.35,
                    color="#C44E52", label="10-day rolling vol (ann.)")
ax_bot.set_ylabel("Rolling Ann. Vol (cross-asset)", fontsize=10)
ax_bot.set_xlabel("Trading Days (Test Period)", fontsize=10)
ax_bot.legend(fontsize=9, loc="upper right")
ax_bot.grid(alpha=0.25)

# Bear-regime shading on both panels
in_bear, bstart = False, 0
for i, reg in enumerate(test_regimes[:T_trace]):
    if reg == 1 and not in_bear:
        bstart = i; in_bear = True
    elif reg == 0 and in_bear:
        for ax in [ax_top, ax_bot]:
            ax.axvspan(bstart, i, alpha=0.10, color="red")
        in_bear = False
if in_bear:
    for ax in [ax_top, ax_bot]:
        ax.axvspan(bstart, T_trace, alpha=0.10, color="red")

bear_patch = mpatches.Patch(color="red", alpha=0.25, label="Bear regime")
ax_top.legend(handles=[
    mpatches.Patch(color=C_ADP_DSRTC, label="$\\eta_t$ (adaptive)"),
    mpatches.Patch(color=C_FIXED_DSRTC, label="$\\eta=0.01$ (fixed)"),
    bear_patch
], fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_12_eta_trace.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_12 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 13: Cumulative wealth — adaptive best vs. fixed DSR+TC vs. EW
# ─────────────────────────────────────────────────────────────────────────────
adap_dr_k30  = np.array(r10["results"]["dsr_tc_k30"]["daily_returns_seed42"])
fixed_dr     = np.array(r7["daily_returns"]["dsr_tc"])
ew_dr        = np.array(r6["daily_returns"]["equal_weight"])

# Trim to common length
L = min(len(adap_dr_k30), len(fixed_dr), len(ew_dr), T_trace)
adap_dr_k30  = adap_dr_k30[:L]
fixed_dr     = fixed_dr[:L]
ew_dr        = ew_dr[:L]

pv_adap  = np.cumprod(1 + adap_dr_k30)
pv_fixed = np.cumprod(1 + fixed_dr)
pv_ew    = np.cumprod(1 + ew_dr)

fig, ax = plt.subplots(figsize=(12, 5))
t_axis = np.arange(L)

ax.plot(t_axis, pv_adap,  color=C_ADP_DSRTC,   linewidth=2.5,
        label=f"Adaptive DSR+TC ($\\kappa=30$, seed 42)  SR={r10['results']['dsr_tc_k30']['sharpe_mean']:.3f}")
ax.plot(t_axis, pv_fixed, color=C_FIXED_DSRTC,  linewidth=2.0, linestyle="--",
        label=f"Fixed DSR+TC ($\\eta=0.01$, seed 42)  SR={r7['results']['dsr_tc']['sharpe_mean']:.3f}")
ax.plot(t_axis, pv_ew,    color=C_EW,           linewidth=1.5, linestyle=":",
        label=f"Equal-Weight baseline  SR={r6['baselines']['equal_weight']['sharpe_ratio']:.3f}")

# Bear shading
in_bear, bstart = False, 0
for i, reg in enumerate(test_regimes[:L]):
    if reg == 1 and not in_bear:
        bstart = i; in_bear = True
    elif reg == 0 and in_bear:
        ax.axvspan(bstart, i, alpha=0.10, color="red")
        in_bear = False
if in_bear:
    ax.axvspan(bstart, L, alpha=0.10, color="red")

ax.set_xlabel("Trading Days (Test Period: 2022–2024)", fontsize=11)
ax.set_ylabel("Portfolio Value (Initial = 1.0)", fontsize=11)
ax.set_title(
    "Cumulative Wealth: Adaptive vs. Fixed $\\eta$ — DSR+TC (Seed 42)\n"
    "Red shading = data-driven bear regime", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.25)
bear_patch = mpatches.Patch(color="red", alpha=0.25, label="Bear Regime")
handles, lbls = ax.get_legend_handles_labels()
ax.legend(handles + [bear_patch], lbls + ["Bear Regime"], fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_13_adaptive_wealth.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_13 saved")

# ─────────────────────────────────────────────────────────────────────────────
# Print final numbers for paper writing
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Numbers for paper ===")
# Best adaptive config
best_k = "dsr_tc_k30"
v = r10["results"][best_k]
print(f"Adaptive DSR+TC k=30: SR={v['sharpe_mean']:.3f}±{v['sharpe_std']:.3f}, "
      f"cumret={v['cumret_mean']:.3f}, to={v['turnover_mean']:.4f}")
print(f"Fixed DSR+TC:          SR={r7['results']['dsr_tc']['sharpe_mean']:.3f}±"
      f"{r7['results']['dsr_tc']['sharpe_std']:.3f}, "
      f"cumret={r7['results']['dsr_tc']['cumret_mean']:.3f}")

# Improvements
sr_imp = (v['sharpe_mean'] - r7['results']['dsr_tc']['sharpe_mean']) / abs(r7['results']['dsr_tc']['sharpe_mean']) * 100
std_imp = (r7['results']['dsr_tc']['sharpe_std'] - v['sharpe_std']) / r7['results']['dsr_tc']['sharpe_std'] * 100
to_imp  = (r7['results']['dsr_tc']['turnover_mean'] - v['turnover_mean']) / r7['results']['dsr_tc']['turnover_mean'] * 100
print(f"Sharpe improvement: +{sr_imp:.1f}%")
print(f"Stability improvement (std reduction): {std_imp:.1f}%")
print(f"Turnover reduction: {to_imp:.1f}%")

# Best stability config
best_std_k = "dsr_tc_k50"
vs = r10["results"][best_std_k]
std_imp50 = (r7['results']['dsr_tc']['sharpe_std'] - vs['sharpe_std']) / r7['results']['dsr_tc']['sharpe_std'] * 100
print(f"\nAdaptive DSR+TC k=50 (best std): SR={vs['sharpe_mean']:.3f}±{vs['sharpe_std']:.3f}  "
      f"std reduction vs fixed: {std_imp50:.1f}%")

# DSR no TC
print(f"\nAdaptive DSR k=30 (no TC): SR={r10['results']['dsr_k30']['sharpe_mean']:.3f}±"
      f"{r10['results']['dsr_k30']['sharpe_std']:.3f}  "
      f"(fixed DSR: {r7['results']['dsr']['sharpe_mean']:.3f}±{r7['results']['dsr']['sharpe_std']:.3f})")

# eta trace stats
eta_mean = np.mean(eta_trace)
eta_min  = np.min(eta_trace)
eta_max  = np.max(eta_trace)
print(f"\neta_t stats (test period, k=30): mean={eta_mean:.5f}, min={eta_min:.5f}, max={eta_max:.5f}")
print(f"(vs fixed eta=0.01)")
