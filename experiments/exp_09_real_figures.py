"""
EXP-09: Figure Generation for Real Market Data Results
======================================================
Generates 4 publication-quality figures for the real-data extension section:
  fig_07: Sharpe comparison (real data) — bar chart with error bars
  fig_08: Cumulative wealth curves (real data, test period)
  fig_09: Stability comparison — side-by-side Sharpe std: synthetic vs real
  fig_10: Turnover comparison — synthetic vs real (grouped bar)
"""
import os as _os
# Repo-relative root; override with the RESEARCH_ROOT environment variable.
_ROOT = _os.environ.get("RESEARCH_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGURES_DIR = _ROOT + "/figures"
RESULTS_DIR = _ROOT + "/results"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load results ───────────────────────────────────────────────────────────────
with open(f"{RESULTS_DIR}/exp_05_results.json") as f: r5 = json.load(f)
with open(f"{RESULTS_DIR}/exp_06_results.json") as f: r6 = json.load(f)
with open(f"{RESULTS_DIR}/exp_07_results.json") as f: r7 = json.load(f)
with open(f"{RESULTS_DIR}/exp_08_results.json") as f: r8 = json.load(f)
# Synthetic results for comparison
with open(f"{RESULTS_DIR}/exp_02_results.json") as f: r2_syn = json.load(f)
with open(f"{RESULTS_DIR}/exp_03_results.json") as f: r3_syn = json.load(f)

REWARD_TYPES = ["log_return", "dsr", "log_return_tc", "dsr_tc"]

# EXP-07 applies the transaction cost to the realised return for every reward
# type, so the log_return and log_return_tc branches reduce to the same
# objective and produce identical runs. Real-data panels therefore plot a
# single merged LR series; see the note to Table II in the paper.
REAL_REWARD_TYPES = ["log_return", "dsr", "dsr_tc"]

LABEL_MAP = {
    "log_return":    "LR",
    "dsr":           "DSR",
    "log_return_tc": "LR+TC",
    "dsr_tc":        "DSR+TC\n(Proposed)",
}
REAL_LABEL_MAP = dict(LABEL_MAP, log_return="LR\n(R1/R3)")
COLORS = {
    "log_return":    "#4C72B0",
    "dsr":           "#55A868",
    "log_return_tc": "#C44E52",
    "dsr_tc":        "#DD8452",
    "equal_weight":  "#8172B2",
    "min_variance":  "#937860",
    "max_sharpe":    "#20808D",
}

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7: Sharpe comparison on real data (bar + error bars)
# ─────────────────────────────────────────────────────────────────────────────
all_strats   = REAL_REWARD_TYPES + ["equal_weight", "min_variance", "max_sharpe"]
sharpe_means = []
sharpe_stds  = []
for s in all_strats:
    if s in REWARD_TYPES:
        sharpe_means.append(r7["results"][s]["sharpe_mean"])
        sharpe_stds.append(r7["results"][s]["sharpe_std"])
    else:
        sharpe_means.append(r6["baselines"][s]["sharpe_ratio"])
        sharpe_stds.append(0.0)

labels_plot  = [REAL_LABEL_MAP.get(s, s.replace("_", "-").title()) for s in all_strats]
labels_plot[-2] = "Min-Var"
labels_plot[-1] = "Max-SR\n(MVO)"
colors_plot = [COLORS.get(s, "#999999") for s in all_strats]

fig, ax = plt.subplots(figsize=(10, 5))
x    = np.arange(len(labels_plot))
bars = ax.bar(x, sharpe_means, yerr=sharpe_stds, capsize=5,
              color=colors_plot, alpha=0.85, edgecolor="black", linewidth=0.7,
              error_kw={"linewidth": 1.5})
ax.set_xticks(x)
ax.set_xticklabels(labels_plot, fontsize=10)
ax.set_ylabel("Annualized Sharpe Ratio", fontsize=11)
ax.set_title("Out-of-Sample Sharpe Ratio — Real S&P 500 Data (2022–2024)\n"
             "Mean ± std over 3 seeds", fontsize=11)
for bar, m, s in zip(bars, sharpe_means, sharpe_stds):
    label_y = m + s + 0.06
    ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"{m:.3f}",
            ha="center", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
ax.set_ylim(min(0, min(sharpe_means) - 0.3),
            max(sharpe_means) + max(sharpe_stds) + 0.4)
ax.axhline(0, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_07_real_sharpe.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_07 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 8: Cumulative wealth curves — real data test period
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

daily_series = r8["daily_return_series"]
test_regimes = np.array(r8["test_regimes"])

line_styles = {"log_return": "-",  "dsr": "--", "log_return_tc": "-.", "dsr_tc": "-", "equal_weight": ":"}
line_widths  = {"log_return": 1.5, "dsr": 1.5,  "log_return_tc": 1.5, "dsr_tc": 2.5, "equal_weight": 1.5}

for rtype in REAL_REWARD_TYPES + ["equal_weight"]:
    if rtype in daily_series:
        dr = np.array(daily_series[rtype])
        pv = np.cumprod(1 + dr)
        t  = np.arange(len(pv))
        ax.plot(t, pv,
                label=REAL_LABEL_MAP.get(rtype, rtype.replace("_", " ").title()),
                color=COLORS.get(rtype, "#888"),
                linewidth=line_widths.get(rtype, 1.5),
                linestyle=line_styles.get(rtype, "-"))

# Shade bear market periods
in_bear, bear_start = False, 0
T = len(test_regimes)
for i, reg in enumerate(test_regimes):
    if reg == 1 and not in_bear:
        bear_start = i; in_bear = True
    elif reg == 0 and in_bear:
        ax.axvspan(bear_start, i, alpha=0.10, color="red")
        in_bear = False
if in_bear:
    ax.axvspan(bear_start, T, alpha=0.10, color="red")

ax.set_xlabel("Trading Days (Test Period: 2022–2024)", fontsize=11)
ax.set_ylabel("Portfolio Value (Initial = 1.0)", fontsize=11)
ax.set_title("Cumulative Wealth — Real S&P 500 Data (2022–2024)\n"
             "Red shading = data-driven bear regime", fontsize=11)
ax.grid(alpha=0.25)
bear_patch = mpatches.Patch(color="red", alpha=0.25, label="Bear Regime")
handles, lbls = ax.get_legend_handles_labels()
ax.legend(handles + [bear_patch], lbls + ["Bear Regime"], fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_08_real_wealth.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_08 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 9: Stability comparison — synthetic vs real (grouped bars, Sharpe std)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))

def _real(series_key, rt):
    """Real-data value, or NaN for the config EXP-07 does not identify separately."""
    if rt not in REAL_REWARD_TYPES:
        return np.nan
    return r7["results"][rt][series_key]

syn_stds  = [r2_syn["results"][rt]["sharpe_std"] for rt in REWARD_TYPES]
real_stds = [_real("sharpe_std", rt)             for rt in REWARD_TYPES]
labels_r  = [LABEL_MAP[rt].replace("\n", " ") for rt in REWARD_TYPES]
labels_r[REWARD_TYPES.index("log_return_tc")] += "*"

x   = np.arange(len(REWARD_TYPES))
w   = 0.35
b1  = ax.bar(x - w/2, syn_stds,  w, label="Synthetic GBM", color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.7)
b2  = ax.bar(x + w/2, real_stds, w, label="Real S&P 500",  color="#DD8452", alpha=0.85, edgecolor="black", linewidth=0.7)

for bar, v in list(zip(b1, syn_stds)) + list(zip(b2, real_stds)):
    if np.isnan(v):
        continue
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels_r, fontsize=10)
ax.set_ylabel("Sharpe Ratio Std (3 seeds)", fontsize=10)
ax.set_title("Training Stability: Synthetic GBM vs. Real S&P 500\n"
             "Lower = more stable", fontsize=11)
ax.text(0.99, -0.16, "*not separately identified on real data (see paper, Table II)",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5, style="italic")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_09_stability_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_09 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 10: Turnover comparison — synthetic vs real
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))

syn_to  = [r3_syn["turnover"][rt]        for rt in REWARD_TYPES]
real_to = [_real("turnover_mean", rt)    for rt in REWARD_TYPES]

b1  = ax.bar(x - w/2, syn_to,  w, label="Synthetic GBM", color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.7)
b2  = ax.bar(x + w/2, real_to, w, label="Real S&P 500",  color="#DD8452", alpha=0.85, edgecolor="black", linewidth=0.7)

for bar, v in list(zip(b1, syn_to)) + list(zip(b2, real_to)):
    if np.isnan(v):
        continue
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{v:.4f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels_r, fontsize=10)
ax.set_ylabel("Mean Daily Turnover", fontsize=10)
ax.set_title("Portfolio Turnover: Synthetic GBM vs. Real S&P 500", fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_10_turnover_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig_10 saved")

print("\nAll real-data figures saved.")
