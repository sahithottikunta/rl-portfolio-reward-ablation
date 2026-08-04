"""
EXP-15 Aggregator: builds the 4×5 grid summary, generates heatmap figures,
runs statistical tests vs the (0.001, 0.010) baseline, saves exp_15_results.json.

Grid:
  lambdas: [0.0005, 0.001, 0.002, 0.005]
  etas:    [0.002,  0.005, 0.010, 0.020, 0.050]
"""
import os as _os
# Repo-relative root; override with the RESEARCH_ROOT environment variable.
_ROOT = _os.environ.get("RESEARCH_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json, numpy as np, os, itertools, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = _ROOT + "/results"
FIGURES_DIR = _ROOT + "/figures"
SEEDS       = list(range(42, 52))   # n=10 per cell

LAMBDAS = [0.0005, 0.001, 0.002, 0.005]
ETAS    = [0.002,  0.005, 0.010, 0.020, 0.050]

def tc_str(v):
    if   v == 0.0005: return "0.0005"
    elif v == 0.001:  return "0.001"
    elif v == 0.002:  return "0.002"
    elif v == 0.005:  return "0.005"
    return f"{v:.4f}"

def eta_str(v):
    if   v == 0.002:  return "0.002"
    elif v == 0.005:  return "0.005"
    elif v == 0.010:  return "0.010"
    elif v == 0.020:  return "0.020"
    elif v == 0.050:  return "0.050"
    return f"{v:.3f}"

def load_cell(lam, eta):
    ts = tc_str(lam); es = eta_str(eta)
    path = f"{RESULTS_DIR}/exp_15_tc{ts}_eta{es}_42_51.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    with open(path) as f:
        d = json.load(f)
    return d["sharpes"], d["cumrets"], d["tos"]

# ── Load grid ─────────────────────────────────────────────────────────────────
print("Loading grid cells...")
grid_mean  = np.zeros((len(LAMBDAS), len(ETAS)))
grid_std   = np.zeros((len(LAMBDAS), len(ETAS)))
grid_med   = np.zeros((len(LAMBDAS), len(ETAS)))
grid_samps = {}

for i, lam in enumerate(LAMBDAS):
    for j, eta in enumerate(ETAS):
        sharpes, cumrets, tos = load_cell(lam, eta)
        a = np.array(sharpes)
        grid_mean[i, j] = float(a.mean())
        grid_std[i, j]  = float(a.std(ddof=1))
        grid_med[i, j]  = float(np.median(a))
        grid_samps[(lam, eta)] = sharpes
        print(f"  lambda={lam:.4f}  eta={eta:.3f}  SR={a.mean():.3f}+/-{a.std(ddof=1):.3f}")

# ── Find optimal cell ─────────────────────────────────────────────────────────
# Criterion: highest mean SR (primary); lowest std (tiebreak)
best_mean_idx = np.unravel_index(np.argmax(grid_mean), grid_mean.shape)
best_std_idx  = np.unravel_index(np.argmin(grid_std),  grid_std.shape)

# Information ratio: mean / std  (risk-adjusted ranking)
# Guard against negative-mean cells
ir = np.where(grid_mean > 0, grid_mean / (grid_std + 1e-6), -np.inf)
best_ir_idx = np.unravel_index(np.argmax(ir), ir.shape)

print(f"\nBest mean SR: lambda={LAMBDAS[best_mean_idx[0]]:.4f}  eta={ETAS[best_mean_idx[1]]:.3f}"
      f"  SR={grid_mean[best_mean_idx]:.3f} +/- {grid_std[best_mean_idx]:.3f}")
print(f"Best IR:      lambda={LAMBDAS[best_ir_idx[0]]:.4f}  eta={ETAS[best_ir_idx[1]]:.3f}"
      f"  SR={grid_mean[best_ir_idx]:.3f} +/- {grid_std[best_ir_idx]:.3f}"
      f"  IR={ir[best_ir_idx]:.3f}")
print(f"Baseline (0.001, 0.010): SR={grid_mean[LAMBDAS.index(0.001), ETAS.index(0.010)]:.3f}"
      f"  std={grid_std[LAMBDAS.index(0.001), ETAS.index(0.010)]:.3f}")

# ── Statistical test: optimal vs baseline ────────────────────────────────────
baseline_samples = np.array(grid_samps[(0.001, 0.010)])
opt_lam = LAMBDAS[best_mean_idx[0]]; opt_eta = ETAS[best_mean_idx[1]]
opt_samples = np.array(grid_samps[(opt_lam, opt_eta)])

def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return float((a.mean()-b.mean())/(pooled+1e-9))

t_stat, p_val = stats.ttest_ind(opt_samples, baseline_samples, equal_var=False)
d = cohen_d(opt_samples, baseline_samples)
rng = np.random.default_rng(42)
power = sum(1 for _ in range(10_000) if
            stats.ttest_ind(rng.choice(opt_samples,10,replace=True),
                            rng.choice(baseline_samples,10,replace=True),
                            equal_var=False)[1] < 0.05) / 10_000

print(f"\nStat test: optimal ({opt_lam:.4f},{opt_eta:.3f}) vs baseline (0.001,0.010):")
print(f"  Cohen d = {d:+.4f}  Welch t = {t_stat:.4f}  p = {p_val:.4f}  power = {power:.4f}")

# IR-optimal test too
ir_lam = LAMBDAS[best_ir_idx[0]]; ir_eta = ETAS[best_ir_idx[1]]
ir_samples = np.array(grid_samps[(ir_lam, ir_eta)])
t2, p2 = stats.ttest_ind(ir_samples, baseline_samples, equal_var=False)
d2 = cohen_d(ir_samples, baseline_samples)
power2 = sum(1 for _ in range(10_000) if
             stats.ttest_ind(rng.choice(ir_samples,10,replace=True),
                             rng.choice(baseline_samples,10,replace=True),
                             equal_var=False)[1] < 0.05) / 10_000
print(f"\nStat test: IR-optimal ({ir_lam:.4f},{ir_eta:.3f}) vs baseline (0.001,0.010):")
print(f"  Cohen d = {d2:+.4f}  Welch t = {t2:.4f}  p = {p2:.4f}  power = {power2:.4f}")

# ── Heatmaps ──────────────────────────────────────────────────────────────────
LAM_LABELS = [f"{l:.4f}" for l in LAMBDAS]
ETA_LABELS = [f"{e:.3f}" for e in ETAS]

def make_heatmap(data, title, fname, cmap="RdYlGn", fmt=".3f", vmin=None, vmax=None,
                 best_idx=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(ETAS)));    ax.set_xticklabels(ETA_LABELS, fontsize=9)
    ax.set_yticks(range(len(LAMBDAS))); ax.set_yticklabels(LAM_LABELS, fontsize=9)
    ax.set_xlabel(r"$\eta_0$ (EMA adaptation rate)", fontsize=10)
    ax.set_ylabel(r"$\lambda$ (TC penalty)", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    for i in range(len(LAMBDAS)):
        for j in range(len(ETAS)):
            val = data[i, j]
            txt = f"{val:{fmt}}"
            bg = im.cmap(im.norm(val))
            lum = 0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]
            # Use high-contrast black or white; add thin outline stroke for mid-tone cells
            if lum < 0.45:
                color = "white"
            else:
                color = "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color,
                    path_effects=[
                        __import__('matplotlib.patheffects', fromlist=['withStroke'])
                        .withStroke(linewidth=2,
                                    foreground="black" if color=="white" else "white")
                    ])
    # Mark optimal cell
    if best_idx is not None:
        bi, bj = best_idx
        rect = plt.Rectangle((bj-0.5, bi-0.5), 1, 1,
                              linewidth=3, edgecolor="white", facecolor="none")
        ax.add_patch(rect)
    # Baseline marker
    bl_i = LAMBDAS.index(0.001); bl_j = ETAS.index(0.010)
    rect2 = plt.Rectangle((bl_j-0.5, bl_i-0.5), 1, 1,
                           linewidth=2.5, edgecolor="#FFD700", facecolor="none",
                           linestyle="--")
    ax.add_patch(rect2)
    plt.colorbar(im, ax=ax, shrink=0.85)
    # Legend placed BELOW the axes (outside the plot area) for full readability
    fig.text(0.5, 0.01,
             "White border = optimal cell   |   Gold dashed border = baseline (0.001, 0.010)",
             ha="center", va="bottom", fontsize=8, color="#333333")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = f"{FIGURES_DIR}/{fname}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path

print("\nGenerating heatmaps...")
make_heatmap(grid_mean, r"Mean Sharpe Ratio — 2D Grid Search ($n=10$ seeds)",
             "fig_14_grid_mean_sr.png", cmap="RdYlGn",
             vmin=max(-0.5, grid_mean.min()), vmax=min(1.5, grid_mean.max()),
             best_idx=best_mean_idx)

make_heatmap(grid_std,  r"Cross-Seed Std — 2D Grid Search ($n=10$ seeds)",
             "fig_15_grid_std_sr.png", cmap="RdYlGn_r",
             vmin=0, vmax=min(1.0, grid_std.max()),
             best_idx=best_std_idx)

# Information ratio heatmap (clip -inf to -1 for display)
ir_disp = np.where(np.isfinite(ir), ir, -1.0)
make_heatmap(ir_disp, r"Information Ratio (mean/std) — 2D Grid Search",
             "fig_16_grid_ir.png", cmap="RdYlGn",
             best_idx=best_ir_idx)

# ── Save results JSON ─────────────────────────────────────────────────────────
def cell_summary(lam, eta):
    s = np.array(grid_samps[(lam, eta)])
    return {"lambda": lam, "eta0": eta, "mean": float(s.mean()),
            "std": float(s.std(ddof=1)), "median": float(np.median(s)),
            "min": float(s.min()), "max": float(s.max()), "samples": list(s)}

all_cells = []
for lam in LAMBDAS:
    for eta in ETAS:
        all_cells.append(cell_summary(lam, eta))

out = {
    "lambdas": LAMBDAS, "etas": ETAS, "n_per_cell": 10, "seeds": SEEDS,
    "grid_mean": grid_mean.tolist(), "grid_std": grid_std.tolist(),
    "grid_median": grid_med.tolist(), "grid_ir": ir_disp.tolist(),
    "best_mean": {"lambda": opt_lam, "eta0": opt_eta,
                  "mean": float(grid_mean[best_mean_idx]),
                  "std":  float(grid_std[best_mean_idx])},
    "best_ir":   {"lambda": ir_lam,  "eta0": ir_eta,
                  "mean": float(grid_mean[best_ir_idx]),
                  "std":  float(grid_std[best_ir_idx]),
                  "ir":   float(ir[best_ir_idx])},
    "baseline":  {"lambda": 0.001, "eta0": 0.010,
                  "mean": float(grid_mean[LAMBDAS.index(0.001), ETAS.index(0.010)]),
                  "std":  float(grid_std[LAMBDAS.index(0.001), ETAS.index(0.010)])},
    "stat_tests": {
        "mean_optimal_vs_baseline": {
            "cohen_d": round(d, 4), "t_stat": round(float(t_stat), 4),
            "p_value": round(float(p_val), 4), "power": round(power, 4), "n": 10},
        "ir_optimal_vs_baseline": {
            "cohen_d": round(d2, 4), "t_stat": round(float(t2), 4),
            "p_value": round(float(p2), 4), "power": round(power2, 4), "n": 10},
    },
    "all_cells": all_cells,
}

with open(f"{RESULTS_DIR}/exp_15_results.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n=== GRID SUMMARY ===")
print(f"{'lambda':>8}  {'eta0':>6}  {'SR mean':>8}  {'SR std':>8}  {'IR':>6}")
print("-"*48)
for lam in LAMBDAS:
    for eta in ETAS:
        i, j = LAMBDAS.index(lam), ETAS.index(eta)
        irv = ir[i, j] if np.isfinite(ir[i, j]) else float("nan")
        mark = " <-- OPTIMAL" if (i,j)==best_mean_idx else \
               (" <-- IR-OPT" if (i,j)==best_ir_idx else \
               (" <-- BASELINE" if (lam==0.001 and eta==0.010) else ""))
        print(f"  {lam:.4f}  {eta:.3f}  {grid_mean[i,j]:>8.3f}  {grid_std[i,j]:>8.3f}  {irv:>6.3f}{mark}")

print("\nSaved -> exp_15_results.json")
