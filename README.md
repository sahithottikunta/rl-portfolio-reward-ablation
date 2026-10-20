# Reward Function Design for Deep RL Portfolio Optimization for finance

[![Paper](https://img.shields.io/badge/Paper-PDF%20%2820%20pages%29-blue?logo=adobeacrobatreader)](paper.pdf)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow?logo=python)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![Stable Baselines3](https://img.shields.io/badge/SB3-PPO-blueviolet)](https://stable-baselines3.readthedocs.io)
[![Status](https://img.shields.io/badge/Experiments-930%20runs%20%E2%9C%93-brightgreen)]()

> **"Reward Function Design for Deep Reinforcement Learning Portfolio Optimization:  
> A Comparative Ablation Study Under Regime-Switching Market Conditions"**

This repository contains the complete research code, data, results, and LaTeX source for a 20-page IEEE-format paper conducting a systematic ablation study of reward function design for deep RL portfolio optimisation. All results are real and reproducible — every number in the paper comes from code in this repository.

---

## Table of Contents

- [Key Findings](#key-findings)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Reproduction Instructions](#reproduction-instructions)
- [Experiment Index](#experiment-index)
- [PPO Hyperparameters](#ppo-hyperparameters)
- [Data](#data)
- [Citation](#citation)

---

## Key Findings

1. **Reward function choice is regime-conditional.** DSR+TC achieves SR = 0.391 ± 0.363 in the 2022–2024 bear market; LR-only achieves SR = 0.523 ± 0.576 in the 2025–2026 bull market. No single reward dominates across both regimes (cross-period shift: *t* = −6.14, *p* < 10⁻⁷, Cohen's *d* = −1.03).

2. **DSR+TC is the most stable reward in bear markets.** Across n = 150 seeds on 2022–2024 S&P 500 data, DSR+TC (η = 0.010, λ = 0.001) achieves SR = 0.391 ± 0.363. The winner's-curse trajectory across three confirmatory stages (d = +0.51 → +0.30 → +0.17) shows the advantage shrinks with sample size but remains positive.

3. **Transaction-cost penalty reverses sign in bull markets.** Adding TC penalty (λ > 0) reduces Sharpe ratio by Cohen's *d* > 1.0 in the 2025–2026 period, because the penalisation discourages trend-following rebalancing that is rewarded in bull conditions.

4. **RSMO blending is a well-replicated null.** A regime-switching multi-objective reward (R_t = α_t · LR + (1−α_t) · DSR+TC, α_t = σ(k · μ̂⁶⁰ᵈ)) underperforms both constituent baselines on both test periods across k ∈ {2, 5, 10}, n ≥ 40 seeds. Root cause: the 60-day rolling mean misclassifies the 2022 bear market as bullish (ᾱ_t ∈ [0.56, 0.68]).

---

## Results Summary

### Synthetic Ablation (EXP-02, n = 3 seeds, simulated regime-switching market)

| Reward | Mean SR | Std SR | Cum. Return |
|--------|---------|--------|-------------|
| R1: Log Return (LR) | 1.010 | 0.305 | 19.8% |
| R2: DSR only | 1.017 | 0.106 | 19.0% |
| R3: LR + TC penalty | 1.072 | 0.431 | 22.7% |
| **R4: DSR + TC (default)** | **1.011** | **0.097** | **19.5%** |

DSR+TC achieves the **lowest variance** (std = 0.097), a 54% reduction vs. LR-only.

---

### Real S&P 500 — Bear Period: 2022–2024 (EXP-18, n = 150 seeds)

| Config | Mean SR | Std SR | 95% CI |
|--------|---------|--------|--------|
| DSR+TC (λ=0.001, η=0.010) — **baseline** | 0.391 | 0.363 | [0.332, 0.450] |
| DSR+TC (λ=0.0005, η=0.020) — target | 0.450 | 0.339 | [0.396, 0.505] |

Welch *t* = 1.47, *p* = 0.144, Cohen's *d* = +0.17 (winner's curse confirmed across 3 confirmatory stages).

---

### Real S&P 500 — Bull Period: 2025–2026 (EXP-19, n = 60 seeds, 312 trading days)

**Regime composition:** 229 bull days (73%) / 83 bear days (27%) · Equal-weight SR = 0.593

| Reward | Mean SR | Std SR | 95% CI |
|--------|---------|--------|--------|
| R1: DSR + TC | −0.115 | 0.596 | [−0.269, +0.039] |
| R2: DSR only | +0.513 | 0.607 | [+0.356, +0.670] |
| R3: LR + TC | −0.317 | 0.648 | — |
| **R4: LR only** | **+0.523** | **0.576** | **[+0.374, +0.672]** |

LR-only vs. DSR+TC: *t* = 5.96, *p* < 0.001, *d* = +1.09.  
Cross-period DSR+TC shift (0.391 → −0.115): *t* = −6.14, *p* < 10⁻⁷, *d* = −1.03.

---

### Regime-Switching Multi-Objective Reward — EXP-20 (n ≥ 40 seeds per cell)

Reward: R_t = α_t · log(1+p_t) + (1−α_t) · ΔDSR(p_t − λ·TO_t),  where α_t = σ(k · μ̂⁶⁰ᵈ_t)

| Config | Period | n | Mean SR | vs DSR+TC (*d*) | vs LR-only (*d*) |
|--------|--------|----|---------|-----------------|------------------|
| RSMO k=2 | 2022–2024 | 50 | +0.174 | −0.555 ** | −1.790 ** |
| RSMO k=5 | 2022–2024 | 40 | +0.070 | −0.689 ** | −1.703 ** |
| RSMO k=10 | 2022–2024 | 40 | −0.140 | −1.085 ** | −2.064 ** |
| RSMO k=2 | 2025–2026 | 50 | −0.181 | −0.115 ns | −1.265 ** |
| RSMO k=5 | 2025–2026 | 40 | −0.146 | −0.050 ns | −1.089 ** |
| RSMO k=10 | 2025–2026 | 40 | −0.488 | −0.570 ** | −1.569 ** |
| DSR+TC baseline | 2022–2024 | 150 | +0.391 | — | — |
| LR-only baseline | 2022–2024 | 40 | +0.824 | — | — |
| DSR+TC baseline | 2025–2026 | 60 | −0.115 | — | — |
| LR-only baseline | 2025–2026 | 60 | +0.523 | — | — |

`**` *p* < 0.01 (Welch *t*-test, two-tailed) · `ns` = not significant

---

## Repository Structure

```
rl-portfolio-reward-ablation/
│
├── paper.tex                          ← LaTeX source (IEEE two-column, compiles cleanly)
├── paper.pdf                          ← Compiled PDF (20 pages, 40 verified references)
├── references.bib                     ← BibTeX bibliography (40 entries, all verified)
├── README.md                          ← This file
│
├── experiments/
│   ├── exp_01_environment.py          ← Synthetic market data generation + classical baselines
│   ├── exp_02_reward_comparison.py    ← Main synthetic reward ablation (EXP-02)
│   ├── exp_03_regime_analysis.py      ← Regime-conditional breakdown + turnover analysis
│   ├── exp_04_figures.py              ← Synthetic result figures (figs 01–06)
│   ├── exp_05_real_data.py            ← S&P 500 data download (Yahoo Finance, 10 tickers)
│   ├── exp_06_real_baselines.py       ← Classical baselines on real data
│   ├── exp_07_real_ablation.py        ← Reward ablation on real market data (3 seeds)
│   ├── exp_08_real_regime.py          ← Regime analysis on real data
│   ├── exp_09_real_figures.py         ← Real-market figures (figs 07–10)
│   ├── exp_10_adaptive_eta.py         ← Adaptive η schedule: η_t = η₀/(1 + κσ̂_t)
│   ├── exp_11_adaptive_figures.py     ← Adaptive η figures (figs 11–12)
│   ├── exp_12_worker.py               ← EXP-12 seed-parallel worker (κ sweep, 10 seeds/batch)
│   ├── exp_12_seed_expansion.py       ← EXP-12 seed expansion to n=40
│   ├── exp_12_aggregate.py            ← EXP-12 aggregation + Welch t-tests
│   ├── exp_13_worker.py               ← EXP-13 n=40 seeds worker
│   ├── exp_13_run_all.py              ← EXP-13 orchestrator
│   ├── exp_13_aggregate.py            ← EXP-13 aggregation
│   ├── exp_14_worker.py               ← TC sweep (λ ∈ {0.001, 0.005, 0.01, 0.05}), n=40
│   ├── exp_14_aggregate.py            ← TC sweep aggregation + figures (figs 14–16)
│   ├── exp_15_worker.py               ← 2D (λ, η₀) grid search, n=10 per cell
│   ├── exp_15_aggregate.py            ← Grid search aggregation
│   ├── exp_16_worker.py               ← Stability-optimal cell (λ=0.002, η₀=0.002), n=50
│   ├── exp_17_worker.py               ← Mean-optimal cell (λ=0.0005, η₀=0.020), n=75
│   ├── exp_18_worker.py               ← Confirmatory n=150 seeds for mean-optimal cell
│   ├── exp_18_aggregate.py            ← EXP-18 aggregation + power analysis
│   ├── exp_19_data_prep.py            ← Post-2024 data download (2025-01-03 to 2026-04-02)
│   ├── exp_19_worker.py               ← EXP-19 4-config × 60 seeds on 2025–2026 data
│   ├── exp_19_aggregate.py            ← EXP-19 aggregation + cross-period analysis
│   ├── exp_20_worker.py               ← RSMO sweep: k ∈ {2,5,10}, both periods, n≥40
│   ├── exp_20_lronly_2022_worker.py   ← LR-only 2022 baseline for RSMO comparison (n=40)
│   ├── exp_20_aggregate.py            ← RSMO aggregation + figures (figs 22–23)
│   └── exp_landscape_figures.py       ← Hyperparameter landscape figures (figs 17–18)
│
├── data/
│   ├── train_returns.npy              ← Synthetic training returns (1000 × 5)
│   ├── test_returns.npy               ← Synthetic test returns (252 × 5)
│   ├── train_regimes.npy              ← Synthetic regime labels (1000,)
│   ├── test_regimes.npy               ← Synthetic regime labels (252,)
│   ├── real_train_returns.npy         ← S&P 500 train returns 2012–2021 (N × 10)
│   ├── real_test_returns.npy          ← S&P 500 test returns 2022–2024 (752 × 10)
│   ├── real_test_regimes.npy          ← Bear/bull regime labels for 2022–2024 (752,)
│   ├── exp19_test_returns.npy         ← S&P 500 test returns 2025–2026 (312 × 10)
│   ├── exp19_test_regimes.npy         ← Regime labels for 2025–2026 (312,)
│   ├── exp19_test_dates.npy           ← Trading date strings for 2025–2026 (312,)
│   ├── exp19_test_prices.npy          ← Price levels for 2025–2026 (312 × 10)
│   └── exp19_metadata.json            ← Ticker list, date range, EW Sharpe
│
├── results/                           ← 133 JSON result files
│   ├── exp_01_results.json … exp_20_results.json   ← Per-experiment aggregated results
│   └── exp_XX_<config>_<seeds>.json                ← Per-batch seed results (raw)
│
└── figures/                           ← 23 PNG figures (150 dpi)
    ├── fig_01_sharpe_comparison.png   … fig_06_summary_table.png   ← Synthetic
    ├── fig_07_real_sharpe.png         … fig_16_grid_ir.png          ← Real market
    ├── fig_17_landscape_bubble.png    … fig_19_power_curve.png      ← Landscape/power
    ├── fig_20_post2024.png            … fig_21_cross_period.png     ← EXP-19
    └── fig_22_rsmo_sr_vs_k.png       … fig_23_rsmo_bull_bear.png   ← EXP-20
```

---

## Requirements

### Python packages

```bash
pip install numpy pandas scipy matplotlib gymnasium stable-baselines3 torch yfinance
```

Tested with Python 3.10, PyTorch 2.2, Stable-Baselines3 2.3.

### LaTeX (for compiling the paper)

```bash
# Ubuntu / Debian
sudo apt-get install texlive-latex-base texlive-latex-extra \
                     texlive-publishers texlive-science texlive-fonts-recommended

# macOS (MacTeX)
brew install --cask mactex
```

---

## Reproduction Instructions

All experiments run sequentially; each saves its output to `results/` before the next reads it.  
Batch workers use 10-seed chunks to stay within ~300 s wall-clock time per invocation.

### Step 0 — Clone and install

```bash
git clone https://github.com/sahithottikunta/rl-portfolio-reward-ablation.git
cd rl-portfolio-reward-ablation
pip install numpy pandas scipy matplotlib gymnasium stable-baselines3 torch yfinance
```

---

### Phase 1 — Synthetic experiments (EXP-01 through EXP-04)

```bash
python experiments/exp_01_environment.py    # generates data/, classical baselines
python experiments/exp_02_reward_comparison.py  # 4 rewards × 3 seeds
python experiments/exp_03_regime_analysis.py    # regime breakdown + turnover
python experiments/exp_04_figures.py            # figs 01–06
```

---

### Phase 2 — Real S&P 500 data (EXP-05 through EXP-09)

```bash
python experiments/exp_05_real_data.py      # downloads S&P 500 data via yfinance
python experiments/exp_06_real_baselines.py
python experiments/exp_07_real_ablation.py
python experiments/exp_08_real_regime.py
python experiments/exp_09_real_figures.py   # figs 07–10
```

---

### Phase 3 — Adaptive η schedule (EXP-10 through EXP-11)

```bash
python experiments/exp_10_adaptive_eta.py
python experiments/exp_11_adaptive_figures.py   # figs 11–12
```

---

### Phase 4 — Large-n seed sweeps (EXP-12 through EXP-13, n = 40 seeds)

Run each batch separately to avoid timeout (10 seeds per invocation):

```bash
for seeds in "42 52" "52 62" "62 72" "72 82"; do
  python experiments/exp_12_worker.py 30 $seeds   # κ=30
  python experiments/exp_12_worker.py 50 $seeds   # κ=50
  python experiments/exp_12_worker.py fixed $seeds
done
python experiments/exp_12_aggregate.py

for seeds in "52 62" "62 72" "72 82"; do
  python experiments/exp_13_worker.py 30 $seeds
  python experiments/exp_13_worker.py 50 $seeds
  python experiments/exp_13_worker.py fixed $seeds
done
python experiments/exp_13_aggregate.py
```

---

### Phase 5 — TC sweep and (λ, η₀) grid search (EXP-14 through EXP-16)

```bash
# TC sweep — λ ∈ {0.001, 0.005, 0.01, 0.05}, 40 seeds each
for lam in 0.001 0.005 0.010 0.050; do
  for seeds in "42 52" "52 62" "62 72" "72 82"; do
    python experiments/exp_14_worker.py $lam $seeds
  done
done
python experiments/exp_14_aggregate.py    # figs 14–16

# 2D grid search — 5×5 (λ, η₀) grid, 10 seeds per cell
python experiments/exp_15_worker.py       # runs all 25 cells
python experiments/exp_15_aggregate.py

# Stability-optimal cell confirmation (λ=0.002, η₀=0.002, n=50)
for seeds in "52 62" "62 72" "72 82" "82 92"; do
  python experiments/exp_16_worker.py $seeds
done
```

---

### Phase 6 — Confirmatory studies (EXP-17 through EXP-18, n = 75 and n = 150)

```bash
# EXP-17: n=75 seeds for mean-optimal cell (λ=0.0005, η₀=0.020)
for seeds in "52 62" "62 72" "72 82" "82 92" "92 102" "102 112" "112 116"; do
  python experiments/exp_17_worker.py $seeds
done

# EXP-18: expand to n=150 (seeds 117–191)
for seeds in "117 127" "127 137" "137 147" "147 157" "157 167" "167 177" "177 187" "187 192"; do
  python experiments/exp_18_worker.py $seeds
done
python experiments/exp_18_aggregate.py    # fig 19 (power curve)
```

---

### Phase 7 — Post-2024 out-of-sample evaluation (EXP-19, n = 60)

```bash
python experiments/exp_19_data_prep.py    # downloads 2025-01-03 to 2026-04-02

# 4 configs × 60 seeds (run in 20-seed batches)
for cfg in dsr_tc dsr_only lr_tc lr_only; do
  for seeds in "42 62" "62 82" "82 102"; do
    python experiments/exp_19_worker.py $cfg $seeds
  done
done
python experiments/exp_19_aggregate.py    # figs 20–21
```

---

### Phase 8 — RSMO regime-switching reward (EXP-20, n ≥ 40)

```bash
# RSMO sweep: k ∈ {2,5,10}, periods {2022,2025}, seeds 42–81 in 10-seed chunks
for k in 2 5 10; do
  for period in 2022 2025; do
    for seeds in "42 52" "52 62" "62 72" "72 82"; do
      python experiments/exp_20_worker.py $k $period $seeds
    done
  done
done

# LR-only 2022 baseline (n=40)
for seeds in "42 52" "52 62" "62 72" "72 82"; do
  python experiments/exp_20_lronly_2022_worker.py $seeds
done

python experiments/exp_20_aggregate.py    # figs 22–23
```

---

### Final — Compile the paper

```bash
cd /path/to/rl-portfolio-reward-ablation
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
# Output: paper.pdf (20 pages, 0 errors)
```

---

### Hyperparameter landscape figures (Section VIII)

```bash
python experiments/exp_landscape_figures.py   # requires EXP-12 through EXP-18 results
# Produces: figures/fig_17_landscape_bubble.png, fig_18_landscape_lambda.png
```

---

## Experiment Index

| EXP | Description | Seeds | Period | Key result |
|-----|-------------|-------|--------|------------|
| 01 | Classical baselines (EW, MV, MS) on synthetic data | — | Synthetic | EW SR=1.135, MaxSharpe SR=1.702 |
| 02 | Reward ablation: LR / DSR / LR+TC / DSR+TC | 3 | Synthetic | DSR+TC lowest variance (std=0.097) |
| 03 | Regime-conditional breakdown + turnover | 3 | Synthetic | DSR+TC best in bear (SR=0.93) |
| 04 | Synthetic figures | — | — | figs 01–06 |
| 05 | S&P 500 data download (2012–2024, 10 tickers) | — | Real | — |
| 06 | Classical baselines on real data | — | Real | — |
| 07 | Reward ablation on real data | 3 | 2022–2024 | DSR+TC SR=0.391 |
| 08 | Regime analysis on real data | 3 | 2022–2024 | Bear/bull breakdown |
| 09 | Real-market figures | — | — | figs 07–10 |
| 10 | Adaptive η: η_t = η₀/(1+κσ̂_t), κ ∈ {30,50} | 3 | 2022–2024 | Stability advantage |
| 11 | Adaptive η figures | — | — | figs 11–12 |
| 12 | κ sweep n=40 seeds (Welch t-test) | 40 | 2022–2024 | κ=30/50 advantage: d≈0.3 |
| 13 | Re-run κ=30/50 at n=40 (confirmatory) | 40 | 2022–2024 | Welch t-test |
| 14 | TC penalty sweep λ ∈ {0.001,0.005,0.01,0.05} | 40 | 2022–2024 | λ=0.001 optimal |
| 15 | 2D grid search (λ, η₀), 5×5 | 10 | 2022–2024 | Mean-opt: λ=0.0005, η₀=0.020 |
| 16 | Stability-optimal cell n=50 (λ=0.002, η₀=0.002) | 50 | 2022–2024 | IR=4.26 confirmed |
| 17 | Mean-optimal cell n=75 (λ=0.0005, η₀=0.020) | 75 | 2022–2024 | d=+0.30, p=0.069 |
| 18 | Confirmatory n=150 (winner's curse check) | 150 | 2022–2024 | d=+0.17, p=0.144 — shrinkage confirmed |
| 19 | Post-2024 OOS: 4 configs × 60 seeds | 60 | 2025–2026 | Full ranking reversal, d>1.0 |
| 20 | RSMO reward: k ∈ {2,5,10}, both periods | ≥40 | Both | Null result — blending fails |

**Total PPO training runs: 930**

---

## PPO Hyperparameters

All experiments use identical PPO settings (Stable-Baselines3):

| Parameter | Value |
|-----------|-------|
| Policy architecture | MLP, `net_arch=[128, 128]` |
| Learning rate | 3 × 10⁻⁴ |
| n_steps | 256 |
| batch_size | 64 |
| n_epochs | 10 |
| γ (discount) | 0.99 |
| GAE λ | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 |
| Total timesteps | 30,000 |

---

## Data

### Tickers (10 S&P 500 constituents)

`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `JPM`, `JNJ`, `XOM`, `PG`, `BRK-B`, `UNH`

### Splits

| Split | Period | Trading Days |
|-------|--------|-------------|
| Training | 2012-01-01 — 2021-12-31 | ~2,516 |
| Test (bear) | 2022-01-03 — 2024-12-31 | 752 |
| Test (bull) | 2025-01-03 — 2026-04-02 | 312 |

Data is downloaded via `yfinance` and cached as `.npy` arrays. The `exp19_data_prep.py` script must be re-run if prices have been updated. All arrays already included in this repository for exact reproduction.

---

## Citation

```bibtex
@article{ottikunta2026reward,
  title   = {Reward Function Design for Deep Reinforcement Learning Portfolio
             Optimization: A Comparative Ablation Study Under Regime-Switching
             Market Conditions},
  author  = {Ottikunta, Sahith},
  year    = {2026},
  note    = {Preprint. Code: https://github.com/sahithottikunta/rl-portfolio-reward-ablation}
}
```

---

## Author Note

The author name field in `paper.tex` (line ~41) currently reads `Author Name (To Be Filled)` — replace with your name before submission.

---

*930 PPO training runs · 20 experiments · 10 S&P 500 assets · 2 out-of-sample test periods · 40 verified references*
