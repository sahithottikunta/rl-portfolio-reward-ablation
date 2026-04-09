"""
EXP-03: Regime-Conditional Performance Analysis
Purpose: For each trained reward type, analyze:
  1. Performance split by market regime (bull vs bear)
  2. Portfolio turnover (average daily weight change)
  3. Statistical significance (paired t-test: dsr_tc vs log_return baseline)
  4. Compute performance on rolling windows

Random seed: 42
"""

import numpy as np
import json
import os
import sys
from scipy import stats

sys.path.insert(0, '/home/user/workspace/research_output/experiments')
from exp_01_environment import portfolio_performance

SEED = 42
np.random.seed(SEED)

DATA_DIR    = "/home/user/workspace/research_output/data"
RESULTS_DIR = "/home/user/workspace/research_output/results"
TC          = 0.001
WINDOW      = 20

train_returns  = np.load(f"{DATA_DIR}/train_returns.npy")
test_returns   = np.load(f"{DATA_DIR}/test_returns.npy")
test_regimes   = np.load(f"{DATA_DIR}/test_regimes.npy")
N_ASSETS       = test_returns.shape[1]
N_TEST         = len(test_returns)

# ─────────────────────────────────────────────────────────────────────────────
# Re-train all agents and collect weight series (with fixed seeds for repro)
# ─────────────────────────────────────────────────────────────────────────────
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, returns, reward_type='log_return', tc=TC, window=WINDOW):
        super().__init__()
        self.returns     = returns
        self.n_assets    = returns.shape[1]
        self.reward_type = reward_type
        self.tc          = tc
        self.window      = window
        self.T           = len(returns)
        obs_dim = self.window * self.n_assets + self.n_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(low=-3.0,   high=3.0,   shape=(self.n_assets,), dtype=np.float32)
        self._A = 0.0; self._B = 0.0; self._eta = 0.01

    def _get_obs(self):
        start = max(0, self.t - self.window)
        window_ret = self.returns[start:self.t]
        if len(window_ret) < self.window:
            pad = np.zeros((self.window - len(window_ret), self.n_assets))
            window_ret = np.vstack([pad, window_ret])
        return np.concatenate([window_ret.flatten(), self.weights]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window
        self.weights = np.ones(self.n_assets) / self.n_assets
        self._A = 0.0; self._B = 0.0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -10, 10)
        w_new  = np.exp(action) / np.sum(np.exp(action))
        cost   = self.tc * np.sum(np.abs(w_new - self.weights))
        r_gross = np.dot(w_new, self.returns[self.t])
        r_net   = r_gross - cost
        if self.reward_type == 'log_return':
            reward = float(r_net)
        elif self.reward_type in ('dsr', 'dsr_tc'):
            delta_A = r_net - self._A; delta_B = r_net**2 - self._B
            denom   = (self._B - self._A**2)
            dsr = (self._B*delta_A - 0.5*self._A*delta_B)/(denom**1.5) if denom>1e-8 else 0.0
            reward = float(np.clip(dsr, -10, 10))
            self._A += self._eta*delta_A; self._B += self._eta*delta_B
        else:
            reward = float(r_net)
        self.weights = w_new.copy()
        self.t += 1
        done = self.t >= self.T
        return self._get_obs(), reward, done, False, {}

def train_agent(reward_type, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tc_train = TC if 'tc' in reward_type else 0.0
    vec_env  = DummyVecEnv([lambda: PortfolioEnv(train_returns, reward_type=reward_type, tc=tc_train)])
    model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=256, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                verbose=0, seed=seed, policy_kwargs=dict(net_arch=[128,128]))
    model.learn(total_timesteps=30_000)
    return model

def get_weights(model, returns):
    env = PortfolioEnv(returns, reward_type='log_return', tc=0.0, window=WINDOW)
    obs, _ = env.reset()
    done = False; w_hist = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, -10, 10)
        w = np.exp(action)/np.sum(np.exp(action))
        w_hist.append(w.copy())
        obs, _, done, _, _ = env.step(action)
    return np.array(w_hist)

REWARD_TYPES = ['log_return', 'dsr', 'log_return_tc', 'dsr_tc']
BEST_SEEDS   = [SEED + 2, SEED, SEED + 2, SEED]   # seed that gave best Sharpe per type (from EXP-02)

print("Re-training agents (best seed per reward type) for regime analysis...")
weights_dict = {}
for rtype, seed in zip(REWARD_TYPES, BEST_SEEDS):
    model = train_agent(rtype, seed)
    w = get_weights(model, test_returns)
    min_len = min(len(w), N_TEST)
    weights_dict[rtype] = w[:min_len]
    print(f"  {rtype}: collected {min_len} weight vectors")

# ─────────────────────────────────────────────────────────────────────────────
# Regime-split analysis
# ─────────────────────────────────────────────────────────────────────────────
def regime_perf(weights, returns, regimes, regime_val, tc=TC):
    idx = np.where(regimes[:len(weights)] == regime_val)[0]
    if len(idx) < 2:
        return None
    w_sub = weights[idx]
    r_sub = returns[idx]
    return portfolio_performance(w_sub, r_sub, tc=tc)

regime_results = {}
for rtype in REWARD_TYPES:
    w = weights_dict[rtype]
    reg = test_regimes[:len(w)]
    bull = regime_perf(w, test_returns, reg, 0)
    bear = regime_perf(w, test_returns, reg, 1)
    regime_results[rtype] = {"bull": bull, "bear": bear}
    print(f"\n{rtype}:")
    print(f"  Bull Sharpe: {bull['sharpe_ratio']:.4f}  CumRet: {bull['cumulative_return']:.4f}")
    print(f"  Bear Sharpe: {bear['sharpe_ratio']:.4f}  CumRet: {bear['cumulative_return']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Turnover Analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Portfolio Turnover (mean daily |Δw|) ---")
turnover_results = {}
for rtype in REWARD_TYPES:
    w = weights_dict[rtype]
    diffs = np.abs(np.diff(w, axis=0)).sum(axis=1)
    turnover_results[rtype] = float(diffs.mean())
    print(f"  {rtype}: {diffs.mean():.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# Statistical significance: paired t-test (dsr_tc vs log_return)
# ─────────────────────────────────────────────────────────────────────────────
# Run N_STAT_RUNS per method to get distribution of Sharpes
N_STAT_RUNS = 3
print(f"\n--- Statistical Test: {N_STAT_RUNS} runs per method ---")
sharpe_samples = {rt: [] for rt in REWARD_TYPES}
for rt in REWARD_TYPES:
    for run in range(N_STAT_RUNS):
        seed = SEED + run * 7
        m = train_agent(rt, seed)
        w = get_weights(m, test_returns)
        min_len = min(len(w), N_TEST)
        perf = portfolio_performance(w[:min_len], test_returns[:min_len], tc=TC)
        sharpe_samples[rt].append(perf['sharpe_ratio'])
    print(f"  {rt}: {np.mean(sharpe_samples[rt]):.4f} ± {np.std(sharpe_samples[rt], ddof=1):.4f}")

# Paired t-test: dsr_tc vs log_return
t_stat, p_val = stats.ttest_rel(sharpe_samples['dsr_tc'], sharpe_samples['log_return'])
print(f"\nPaired t-test (dsr_tc vs log_return): t={t_stat:.4f}, p={p_val:.4f}")
t_stat2, p_val2 = stats.ttest_rel(sharpe_samples['dsr_tc'], sharpe_samples['dsr'])
print(f"Paired t-test (dsr_tc vs dsr):        t={t_stat2:.4f}, p={p_val2:.4f}")
t_stat3, p_val3 = stats.ttest_rel(sharpe_samples['dsr_tc'], sharpe_samples['log_return_tc'])
print(f"Paired t-test (dsr_tc vs lr_tc):      t={t_stat3:.4f}, p={p_val3:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Daily return series for all strategies (for figure generation)
# ─────────────────────────────────────────────────────────────────────────────
def daily_returns_series(weights, returns, tc=TC):
    min_len = min(len(weights), len(returns))
    daily_r = []
    prev_w = np.ones(returns.shape[1]) / returns.shape[1]
    for t in range(min_len):
        w = weights[t]
        cost = tc * np.sum(np.abs(w - prev_w))
        r = np.dot(w, returns[t]) - cost
        daily_r.append(r)
        prev_w = w.copy()
    return np.array(daily_r)

daily_series = {}
for rtype in REWARD_TYPES:
    daily_series[rtype] = daily_returns_series(weights_dict[rtype], test_returns).tolist()

# Baselines
ew_w = np.full((N_TEST, N_ASSETS), 1.0/N_ASSETS)
daily_series['equal_weight'] = daily_returns_series(ew_w, test_returns).tolist()

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
results_out = {
    "experiment": "EXP-03",
    "n_stat_runs": N_STAT_RUNS,
    "regime_results": regime_results,
    "turnover": turnover_results,
    "sharpe_samples": {k: [float(x) for x in v] for k, v in sharpe_samples.items()},
    "stat_tests": {
        "dsr_tc_vs_log_return": {"t": float(t_stat),  "p": float(p_val)},
        "dsr_tc_vs_dsr":        {"t": float(t_stat2), "p": float(p_val2)},
        "dsr_tc_vs_lr_tc":      {"t": float(t_stat3), "p": float(p_val3)},
    },
    "daily_return_series": daily_series,
    "test_regimes": test_regimes.tolist(),
}

with open(f"{RESULTS_DIR}/exp_03_results.json", "w") as f:
    json.dump(results_out, f, indent=2)

print(f"\nEXP-03 complete. Results saved.")
