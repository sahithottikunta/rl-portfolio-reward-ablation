"""
EXP-02: Reward Function Ablation Study
Purpose: Compare four reward functions for PPO-based portfolio agent:
  R1 — Log return (simple return)
  R2 — Differential Sharpe ratio (DSR)
  R3 — Log return + transaction cost penalty
  R4 — DSR + transaction cost penalty (our proposed reward)

The gap we address: no prior work systematically isolates the JOINT effect of
differential Sharpe ratio AND transaction costs in a single reward function
on training stability and out-of-sample Sharpe for a multi-asset regime-switching market.

Random seed: 42
"""
import os as _os
# Repo-relative root; override with the RESEARCH_ROOT environment variable.
_ROOT = _os.environ.get("RESEARCH_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import json, os

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Load data from EXP-01
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = _ROOT + "/data"
train_returns = np.load(f"{DATA_DIR}/train_returns.npy")
test_returns  = np.load(f"{DATA_DIR}/test_returns.npy")
N_ASSETS      = train_returns.shape[1]
WINDOW        = 20      # lookback window for state features
TC            = 0.001   # transaction cost rate

# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Gym Environment
# ─────────────────────────────────────────────────────────────────────────────
class PortfolioEnv(gym.Env):
    """
    Multi-asset portfolio environment with configurable reward function.
    State: flattened (WINDOW × N_ASSETS) return matrix + current weights
    Action: raw scores → softmax → portfolio weights (long-only, sum=1)
    """
    metadata = {'render_modes': []}

    def __init__(self, returns, reward_type='log_return', tc=TC, window=WINDOW):
        super().__init__()
        self.returns     = returns
        self.n_assets    = returns.shape[1]
        self.reward_type = reward_type
        self.tc          = tc
        self.window      = window
        self.T           = len(returns)

        # observation: flattened window of returns + current weights
        obs_dim = self.window * self.n_assets + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # action: raw logits → softmax
        self.action_space = spaces.Box(
            low=-3.0, high=3.0, shape=(self.n_assets,), dtype=np.float32
        )
        # DSR state variables
        self._A = 0.0
        self._B = 0.0
        self._eta = 0.01  # DSR adaptation rate

    def _get_obs(self):
        start = max(0, self.t - self.window)
        window_ret = self.returns[start:self.t]
        if len(window_ret) < self.window:
            pad = np.zeros((self.window - len(window_ret), self.n_assets))
            window_ret = np.vstack([pad, window_ret])
        return np.concatenate([window_ret.flatten(), self.weights]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t       = self.window
        self.weights = np.ones(self.n_assets) / self.n_assets
        self._A      = 0.0
        self._B      = 0.0
        self.portfolio_value = 1.0
        return self._get_obs(), {}

    def step(self, action):
        # softmax to get valid weights
        action = np.clip(action, -10, 10)
        w_new  = np.exp(action) / np.sum(np.exp(action))

        # transaction cost
        cost   = self.tc * np.sum(np.abs(w_new - self.weights))
        r_gross = np.dot(w_new, self.returns[self.t])
        r_net   = r_gross - cost

        # ---- Reward computation ----
        if self.reward_type == 'log_return':
            reward = float(r_net)

        elif self.reward_type == 'dsr':
            # Differential Sharpe Ratio (Moody & Saffell, 2001)
            delta_A = r_net - self._A
            delta_B = r_net**2 - self._B
            denom   = (self._B - self._A**2)
            if denom > 1e-8:
                dsr = (self._B * delta_A - 0.5 * self._A * delta_B) / (denom ** 1.5)
            else:
                dsr = 0.0
            reward = float(np.clip(dsr, -10, 10))
            self._A += self._eta * delta_A
            self._B += self._eta * delta_B

        elif self.reward_type == 'log_return_tc':
            reward = float(r_net)   # r_net already includes TC cost

        elif self.reward_type == 'dsr_tc':
            # DSR with transaction costs baked into r_net
            delta_A = r_net - self._A
            delta_B = r_net**2 - self._B
            denom   = (self._B - self._A**2)
            if denom > 1e-8:
                dsr = (self._B * delta_A - 0.5 * self._A * delta_B) / (denom ** 1.5)
            else:
                dsr = 0.0
            reward = float(np.clip(dsr, -10, 10))
            self._A += self._eta * delta_A
            self._B += self._eta * delta_B

        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        self.weights = w_new.copy()
        self.portfolio_value *= (1 + r_net)
        self.t += 1
        done = self.t >= self.T
        return self._get_obs(), reward, done, False, {}

# ─────────────────────────────────────────────────────────────────────────────
# Training callback — track episode rewards
# ─────────────────────────────────────────────────────────────────────────────
class RewardTracker(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._ep_reward = 0.0
    def _on_step(self):
        self._ep_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
        return True

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation function
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_agent(model, returns, tc=TC, window=WINDOW):
    """Run trained model on returns, collect weights, compute metrics."""
    n_assets = returns.shape[1]
    T        = len(returns)
    weights_hist = []
    obs_hist     = []

    env = PortfolioEnv(returns, reward_type='log_return', tc=0.0, window=window)
    obs, _ = env.reset()
    done   = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # softmax weights
        action = np.clip(action, -10, 10)
        w = np.exp(action) / np.sum(np.exp(action))
        weights_hist.append(w.copy())
        obs, _, done, _, _ = env.step(action)

    weights_hist = np.array(weights_hist)

    # compute metrics with actual TC
    from exp_01_environment import portfolio_performance
    return portfolio_performance(weights_hist, returns, tc=tc)

# ─────────────────────────────────────────────────────────────────────────────
# Main: train & evaluate all four reward functions
# ─────────────────────────────────────────────────────────────────────────────
REWARD_TYPES   = ['log_return', 'dsr', 'log_return_tc', 'dsr_tc']
TOTAL_TIMESTEPS = 30_000
N_RUNS          = 3   # multiple runs for statistical robustness
RESULTS_DIR     = _ROOT + "/results"

all_results = {}
training_curves = {}

for rtype in REWARD_TYPES:
    print(f"\n{'='*60}")
    print(f"Training PPO with reward: {rtype}")
    print(f"{'='*60}")
    run_sharpes   = []
    run_cum_rets  = []
    run_max_dds   = []
    run_vols      = []
    run_curves    = []

    for run_id in range(N_RUNS):
        seed = SEED + run_id
        np.random.seed(seed)
        torch.manual_seed(seed)

        def make_env():
            e = PortfolioEnv(train_returns, reward_type=rtype,
                             tc=TC if 'tc' in rtype else 0.0)
            return e

        vec_env  = DummyVecEnv([make_env])
        callback = RewardTracker()

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            seed=seed,
            policy_kwargs=dict(net_arch=[128, 128])
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

        # Evaluate on test set
        import sys
        sys.path.insert(0, _ROOT + '/experiments')
        from exp_01_environment import portfolio_performance

        # collect weights on test env
        test_env = PortfolioEnv(test_returns, reward_type='log_return', tc=0.0, window=WINDOW)
        obs, _ = test_env.reset()
        done   = False
        w_hist = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(action, -10, 10)
            w = np.exp(action) / np.sum(np.exp(action))
            w_hist.append(w.copy())
            obs, _, done, _, _ = test_env.step(action)

        w_hist = np.array(w_hist)
        # Trim to the shorter length to handle off-by-one from env reset window
        min_len = min(len(w_hist), len(test_returns))
        w_hist  = w_hist[:min_len]
        r_hist  = test_returns[:min_len]
        perf   = portfolio_performance(w_hist, r_hist, tc=TC)

        run_sharpes.append(perf['sharpe_ratio'])
        run_cum_rets.append(perf['cumulative_return'])
        run_max_dds.append(perf['max_drawdown'])
        run_vols.append(perf['annualized_volatility'])
        run_curves.append(callback.episode_rewards[:])

        print(f"  Run {run_id+1}: Sharpe={perf['sharpe_ratio']:.4f}  "
              f"CumRet={perf['cumulative_return']:.4f}  "
              f"MaxDD={perf['max_drawdown']:.4f}")

    all_results[rtype] = {
        "sharpe_mean":   float(np.mean(run_sharpes)),
        "sharpe_std":    float(np.std(run_sharpes, ddof=1)),
        "cumret_mean":   float(np.mean(run_cum_rets)),
        "cumret_std":    float(np.std(run_cum_rets, ddof=1)),
        "maxdd_mean":    float(np.mean(run_max_dds)),
        "vol_mean":      float(np.mean(run_vols)),
        "all_sharpes":   [float(x) for x in run_sharpes],
    }
    # Save last run's curve
    training_curves[rtype] = [float(x) for x in (run_curves[-1] if run_curves[-1] else [])]

print("\n{'='*60}")
print("SUMMARY — Out-of-sample Sharpe Ratio")
print("{'='*60}")
for rtype, res in all_results.items():
    print(f"{rtype:20s}: {res['sharpe_mean']:.4f} ± {res['sharpe_std']:.4f}")

results_out = {
    "experiment": "EXP-02",
    "reward_types": REWARD_TYPES,
    "total_timesteps": TOTAL_TIMESTEPS,
    "n_runs": N_RUNS,
    "seed_base": SEED,
    "transaction_cost": TC,
    "results": all_results,
    "training_curves": training_curves,
}
with open(f"{RESULTS_DIR}/exp_02_results.json", "w") as f:
    json.dump(results_out, f, indent=2)

print(f"\nEXP-02 complete. Results saved to {RESULTS_DIR}/exp_02_results.json")
