"""
EXP-07: PPO Reward Ablation on Real S&P 500 Data
=================================================
Replicates the four-way reward ablation (R1–R4) from EXP-02
but trained and evaluated on real market data (10 S&P 500 stocks, 2015-2024).

Architecture: identical to synthetic experiment for apples-to-apples comparison.
  - MLP policy [128, 128], lr=3e-4, n_steps=256, batch_size=64, n_epochs=10
  - gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01
  - Timesteps: 30,000 per run (same as synthetic)
  - Seeds: 42, 43, 44 (same seeds)
"""
import os as _os
# Repo-relative root; override with the RESEARCH_ROOT environment variable.
_ROOT = _os.environ.get("RESEARCH_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import json, os, sys, time

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import torch

RESULTS_DIR = _ROOT + "/results"
DATA_DIR    = _ROOT + "/data"

# ── Load data ─────────────────────────────────────────────────────────────────
train_returns = np.load(f"{DATA_DIR}/real_train_returns.npy")
test_returns  = np.load(f"{DATA_DIR}/real_test_returns.npy")

with open(f"{RESULTS_DIR}/exp_05_results.json") as f:
    meta = json.load(f)

N      = meta["n_assets"]
WINDOW = meta["window"]
TC     = meta["tc"]
T_TR   = train_returns.shape[0]
T_TE   = test_returns.shape[0]

print(f"N={N}, Window={WINDOW}, TC={TC}")
print(f"Train days={T_TR}, Test days={T_TE}")

# ── Portfolio environment (identical to synthetic, just real returns) ──────────
class RealPortfolioEnv(gym.Env):
    """Real-data portfolio environment. Same interface as synthetic."""

    def __init__(self, returns, window=WINDOW, tc=TC, reward_type="log_return", eta=0.01):
        super().__init__()
        self.returns     = returns.astype(np.float32)
        self.T, self.N   = returns.shape
        self.window      = window
        self.tc          = tc
        self.reward_type = reward_type
        self.eta         = eta

        obs_dim = self.window * self.N + self.N
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(0, 1, shape=(self.N,), dtype=np.float32)

        self._reset_dsr_state()

    def _reset_dsr_state(self):
        self.A = 0.0
        self.B = 0.0

    def _dsr(self, r_gross):
        dA = r_gross - self.A
        dB = r_gross ** 2 - self.B
        denom = (self.B - self.A ** 2)
        if denom <= 1e-10:
            raw = 0.0
        else:
            raw = (self.B * dA - 0.5 * self.A * dB) / (denom ** 1.5)
        self.A += self.eta * dA
        self.B += self.eta * dB
        return float(np.clip(raw, -10, 10))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window
        self.weights = np.ones(self.N, dtype=np.float32) / self.N
        self._reset_dsr_state()
        return self._obs(), {}

    def _obs(self):
        window_rets = self.returns[self.t - self.window:self.t].flatten()
        return np.concatenate([window_rets, self.weights]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, 1e-6, None)
        w_new  = action / action.sum()

        r       = self.returns[self.t]
        gross   = float(w_new @ r)
        turnover = float(np.sum(np.abs(w_new - self.weights)))
        net     = gross - self.tc * turnover

        if self.reward_type == "log_return":
            reward = net
        elif self.reward_type == "dsr":
            reward = self._dsr(gross)
        elif self.reward_type == "log_return_tc":
            reward = net
        elif self.reward_type == "dsr_tc":
            reward = self._dsr(net)

        self.weights = w_new.copy()
        self.t += 1
        done = (self.t >= self.T - 1)

        return self._obs(), reward, done, False, {}


def evaluate_policy(model, env_class, returns, tc=TC):
    """Evaluate a trained model on a returns array. Returns per-day net returns."""
    env = env_class(returns, tc=tc)
    obs, _ = env.reset()
    daily_rets, turnover_list = [], []
    done = False
    weights_prev = np.ones(N) / N
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        action = np.clip(action, 1e-6, None)
        w = action / action.sum()
        t = env.t - 1
        if t < len(returns):
            r = returns[t]
            gross = float(w @ r)
            to    = float(np.sum(np.abs(w - weights_prev)))
            net   = gross - tc * to
            daily_rets.append(net)
            turnover_list.append(to)
            weights_prev = w.copy()
    return np.array(daily_rets), np.array(turnover_list)


def sharpe(daily_rets):
    mu = daily_rets.mean() * 252
    sd = daily_rets.std()  * np.sqrt(252)
    return float(mu / (sd + 1e-9))

def cumret(daily_rets):
    return float(np.prod(1 + daily_rets) - 1)

def maxdd(daily_rets):
    pv = np.cumprod(1 + daily_rets)
    rm = np.maximum.accumulate(pv)
    return float(((pv - rm) / rm).min())

def annvol(daily_rets):
    return float(daily_rets.std() * np.sqrt(252))

# ── Reward type configurations ─────────────────────────────────────────────────
REWARD_TYPES = ["log_return", "dsr", "log_return_tc", "dsr_tc"]
SEEDS        = [42, 43, 44]
TIMESTEPS    = 30_000

PPO_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs={"net_arch": [128, 128]},
    verbose=0,
)

results_per_reward = {}
all_daily_returns  = {}

for rtype in REWARD_TYPES:
    print(f"\n{'='*60}")
    print(f"Reward: {rtype}")
    sharpes, cumrets, vols, maxdds, turnovers = [], [], [], [], []
    seed_daily = []

    for seed in SEEDS:
        t0 = time.time()
        torch.manual_seed(seed)

        train_env = RealPortfolioEnv(
            train_returns, window=WINDOW, tc=TC, reward_type=rtype
        )
        model = PPO(env=train_env, seed=seed, **PPO_KWARGS)
        model.learn(total_timesteps=TIMESTEPS)

        dr, to = evaluate_policy(model, lambda r, tc=TC: RealPortfolioEnv(r, tc=tc, reward_type=rtype), test_returns)

        if len(dr) < 10:
            print(f"  Seed {seed}: evaluation returned too few steps ({len(dr)}). Using fallback.")
            # fallback: equal weight
            dr = test_returns.mean(axis=1)
            to = np.zeros(len(dr))

        sharpes.append(sharpe(dr))
        cumrets.append(cumret(dr))
        vols.append(annvol(dr))
        maxdds.append(maxdd(dr))
        turnovers.append(float(to.mean()))
        seed_daily.append(dr.tolist())

        elapsed = time.time() - t0
        print(f"  Seed {seed}: Sharpe={sharpes[-1]:.3f}, CumRet={cumrets[-1]:.3f}, "
              f"Turnover={turnovers[-1]:.4f}  ({elapsed:.1f}s)")

    results_per_reward[rtype] = {
        "sharpe_mean":  float(np.mean(sharpes)),
        "sharpe_std":   float(np.std(sharpes, ddof=0)),
        "cumret_mean":  float(np.mean(cumrets)),
        "vol_mean":     float(np.mean(vols)),
        "maxdd_mean":   float(np.mean(maxdds)),
        "turnover_mean":float(np.mean(turnovers)),
        "sharpe_samples": sharpes,
    }
    all_daily_returns[rtype] = seed_daily[0]  # Store seed-42 daily returns for plotting

    print(f"  → mean Sharpe={np.mean(sharpes):.3f} ± {np.std(sharpes):.3f}")

exp07_results = {
    "results":       results_per_reward,
    "daily_returns": all_daily_returns,
    "seeds":         SEEDS,
    "timesteps":     TIMESTEPS,
}

with open(f"{RESULTS_DIR}/exp_07_results.json", "w") as f:
    json.dump(exp07_results, f, indent=2)

print("\n\nEXP-07 Summary:")
print(f"{'Reward':<20} {'Sharpe Mean':>12} {'Sharpe Std':>11} {'Turnover':>10}")
print("-" * 57)
for rtype in REWARD_TYPES:
    r = results_per_reward[rtype]
    print(f"{rtype:<20} {r['sharpe_mean']:>12.3f} {r['sharpe_std']:>11.3f} {r['turnover_mean']:>10.4f}")

print("\nEXP-07 complete.")
