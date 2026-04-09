"""
LR-only baseline on 2022 test period (for RSMO comparison).
Train on real_train_returns.npy, test on real_test_returns.npy (2022-2024).
R_t = log(1 + net_portfolio_return_t)  [no TC, no DSR]
Usage: python3 exp_20_lronly_2022_worker.py <seed_start> <seed_end_exclusive>
Saves: results/exp_20_lronly_2022_{seed_start}_{seed_end-1}.json
"""
import sys, json, time, os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

SEED_START = int(sys.argv[1]) if len(sys.argv) > 1 else 42
SEED_END   = int(sys.argv[2]) if len(sys.argv) > 2 else 52
SEEDS      = list(range(SEED_START, SEED_END))

RESULTS_DIR = "/home/user/workspace/research_output/results"
DATA_DIR    = "/home/user/workspace/research_output/data"

outpath = f"{RESULTS_DIR}/exp_20_lronly_2022_{SEED_START}_{SEED_END-1}.json"
if os.path.exists(outpath):
    print(f"[SKIP] {outpath} already exists.")
    sys.exit(0)

train_returns = np.load(f"{DATA_DIR}/real_train_returns.npy")
test_returns  = np.load(f"{DATA_DIR}/real_test_returns.npy")   # 2022-2024 period

with open(f"{RESULTS_DIR}/exp_05_results.json") as f: meta = json.load(f)
N      = meta["n_assets"]
WINDOW = meta["window"]

with open(f"{RESULTS_DIR}/exp_10_results.json") as f: r10 = json.load(f)
TIMESTEPS = r10["timesteps"]

# Load regime data for 2022 period
regime_path = f"{DATA_DIR}/real_test_regimes.npy"
if os.path.exists(regime_path):
    test_regimes = np.load(regime_path)
else:
    # Compute simple bull/bear from 60-day rolling mean
    ew_ret = test_returns.mean(axis=1)
    test_regimes = np.ones(len(ew_ret), dtype=int)
    for t in range(len(ew_ret)):
        start = max(0, t-60)
        mu60 = ew_ret[start:t+1].mean() * 252
        test_regimes[t] = 1 if mu60 > 0 else 0

PPO_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=256, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    policy_kwargs={"net_arch": [128, 128]},
    verbose=0,
)

class LROnlyEnv(gym.Env):
    def __init__(self, returns, window, n_assets):
        super().__init__()
        self.returns = returns
        self.window  = window
        self.n       = n_assets
        self.T       = len(returns) - 1
        obs_dim = window * n_assets + n_assets
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(n_assets,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window
        self.weights = np.ones(self.n) / self.n
        return self._obs(), {}

    def _obs(self):
        hist = self.returns[self.t - self.window:self.t].flatten()
        return np.concatenate([hist, self.weights]).astype(np.float32)

    def step(self, action):
        w = np.abs(action)
        s = w.sum()
        w = w / s if s > 1e-8 else np.ones(self.n)/self.n
        r = self.returns[self.t]
        net = float(np.dot(w, r))
        reward = float(np.log1p(net))
        self.weights = w
        self.t += 1
        done = self.t >= self.T
        return self._obs(), reward, done, False, {}

def evaluate(model, returns, regimes, window, n):
    env = LROnlyEnv(returns, window, n)
    obs, _ = env.reset()
    port_rets = []
    bull_rets, bear_rets = [], []
    weights_prev = np.ones(n)/n
    t = window
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        w = np.abs(action); s = w.sum()
        w = w / s if s > 1e-8 else np.ones(n)/n
        r_t = returns[t]
        net = float(np.dot(w, r_t))
        port_rets.append(net)
        regime = int(regimes[t]) if t < len(regimes) else 1
        if regime == 1:
            bull_rets.append(net)
        else:
            bear_rets.append(net)
        weights_prev = w
        t += 1
        if done:
            break
    port_rets = np.array(port_rets)
    cr = float(np.prod(1 + port_rets) - 1)
    sr = float(port_rets.mean() / (port_rets.std(ddof=1) + 1e-9) * np.sqrt(252))
    def sr_of(arr):
        arr = np.array(arr)
        if len(arr) < 2: return 0.0
        return float(arr.mean() / (arr.std(ddof=1) + 1e-9) * np.sqrt(252))
    return sr, cr, sr_of(bull_rets), sr_of(bear_rets)

print(f"LR-only 2022 | seeds {SEED_START}-{SEED_END-1}")
print(f"  Train: {train_returns.shape}, Test: {test_returns.shape}")

sharpes, cumrets, bull_srs, bear_srs = [], [], [], []
for seed in SEEDS:
    t0 = time.time()
    train_env = LROnlyEnv(train_returns, WINDOW, N)
    model = PPO(env=train_env, seed=seed, **PPO_KWARGS)
    model.learn(total_timesteps=TIMESTEPS)
    sr, cr, bull, bear = evaluate(model, test_returns, test_regimes, WINDOW, N)
    sharpes.append(sr); cumrets.append(cr); bull_srs.append(bull); bear_srs.append(bear)
    print(f"  seed={seed} SR={sr:.3f} CR={cr:.3f} bull={bull:.3f} bear={bear:.3f} ({time.time()-t0:.1f}s)")

mean_sr = float(np.mean(sharpes))
print(f"  => mean SR={mean_sr:.4f}  std={float(np.std(sharpes,ddof=1)):.4f}")

out = {
    'config': 'lr_only', 'test_period': '2022',
    'seeds': SEEDS, 'sharpes': sharpes, 'cumrets': cumrets,
    'bull_sharpes': bull_srs, 'bear_sharpes': bear_srs,
    'mean_sr': mean_sr,
}
with open(outpath, 'w') as f:
    json.dump(out, f, indent=2)
print(f"Saved -> {outpath}")
