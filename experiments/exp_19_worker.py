"""
EXP-19 Worker: Post-2024 out-of-sample evaluation.

Trains on real_train_returns.npy (2015-2021), evaluates on exp19_test_returns.npy (2025-2026).
Runs all 4 reward configurations:
  R1: DSR+TC   (lambda=0.001, eta=0.010)  -- baseline / best real-market config
  R2: DSR-only (lambda=0.000, eta=0.010)
  R3: LR+TC    (lambda=0.001, eta=None)   -- log-return + TC penalty
  R4: LR-only  (lambda=0.000, eta=None)

Usage: python3 exp_19_worker.py <config_name> <seed_start> <seed_end_exclusive>
  config_name: dsr_tc | dsr_only | lr_tc | lr_only
  Saves: results/exp_19_{config}_{seed_start}_{seed_end-1}.json
  Skips if output file already exists.
"""
import os as _os
# Repo-relative root; override with the RESEARCH_ROOT environment variable.
_ROOT = _os.environ.get("RESEARCH_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, json, time, os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

CONFIG_NAME = sys.argv[1]
SEED_START  = int(sys.argv[2]) if len(sys.argv) > 2 else 42
SEED_END    = int(sys.argv[3]) if len(sys.argv) > 3 else 52
SEEDS       = list(range(SEED_START, SEED_END))

RESULTS_DIR = _ROOT + "/results"
DATA_DIR    = _ROOT + "/data"

outpath = f"{RESULTS_DIR}/exp_19_{CONFIG_NAME}_{SEED_START}_{SEED_END-1}.json"
if os.path.exists(outpath):
    print(f"[SKIP] {outpath} already exists.")
    sys.exit(0)

# Config parameters
CONFIGS = {
    'dsr_tc':   {'tc': 0.001, 'eta': 0.010, 'use_dsr': True},
    'dsr_only': {'tc': 0.000, 'eta': 0.010, 'use_dsr': True},
    'lr_tc':    {'tc': 0.001, 'eta': None,  'use_dsr': False},
    'lr_only':  {'tc': 0.000, 'eta': None,  'use_dsr': False},
}
assert CONFIG_NAME in CONFIGS, f"Unknown config: {CONFIG_NAME}"
cfg = CONFIGS[CONFIG_NAME]
TC_VAL  = cfg['tc']
ETA_VAL = cfg['eta']
USE_DSR = cfg['use_dsr']

# Load data
train_returns = np.load(f"{DATA_DIR}/real_train_returns.npy")
test_returns  = np.load(f"{DATA_DIR}/exp19_test_returns.npy")   # 2025-2026

with open(f"{RESULTS_DIR}/exp_05_results.json") as f: meta = json.load(f)
N      = meta["n_assets"]
WINDOW = meta["window"]

with open(f"{RESULTS_DIR}/exp_10_results.json") as f: r10 = json.load(f)
TIMESTEPS = r10["timesteps"]

PPO_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=256, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    policy_kwargs={"net_arch": [128, 128]},
    verbose=0,
)


class DSRTCEnv(gym.Env):
    """DSR reward with optional TC penalty."""
    def __init__(self, returns, window=WINDOW, tc=TC_VAL, eta=ETA_VAL):
        super().__init__()
        self.returns = returns.astype(np.float32)
        self.T, self.N = returns.shape
        self.window = window; self.tc = tc; self.eta = eta
        obs_dim = self.window * self.N + self.N
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(0, 1, shape=(self.N,), dtype=np.float32)
        self.A = 0.0; self.B = 0.0

    def _dsr(self, r):
        dA = r - self.A; dB = r**2 - self.B
        var = self.B - self.A**2
        raw = 0.0 if var <= 1e-10 else (self.B*dA - 0.5*self.A*dB) / (var**1.5)
        self.A += self.eta * dA; self.B += self.eta * dB
        return float(np.clip(raw, -10, 10))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window
        self.weights = np.ones(self.N, dtype=np.float32) / self.N
        self.A = 0.0; self.B = 0.0
        return self._obs(), {}

    def _obs(self):
        return np.concatenate([
            self.returns[self.t-self.window:self.t].flatten(), self.weights
        ]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, 1e-6, None)
        w = action / action.sum()
        r = self.returns[self.t]
        gross = float(w @ r)
        to    = float(np.sum(np.abs(w - self.weights)))
        net   = gross - self.tc * to
        reward = self._dsr(net)
        self.weights = w.copy(); self.t += 1
        done = (self.t >= self.T - 1)
        return self._obs(), reward, done, False, {}


class LRTCEnv(gym.Env):
    """Log-return reward with optional TC penalty."""
    def __init__(self, returns, window=WINDOW, tc=TC_VAL):
        super().__init__()
        self.returns = returns.astype(np.float32)
        self.T, self.N = returns.shape
        self.window = window; self.tc = tc
        obs_dim = self.window * self.N + self.N
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(0, 1, shape=(self.N,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window
        self.weights = np.ones(self.N, dtype=np.float32) / self.N
        return self._obs(), {}

    def _obs(self):
        return np.concatenate([
            self.returns[self.t-self.window:self.t].flatten(), self.weights
        ]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, 1e-6, None)
        w = action / action.sum()
        r = self.returns[self.t]
        gross = float(w @ r)
        to    = float(np.sum(np.abs(w - self.weights)))
        net   = gross - self.tc * to
        reward = float(np.log1p(np.clip(net, -0.999, None)))
        self.weights = w.copy(); self.t += 1
        done = (self.t >= self.T - 1)
        return self._obs(), reward, done, False, {}


def make_train_env():
    if USE_DSR:
        return DSRTCEnv(train_returns, tc=TC_VAL, eta=ETA_VAL)
    else:
        return LRTCEnv(train_returns, tc=TC_VAL)


def evaluate(model):
    """Evaluate trained model on 2025-2026 test returns."""
    if USE_DSR:
        env = DSRTCEnv(test_returns, tc=TC_VAL, eta=ETA_VAL)
    else:
        env = LRTCEnv(test_returns, tc=TC_VAL)
    obs, _ = env.reset()
    w_prev = np.ones(N) / N
    daily_rets, tos, weights_hist = [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        action = np.clip(action, 1e-6, None)
        w = action / action.sum()
        t = env.t - 1
        if t < len(test_returns):
            r   = test_returns[t]
            to  = float(np.sum(np.abs(w - w_prev)))
            net = float(w @ r) - TC_VAL * to
            daily_rets.append(net)
            tos.append(to)
            weights_hist.append(w.copy())
            w_prev = w.copy()
    dr = np.array(daily_rets)
    if len(dr) < 10:
        dr = test_returns.mean(axis=1)
    return dr, np.array(tos), np.array(weights_hist)


def sharpe(dr):
    return float(dr.mean() * 252 / (dr.std() * np.sqrt(252) + 1e-9))

def cumret(dr):
    return float(np.prod(1 + dr) - 1)

def regime_sharpe(dr, regimes):
    bull_mask = regimes[:len(dr)] == 1
    bear_mask = ~bull_mask
    bull_dr = dr[bull_mask]
    bear_dr = dr[bear_mask]
    sr_bull = sharpe(bull_dr) if len(bull_dr) > 10 else float('nan')
    sr_bear = sharpe(bear_dr) if len(bear_dr) > 10 else float('nan')
    return sr_bull, sr_bear


# Load regimes for test period
regimes = np.load(f"{DATA_DIR}/exp19_test_regimes.npy")

print(f"EXP-19 | config={CONFIG_NAME}  tc={TC_VAL}  eta={ETA_VAL}  use_dsr={USE_DSR}", flush=True)
print(f"  Train: real_train_returns.npy  ({train_returns.shape[0]} days, 2015-2021)", flush=True)
print(f"  Test : exp19_test_returns.npy  ({test_returns.shape[0]} days, 2025-2026)", flush=True)

sharpes, cumrets, tos_mean = [], [], []
bull_sharpes, bear_sharpes = [], []

for seed in SEEDS:
    t0 = time.time()
    torch.manual_seed(seed)
    env = make_train_env()
    model = PPO(env=env, seed=seed, **PPO_KWARGS)
    model.learn(total_timesteps=TIMESTEPS)
    dr, to, _ = evaluate(model)
    sr = sharpe(dr); cr = cumret(dr); tom = float(to.mean())
    sr_bull, sr_bear = regime_sharpe(dr, regimes)
    sharpes.append(sr); cumrets.append(cr); tos_mean.append(tom)
    bull_sharpes.append(sr_bull); bear_sharpes.append(sr_bear)
    print(f"  seed={seed} SR={sr:.3f} CR={cr:.3f} bull={sr_bull:.3f} bear={sr_bear:.3f} ({time.time()-t0:.1f}s)", flush=True)

print(f"  => mean SR={np.mean(sharpes):.3f}  std={np.std(sharpes,ddof=1):.3f}", flush=True)

out = {
    "config": CONFIG_NAME, "tc": TC_VAL, "eta": ETA_VAL, "use_dsr": USE_DSR,
    "seeds": SEEDS,
    "sharpes": sharpes, "cumrets": cumrets, "tos": tos_mean,
    "bull_sharpes": [float(x) for x in bull_sharpes],
    "bear_sharpes": [float(x) for x in bear_sharpes],
    "test_period": "2025-01-03 to 2026-04-02",
}
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"  Saved -> {outpath}", flush=True)
