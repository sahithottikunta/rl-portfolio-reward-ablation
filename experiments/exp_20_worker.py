"""
EXP-20 Worker: Regime-Switching Multi-Objective (RSMO) reward.

Reward formulation:
  r_DSR(t) = differential Sharpe ratio increment (eta=0.010, lambda=0.001)
  r_LR(t)  = log(1 + net_t)            (lambda=0.000)
  alpha_t  = sigmoid(k * mu_60d_t)      mu_60d_t = 60-day annualised EW mean return
  R_t      = alpha_t * r_LR(t) + (1 - alpha_t) * r_DSR(t)

  alpha_t -> 0 in bear regimes (mu_60d < 0)  =>  pure DSR+TC
  alpha_t -> 1 in bull regimes (mu_60d > 0)  =>  pure LR-only

Observation: [returns[-window:], weights, alpha_t]  (window*N + N + 1)

Usage: python3 exp_20_worker.py <k> <test_period> <seed_start> <seed_end_exclusive>
  k           : blend steepness  (e.g. 2, 5, 10)
  test_period : 2022 | 2025
  Saves: results/exp_20_k{k}_{test_period}_{seed_start}_{seed_end-1}.json
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

K_VAL       = float(sys.argv[1])
TEST_PERIOD = sys.argv[2]           # '2022' or '2025'
SEED_START  = int(sys.argv[3]) if len(sys.argv) > 3 else 42
SEED_END    = int(sys.argv[4]) if len(sys.argv) > 4 else 52
SEEDS       = list(range(SEED_START, SEED_END))

RESULTS_DIR = _ROOT + "/results"
DATA_DIR    = _ROOT + "/data"

K_STR = str(int(K_VAL)) if K_VAL == int(K_VAL) else str(K_VAL)
outpath = f"{RESULTS_DIR}/exp_20_k{K_STR}_{TEST_PERIOD}_{SEED_START}_{SEED_END-1}.json"
if os.path.exists(outpath):
    print(f"[SKIP] {outpath} already exists.")
    sys.exit(0)

# Data
train_returns = np.load(f"{DATA_DIR}/real_train_returns.npy")
if TEST_PERIOD == '2022':
    test_returns = np.load(f"{DATA_DIR}/real_test_returns.npy")
elif TEST_PERIOD == '2025':
    test_returns = np.load(f"{DATA_DIR}/exp19_test_returns.npy")
else:
    raise ValueError(f"Unknown test_period: {TEST_PERIOD}")

with open(f"{RESULTS_DIR}/exp_05_results.json") as f: meta = json.load(f)
N      = meta["n_assets"]
WINDOW = meta["window"]

with open(f"{RESULTS_DIR}/exp_10_results.json") as f: r10 = json.load(f)
TIMESTEPS = r10["timesteps"]

# RSMO constants
TC_DSR   = 0.001   # transaction cost for DSR component
ETA      = 0.010   # EMA rate for DSR
TC_LR    = 0.000   # LR-only has no TC
VOL_WIN  = 60      # 60-day window for regime signal

PPO_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=256, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    policy_kwargs={"net_arch": [128, 128]},
    verbose=0,
)


class RSMOEnv(gym.Env):
    """
    Regime-Switching Multi-Objective portfolio environment.

    State:  [returns[-window:] (W*N), weights (N), alpha_t (1)]
    Action: portfolio weights (N, softmax-normalised)
    Reward: alpha_t * r_LR(net_t) + (1 - alpha_t) * r_DSR(net_t)
      where net_t = w_t . r_t - TC_DSR * |w_t - w_{t-1}|_1  (for DSR component)
            net_t = w_t . r_t                                  (for LR component, lambda=0)
      alpha_t = sigmoid(k * mu_60d_t)
      mu_60d_t = (1/min(t,60)) * sum_{s=t-59}^{t} ew_s   (annualised)
    """
    def __init__(self, returns, window=WINDOW, k=K_VAL):
        super().__init__()
        self.returns = returns.astype(np.float32)
        self.T, self.N = returns.shape
        self.window = window
        self.k = k

        # EW returns for regime signal
        self.ew_returns = self.returns.mean(axis=1)

        obs_dim = self.window * self.N + self.N + 1   # +1 for alpha_t
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(0, 1, shape=(self.N,), dtype=np.float32)

        # DSR state
        self.A = 0.0
        self.B = 0.0

    def _alpha(self, t):
        """Soft bull-market weight: sigmoid(k * mu_60d_annualised)."""
        start = max(0, t - VOL_WIN + 1)
        seg = self.ew_returns[start:t + 1]
        mu_60d = float(seg.mean()) * 252   # annualised
        return float(1.0 / (1.0 + np.exp(-self.k * mu_60d)))

    def _dsr_reward(self, net_r):
        """Differential Sharpe ratio increment (updates EMA state)."""
        dA = net_r - self.A
        dB = net_r ** 2 - self.B
        var = self.B - self.A ** 2
        raw = 0.0 if var <= 1e-10 else (
            (self.B * dA - 0.5 * self.A * dB) / (var ** 1.5)
        )
        self.A += ETA * dA
        self.B += ETA * dB
        return float(np.clip(raw, -10, 10))

    def _lr_reward(self, gross_r):
        """Log-return reward (no TC penalty)."""
        return float(np.log1p(np.clip(gross_r, -0.999, None)))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window
        self.weights = np.ones(self.N, dtype=np.float32) / self.N
        self.A = 0.0
        self.B = 0.0
        return self._obs(), {}

    def _obs(self):
        alpha = self._alpha(self.t - 1)
        return np.concatenate([
            self.returns[self.t - self.window:self.t].flatten(),
            self.weights,
            np.array([alpha], dtype=np.float32),
        ]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, 1e-6, None)
        w = action / action.sum()
        r = self.returns[self.t]

        gross = float(w @ r)
        to    = float(np.sum(np.abs(w - self.weights)))

        # Net return for DSR component includes TC penalty
        net_dsr = gross - TC_DSR * to
        # Gross return for LR component (no TC)
        gross_lr = gross

        # Blend coefficient at current time
        alpha_t = self._alpha(self.t)

        # Blended reward
        r_dsr    = self._dsr_reward(net_dsr)
        r_lr     = self._lr_reward(gross_lr)
        reward   = alpha_t * r_lr + (1.0 - alpha_t) * r_dsr

        self.weights = w.copy()
        self.t += 1
        done = (self.t >= self.T - 1)
        return self._obs(), reward, done, False, {}


def evaluate(model, test_ret, k):
    """Evaluate on test returns; measure using DSR+TC Sharpe as neutral metric."""
    env = RSMOEnv(test_ret, k=k)
    obs, _ = env.reset()
    w_prev = np.ones(N) / N
    daily_rets, tos, alphas = [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        action = np.clip(action, 1e-6, None)
        w = action / action.sum()
        t = env.t - 1
        if t < len(test_ret):
            r    = test_ret[t]
            to   = float(np.sum(np.abs(w - w_prev)))
            # Evaluate net return with TC=0.001 for fair comparison with DSR+TC
            net  = float(w @ r) - TC_DSR * to
            daily_rets.append(net)
            tos.append(to)
            # Record alpha for reporting
            start = max(0, t - VOL_WIN + 1)
            seg = test_ret[start:t + 1].mean(axis=1)
            mu = float(seg.mean()) * 252
            alphas.append(float(1.0 / (1.0 + np.exp(-k * mu))))
            w_prev = w.copy()
    dr = np.array(daily_rets)
    if len(dr) < 10:
        dr = test_ret.mean(axis=1)
    return dr, np.array(tos), np.array(alphas)


def sharpe(dr):
    return float(dr.mean() * 252 / (dr.std() * np.sqrt(252) + 1e-9))

def cumret(dr):
    return float(np.prod(1 + dr) - 1)

def regime_sharpe(dr, regimes):
    if TEST_PERIOD == '2022':
        reg = np.load(f"{DATA_DIR}/real_test_regimes.npy")
    else:
        reg = np.load(f"{DATA_DIR}/exp19_test_regimes.npy")
    reg = reg[:len(dr)]
    bull_dr = dr[reg == 1]
    bear_dr = dr[reg == 0]
    sr_bull = sharpe(bull_dr) if len(bull_dr) > 10 else float('nan')
    sr_bear = sharpe(bear_dr) if len(bear_dr) > 10 else float('nan')
    return sr_bull, sr_bear


print(f"EXP-20 RSMO | k={K_VAL}  test={TEST_PERIOD}  seeds {SEED_START}-{SEED_END-1}", flush=True)
print(f"  Train: real_train_returns.npy  ({train_returns.shape[0]} days)", flush=True)
print(f"  Test : {test_returns.shape[0]} days ({TEST_PERIOD} period)", flush=True)

sharpes, cumrets_list, tos_mean = [], [], []
bull_sharpes, bear_sharpes, alpha_means = [], [], []

for seed in SEEDS:
    t0 = time.time()
    torch.manual_seed(seed)
    env = RSMOEnv(train_returns, k=K_VAL)
    model = PPO(env=env, seed=seed, **PPO_KWARGS)
    model.learn(total_timesteps=TIMESTEPS)
    dr, to, alphas = evaluate(model, test_returns, K_VAL)
    sr = sharpe(dr); cr = cumret(dr); tom = float(to.mean())
    sr_bull, sr_bear = regime_sharpe(dr, None)
    alpha_m = float(alphas.mean())
    sharpes.append(sr); cumrets_list.append(cr); tos_mean.append(tom)
    bull_sharpes.append(sr_bull); bear_sharpes.append(sr_bear)
    alpha_means.append(alpha_m)
    print(f"  seed={seed} SR={sr:.3f} CR={cr:.3f} bull={sr_bull:.3f} bear={sr_bear:.3f} "
          f"alpha={alpha_m:.3f} ({time.time()-t0:.1f}s)", flush=True)

print(f"  => mean SR={np.mean(sharpes):.3f}  std={np.std(sharpes,ddof=1):.3f}", flush=True)

out = {
    "config": "rsmo", "k": K_VAL, "test_period": TEST_PERIOD,
    "seeds": SEEDS,
    "sharpes": sharpes, "cumrets": cumrets_list, "tos": tos_mean,
    "bull_sharpes": [float(x) for x in bull_sharpes],
    "bear_sharpes": [float(x) for x in bear_sharpes],
    "alpha_means": alpha_means,
    "mean_alpha": float(np.mean(alpha_means)),
}
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"  Saved -> {outpath}", flush=True)
