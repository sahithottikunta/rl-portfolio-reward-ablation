"""
EXP-18 Worker: n=150 confirmatory run for mean-optimal cell (lambda=0.0005, eta0=0.020)
and baseline (lambda=0.001, eta0=0.010).

Usage: python3 exp_18_worker.py <lambda> <eta0> <seed_start> <seed_end_exclusive>
  Saves: results/exp_18_tc{lambda}_eta{eta0}_{seed_start}_{seed_end-1}.json
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

TC_STR  = sys.argv[1]
ETA_STR = sys.argv[2]
TC_VAL  = float(TC_STR)
ETA_VAL = float(ETA_STR)
SEED_START = int(sys.argv[3]) if len(sys.argv) > 3 else 117
SEED_END   = int(sys.argv[4]) if len(sys.argv) > 4 else 127
SEEDS      = list(range(SEED_START, SEED_END))

RESULTS_DIR = _ROOT + "/results"
DATA_DIR    = _ROOT + "/data"

outpath = f"{RESULTS_DIR}/exp_18_tc{TC_STR}_eta{ETA_STR}_{SEED_START}_{SEED_END-1}.json"
if os.path.exists(outpath):
    print(f"[SKIP] {outpath} already exists.")
    sys.exit(0)

train_returns = np.load(f"{DATA_DIR}/real_train_returns.npy")
test_returns  = np.load(f"{DATA_DIR}/real_test_returns.npy")

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


def evaluate(model, test_ret):
    env = DSRTCEnv(test_ret)
    obs, _ = env.reset()
    w_prev = np.ones(N) / N
    daily_rets, tos = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        action = np.clip(action, 1e-6, None)
        w = action / action.sum()
        t = env.t - 1
        if t < len(test_ret):
            r  = test_ret[t]
            to = float(np.sum(np.abs(w - w_prev)))
            net= float(w @ r) - TC_VAL * to
            daily_rets.append(net); tos.append(to)
            w_prev = w.copy()
    dr = np.array(daily_rets)
    if len(dr) < 10: dr = test_ret.mean(axis=1)
    return dr, np.array(tos)

def sharpe(dr):
    return float(dr.mean() * 252 / (dr.std() * np.sqrt(252) + 1e-9))
def cumret(dr):
    return float(np.prod(1 + dr) - 1)

print(f"EXP-18 | lambda={TC_VAL}  eta0={ETA_VAL}  seeds {SEED_START}-{SEED_END-1}", flush=True)
sharpes, cumrets, tos_mean = [], [], []

for seed in SEEDS:
    t0 = time.time()
    torch.manual_seed(seed)
    env = DSRTCEnv(train_returns)
    model = PPO(env=env, seed=seed, **PPO_KWARGS)
    model.learn(total_timesteps=TIMESTEPS)
    dr, to = evaluate(model, test_returns)
    sr = sharpe(dr); cr = cumret(dr); tom = float(to.mean())
    sharpes.append(sr); cumrets.append(cr); tos_mean.append(tom)
    print(f"  seed={seed} SR={sr:.3f} CR={cr:.3f} TO={tom:.3f}  ({time.time()-t0:.1f}s)", flush=True)

print(f"  => mean SR={np.mean(sharpes):.3f}  std={np.std(sharpes,ddof=1):.3f}", flush=True)

out = {"tc": TC_VAL, "tc_str": TC_STR, "eta0": ETA_VAL, "eta_str": ETA_STR,
       "seeds": SEEDS, "sharpes": sharpes, "cumrets": cumrets, "tos": tos_mean}
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"  Saved -> {outpath}", flush=True)
