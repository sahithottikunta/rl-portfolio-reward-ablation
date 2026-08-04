"""
EXP-13 Master runner: runs all pending batches sequentially,
skipping any that already have a saved results file.
Saves each (config, seed_range) batch immediately on completion.
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

RESULTS_DIR = _ROOT + "/results"
DATA_DIR    = _ROOT + "/data"

train_returns = np.load(f"{DATA_DIR}/real_train_returns.npy")
test_returns  = np.load(f"{DATA_DIR}/real_test_returns.npy")

with open(f"{RESULTS_DIR}/exp_05_results.json") as f: meta = json.load(f)
with open(f"{RESULTS_DIR}/exp_10_results.json") as f: r10  = json.load(f)

N       = meta["n_assets"]
WINDOW  = meta["window"]
TC      = meta["tc"]
ETA0    = r10["eta0"]
VOLWIN  = r10["volwin"]
TIMESTEPS = r10["timesteps"]

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

class FixedEtaPortfolioEnv(gym.Env):
    def __init__(self, returns, window=WINDOW, tc=TC, reward_type="dsr_tc", eta=0.01):
        super().__init__()
        self.returns     = returns.astype(np.float32)
        self.T, self.N   = returns.shape
        self.window      = window; self.tc = tc
        self.reward_type = reward_type; self.eta = eta
        obs_dim = self.window * self.N + self.N
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(0, 1, shape=(self.N,), dtype=np.float32)
        self.A = 0.0; self.B = 0.0

    def _dsr(self, r):
        dA = r - self.A; dB = r**2 - self.B
        var = self.B - self.A**2
        raw = 0.0 if var <= 1e-10 else (self.B*dA - 0.5*self.A*dB) / (var**1.5)
        self.A += self.eta*dA; self.B += self.eta*dB
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
        reward = self._dsr(net) if self.reward_type == "dsr_tc" else net
        self.weights = w.copy(); self.t += 1
        return self._obs(), reward, (self.t >= self.T - 1), False, {}


class AdaptiveEtaPortfolioEnv(gym.Env):
    def __init__(self, returns, window=WINDOW, tc=TC, reward_type="dsr_tc",
                 eta0=ETA0, kappa=30, volwin=VOLWIN):
        super().__init__()
        self.returns     = returns.astype(np.float32)
        self.T, self.N   = returns.shape
        self.window      = window; self.tc = tc
        self.reward_type = reward_type; self.eta0 = eta0
        self.kappa       = kappa; self.volwin = volwin
        obs_dim = self.window * self.N + self.N
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(0, 1, shape=(self.N,), dtype=np.float32)

    def _reset_state(self):
        self.A = 0.0; self.B = 0.0; self._vol_buf = []

    def _eta(self, r_mean):
        self._vol_buf.append(float(r_mean))
        if len(self._vol_buf) > self.volwin: self._vol_buf.pop(0)
        sig = float(np.std(self._vol_buf, ddof=1)) if len(self._vol_buf) >= 2 else 0.0
        return self.eta0 / (1.0 + self.kappa * sig)

    def _dsr(self, r_input, r_mean):
        eta = self._eta(r_mean)
        dA  = r_input - self.A; dB = r_input**2 - self.B
        var = self.B - self.A**2
        raw = 0.0 if var <= 1e-10 else (self.B*dA - 0.5*self.A*dB)/(var**1.5)
        self.A += eta*dA; self.B += eta*dB
        return float(np.clip(raw, -10, 10))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window
        self.weights = np.ones(self.N, dtype=np.float32) / self.N
        self._reset_state()
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
        reward = self._dsr(net, float(r.mean()))
        self.weights = w.copy(); self.t += 1
        return self._obs(), reward, (self.t >= self.T - 1), False, {}


def evaluate(model, env_class, test_ret, **env_kwargs):
    env = env_class(test_ret, **env_kwargs)
    obs, _ = env.reset()
    daily_rets, tos = [], []
    w_prev = np.ones(N) / N
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
            net= float(w @ r) - TC * to
            daily_rets.append(net); tos.append(to)
            w_prev = w.copy()
    dr = np.array(daily_rets)
    if len(dr) < 10: dr = test_ret.mean(axis=1)
    return dr, np.array(tos)

def sharpe(dr):
    return float(dr.mean()*252 / (dr.std()*np.sqrt(252)+1e-9))
def cumret(dr):
    return float(np.prod(1+dr)-1)

CONFIGS_MAP = {
    "fixed_dsr_tc": (FixedEtaPortfolioEnv,   dict(reward_type="dsr_tc", eta=0.01)),
    "adaptive_k30": (AdaptiveEtaPortfolioEnv, dict(reward_type="dsr_tc", kappa=30)),
    "adaptive_k50": (AdaptiveEtaPortfolioEnv, dict(reward_type="dsr_tc", kappa=50)),
}

# All batches needed: 3 configs × 3 batches = 9 total
# Seeds 52–61 (A), 62–71 (B), 72–81 (C)
BATCHES = [
    ("fixed_dsr_tc", 52, 62),
    ("fixed_dsr_tc", 62, 72),
    ("fixed_dsr_tc", 72, 82),
    ("adaptive_k30",  52, 62),
    ("adaptive_k30",  62, 72),
    ("adaptive_k30",  72, 82),
    ("adaptive_k50",  52, 62),
    ("adaptive_k50",  62, 72),
    ("adaptive_k50",  72, 82),
]

grand_start = time.time()

for (cfg_name, seed_start, seed_end) in BATCHES:
    seeds = list(range(seed_start, seed_end))
    tag   = f"{seed_start}_{seed_end-1}"
    outpath = f"{RESULTS_DIR}/exp_13_{cfg_name}_{seed_start}_{seed_end-1}.json"

    if os.path.exists(outpath):
        print(f"[SKIP] {cfg_name} seeds {seed_start}–{seed_end-1} already done.", flush=True)
        continue

    print(f"\n{'='*60}", flush=True)
    print(f"Config={cfg_name}  Seeds={seed_start}–{seed_end-1}", flush=True)
    EnvClass, env_kwargs = CONFIGS_MAP[cfg_name]
    seed_sharpes, seed_cumrets, seed_tos = [], [], []

    for seed in seeds:
        t0 = time.time()
        torch.manual_seed(seed)
        train_env = EnvClass(train_returns, **env_kwargs)
        model = PPO(env=train_env, seed=seed, **PPO_KWARGS)
        model.learn(total_timesteps=TIMESTEPS)
        dr, to = evaluate(model, EnvClass, test_returns, **env_kwargs)
        sr = sharpe(dr); cr = cumret(dr)
        seed_sharpes.append(sr); seed_cumrets.append(cr)
        seed_tos.append(float(to.mean()))
        print(f"  seed={seed} SR={sr:.3f} CR={cr:.3f} TO={to.mean():.3f} ({time.time()-t0:.1f}s)", flush=True)

    out = {"config": cfg_name, "seeds": seeds,
           "sharpes": seed_sharpes, "cumrets": seed_cumrets, "tos": seed_tos}
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {outpath}", flush=True)

print(f"\nAll batches done in {time.time()-grand_start:.0f}s", flush=True)
