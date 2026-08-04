"""
EXP-12: Seed Expansion — n=3 → n=10 for Adaptive η Configs
============================================================
Runs seeds 45–51 (7 new seeds) for three configurations:
  1. Fixed DSR+TC   (eta=0.01, kappa=N/A)
  2. Adaptive DSR+TC kappa=30
  3. Adaptive DSR+TC kappa=50

Seeds 42–44 already exist in exp_07 and exp_10 results.
After running, this script aggregates to n=10 per config and computes:
  - Two-sample Welch t-test (adaptive vs. fixed)
  - Cohen's d (effect size)
  - Achieved statistical power (post-hoc, via scipy)

Output: exp_12_results.json
"""
import os as _os
# Repo-relative root; override with the RESEARCH_ROOT environment variable.
_ROOT = _os.environ.get("RESEARCH_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import json, os, time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import torch
from scipy import stats

RESULTS_DIR = _ROOT + "/results"
DATA_DIR    = _ROOT + "/data"

train_returns = np.load(f"{DATA_DIR}/real_train_returns.npy")
test_returns  = np.load(f"{DATA_DIR}/real_test_returns.npy")

with open(f"{RESULTS_DIR}/exp_05_results.json") as f: meta = json.load(f)
with open(f"{RESULTS_DIR}/exp_07_results.json") as f: r7   = json.load(f)
with open(f"{RESULTS_DIR}/exp_10_results.json") as f: r10  = json.load(f)

N      = meta["n_assets"]
WINDOW = meta["window"]
TC     = meta["tc"]
ETA0   = r10["eta0"]
VOLWIN = r10["volwin"]
KAPPAS_TO_RUN = [30, 50]
NEW_SEEDS  = [45, 46, 47, 48, 49, 50, 51]   # 7 new seeds → total n=10
TIMESTEPS  = r10["timesteps"]

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

# ── Environments (same as exp_07 and exp_10) ──────────────────────────────────
class FixedEtaPortfolioEnv(gym.Env):
    def __init__(self, returns, window=WINDOW, tc=TC,
                 reward_type="dsr_tc", eta=0.01):
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
            self.returns[self.t-self.window:self.t].flatten(),
            self.weights
        ]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, 1e-6, None)
        w = action / action.sum()
        r = self.returns[self.t]
        gross = float(w @ r)
        to    = float(np.sum(np.abs(w - self.weights)))
        net   = gross - self.tc * to
        reward = self._dsr(net) if self.reward_type == "dsr_tc" else net
        self.weights = w.copy()
        self.t += 1
        done = (self.t >= self.T - 1)
        return self._obs(), reward, done, False, {}


class AdaptiveEtaPortfolioEnv(gym.Env):
    def __init__(self, returns, window=WINDOW, tc=TC,
                 reward_type="dsr_tc", eta0=ETA0, kappa=30, volwin=VOLWIN):
        super().__init__()
        self.returns     = returns.astype(np.float32)
        self.T, self.N   = returns.shape
        self.window      = window
        self.tc          = tc
        self.reward_type = reward_type
        self.eta0        = eta0
        self.kappa       = kappa
        self.volwin      = volwin
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
            self.returns[self.t-self.window:self.t].flatten(),
            self.weights
        ]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, 1e-6, None)
        w = action / action.sum()
        r = self.returns[self.t]
        gross = float(w @ r)
        to    = float(np.sum(np.abs(w - self.weights)))
        net   = gross - self.tc * to
        reward = self._dsr(net, float(r.mean()))
        self.weights = w.copy()
        self.t += 1
        done = (self.t >= self.T - 1)
        return self._obs(), reward, done, False, {}


# ── Evaluation ────────────────────────────────────────────────────────────────
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
            r   = test_ret[t]
            to  = float(np.sum(np.abs(w - w_prev)))
            net = float(w @ r) - TC * to
            daily_rets.append(net); tos.append(to)
            w_prev = w.copy()
    dr = np.array(daily_rets)
    if len(dr) < 10:
        dr = test_ret.mean(axis=1)  # EW fallback
    return dr, np.array(tos)

def sharpe(dr):
    return float(dr.mean()*252 / (dr.std()*np.sqrt(252)+1e-9))
def cumret(dr):
    return float(np.prod(1+dr)-1)

# ── Training configs ──────────────────────────────────────────────────────────
CONFIGS = [
    ("fixed_dsr_tc",   FixedEtaPortfolioEnv,    dict(reward_type="dsr_tc", eta=0.01)),
    ("adaptive_k30",   AdaptiveEtaPortfolioEnv,  dict(reward_type="dsr_tc", kappa=30)),
    ("adaptive_k50",   AdaptiveEtaPortfolioEnv,  dict(reward_type="dsr_tc", kappa=50)),
]

new_results = {}   # config_name -> list of sharpe values for new seeds
total_elapsed = 0

for cfg_name, EnvClass, env_kwargs in CONFIGS:
    print(f"\n{'='*65}")
    print(f"Config: {cfg_name}")
    seed_sharpes = []

    for seed in NEW_SEEDS:
        t0 = time.time()
        torch.manual_seed(seed)

        train_env = EnvClass(train_returns, **env_kwargs)
        model = PPO(env=train_env, seed=seed, **PPO_KWARGS)
        model.learn(total_timesteps=TIMESTEPS)

        dr, to = evaluate(model, EnvClass, test_returns, **env_kwargs)
        sr = sharpe(dr)
        seed_sharpes.append(sr)

        elapsed = time.time() - t0
        total_elapsed += elapsed
        print(f"  Seed {seed}: SR={sr:.3f}  CumRet={cumret(dr):.3f}  ({elapsed:.1f}s)")

    new_results[cfg_name] = seed_sharpes
    print(f"  New seeds mean={np.mean(seed_sharpes):.3f}  std={np.std(seed_sharpes):.3f}")

print(f"\nTotal training time: {total_elapsed:.0f}s")

# ── Aggregate n=10 per config ─────────────────────────────────────────────────
print("\n" + "="*65)
print("AGGREGATING n=10 RESULTS")
print("="*65)

# Pull existing n=3 samples
old_fixed   = r7["results"]["dsr_tc"]["sharpe_samples"]          # seeds 42,43,44
old_k30     = r10["results"]["dsr_tc_k30"]["sharpe_samples"]     # seeds 42,43,44
old_k50     = r10["results"]["dsr_tc_k50"]["sharpe_samples"]     # seeds 42,43,44

all_fixed   = old_fixed + new_results["fixed_dsr_tc"]   # n=10
all_k30     = old_k30   + new_results["adaptive_k30"]   # n=10
all_k50     = old_k50   + new_results["adaptive_k50"]   # n=10

print(f"\nSeed samples (n=10):")
print(f"  fixed_dsr_tc: {[round(x,3) for x in all_fixed]}")
print(f"  adaptive_k30: {[round(x,3) for x in all_k30]}")
print(f"  adaptive_k50: {[round(x,3) for x in all_k50]}")

def summary_stats(samples, label):
    arr = np.array(samples)
    return {
        "label":  label,
        "n":      len(arr),
        "mean":   float(arr.mean()),
        "std":    float(arr.std(ddof=1)),
        "median": float(np.median(arr)),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "samples": samples,
    }

stats_fixed = summary_stats(all_fixed, "fixed_dsr_tc")
stats_k30   = summary_stats(all_k30,   "adaptive_k30")
stats_k50   = summary_stats(all_k50,   "adaptive_k50")

print(f"\n{'Config':<18} {'Mean':>7} {'Std':>7} {'Median':>8} {'Min':>7} {'Max':>7}")
print("-"*60)
for s in [stats_fixed, stats_k30, stats_k50]:
    print(f"{s['label']:<18} {s['mean']:>7.3f} {s['std']:>7.3f} "
          f"{s['median']:>8.3f} {s['min']:>7.3f} {s['max']:>7.3f}")

# ── Statistical tests ─────────────────────────────────────────────────────────
print("\n=== STATISTICAL TESTS (n=10 vs n=10) ===")

def cohen_d(a, b):
    """Cohen d for two independent samples (pooled std)."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.var(a,ddof=1) + (nb-1)*np.var(b,ddof=1)) / (na+nb-2))
    return float((np.mean(a) - np.mean(b)) / (pooled + 1e-9))

def welch_t(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)

def power_posthoc(a, b):
    """Post-hoc power: proportion of 10k bootstrap t-tests that reject H0 at alpha=0.05."""
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n_bootstrap = 10_000
    rejections = 0
    for _ in range(n_bootstrap):
        s1 = rng.choice(a, size=len(a), replace=True)
        s2 = rng.choice(b, size=len(b), replace=True)
        _, p = stats.ttest_ind(s1, s2, equal_var=False)
        if p < 0.05:
            rejections += 1
    return rejections / n_bootstrap

test_results = {}
for label, adap_samples in [("k30", all_k30), ("k50", all_k50)]:
    d    = cohen_d(adap_samples, all_fixed)
    t, p = welch_t(adap_samples, all_fixed)
    pwr  = power_posthoc(np.array(adap_samples), np.array(all_fixed))
    test_results[label] = {
        "cohen_d":   d,
        "t_stat":    t,
        "p_value":   p,
        "power":     pwr,
        "n":         10,
        "df_approx": 18,  # Welch approx for equal n
    }
    sig = "✓ p<0.05" if p < 0.05 else ("~ p<0.10" if p < 0.10 else "✗ not sig")
    effect = "large" if abs(d)>0.8 else ("medium" if abs(d)>0.5 else "small")
    print(f"\n  Adaptive {label} vs. Fixed DSR+TC:")
    print(f"    Cohen d = {d:+.3f}  ({effect} effect)")
    print(f"    Welch t = {t:.3f},  p = {p:.4f}  {sig}")
    print(f"    Post-hoc bootstrap power = {pwr:.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
exp12_out = {
    "new_seeds":       NEW_SEEDS,
    "new_results":     new_results,
    "aggregated_n10": {
        "fixed_dsr_tc":  stats_fixed,
        "adaptive_k30":  stats_k30,
        "adaptive_k50":  stats_k50,
    },
    "statistical_tests": test_results,
    "timesteps":   TIMESTEPS,
    "total_seeds": 10,
}

with open(f"{RESULTS_DIR}/exp_12_results.json", "w") as f:
    json.dump(exp12_out, f, indent=2)

print("\nEXP-12 complete. Results saved to exp_12_results.json")
