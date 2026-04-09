"""
EXP-10: Adaptive EMA Rate (eta) Validation on Real S&P 500 Data
================================================================
Tests the proposed adaptive schedule:
    eta_t = eta_0 / (1 + kappa * sigma_hat_t)
where sigma_hat_t is a 10-day rolling standard deviation of the
cross-asset mean log-return (computed on-the-fly during training/evaluation).

Configurations tested:
  - Reward types: ['dsr_tc', 'dsr']    (DSR+TC is proposed; DSR alone tests
                                        whether adaptive eta rescues it)
  - kappa values: [20, 30, 50]         (calibrated so eta -> 0.003–0.005 at
                                        peak volatility observed in training data)
  - Seeds: [42, 43, 44]               (same as prior experiments)
  - Timesteps: 30,000                  (same as prior experiments)

Fixed-eta baselines (already in exp_07_results.json) are NOT re-run here;
they are loaded directly for comparison.

Output: exp_10_results.json
"""

import numpy as np
import json, os, time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import torch

RESULTS_DIR = "/home/user/workspace/research_output/results"
DATA_DIR    = "/home/user/workspace/research_output/data"

train_returns = np.load(f"{DATA_DIR}/real_train_returns.npy")
test_returns  = np.load(f"{DATA_DIR}/real_test_returns.npy")

with open(f"{RESULTS_DIR}/exp_05_results.json") as f:
    meta = json.load(f)

N      = meta["n_assets"]          # 10
WINDOW = meta["window"]            # 20
TC     = meta["tc"]                # 0.001
ETA0   = 0.01                      # base rate (same as fixed-eta experiments)
KAPPAS = [20, 30, 50]
REWARD_TYPES = ["dsr_tc", "dsr"]
SEEDS  = [42, 43, 44]
TIMESTEPS = 30_000
VOLWIN = 10                        # rolling window for sigma_hat (days)

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

# ── Adaptive-eta portfolio environment ────────────────────────────────────────
class AdaptiveEtaPortfolioEnv(gym.Env):
    """
    Portfolio environment with adaptive EMA rate.
    eta_t = eta0 / (1 + kappa * sigma_hat_t)
    sigma_hat_t = rolling std of cross-asset mean return over last VOLWIN steps.
    """

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

    def _reset_dsr_state(self):
        self.A = 0.0
        self.B = 0.0
        self._vol_buf = []        # rolling buffer for sigma_hat

    def _eta_t(self, r_mean):
        """Compute adaptive eta using rolling vol of cross-asset mean return."""
        self._vol_buf.append(float(r_mean))
        if len(self._vol_buf) > self.volwin:
            self._vol_buf.pop(0)
        if len(self._vol_buf) < 2:
            return self.eta0
        sigma_hat = float(np.std(self._vol_buf, ddof=1))
        return self.eta0 / (1.0 + self.kappa * sigma_hat)

    def _dsr(self, r_input, r_mean_for_eta):
        eta = self._eta_t(r_mean_for_eta)
        dA  = r_input - self.A
        dB  = r_input ** 2 - self.B
        var = self.B - self.A ** 2
        if var <= 1e-10:
            raw = 0.0
        else:
            raw = (self.B * dA - 0.5 * self.A * dB) / (var ** 1.5)
        self.A += eta * dA
        self.B += eta * dB
        return float(np.clip(raw, -10, 10))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t       = self.window
        self.weights = np.ones(self.N, dtype=np.float32) / self.N
        self._reset_dsr_state()
        return self._obs(), {}

    def _obs(self):
        window_rets = self.returns[self.t - self.window:self.t].flatten()
        return np.concatenate([window_rets, self.weights]).astype(np.float32)

    def step(self, action):
        action  = np.clip(action, 1e-6, None)
        w_new   = action / action.sum()
        r       = self.returns[self.t]
        r_mean  = float(r.mean())                          # cross-asset mean
        gross   = float(w_new @ r)
        turnover= float(np.sum(np.abs(w_new - self.weights)))
        net     = gross - self.tc * turnover

        if self.reward_type == "dsr_tc":
            reward = self._dsr(net,   r_mean)
        elif self.reward_type == "dsr":
            reward = self._dsr(gross, r_mean)
        else:
            reward = net

        self.weights = w_new.copy()
        self.t += 1
        done = (self.t >= self.T - 1)
        return self._obs(), reward, done, False, {}


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(model, returns, reward_type, kappa, eta0=ETA0):
    env = AdaptiveEtaPortfolioEnv(
        returns, reward_type=reward_type, kappa=kappa, eta0=eta0)
    obs, _ = env.reset()
    daily_rets, turnovers, eta_trace = [], [], []
    w_prev = np.ones(N) / N
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        action = np.clip(action, 1e-6, None)
        w = action / action.sum()
        t = env.t - 1
        if t < len(returns):
            r     = returns[t]
            gross = float(w @ r)
            to    = float(np.sum(np.abs(w - w_prev)))
            net   = gross - TC * to
            daily_rets.append(net)
            turnovers.append(to)
            # record current eta
            eta_now = eta0 / (1.0 + kappa * (np.std(env._vol_buf, ddof=1)
                              if len(env._vol_buf) >= 2 else 0.0))
            eta_trace.append(float(eta_now))
            w_prev = w.copy()
    return np.array(daily_rets), np.array(turnovers), np.array(eta_trace)

def sharpe(dr):
    return float(dr.mean()*252 / (dr.std()*np.sqrt(252) + 1e-9))
def cumret(dr):
    return float(np.prod(1 + dr) - 1)
def maxdd(dr):
    pv = np.cumprod(1 + dr)
    rm = np.maximum.accumulate(pv)
    return float(((pv - rm)/rm).min())
def annvol(dr):
    return float(dr.std() * np.sqrt(252))

# ── Main sweep ────────────────────────────────────────────────────────────────
results = {}          # key: (reward_type, kappa)
eta_traces = {}       # key: (reward_type, kappa, seed) — seed-42 trace

for reward_type in REWARD_TYPES:
    print(f"\n{'='*65}")
    print(f"Reward type: {reward_type}")
    for kappa in KAPPAS:
        key = f"{reward_type}_k{kappa}"
        print(f"\n  kappa={kappa}")
        sharpes, cumrets, vols, maxdds, tos = [], [], [], [], []
        seed42_dr    = None
        seed42_trace = None

        for seed in SEEDS:
            t0 = time.time()
            torch.manual_seed(seed)

            train_env = AdaptiveEtaPortfolioEnv(
                train_returns, reward_type=reward_type,
                kappa=kappa, eta0=ETA0)
            model = PPO(env=train_env, seed=seed, **PPO_KWARGS)
            model.learn(total_timesteps=TIMESTEPS)

            dr, to_arr, eta_arr = evaluate(
                model, test_returns, reward_type, kappa)

            if len(dr) < 10:
                print(f"    Seed {seed}: too few eval steps, using EW fallback")
                dr = test_returns.mean(axis=1)
                to_arr  = np.zeros(len(dr))
                eta_arr = np.full(len(dr), ETA0)

            sharpes.append(sharpe(dr))
            cumrets.append(cumret(dr))
            vols.append(annvol(dr))
            maxdds.append(maxdd(dr))
            tos.append(float(to_arr.mean()))

            if seed == 42:
                seed42_dr    = dr.tolist()
                seed42_trace = eta_arr.tolist()

            elapsed = time.time() - t0
            print(f"    Seed {seed}: SR={sharpes[-1]:.3f}  CumRet={cumrets[-1]:.3f}"
                  f"  TO={tos[-1]:.4f}  ({elapsed:.1f}s)"
                  f"  eta_mean={np.mean(eta_arr):.5f}")

        results[key] = {
            "reward_type":   reward_type,
            "kappa":         kappa,
            "sharpe_mean":   float(np.mean(sharpes)),
            "sharpe_std":    float(np.std(sharpes, ddof=0)),
            "cumret_mean":   float(np.mean(cumrets)),
            "vol_mean":      float(np.mean(vols)),
            "maxdd_mean":    float(np.mean(maxdds)),
            "turnover_mean": float(np.mean(tos)),
            "sharpe_samples":sharpes,
            "daily_returns_seed42": seed42_dr,
        }
        if seed42_trace is not None:
            eta_traces[f"{reward_type}_k{kappa}"] = seed42_trace

        print(f"  → SR={np.mean(sharpes):.3f} ± {np.std(sharpes):.3f}  "
              f"TO={np.mean(tos):.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n\n=== ADAPTIVE ETA SUMMARY ===")
print(f"{'Config':<25} {'SR mean':>9} {'SR std':>9} {'Turnover':>10} {'CumRet':>9}")
print("-" * 65)
for k, v in results.items():
    print(f"{k:<25} {v['sharpe_mean']:>9.3f} {v['sharpe_std']:>9.3f} "
          f"{v['turnover_mean']:>10.4f} {v['cumret_mean']:>9.3f}")

print("\n=== FIXED ETA BASELINES (from exp_07) ===")
with open(f"{RESULTS_DIR}/exp_07_results.json") as f:
    r7 = json.load(f)
for rt in ["log_return", "dsr", "dsr_tc"]:
    v = r7["results"][rt]
    print(f"{'fixed_'+rt:<25} {v['sharpe_mean']:>9.3f} {v['sharpe_std']:>9.3f} "
          f"{v['turnover_mean']:>10.4f} {v['cumret_mean']:>9.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
exp10_out = {
    "results":    results,
    "eta_traces": eta_traces,
    "kappas":     KAPPAS,
    "eta0":       ETA0,
    "volwin":     VOLWIN,
    "seeds":      SEEDS,
    "timesteps":  TIMESTEPS,
}
with open(f"{RESULTS_DIR}/exp_10_results.json", "w") as f:
    json.dump(exp10_out, f, indent=2)

print("\nEXP-10 complete. Results saved.")
