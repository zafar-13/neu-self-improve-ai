# Part-1

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

class MDP:
    def reset(self): ...
    def step(self, state, action) -> Tuple[Any, float, bool, Dict]: ...
    def actions(self, state) -> List[int]: ...

class StatelessBanditMDP(MDP):
    def __init__(self, k: int, rng: np.random.Generator):
        self.k = k
        self.rng = rng
        self.q_star = self.rng.normal(0.0, 1.0, size=k)
        self._state = 0
    def reset(self):
        self._state = 0
        return self._state
    def step(self, state, action):
        reward = float(self.rng.normal(self.q_star[action], 1.0))
        return self._state, reward, False, {}
    def actions(self, state):
        return list(range(self.k))

@dataclass
class EpsilonGreedyAgent:
    n_actions: int
    epsilon: float
    rng: np.random.Generator
    def __post_init__(self):
        self.Q = np.zeros(self.n_actions, dtype=float)
        self.N = np.zeros(self.n_actions, dtype=int)
    def reset(self):
        self.Q[:] = 0.0
        self.N[:] = 0
    def select_action(self, state) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        # random tie-break
        maxQ = self.Q.max()
        idx = np.flatnonzero(np.isclose(self.Q, maxQ))
        return int(self.rng.choice(idx))
    def observe(self, state, action, reward, next_state, done):
        self.N[action] += 1
        alpha = 1.0 / self.N[action]
        self.Q[action] += alpha * (reward - self.Q[action])


def simulate_epsilon_greedy_vectorized(
    k=10, steps=1000, runs=5000, epsilons=(0.0, 0.01, 0.1), seed=0
):
    rng = np.random.default_rng(seed)
    out = {}
    for eps in epsilons:
        q_star = rng.normal(0.0, 1.0, size=(runs, k))
        optimal = np.argmax(q_star, axis=1)
        Q = np.zeros((runs, k))
        N = np.zeros((runs, k), dtype=int)
        avg_rew = np.zeros(steps)
        pct_opt = np.zeros(steps)
        for t in range(steps):
            explore = rng.random(runs) < eps
            a_rand = rng.integers(0, k, size=runs)
            greedy = np.argmax(Q + rng.uniform(-1e-8, 1e-8, size=(runs, k)), axis=1)
            A = np.where(explore, a_rand, greedy)
            idx = (np.arange(runs), A)
            rewards = rng.normal(q_star[idx], 1.0)
            N[idx] += 1
            Q[idx] += (rewards - Q[idx]) / N[idx]
            avg_rew[t] = rewards.mean()
            pct_opt[t] = (A == optimal).mean() * 100.0
        out[eps] = {"avg_reward": avg_rew, "pct_optimal": pct_opt}
    return out

def plot_results(results):
    steps = len(next(iter(results.values()))["avg_reward"])
    x = np.arange(1, steps + 1)
    plt.figure(figsize=(7, 4.5))
    for eps in sorted(results.keys()):
        plt.plot(x, results[eps]["avg_reward"], label=f"ε={eps}")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bandit_avg_reward.png", dpi=160)
    plt.figure(figsize=(7, 4.5))
    for eps in sorted(results.keys()):
        plt.plot(x, results[eps]["pct_optimal"], label=f"ε={eps}")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig("bandit_pct_optimal.png", dpi=160)

if __name__ == "__main__":
    res = simulate_epsilon_greedy_vectorized()
    plot_results(res)
    print("Saved: bandit_avg_reward.png, bandit_pct_optimal.png")

# Part-2

import numpy as np
import matplotlib.pyplot as plt


def simulate_epsilon_greedy_vectorized(k=10, steps=1000, runs=5000,
                                       epsilons=(0.0, 0.01, 0.1), seed=42):
    rng = np.random.default_rng(seed)
    out = {}
    q_star = rng.normal(0.0, 1.0, size=(runs, k))  
    optimal = np.argmax(q_star, axis=1)
    for eps in epsilons:
        Q = np.zeros((runs, k))
        N = np.zeros((runs, k), dtype=int)
        avg_rew = np.zeros(steps)
        pct_opt = np.zeros(steps)
        for t in range(steps):
            explore = rng.random(runs) < eps
            a_rand = rng.integers(0, k, size=runs)
            greedy = np.argmax(Q + rng.uniform(-1e-8, 1e-8, size=(runs, k)), axis=1)
            A = np.where(explore, a_rand, greedy)
            idx = (np.arange(runs), A)
            rewards = rng.normal(q_star[idx], 1.0)
            N[idx] += 1
            Q[idx] += (rewards - Q[idx]) / N[idx]
            avg_rew[t] = rewards.mean()
            pct_opt[t] = (A == optimal).mean() * 100.0
        out[f"ε={eps}"] = {"avg_reward": avg_rew, "pct_optimal": pct_opt}
    return out, q_star, optimal


def _softmax_rows(H):
    Hs = H - H.max(axis=1, keepdims=True)
    expH = np.exp(Hs)
    return expH / expH.sum(axis=1, keepdims=True)

def _sample_from_rows(probs, rng):
    cum = np.cumsum(probs, axis=1)
    r = rng.random(probs.shape[0])[:, None]
    return (cum >= r).argmax(axis=1)

def simulate_gradient_bandit_vectorized(k=10, steps=1000, runs=5000,
                                        alphas=(0.1, 0.4), baselines=(False, True),
                                        seed=123, q_star=None):
    rng = np.random.default_rng(seed)
    if q_star is None:
        q_star = rng.normal(0.0, 1.0, size=(runs, k))
    optimal = np.argmax(q_star, axis=1)
    results = {}
    for use_baseline in baselines:
        for alpha in alphas:
            H = np.zeros((runs, k))
            pi = np.full((runs, k), 1.0/k)      
            baseline = np.zeros(runs)            
            tcount = np.zeros(runs, dtype=int)
            avg_rew = np.zeros(steps)
            pct_opt = np.zeros(steps)
            for t in range(steps):
                pi = _softmax_rows(H)
                A = _sample_from_rows(pi, rng)
                idx = (np.arange(runs), A)
                rewards = rng.normal(q_star[idx], 1.0)
                tcount += 1
                if use_baseline:
                    baseline += (rewards - baseline) / tcount
                    adv = rewards - baseline
                else:
                    adv = rewards
                H -= alpha * (adv[:, None] * pi)
                H[idx] += alpha * adv * (1.0 - pi[idx])
                avg_rew[t] = rewards.mean()
                pct_opt[t] = (A == optimal).mean() * 100.0
            tag = f"Gradient α={alpha} " + ("(baseline)" if use_baseline else "(no baseline)")
            results[tag] = {"avg_reward": avg_rew, "pct_optimal": pct_opt}
    return results


if __name__ == "__main__":
    eps_res, q_star_fixed, _ = simulate_epsilon_greedy_vectorized()
    grad_res = simulate_gradient_bandit_vectorized(q_star=q_star_fixed)
    results = {**eps_res, **grad_res}
    x = np.arange(1, len(next(iter(results.values()))["avg_reward"]) + 1)
    # Average reward
    plt.figure(figsize=(8.5, 5))
    for label in sorted(results.keys()):
        plt.plot(x, results[label]["avg_reward"], label=label)
    plt.xlabel("Steps"); plt.ylabel("Average reward"); plt.legend(); plt.tight_layout()
    plt.savefig("bandit_avg_reward_with_gradient.png", dpi=160)
    # % Optimal action
    plt.figure(figsize=(8.5, 5))
    for label in sorted(results.keys()):
        plt.plot(x, results[label]["pct_optimal"], label=label)
    plt.xlabel("Steps"); plt.ylabel("% Optimal action"); plt.ylim(0, 100)
    plt.legend(); plt.tight_layout()
    plt.savefig("bandit_pct_optimal_with_gradient.png", dpi=160)
    print("Saved: bandit_avg_reward_with_gradient.png, bandit_pct_optimal_with_gradient.png")

