from __future__ import annotations
from typing import Dict, Tuple, Iterable, Callable, Any
import random
import numpy as np

Action = int  # 0:up,1:down,2:left,3:right


# utilities 

def _argmax_random_tie(qs):
    m = max(qs)
    ids = [i for i, x in enumerate(qs) if abs(x - m) < 1e-12]
    return random.choice(ids)

def _actions(env, s):

    return list(env.actions(s))

def _states(env):

    return list(env.all_states())


# DP: Value Iteration

def value_iteration(
    env: Any,
    gamma: float = 1.0,
    theta: float = 1e-6,
    samples_per_backup: int = 16,
) -> Tuple[Dict, Dict]:
    
    states = _states(env)
    V = {s: 0.0 for s in states}

    while True:
        delta = 0.0
        for s in states:
            acts = _actions(env, s)
            if not acts:
                continue
            qvals = []
            for a in acts:
                tot = 0.0
                for _ in range(samples_per_backup):
                    ns, r, done = env.step(s, a)
                    tot += r + (0.0 if done else gamma * V[ns])
                qvals.append(tot / samples_per_backup)
            v_new = max(qvals)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break

    pi = {}
    for s in states:
        acts = _actions(env, s)
        if not acts:
            continue
        qvals = []
        for a in acts:
            tot = 0.0
            for _ in range(max(32, samples_per_backup)):
                ns, r, done = env.step(s, a)
                tot += r + (0.0 if done else gamma * V[ns])
            qvals.append(tot / max(32, samples_per_backup))
        pi[s] = _argmax_random_tie(qvals)
    return V, pi


# MC: On-Policy 

def mc_on_policy(
    env: Any,
    episodes: int = 2000,
    gamma: float = 1.0,
    eps: float = 0.1,
    max_steps: int = 200,
    first_visit: bool = True,
) -> Tuple[Dict, Dict]:
    """
    First-visit (or every-visit) Monte Carlo control with ε-greedy behavior/target.
    Returns state-value V and greedy policy π.
    """
    states = _states(env)
    Q = {(s, a): 0.0 for s in states for a in _actions(env, s)}
    N = {(s, a): 0 for s in states for a in _actions(env, s)}

    def policy(s):
        acts = _actions(env, s)
        if not acts:
            return None
        if random.random() < eps:
            return random.choice(acts)
        qs = [Q[(s, a)] for a in acts]
        return _argmax_random_tie(qs)

    for _ in range(episodes):
        s = env.reset()
        traj = []
        for _step in range(max_steps):
            a = policy(s)
            ns, r, done = env.step(s, a)
            traj.append((s, a, r))
            s = ns
            if done:
                break

        G = 0.0
        seen = set()
        for t in reversed(range(len(traj))):
            s, a, r = traj[t]
            G = gamma * G + r
            key = (s, a)
            if first_visit:
                if key in seen:
                    continue
                seen.add(key)
            N[key] += 1
            Q[key] += (G - Q[key]) / N[key]

    pi = {}
    V = {}
    for s in states:
        acts = _actions(env, s)
        if not acts:
            continue
        qs = [Q[(s, a)] for a in acts]
        V[s] = max(qs)
        pi[s] = _argmax_random_tie(qs)
    return V, pi


# MC: Off-Policy

def mc_off_policy(
    env: Any,
    episodes: int = 4000,
    gamma: float = 1.0,
    b_eps: float = 0.2,
    weighted: bool = True,
    max_steps: int = 200,
) -> Tuple[Dict, Dict]:
    """
    Off-policy MC control. Target π is greedy w.r.t. Q (deterministic).
    Behavior b is ε-soft with ε=b_eps. Importance sampling can be ordinary
    (weighted=False) or weighted (weighted=True).
    """
    states = _states(env)
    Q = {(s, a): 0.0 for s in states for a in _actions(env, s)}
    C = {(s, a): 0.0 for s in states for a in _actions(env, s)}  

    def greedy_a(s):
        acts = _actions(env, s)
        if not acts:
            return None
        qs = [Q[(s, a)] for a in acts]
        return _argmax_random_tie(qs)

    for _ in range(episodes):

        s = env.reset()
        episode = []
        for _step in range(max_steps):
            acts = _actions(env, s)
            if not acts:
                a = None
            else:
                if random.random() < b_eps:
                    a = random.choice(acts)
                else:
                    a = greedy_a(s)
            ns, r, done = env.step(s, a)
            episode.append((s, a, r, acts))
            s = ns
            if done:
                break

        G = 0.0
        W = 1.0
        for t in reversed(range(len(episode))):
            s, a, r, acts = episode[t]
            G = gamma * G + r
            if a is None:
                break
            b_prob = (b_eps / len(acts)) + (1 - b_eps if a == greedy_a(s) else 0.0)
            W *= 1.0 / b_prob  # target prob is 1 for greedy, 0 otherwise
            if weighted:
                C[(s, a)] += W
                Q[(s, a)] += (W / max(C[(s, a)], 1e-12)) * (G - Q[(s, a)])
            else:

                alpha = 1.0 / (C[(s, a)] + 1.0)
                Q[(s, a)] += alpha * W * (G - Q[(s, a)])
                C[(s, a)] += 1.0
            if a != greedy_a(s):
                break

    pi = {}
    V = {}
    for s in states:
        acts = _actions(env, s)
        if not acts:
            continue
        qs = [Q[(s, a)] for a in acts]
        V[s] = max(qs)
        pi[s] = _argmax_random_tie(qs)
    return V, pi


# TD(0): On-Policy SARSA

def sarsa(
    env: Any,
    episodes: int = 3000,
    alpha: float = 0.5,
    gamma: float = 1.0,
    eps: float = 0.1,
    max_steps: int = 200,
) -> Tuple[Dict, Dict]:
    """On-policy TD(0) control (SARSA) with ε-greedy policy."""
    states = _states(env)
    Q = {(s, a): 0.0 for s in states for a in _actions(env, s)}

    def eps_greedy(s):
        acts = _actions(env, s)
        if not acts:
            return None
        if random.random() < eps:
            return random.choice(acts)
        qs = [Q[(s, a)] for a in acts]
        return _argmax_random_tie(qs)

    for _ in range(episodes):
        s = env.reset()
        a = eps_greedy(s)
        for _step in range(max_steps):
            ns, r, done = env.step(s, a)
            if done:
                Q[(s, a)] += alpha * (r - Q[(s, a)])
                break
            na = eps_greedy(ns)
            td_target = r + gamma * Q[(ns, na)]
            Q[(s, a)] += alpha * (td_target - Q[(s, a)])
            s, a = ns, na

    pi = {}
    V = {}
    for s in states:
        acts = _actions(env, s)
        if not acts:
            continue
        qs = [Q[(s, a)] for a in acts]
        V[s] = max(qs)
        pi[s] = _argmax_random_tie(qs)
    return V, pi


# TD(0): Off-Policy 

def offpolicy_sarsa_is(
    env: Any,
    episodes: int = 4000,
    alpha: float = 0.3,
    gamma: float = 1.0,
    b_eps: float = 0.2,
    weighted: bool = False,
    max_steps: int = 200,
) -> Tuple[Dict, Dict]:
    """Off-policy SARSA using per-decision importance sampling."""
    states = _states(env)
    Q = {(s, a): 0.0 for s in states for a in _actions(env, s)}
    C = {(s, a): 0.0 for s in states for a in _actions(env, s)}  # for weighted

    def greedy_a(s):
        acts = _actions(env, s)
        if not acts:
            return None
        qs = [Q[(s, a)] for a in acts]
        return _argmax_random_tie(qs)

    def behavior(s):
        acts = _actions(env, s)
        if not acts:
            return None
        if random.random() < b_eps:
            return random.choice(acts)
        return greedy_a(s)

    for _ in range(episodes):
        s = env.reset()
        a = behavior(s)
        rho = 1.0
        for _step in range(max_steps):
            ns, r, done = env.step(s, a)
            if done:
                if weighted:
                    C[(s, a)] += rho
                    Q[(s, a)] += (rho / max(C[(s, a)], 1e-12)) * (r - Q[(s, a)])
                else:
                    Q[(s, a)] += alpha * rho * (r - Q[(s, a)])
                break

            an = greedy_a(ns)   # target
            ab = behavior(ns)   # behavior

            acts = _actions(env, s)
            b_prob = (b_eps / len(acts)) + (1 - b_eps if a == greedy_a(s) else 0.0)
            pi_prob = 1.0 if a == greedy_a(s) else 0.0
            w = 0.0 if b_prob == 0 else pi_prob / b_prob
            rho *= w

            td_target = r + gamma * Q[(ns, an)]
            if weighted:
                C[(s, a)] += rho
                Q[(s, a)] += (rho / max(C[(s, a)], 1e-12)) * (td_target - Q[(s, a)])
            else:
                Q[(s, a)] += alpha * rho * (td_target - Q[(s, a)])
            s, a = ns, ab

    pi = {}
    V = {}
    for s in states:
        acts = _actions(env, s)
        if not acts:
            continue
        qs = [Q[(s, a)] for a in acts]
        V[s] = max(qs)
        pi[s] = _argmax_random_tie(qs)
    return V, pi
